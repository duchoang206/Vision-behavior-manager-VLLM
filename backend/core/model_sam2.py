import ctypes
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace

from core.model_track_masks import TrackMaskCache, bbox, boundary_polygons, box_iou, compatible_boxes, track_key
from core.model_label_prompts import LabelPromptTracks
from core.phase0_baseline import mark as baseline_mark

def reference_identity(identity, expected):
    if identity and identity.get("accepted") and expected and identity.get("label", "").casefold() != expected.casefold():
        return dict(identity, accepted=False, reason="reference_identity_mismatch", label=None, expected_label=expected)
    return identity


class ModelSAM2:
    def __init__(self, label_directory, reply, on_mask_ready=None):
        from core.model_label_session import ModelLabelSession
        self.labels = ModelLabelSession(reply)
        self.on_mask_ready = on_mask_ready
        self.label_directory = label_directory
        self.metric = None
        try:
            manifest = Path(label_directory) / "gallery.json"
            document = json.loads(manifest.read_text()) if manifest.exists() else {}
            self.initial_categories = {sample["category"] for sample in document.get("samples", [])}
            self.initial_require_labels = bool(document.get("require_labels"))
        except (OSError, ValueError, KeyError):
            self.initial_categories = set()
            self.initial_require_labels = True
        self.cache = TrackMaskCache()
        self.prompt_tracks = LabelPromptTracks()
        self.pending = {}
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.stopping = threading.Event()
        self.ready = False
        self.error = None
        self.completed = {}
        self.revoked = {}
        self.last_submit = {}
        self.dropped_stale = 0
        self.replaced_pending = 0
        self.worker_count = min(2, max(1, int(os.getenv('CUSTOM_SAM2_WORKERS', '2'))))
        self.camera_workers = {}
        self.worker_loads = [0] * self.worker_count
        self.worker_threads = []
        self.worker_errors = {}
        self.copy_stream = None
        self.thread = threading.Thread(target=self._run, name="model-sam2-cuda", daemon=True)
        self.thread.start()

    def status(self, camera_id):
        with self.lock:
            now = time.time() * 1000
            self.revoked = {key: expiry for key, expiry in self.revoked.items() if expiry > now}
            result = {"ready": self.ready, "error": self.error, "device": "cuda",
                      "pending_cameras": len(self.pending), "dropped_stale": self.dropped_stale,
                      "workers": self.worker_count, "worker_errors": dict(self.worker_errors),
                      "replaced_pending": self.replaced_pending, **self.completed.get(camera_id, {}),
                      "revoked_track_ids": [identifier for (camera, identifier) in self.revoked if camera == camera_id]}
        result["labels"] = self.metric.status() if self.metric else {"state": "starting"}
        return result

    def request(self, request):
        if not self.ready:
            self.labels.reply(dict(id=request["id"], error=self.error or "SAM2 GPU đang nạp; hãy thử lại."))
            return
        self.labels.request(request)
        self.wake.set()

    def activate_label(self, camera_id, sample_id):
        sample = self.metric.saved_sample(camera_id, sample_id) if self.metric else None
        if not sample:
            raise ValueError('Mẫu Label chưa được lưu hoặc đã bị xóa.')
        receipt = self.labels._get(self.labels.previews, sample_id, camera_id)['sample']
        now = time.time() * 1000
        captured_at = receipt.get('captured_at', 0)
        frame_info = receipt.get('frame_info', {})
        mode = 'tracking_queued'
        with self.lock:
            self.prompt_tracks.attempts.pop((camera_id, sample_id), None)
            if frame_info.get('model_id') and 0 <= now - captured_at <= self.prompt_tracks.max_age_ms:
                targets = self.prompt_tracks.candidates(camera_id, self.metric.prompt_samples(camera_id), frame_info, captured_at)
                target = next((obj for obj in targets if obj['label_prompt_id'] == sample_id), None)
                if target:
                    target = dict(target, **dict(zip(('x', 'y', 'w', 'h'), sample['bbox'])))
                    policy = self.metric.policy(sample['category'])
                    identity = dict(accepted=True, label=sample['label'], class_name=sample['class_name'],
                                    score=1., revision=policy['revision'], version=policy['version'], match_source='user_reference')
                    self.prompt_tracks.observe(camera_id, target, sample['mask'], identity, captured_at, sample['frame_id'])
                    self.cache.store(camera_id, target, sample['mask'], sample['frame_id'], captured_at, identity)
                    self.revoked.pop((camera_id, target['id']), None)
                    mode = 'mask_initialized'
        self.wake.set()
        return dict(mode=mode, sample_id=sample_id, label=sample['label'], source_frame_id=sample['frame_id'],
                    source_age_ms=round(max(0, now - captured_at)), requires_yolo_detection=False)

    def _take_pending(self, worker_id=None):
        with self.lock:
            if not self.pending:
                return None, None
            camera_id = next((identifier for identifier in self.pending
                              if worker_id is None or self.camera_workers[identifier] == worker_id), None)
            if camera_id is None:
                return None, None
            item = self.pending.pop(camera_id)
            if self.pending:
                self.wake.set()
            return camera_id, item

    def _publish_result(self, camera_id, frame_id, captured_at, detections, result):
        if time.time() * 1000 - captured_at > self.cache.max_age_ms:
            return
        obj, mask, identity = result
        with self.lock:
            if self.cache.has_newer(camera_id, obj, captured_at):
                return
            self.prompt_tracks.observe(camera_id, obj, mask, identity, captured_at, frame_id, detections)
            if identity and not identity['accepted']:
                self.cache.reject(camera_id, obj)
                self.revoked[(camera_id, obj['id'])] = time.time() * 1000 + 1500
            elif mask:
                self.cache.store(camera_id, obj, mask, frame_id, captured_at, identity)
                self.revoked.pop((camera_id, obj['id']), None)
        on_mask_ready = getattr(self, "on_mask_ready", None)
        if on_mask_ready:
            try:
                on_mask_ready(camera_id, frame_id, captured_at, obj, mask, identity)
            except Exception:
                logging.exception("SAM2 mask-ready callback failed")

    def attach(self, camera_id, objects, frame_id, now):
        if self.metric and hasattr(self, "prompt_tracks"):
            samples = self.metric.prompt_samples(camera_id)
            with self.lock:
                objects = objects + self.prompt_tracks.live(camera_id, samples, now, detections=objects)
                self.cache.attach(camera_id, objects, frame_id, now)
        else:
            self.cache.attach(camera_id, objects, frame_id, now)
        if not self.metric:
            return [obj for obj in objects if not self.initial_require_labels
                    and obj.get("category") not in self.initial_categories and not obj.get("requires_label_verification")]
        output = []
        for obj in objects:
            policy = self.metric.policy(obj.get("category"))
            identity = obj.pop("identity_verification", None)
            if policy["required"] or obj.get("requires_label_verification"):
                if (policy["error"] or not identity or not identity.get("accepted")
                        or not policy.get('continuity_revision', policy['revision']) <= identity['revision'] <= policy['revision']
                        or (identity.get("match_source") not in {"user_gallery", "user_reference"} and identity.get("version") != policy.get("version"))):
                    continue
                obj.update(label=identity["label"], identity_verified=True, identity_source="model_label_triplet",
                           identity_score=identity["score"], identity_revision=identity["revision"],
                           identity_match_source=identity.get("match_source", "triplet"),
                           identity_class=identity["class_name"], detector_class=obj["class"])
            obj.pop("requires_label_verification", None)
            output.append(obj)
        winners = {}
        for obj in output:
            if obj.get("identity_verified"):
                label = obj["label"].casefold()
                previous = winners.get(label)
                priority = lambda target: (bool(target.get("label_prompt_id")), target.get("identity_match_source") in {"user_gallery", "user_reference"}, target["identity_score"], target["confidence"])
                if not previous or priority(obj) > priority(previous):
                    winners[label] = obj
        return sorted((obj for obj in output if not obj.get("identity_verified") or winners[obj["label"].casefold()] is obj),
                      key=lambda obj: (obj.get("identity_match_source") == "user_gallery", bool(obj.get("identity_verified"))), reverse=True)

    def _verified_labels(self, camera_id, objects, now):
        policies = {obj.get("category"): self.metric.policy(obj.get("category")) for obj in objects} if self.metric else {}
        return self.cache.verified_labels(camera_id, objects, now, policies)

    def submit(self, buffer, batch_id, camera_id, frame_id, captured_at, objects, frame_info=None, snapshot_objects=None):
        targets = [dict(obj) for obj in objects if obj.get("category") in {"robot", "rack"}
                   or obj.get("requires_label_verification")
                   or (self.metric and self.metric.policy(obj.get("category"))["required"])]
        samples = self.metric.prompt_samples(camera_id) if self.metric else []
        with self.lock:
            needs_prompt = self.prompt_tracks.needs_frame(camera_id, samples, captured_at)
        if not self.ready or (not targets and not needs_prompt and not self.labels.needs_frame(camera_id)) or captured_at - self.last_submit.get(camera_id, 0) < 30:
            return
        import cupy
        import pyds

        baseline_mark(camera_id, frame_id, 'submit_enter', targets=len(targets))
        try:
            dtype, shape, strides, capsule, size = pyds.get_nvds_buf_surface_gpu(hash(buffer), batch_id)
            get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
            get_pointer.restype = ctypes.c_void_p
            get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
            memory = cupy.cuda.UnownedMemory(get_pointer(capsule, None), size, buffer)
            view = cupy.ndarray(shape, dtype=dtype, memptr=cupy.cuda.MemoryPointer(memory, 0), strides=strides)
            if self.copy_stream is None:
                self.copy_stream = cupy.cuda.Stream(non_blocking=True)
            with self.copy_stream as copy_stream:
                owned = view[:, :, :3].copy()
                copy_stream.synchronize()
            baseline_mark(camera_id, frame_id, 'gpu_copy_done')
            with self.lock:
                if camera_id not in self.camera_workers:
                    worker_id = min(range(self.worker_count), key=self.worker_loads.__getitem__)
                    self.camera_workers[camera_id] = worker_id
                    self.worker_loads[worker_id] += max(1, len(samples), len(targets))
                if camera_id in self.pending:
                    baseline_mark(camera_id, self.pending[camera_id][1], 'replaced_pending')
                    self.replaced_pending += 1
                baseline_mark(camera_id, frame_id, 'enqueued')
                self.pending[camera_id] = (owned, frame_id, captured_at, targets, frame_info,
                                          snapshot_objects if self.labels.needs_frame(camera_id) else None)
            self.last_submit[camera_id] = captured_at
            self.wake.set()
        except Exception as error:
            self.error = f"GPU surface: {error}"
            self.last_submit[camera_id] = captured_at + 1000
            logging.exception("SAM2 GPU surface")

    def _run(self):
        try:
            import torch
            runtime = CudaMaskRuntime(self.label_directory)
            self.metric = runtime.metric
            runtimes = [runtime]
            for _worker in range(1, self.worker_count):
                runtimes.append(CudaMaskRuntime(self.label_directory, metric=self.metric))
            initialized = torch.cuda.Event()
            initialized.record()
            for worker_id, worker_runtime in enumerate(runtimes):
                thread = threading.Thread(target=self._work, args=(worker_runtime, worker_id, initialized),
                                          name=f'model-sam2-cuda-{worker_id}', daemon=True)
                self.worker_threads.append(thread)
                thread.start()
            self.ready = True
            for thread in self.worker_threads:
                thread.join()
            return
        except Exception as error:
            self.error = str(error)
            logging.exception("Model SAM2 initialization")
            return

    def _work(self, runtime, worker_id, initialized):
        import torch

        inference_stream = torch.cuda.Stream(priority=-1)
        inference_stream.wait_event(initialized)
        while not self.stopping.is_set():
            if worker_id == 0:
                with torch.cuda.stream(inference_stream):
                    self.labels.service(runtime)
            camera_id, item = self._take_pending(worker_id)
            if item is None:
                self.wake.wait(.005)
                self.wake.clear()
                continue
            image, frame_id, captured_at, objects, frame_info, snapshot_objects = item
            if time.time() * 1000 - captured_at > 250:
                baseline_mark(camera_id, frame_id, 'dropped_stale')
                with self.lock:
                    self.dropped_stale += 1
                continue
            started = time.monotonic()
            dequeued_at = time.time() * 1000
            try:
                samples = self.metric.prompt_samples(camera_id)
                with self.lock:
                    prompts = self.prompt_tracks.candidates(camera_id, samples, frame_info or {}, captured_at) if frame_info else []
                detections = objects
                objects = prompts + objects
                with torch.cuda.stream(inference_stream):
                    if objects:
                        masks = runtime.segment(camera_id, image, objects, captured_at,
                            on_result=lambda result: self._publish_result(camera_id, frame_id, captured_at, detections, result))
                        segment_stats = dict(runtime.last_segment_stats)
                    else:
                        masks = []
                        segment_stats = {"backbone_inferences": 0, "decoder_inferences": 0, "propagated_masks": 0}
                    if self.labels.needs_frame(camera_id):
                        predictions = {obj["id"]: dict(obj) for obj in (snapshot_objects or objects)}
                        for obj, mask, identity in masks:
                            if obj["id"] in predictions and mask:
                                predictions[obj["id"]]["mask"] = dict(mask, frame_id=frame_id, observed_at=captured_at)
                        self.labels.capture(camera_id, image, frame_id, captured_at, list(predictions.values()), frame_info)
                with self.lock:
                    self.worker_errors.pop(worker_id, None)
                    self.error = next(iter(self.worker_errors.values()), None)
                    self.completed[camera_id] = {"frame_id": frame_id, "observed_at": captured_at,
                                                 "masks": sum(bool(mask) and (not identity or identity["accepted"]) for _, mask, identity in masks),
                                                 "identity_rejected": [dict(track_id=obj["local_id"], **identity) for obj, _, identity in masks if identity and not identity["accepted"]],
                                                 "targets": len(objects),
                                                 "label_prompts": len(prompts),
                                                 "capacity": runtime.max_tracks,
                                                 "backbone_inferences": segment_stats["backbone_inferences"],
                                                 "decoder_inferences": segment_stats["decoder_inferences"],
                                                 "propagated_masks": segment_stats["propagated_masks"],
                                                 "worker_id": worker_id,
                                                 "reference_memories": runtime.reference_status(),
                                                 "queue_wait_ms": round(max(0, dequeued_at - captured_at), 1),
                                                 "inference_ms": round((time.monotonic() - started) * 1000, 1),
                                                 "end_to_end_ms": round(max(0, time.time() * 1000 - captured_at), 1)}
            except Exception as error:
                with self.lock:
                    self.error = str(error)
                    self.worker_errors[worker_id] = str(error)
                logging.exception("SAM2 live model tracks")
                runtime.reset(camera_id)
                self.stopping.wait(.5)

    def stop(self):
        self.stopping.set()
        self.wake.set()
        self.thread.join(timeout=3)
        if self.metric:
            self.metric.stop()


class CudaMaskRuntime:
    def __init__(self, label_directory=None, metric=None):
        import torch
        from ultralytics.models.sam import SAM2Predictor

        if not torch.cuda.is_available():
            raise RuntimeError("SAM2 yêu cầu CUDA; không chuyển xử lý ảnh sang CPU.")
        model_path = os.getenv("CUSTOM_SAM2_MODEL", str(Path(__file__).resolve().parents[1] / "models/sam2.1_t.pt"))
        if not Path(model_path).is_file():
            raise RuntimeError(f"Thiếu SAM2 weights: {model_path}")
        torch.set_num_threads(2)
        torch.backends.cudnn.enabled = os.getenv("REGISTERED_MASK_CUDNN", "0") == "1"
        self.size = int(os.getenv("CUSTOM_SAM2_IMGSZ", "640"))
        self.max_tracks = max(1, int(os.getenv("CUSTOM_SAM2_MAX_TRACKS", "32")))
        self.options = dict(model=model_path, task="segment", mode="predict", device=0, quantize=16,
                            imgsz=self.size, verbose=False, conf=0.0, save=False)
        self.base = SAM2Predictor(overrides=self.options)
        self.base.setup_model(verbose=False)
        self.base.model.set_imgsz([self.size, self.size])
        self.entries = {}
        self.last_segment_stats = {"backbone_inferences": 0, "decoder_inferences": 0, "propagated_masks": 0}
        with open(model_path, "rb") as weights:
            self.signature = f"sam2-rgb-masked-v2:{hashlib.file_digest(weights, 'sha256').hexdigest()}:{self.size}"
        from core.model_reference_memory import LabelReferenceMemory
        self.references = LabelReferenceMemory(label_directory)
        self.reference_revision = -1
        self.reference_limit = min(3, max(1, int(os.getenv('CUSTOM_SAM2_REFERENCE_VIEWS', '2'))))
        self.reference_choices = {}
        self.reference_ids = set()
        self.rejected_tracks = {}
        self.metric = metric
        if label_directory and metric is None:
            from core.model_label_metric import ModelLabelMetric
            self.metric = ModelLabelMetric(label_directory, self.signature, start=False)
            path = Path(label_directory) / "gallery.json"
            self.metric.load(json.loads(path.read_text()) if path.exists() else {"revision": 0, "samples": []})
            self.metric.start()

    def _image(self, image):
        import torch
        import torch.nn.functional as functional
        tensor = torch.from_dlpack(image)
        height, width = tensor.shape[:2]
        ratio = min(self.size / height, self.size / width)
        resized = round(height * ratio), round(width * ratio)
        tensor = functional.interpolate(tensor.permute(2, 0, 1).unsqueeze(0).float(), size=resized, mode="bilinear", align_corners=False)
        tensor = functional.pad(tensor, (0, self.size - resized[1], 0, self.size - resized[0]), value=114)
        tensor = ((tensor - self.base.mean) / self.base.std).half()
        return tensor, self.base.model.forward_image(tensor), resized, (height, width)

    def _predictor(self, backbone, shape):
        from core.sam2_live_predictor import LiveSAM2Predictor
        predictor = LiveSAM2Predictor(overrides=self.options, max_obj_num=1)
        for attribute in ("model", "device", "mean", "std", "done_warmup", "torch_dtype"):
            setattr(predictor, attribute, getattr(self.base, attribute))
        predictor.imgsz = [self.size, self.size]
        predictor.batch = (None, [SimpleNamespace(shape=(*shape, 3))], None)
        predictor.backbone_out = backbone
        return predictor

    def _descriptor(self, backbone, binary, shape):
        _, features, _, sizes = self.base.model._prepare_backbone_features(backbone)
        features = features[-1].permute(1, 2, 0).reshape(1, -1, *sizes[-1])
        return self._descriptor_from_features(features, binary, shape)

    def _descriptor_from_features(self, features, binary, shape):
        from core.identity_metric import masked_descriptor
        return masked_descriptor(features, binary, shape, (self.size, self.size))

    def preview(self, image, box, points, labels):
        import torch
        with torch.inference_mode():
            tensor, backbone, resized, shape = self._image(image)
            predictor = self._predictor(backbone, shape)
            height, width = shape
            prompts = dict(bboxes=[[box[0] * width, box[1] * height, (box[0] + box[2]) * width, (box[1] + box[3]) * height]])
            if points:
                prompts.update(points=[[[point[0] * width, point[1] * height] for point in points]], labels=[labels])
            masks, scores = predictor.inference(tensor, **prompts, obj_ids=[0], update_memory=True)
            result = self._payload(masks[0], scores[0], box, resized, shape)
            if not result:
                raise ValueError("SAM2 chưa tách được vật rõ ràng. Khoanh sát vật, thêm điểm chọn/loại rồi thử lại.")
            payload, binary = result
            return payload, self._descriptor(backbone, binary, shape)

    def reset(self, camera_id):
        self.entries = {key: value for key, value in self.entries.items() if key[0] != camera_id}

    def reference_status(self):
        return dict(encoded=self.references.encoded, cached=len(self.references.entries),
                    failed=len(self.references.failures), revision=self.reference_revision,
                    errors=dict(list(self.references.failures.items())[:4]),
                    tracks=sum(bool(entry.get('reference_count')) for entry in self.entries.values()))

    def _references_for(self, camera_id, label):
        if not self.metric or not label:
            return []
        key = camera_id, label.casefold()
        if key not in self.reference_choices:
            self.reference_choices[key] = self.metric.reference_samples(label, camera_id, self.reference_limit)
        return self.reference_choices[key]

    def _seed_references(self, predictor, samples):
        memories = [memory for sample in samples if (memory := self.references.get(self, sample)) is not None]
        if memories:
            predictor.memory_bank[:] = memories
            predictor.obj_idx_set.add(0)
        return len(memories)

    def _sync_references(self, revision):
        active_ids = self.metric.reference_ids() if self.metric else set()
        if self.reference_ids - active_ids:
            self.entries.clear()
        self.reference_choices.clear()
        self.rejected_tracks.clear()
        self.references.prune(active_ids)
        self.reference_ids = active_ids
        self.reference_revision = revision

    def segment(self, camera_id, image, objects, captured_at, on_result=None):
        import torch

        self.last_segment_stats = {"backbone_inferences": 0, "decoder_inferences": 0, "propagated_masks": 0}
        active = {track_key(obj) for obj in objects}
        revision = self.metric.status()['revision'] if self.metric else -1
        if revision != self.reference_revision:
            self._sync_references(revision)
        self.entries = {key: entry for key, entry in self.entries.items()
                        if captured_at - entry["seen"] < 1500 and (key[0] != camera_id or key[1] in active)}
        self.rejected_tracks = {key: entry for key, entry in self.rejected_tracks.items() if captured_at < entry['retry_at']}
        output = []
        def publish(obj, mask, identity):
            result = (obj, mask, identity)
            output.append(result)
            if on_result:
                on_result(result)

        with torch.inference_mode():
            tensor, backbone, resized, shape = self._image(image)
            self.last_segment_stats["backbone_inferences"] = 1
            height, width = shape
            descriptor_features = None
            if self.metric and any(self.metric.policy(obj.get("category"))["required"] for obj in objects):
                _, features, _, sizes = self.base.model._prepare_backbone_features(backbone)
                descriptor_features = features[-1].permute(1, 2, 0).reshape(1, -1, *sizes[-1])
            for obj in objects:
                key = camera_id, track_key(obj)
                current = bbox(obj)
                if not obj.get('label_prompt_id') and any(previous.get('label_prompt_id') and mask
                        and previous.get('category') == obj.get('category') and box_iou(bbox(previous), current) >= .5
                        for previous, mask, identity in output):
                    continue
                rejection = self.rejected_tracks.get(key)
                if rejection and compatible_boxes(rejection['box'], current):
                    continue
                entry = self.entries.get(key)
                if entry and not compatible_boxes(entry["box"], current):
                    del self.entries[key]
                    entry = None
                if entry is None:
                    if len(self.entries) >= self.max_tracks:
                        continue
                    predictor = self._predictor(backbone, shape)
                    references = self._references_for(camera_id, obj.get('label_prompt_expected'))
                    count = self._seed_references(predictor, references)
                    entry = {"predictor": predictor, "prompt_at": 0, "seen": captured_at, "box": current, "failures": 0,
                             "reference_count": count, "reference_label": obj.get('label_prompt_expected') if count else None,
                             "reference_revision": revision}
                    self.entries[key] = entry
                predictor = entry["predictor"]
                if entry.get('reference_label') and entry.get('reference_revision') != revision:
                    live_memory = predictor.memory_bank[entry['reference_count']:][-1:]
                    references = self._references_for(camera_id, entry['reference_label'])
                    count = self._seed_references(predictor, references)
                    if count:
                        predictor.memory_bank.extend(live_memory)
                        entry['reference_count'] = count
                    entry['reference_revision'] = revision
                predictor.batch = (None, [SimpleNamespace(shape=(height, width, 3))], None)
                predictor.backbone_out = backbone
                movement = max(abs(current[index] - entry["box"][index]) / max(current[index + 2], .001) for index in (0, 1))
                label_prompt = bool(obj.get("label_prompt_id"))
                prompt = not predictor.memory_bank or (not entry.get('reference_count') and
                    (entry["failures"] > 0 or (not label_prompt and (captured_at - entry["prompt_at"] > 650 or movement > .15))))
                prompt_box = None
                if prompt:
                    prompt_box = [[current[0] * width, current[1] * height,
                                   (current[0] + current[2]) * width, (current[1] + current[3]) * height]]
                    entry["prompt_at"] = captured_at
                masks, scores = predictor.infer_live(tensor, box=prompt_box)
                self.last_segment_stats["decoder_inferences"] += 1
                predictor.backbone_out = None
                result = self._binary(masks[0], scores[0], current, resized, shape,
                                      allow_relocation=label_prompt and bool(entry.get('reference_count')))
                entry.update(seen=captured_at, box=current)
                if result:
                    binary, score = result
                    expected = entry.get('reference_label')
                    identity = self.metric.verify(self._descriptor_from_features(descriptor_features, binary, shape), obj.get("category"),
                                                  expected_label=expected) if descriptor_features is not None and self.metric.policy(obj.get("category"))["required"] else None
                    identity = reference_identity(identity, entry.get("reference_label") or obj.get("label_prompt_expected"))
                    if identity and not identity["accepted"]:
                        predictor.live_output = None
                        entry['failures'] += 1
                        if entry.get('last_verified') and identity.get('reason') in {'appearance_mismatch', 'ambiguous_identity'} and entry['failures'] < 3:
                            continue
                        self.entries.pop(key, None)
                        if not label_prompt:
                            self.rejected_tracks[key] = dict(box=current, retry_at=captured_at + 250)
                        publish(obj, None, identity)
                        continue
                    polygons = self._polygons(binary)
                    if polygons:
                        if identity and identity.get('accepted') and not entry.get('reference_count'):
                            references = self._references_for(camera_id, identity['label'])
                            entry['reference_count'] = self._seed_references(predictor, references)
                            entry['reference_label'] = identity['label'] if entry['reference_count'] else None
                        mask = {"polygons": polygons, "confidence": score,
                                "reference_views": entry.get('reference_count', 0)}
                        if label_prompt:
                            points = [point for ring in polygons for point in ring]
                            left, top = min(point[0] for point in points), min(point[1] for point in points)
                            right, bottom = max(point[0] for point in points), max(point[1] for point in points)
                            obj.update(x=left, y=top, w=right-left, h=bottom-top, confidence=score,
                                       observed_at=captured_at, tracking_state="tracked")
                            entry["box"] = bbox(obj)
                        predictor.commit_live_memory(limit=max(3, 1 + entry.get('reference_count', 0)),
                                                     reference_count=entry.get('reference_count', 0))
                        if not prompt:
                            self.last_segment_stats["propagated_masks"] += 1
                        entry["failures"] = 0
                        entry['last_verified'] = captured_at
                        publish(obj, mask, identity)
                    else:
                        result = None
                if not result:
                    predictor.live_output = None
                    entry["failures"] += 1
                    if entry["failures"] >= 3:
                        self.entries.pop(key, None)
                        publish(obj, None, dict(accepted=False, reason='object_not_present',
                                                candidate_label=entry.get('reference_label'), revision=revision))
        return output

    def _payload(self, logits, score, box, resized_shape, source_shape):
        result = self._binary(logits, score, box, resized_shape, source_shape)
        if result is None:
            return None
        binary, score_value = result
        polygons = self._polygons(binary)
        return ({"polygons": polygons, "confidence": score_value}, binary) if polygons else None

    def _binary(self, logits, score, box, resized_shape, source_shape, allow_relocation=False):
        import torch
        import torch.nn.functional as functional

        score_value = float(score)
        if not math.isfinite(score_value) or score_value <= .02:
            return None
        output_width = 640
        output_height = max(1, round(output_width * source_shape[0] / source_shape[1]))
        mask = functional.interpolate(logits[None, None].float(), (self.size, self.size), mode="bilinear", align_corners=False)
        mask = mask[:, :, :resized_shape[0], :resized_shape[1]]
        soft = functional.interpolate(mask, (output_height, output_width), mode="bilinear", align_corners=False)
        binary = functional.avg_pool2d(soft, 3, stride=1, padding=1)[0, 0] > 0
        rows = torch.arange(output_height, device=binary.device)[:, None]
        columns = torch.arange(output_width, device=binary.device)[None, :]
        roi = ((columns >= box[0] * output_width) & (columns <= (box[0] + box[2]) * output_width)
               & (rows >= box[1] * output_height) & (rows <= (box[1] + box[3]) * output_height))
        area = binary.sum()
        inside = (binary & roi).sum()
        area_value, inside_value = torch.stack([area, inside]).tolist()
        box_area = box[2] * box[3] * output_width * output_height
        if (area_value < max(8, box_area * .025) or area_value > min(box_area * 1.6, output_width * output_height * .6)
                or (not allow_relocation and inside_value < area_value * .75)):
            return None
        return binary, score_value

    def _polygons(self, binary):
        import torch
        import torch.nn.functional as functional

        output_height, output_width = binary.shape
        padded = functional.pad(binary, (1, 1, 1, 1))
        neighbors = torch.stack((padded[:-2, 1:-1], padded[1:-1, 2:], padded[2:, 1:-1], padded[1:-1, :-2]))
        positions = torch.nonzero(binary.unsqueeze(0) & ~neighbors)
        if positions.shape[0] > 20000:
            return []
        offsets = torch.tensor(((0, 0, 1, 0), (1, 0, 1, 1), (1, 1, 0, 1), (0, 1, 0, 0)), device=binary.device)
        boundary = positions[:, (2, 1, 2, 1)] + offsets[positions[:, 0]]
        return boundary_polygons(boundary.tolist(), output_width, output_height)
