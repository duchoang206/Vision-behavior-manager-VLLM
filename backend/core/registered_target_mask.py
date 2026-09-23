"""Isolated, prompt-based segmentation for registered non-person identities."""

import atexit
import base64
import hashlib
import math
import multiprocessing
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from core.registered_samples import sample_fingerprint
from core.robot_spatial_identity import robot_spatial_identity, spatial_verification

import cv2
import numpy as np


def eligible_target(target):
    return bool(str(target.get("label") or "").strip()) and target.get("category") in {"robot", "rack"}


def sam2_presence_confidence(score):
    scaled_logit = float(score)
    if not math.isfinite(scaled_logit) or scaled_logit <= 0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-min(32.0, scaled_logit * 32.0)))


def validate_mask(mask):
    if not isinstance(mask, dict) or not isinstance(mask.get("polygons"), list):
        raise ValueError("Mask phải có danh sách polygons.")
    polygons = mask["polygons"]
    if not 1 <= len(polygons) <= 32 or any(not isinstance(ring, list) for ring in polygons) or sum(len(ring) for ring in polygons) > 4096:
        raise ValueError("Mask vượt giới hạn số đường viền/điểm.")
    normalized = []
    for ring in polygons:
        points = np.asarray(ring, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
            raise ValueError("Mỗi đường viền phải có ít nhất 3 điểm [x, y].")
        if not np.isfinite(points).all() or (points < 0).any() or (points > 1).any():
            raise ValueError("Tọa độ mask phải hữu hạn trong khoảng 0..1.")
        if abs(cv2.contourArea(points.astype(np.float32))) < 1e-7:
            raise ValueError("Mask không có diện tích.")
        normalized.append(np.round(points, 6).tolist())
    return {"polygons": normalized}


def mask_bitmap(mask, shape):
    height, width = shape[:2]
    binary = np.zeros((height, width), dtype=np.uint8)
    contours = [np.rint(np.asarray(ring) * [width - 1, height - 1]).astype(np.int32) for ring in mask["polygons"]]
    cv2.fillPoly(binary, contours, 1)
    return binary


def mask_payload(binary, source="sam2", confidence=1.0):
    binary = np.asarray(binary, dtype=np.uint8)
    height, width = binary.shape
    find_result = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # OpenCV 4.x returns (contours, hierarchy); OpenCV 3.x returned (image, contours, hierarchy)
    contours, hierarchy = find_result[-2], find_result[-1]
    if hierarchy is None:
        return None
    rings = []
    for contour in contours:
        if cv2.contourArea(contour) < 12:
            continue
        simplified = cv2.approxPolyDP(contour, max(0.75, cv2.arcLength(contour, True) * 0.002), True)
        if len(simplified) < 3:
            continue
        points = simplified.reshape(-1, 2) / [max(1, width - 1), max(1, height - 1)]
        rings.append(np.round(points, 6).tolist())
    if not rings or len(rings) > 32 or sum(map(len, rings)) > 4096:
        return None
    return {"polygons": rings, "source": source, "confidence": round(float(confidence), 4)}


def decode_frame(encoded):
    if not isinstance(encoded, str) or len(encoded) > 16_000_000:
        raise ValueError("Ảnh không hợp lệ hoặc quá lớn.")
    raw = base64.b64decode(encoded.split(",", 1)[-1], validate=True)
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.size == 0 or max(image.shape[:2]) > 4096:
        raise ValueError("Không đọc được ảnh hoặc ảnh vượt 4096 px.")
    return image


def pixels_bbox(bbox, shape):
    height, width = shape[:2]
    left, top, box_width, box_height = bbox
    return [left * width, top * height, (left + box_width) * width, (top + box_height) * height]



class MaskRuntime:
    def __init__(self, model_path):
        import torch
        from ultralytics.models.sam import SAM2Predictor

        torch.set_num_threads(2)
        if os.getenv("REGISTERED_MASK_CUDNN", "0") != "1":
            torch.backends.cudnn.enabled = False
        self.options = dict(model=model_path, task="segment", mode="predict", device=0,
                            quantize=16, imgsz=int(os.getenv("REGISTERED_MASK_IMGSZ", "640")),
                            verbose=False, conf=0.1, save=False)
        self.preview = SAM2Predictor(overrides=self.options)
        self.preview.setup_model(verbose=False)
        self.cameras = {}
        self.shared_features = os.getenv("REGISTERED_MASK_SHARED_FEATURES", "1").lower() not in {"0", "false", "no"}
        self.smoothing_alpha = float(os.getenv("REGISTERED_MASK_SMOOTHING_ALPHA", "0.72"))
        from core.identity_metric import IdentityMetric
        with open(model_path, "rb") as model_file:
            signature = f'{hashlib.file_digest(model_file, "sha256").hexdigest()[:16]}:{self.options["imgsz"]}'
        self.identity = IdentityMetric(os.getenv("REGISTERED_IDENTITY_DIRECTORY", str(Path(__file__).resolve().parents[1] / "data/identity_metric")), model_signature=signature)

    @staticmethod
    def _identity_descriptor(predictor, bitmap, shape):
        from core.identity_metric import masked_descriptor
        features = predictor.vision_feats[-1]
        features = features.permute(1, 2, 0).reshape(1, -1, *predictor.feat_sizes[-1])
        return masked_descriptor(features, bitmap, shape, predictor.imgsz)

    def describe(self, frame, mask):
        import torch
        from core.identity_metric import masked_descriptor
        with torch.inference_mode():
            predictor = self.preview
            predictor.reset_image()
            predictor.setup_source(frame)
            image = predictor.preprocess([frame])
            backbone = predictor.model.forward_image(image)
            _, features, _, sizes = predictor.model._prepare_backbone_features(backbone)
            features = features[-1].permute(1, 2, 0).reshape(1, -1, *sizes[-1])
            descriptor = masked_descriptor(features, mask_bitmap(mask, frame.shape), frame.shape, image.shape)
            return descriptor.cpu().tolist()

    @staticmethod
    def _bbox_iou(first, second):
        if not first or not second:
            return 0.0
        fx, fy, fw, fh = first
        sx, sy, sw, sh = second
        ix1, iy1 = max(fx, sx), max(fy, sy)
        ix2, iy2 = min(fx + fw, sx + sw), min(fy + fh, sy + sh)
        intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = fw * fh + sw * sh - intersection
        return intersection / union if union > 1e-8 else 0.0

    def _smooth_mask(self, entry, binary, bbox):
        binary = (np.asarray(binary, dtype=np.uint8) > 0).astype(np.uint8)
        height, width = binary.shape
        left, top = max(0, int(bbox[0] * width)), max(0, int(bbox[1] * height))
        right = min(width, int(np.ceil((bbox[0] + bbox[2]) * width)))
        bottom = min(height, int(np.ceil((bbox[1] + bbox[3]) * height)))
        if right <= left or bottom <= top:
            return binary
        crop = binary[top:bottom, left:right]
        reduced = cv2.resize(crop, (160, 160), interpolation=cv2.INTER_NEAREST)
        distance = cv2.distanceTransform(reduced, cv2.DIST_L2, 3) - cv2.distanceTransform(1 - reduced, cv2.DIST_L2, 3)
        previous = entry.get("shape_distance")
        if previous is not None and time.monotonic() - entry.get("last_mask_at", 0) <= 0.5:
            intersection = np.logical_and(previous > 0, reduced > 0).sum()
            union = np.logical_or(previous > 0, reduced > 0).sum()
            if intersection / max(1, union) > 0.6:
                alpha = min(1.0, max(0.7, getattr(self, "smoothing_alpha", 0.72)))
                filtered = alpha * distance + (1 - alpha) * previous
                boundary = np.abs(distance) <= 1.5
                distance[boundary] = filtered[boundary]
                binary[top:bottom, left:right] = (cv2.resize(distance, (right - left, bottom - top), interpolation=cv2.INTER_LINEAR) > 0).astype(np.uint8)
        entry["shape_distance"] = np.clip(distance, -4, 4)
        entry["last_bbox"] = list(bbox)
        entry["last_mask_at"] = time.monotonic()
        return binary

    def segment(self, frame, bbox, points=None, point_labels=None):
        self.preview.reset_image()
        prompts = {"bboxes": [pixels_bbox(bbox, frame.shape)]}
        if points:
            height, width = frame.shape[:2]
            prompts.update(points=[[[point[0] * width, point[1] * height] for point in points]],
                           labels=[point_labels])
        result = self.preview(source=frame, **prompts)[0]
        if result.masks is None or len(result.masks) == 0:
            return None
        best_index = int(result.boxes.conf.argmax())
        return mask_payload(result.masks.data[best_index].cpu().numpy(), confidence=float(result.boxes.conf[best_index]))

    def configure(self, cam_id, targets):
        from core.sam2_live_predictor import LiveSAM2Predictor

        previous = self.cameras.pop(cam_id, {"objects": {}})
        if not targets:
            return
        state = {"objects": {}}
        self.cameras[cam_id] = state
        for target in targets:
            samples = [sample for sample in target.get("samples", []) if sample.get("mask") and sample.get("frame_image")]
            signature = tuple(sample_fingerprint(sample["frame_image"], sample["mask"]) for sample in samples)
            existing = previous["objects"].get(target["label"])
            if existing is not None and existing.get("sample_signature") == signature:
                state["objects"][target["label"]] = existing
                continue
            predictor = LiveSAM2Predictor(overrides=dict(self.options, conf=0.0), max_obj_num=1)
            predictor.setup_model(model=self.preview.model, verbose=False)
            entry = {"predictor": predictor, "seeded": False, "last_binary": None, "last_bbox": None,
                     "last_mask_at": 0.0, "sample_signature": signature, "box_sizes": [],
                     "misses": 0, "recovery_attempts": 0, "next_recovery_at": 0.0}
            state["objects"][target["label"]] = entry
            view_limit = min(16, max(1, int(os.getenv("REGISTERED_MASK_MEMORY_VIEWS", "8"))))
            for sample in samples[-view_limit:]:
                self._reset_predictor_memory(predictor)
                frame = decode_frame(sample["frame_image"])
                bitmap = mask_bitmap(sample["mask"], frame.shape)
                predictor(source=frame, masks=bitmap[..., None][None], obj_ids=[0], update_memory=True)
                descriptor = self._identity_descriptor(predictor, bitmap, frame.shape).detach().cpu().tolist()
                self.identity.learn(target["label"], cam_id, sample_fingerprint(sample["frame_image"], sample["mask"]), descriptor)
                points = np.concatenate([np.asarray(ring) for ring in sample["mask"]["polygons"]])
                dimensions = (points.max(axis=0) - points.min(axis=0)).tolist()
                entry["box_sizes"].append(dimensions)
            self._reset_predictor_memory(predictor)
            entry["seeded"] = False

    @staticmethod
    def _reset_predictor_memory(predictor):
        """Drop SAM2's temporal state after a confirmed identity mismatch."""
        predictor.memory_bank.clear()
        predictor.obj_idx_set.clear()
        predictor.live_output = None
        for mapping_name in ("obj_id_to_idx", "obj_idx_to_id"):
            mapping = getattr(predictor, mapping_name, None)
            if mapping is not None:
                mapping.clear()
                for index in range(int(getattr(predictor, "_max_obj_num", 1))):
                    mapping[index] = index
        predictor.reset_image()

    @contextmanager
    def _frame_features(self, frame, predictors):
        if not self.shared_features or not predictors:
            yield
            return
        import torch

        with torch.inference_mode():
            try:
                encoder = predictors[0]
                encoder.reset_image()
                encoder.setup_source(frame)
                image = encoder.preprocess([frame])
                backbone = encoder.model.forward_image(image)
                for predictor in predictors:
                    predictor.im = image
                    predictor.backbone_out = backbone
                yield
            finally:
                for predictor in predictors:
                    predictor.backbone_out = None
                    predictor.reset_image()

    def track(self, cam_id, frame, seeds, spatial_context=None):
        import torch

        state = self.cameras.get(cam_id)
        if state is None:
            return {}
        now = time.monotonic()
        entries = {label: entry for label, entry in state["objects"].items() if entry["seeded"]}
        recovering = [(label, entry) for label, entry in state["objects"].items()
                      if not entry["seeded"] and (label in seeds or entry.get("box_sizes"))
                      and now >= entry.get("next_recovery_at", 0)]
        if recovering:
            label, entry = min(recovering, key=lambda item: item[1].get("next_recovery_at", 0))
            entries[label] = entry
        with torch.inference_mode(), self._frame_features(frame, [entry["predictor"] for entry in entries.values()]):
            return self._track_objects(frame, entries, seeds, spatial_context)

    def _recovery_bbox(self, label, entry, frame, seeds, spatial_context=None):
        attempt = entry.get("recovery_attempts", 0)
        entry["recovery_attempts"] = attempt + 1
        entry["next_recovery_at"] = time.monotonic() + max(0.1, float(os.getenv("REGISTERED_MASK_RECOVERY_INTERVAL", "0.6")))
        entry["diagnostics"]["recovery_attempts"] = entry["recovery_attempts"]
        nearby = seeds.get(label) or entry.get("last_bbox")
        if nearby is not None and spatial_verification(spatial_context, label, nearby)["accepted"] is False:
            nearby = None
        if nearby is not None and attempt % 4 == 0:
            return nearby
        if not entry.get("box_sizes"):
            return nearby
        from core.registered_mask_recovery import appearance_candidates
        predictor = entry["predictor"]
        predictor.setup_source(frame)
        image = predictor.preprocess([frame])
        predictor.get_im_features(image)
        candidates = appearance_candidates(self.identity, label, predictor, frame.shape, entry["box_sizes"])
        candidates = [bbox for bbox in candidates if spatial_verification(spatial_context, label, bbox)["accepted"] is not False]
        entry["diagnostics"]["recovery_candidates"] = len(candidates)
        if not candidates:
            return nearby
        search_attempt = entry.get("search_attempts", 0)
        entry["search_attempts"] = search_attempt + 1
        return candidates[search_attempt % len(candidates)]

    def _lose_memory(self, entry):
        self._reset_predictor_memory(entry["predictor"])
        entry["seeded"] = False
        entry["last_binary"] = None
        entry.pop("shape_distance", None)
        entry["diagnostics"].update(state="searching", memory_count=0)

    def _track_objects(self, frame, entries, seeds, spatial_context=None):
        observations = {}
        for label, entry in entries.items():
            diagnostics = entry.setdefault("diagnostics", {"frames": 0, "accepted": 0, "rejections": {}})
            diagnostics["frames"] += 1
            diagnostics.update(last_score=None, last_rejection=None)
            predictor = entry["predictor"]
            if not entry["seeded"]:
                bbox = self._recovery_bbox(label, entry, frame, seeds, spatial_context)
                if bbox is None:
                    diagnostics.update(state="searching", last_rejection="no_recovery_candidate")
                    continue
                result = predictor(source=frame, bboxes=[pixels_bbox(bbox, frame.shape)],
                                   obj_ids=[0], update_memory=True)[0]
                entry["seeded"] = True
            else:
                result = predictor(source=frame)[0]
            if result.masks is None or len(result.boxes) == 0:
                self._reject(entry, "no_sam_mask")
                if not entry["seeded"]:
                    observations[label] = {"tracking_lost": True, "reason": "no_sam_mask"}
                continue
            box = result.boxes[0]
            diagnostics["last_score"] = float(box.conf.item())
            if diagnostics["last_score"] <= 0:
                self._reject(entry, "no_object_presence")
                continue
            height, width = frame.shape[:2]
            left, top, right, bottom = box.xyxy[0].cpu().tolist()
            measured_bbox = [left / width, top / height, (right - left) / width, (bottom - top) / height]
            spatial = spatial_verification(spatial_context, label, measured_bbox)
            diagnostics["spatial_identity"] = spatial
            if spatial["accepted"] is False:
                self._reject(entry, spatial["reason"])
                self._lose_memory(entry)
                observations[label] = {"identity_rejected": True, "identity": spatial}
                continue
            descriptor = self._identity_descriptor(predictor, result.masks.data[0], frame.shape)
            verification = self.identity.verify(label, descriptor)
            diagnostics["identity"] = verification
            if not verification["accepted"]:
                self._reject(entry, verification["reason"])
                self._lose_memory(entry)
                observations[label] = {"identity_rejected": True, "identity": verification}
                continue
            binary = result.masks.data[0].cpu().numpy()
            if binary.sum() < 32 or binary.mean() > 0.6:
                self._reject(entry, "mask_area")
                continue
            binary = self._smooth_mask(entry, binary, measured_bbox)
            payload = mask_payload(binary, "sam2_memory", float(box.conf.item()))
            if payload is None:
                self._reject(entry, "invalid_contour")
                continue
            if predictor.commit_live_memory(limit=max(1, int(os.getenv("REGISTERED_MASK_LIVE_MEMORIES", "3")))):
                diagnostics["memory_updates"] = diagnostics.get("memory_updates", 0) + 1
            if entry.get("recovery_attempts"):
                diagnostics["recoveries"] = diagnostics.get("recoveries", 0) + 1
            entry.update(misses=0, recovery_attempts=0, search_attempts=0, next_recovery_at=0.0)
            diagnostics.update(state="tracking", consecutive_misses=0, recovery_attempts=0,
                               memory_count=len(predictor.memory_bank))
            diagnostics["accepted"] += 1
            payload["presence_confidence"] = sam2_presence_confidence(float(box.conf.item()))
            payload["score_type"] = "sam2_positive_object_logit_div32"
            payload["identity"] = verification
            payload["spatial_identity"] = spatial
            observations[label] = {
                "bbox": measured_bbox,
                "mask": payload,
            }
        return observations


    def _reject(self, entry, reason):
        diagnostics = entry["diagnostics"]
        diagnostics["last_rejection"] = reason
        diagnostics["rejections"][reason] = diagnostics["rejections"].get(reason, 0) + 1
        entry["misses"] = entry.get("misses", 0) + 1
        diagnostics["consecutive_misses"] = entry["misses"]
        if entry["misses"] >= max(1, int(os.getenv("REGISTERED_MASK_RESET_MISSES", "3"))):
            self._lose_memory(entry)

def _worker(connection, model_path):
    runtime = None
    try:
        while True:
            request = connection.recv()
            try:
                if runtime is None:
                    runtime = MaskRuntime(model_path)
                action = request["action"]
                if action == "configure":
                    runtime.configure(request["cam_id"], request["targets"])
                    result = True
                elif action == "remove":
                    runtime.cameras.pop(request["cam_id"], None)
                    result = True
                elif action == "preview":
                    result = runtime.segment(request["frame"], request["bbox"], request.get("points"), request.get("point_labels"))
                elif action == "describe":
                    result = runtime.describe(request["frame"], request["mask"])
                elif action == "learn_identity":
                    result = runtime.identity.learn(request["label"], request["cam_id"], request["fingerprint"], request["vector"])
                elif action == "forget_identity":
                    runtime.identity.forget(request["label"], request.get("cam_id"))
                    result = runtime.identity.status()
                else:
                    result = runtime.track(request["cam_id"], request["frame"], request["seeds"], request.get("spatial_context"))
                response = {"result": result, "identity": runtime.identity.status()}
                if action == "track":
                    response["diagnostics"] = {
                        label: entry.get("diagnostics", {})
                        for label, entry in runtime.cameras.get(request["cam_id"], {}).get("objects", {}).items()
                    }
                connection.send(response)
            except Exception as exc:
                import traceback
                traceback.print_exc()
                connection.send({"error": str(exc)})
    except (EOFError, BrokenPipeError):
        pass
    finally:
        connection.close()


class RegisteredTargetMaskSegmenter:
    def __init__(self):
        self.enabled = os.getenv("IDENTITY_TEMPLATE_MASK_ENABLED", "1").lower() not in {"0", "false", "no"}
        self.shared_features = os.getenv("REGISTERED_MASK_SHARED_FEATURES", "1").lower() not in {"0", "false", "no"}
        self.model_path = os.getenv("REGISTERED_MASK_MODEL_PATH", str(Path(__file__).resolve().parents[1] / "models/sam2.1_t.pt"))
        self._lock = threading.RLock()
        self._schedule_lock = threading.Lock()
        self._waiting_cameras = {}
        self._last_slot_at = {}
        self._camera_diagnostics = {}
        self._identity_status = {"state": "initializing", "revision": 0, "training": False}
        self._process = None
        self._connection = None
        self._signatures = {}
        self.last_error = None
        self.retry_at = 0.0
        self.last_ms = 0.0
        atexit.register(self.close)

    def available(self):
        return self.enabled and Path(self.model_path).is_file()

    def frame_slot(self):
        return self._lock

    @contextmanager
    def try_frame_slot(self, cam_id, interval=0.0):
        now = time.monotonic()
        acquired = False
        with self._schedule_lock:
            self._waiting_cameras = {camera: times for camera, times in self._waiting_cameras.items()
                                     if now - times[1] < 1.0}
            if now - self._last_slot_at.get(cam_id, -math.inf) >= interval:
                first_wait = self._waiting_cameras.get(cam_id, (now, now))[0]
                self._waiting_cameras[cam_id] = (first_wait, now)
                next_camera = min(self._waiting_cameras, key=lambda camera: (
                    self._last_slot_at.get(camera, -math.inf), self._waiting_cameras[camera][0]
                ))
                if next_camera == cam_id:
                    acquired = self._lock.acquire(blocking=False)
                    if acquired:
                        self._waiting_cameras.pop(cam_id, None)
                        self._last_slot_at[cam_id] = now
        try:
            yield acquired
        finally:
            if acquired:
                self._lock.release()

    def cancel_frame_wait(self, cam_id):
        with self._schedule_lock:
            self._waiting_cameras.pop(cam_id, None)

    def _request(self, **payload):
        if not self.available():
            raise RuntimeError("Model mask chưa sẵn sàng. Kiểm tra REGISTERED_MASK_MODEL_PATH.")
        with self._lock:
            if self._process is None or not self._process.is_alive():
                self.close()
                context = multiprocessing.get_context("spawn")
                parent, child = context.Pipe()
                self._connection = parent
                self._process = context.Process(target=_worker, args=(child, self.model_path), daemon=True)
                self._process.start()
                child.close()
            self._connection.send(payload)
            if not self._connection.poll(45):
                self.close()
                raise RuntimeError("Mask worker không phản hồi trong 45 giây.")
            response = self._connection.recv()
            if "error" in response:
                raise RuntimeError(response["error"])
            if "diagnostics" in response:
                self._camera_diagnostics.setdefault(payload["cam_id"], {})["targets"] = response["diagnostics"]
            if "identity" in response:
                self._identity_status = response["identity"]
            return response["result"]

    def preview(self, frame, bbox, points=None, point_labels=None):
        return self._request(action="preview", frame=frame, bbox=bbox, points=points, point_labels=point_labels)

    def describe(self, frame, mask):
        return self._request(action="describe", frame=frame, mask=mask)

    def learn_identity(self, label, cam_id, frame_image, mask, vector):
        return self._request(action="learn_identity", label=label, cam_id=cam_id,
                             fingerprint=sample_fingerprint(frame_image, mask), vector=vector)

    def forget_identity(self, label, cam_id=None):
        return self._request(action="forget_identity", label=label, cam_id=cam_id)

    def track(self, cam_id, frame, targets, seeds, observed_at=None):
        selected = [target for target in targets if eligible_target(target)]
        if not selected:
            if cam_id in self._signatures:
                self.remove_camera(cam_id)
            return {}
        if not self.available() or time.monotonic() < self.retry_at:
            return {}
        with self._lock:
            try:
                started = time.monotonic()
                diagnostics = self._camera_diagnostics.setdefault(cam_id, {})
                gap_ms = (started - diagnostics.get("last_started_at", started)) * 1000
                diagnostics.update(last_started_at=started, last_interval_ms=round(gap_ms, 1),
                                   max_interval_ms=round(max(gap_ms, diagnostics.get("max_interval_ms", 0)), 1))
                signature = tuple((target["label"], target.get("mask_revision", 0)) for target in selected)
                if self._process is None or not self._process.is_alive():
                    self._signatures.clear()
                if self._signatures.get(cam_id) != signature:
                    self._request(action="configure", cam_id=cam_id, targets=[
                        {"label": target["label"], "samples": target.get("mask_samples", [])} for target in selected
                    ])
                    self._signatures[cam_id] = signature
                spatial_context = robot_spatial_identity.context(cam_id, selected, time.time() if observed_at is None else observed_at)
                result = self._request(action="track", cam_id=cam_id, frame=frame, seeds=seeds, spatial_context=spatial_context)
                self.last_error = None
                self.last_ms = round((time.monotonic() - started) * 1000, 1)
                observed = [label for label, observation in result.items() if observation.get("mask")]
                diagnostics.update(last_ms=self.last_ms, last_observed_labels=observed,
                                   frames=diagnostics.get("frames", 0) + 1,
                                   empty_frames=diagnostics.get("empty_frames", 0) + int(not observed))
                return result
            except Exception as exc:
                self.last_error = str(exc)
                self.retry_at = time.monotonic() + 10
                self.close()
                print(f"[RegisteredMask] {exc}", flush=True)
                return {}

    def remove_camera(self, cam_id):
        with self._schedule_lock:
            self._waiting_cameras.pop(cam_id, None)
            self._last_slot_at.pop(cam_id, None)
        with self._lock:
            self._signatures.pop(cam_id, None)
            self._camera_diagnostics.pop(cam_id, None)
            if self._process is not None and self._process.is_alive():
                try:
                    self._request(action="remove", cam_id=cam_id)
                except Exception:
                    self.close()

    def status(self, cam_id=None):
        status = {"enabled": self.enabled, "available": self.available(), "model": Path(self.model_path).name,
                "backend": "sam2_registered_memory", "shared_features": self.shared_features,
                "last_error": self.last_error, "last_ms": self.last_ms,
                "identity": dict(self._identity_status),
                "scheduled_cameras": len(self._last_slot_at)}
        if cam_id is not None:
            status["camera"] = {key: value for key, value in self._camera_diagnostics.get(cam_id, {}).items()
                                if key != "last_started_at"}
        return status

    def close(self):
        with self._lock:
            if self._connection is not None:
                self._connection.close()
            if self._process is not None:
                if self._process.is_alive():
                    self._process.terminate()
                self._process.join(timeout=2)
            self._connection = None
            self._process = None
            self._signatures.clear()


registered_target_mask_segmenter = RegisteredTargetMaskSegmenter()
