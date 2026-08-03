"""Isolated, prompt-based segmentation for registered non-person identities."""

import atexit
import base64
import multiprocessing
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np


def eligible_target(target):
    return bool(str(target.get("label") or "").strip()) and target.get("category") in {"robot", "rack"}


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
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
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

def advance_mask(mask, source_bbox, target_bbox):
    if not isinstance(mask, dict) or not isinstance(mask.get("polygons"), list):
        return mask
    source_x, source_y, source_w, source_h = source_bbox
    target_x, target_y, target_w, target_h = target_bbox
    if source_w <= 1e-6 or source_h <= 1e-6:
        return mask
    scale_x = target_w / source_w
    scale_y = target_h / source_h
    polygons = []
    for ring in mask["polygons"]:
        polygons.append([
            [
                round(target_x + (point[0] - source_x) * scale_x, 6),
                round(target_y + (point[1] - source_y) * scale_y, 6),
            ]
            for point in ring
        ])
    return dict(mask, polygons=polygons)


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
        from ultralytics.models.sam import SAM2DynamicInteractivePredictor

        self.cameras.pop(cam_id, None)
        if not targets:
            return
        state = {"objects": {}}
        self.cameras[cam_id] = state
        for target in targets:
            predictor = SAM2DynamicInteractivePredictor(overrides=self.options, max_obj_num=1)
            predictor.setup_model(model=self.preview.model, verbose=False)
            entry = {"predictor": predictor, "seeded": False}
            state["objects"][target["label"]] = entry
            samples = [sample for sample in target.get("samples", []) if sample.get("mask") and sample.get("frame_image")]
            view_limit = min(16, max(1, int(os.getenv("REGISTERED_MASK_MEMORY_VIEWS", "8"))))
            for sample in samples[-view_limit:]:
                frame = decode_frame(sample["frame_image"])
                bitmap = mask_bitmap(sample["mask"], frame.shape)
                predictor(source=frame, masks=bitmap[None], obj_ids=[0], update_memory=True)
                entry["seeded"] = True

    def track(self, cam_id, frame, seeds):
        state = self.cameras.get(cam_id)
        if state is None:
            return {}
        observations = {}
        for label, entry in state["objects"].items():
            predictor = entry["predictor"]
            if not entry["seeded"]:
                if label not in seeds:
                    continue
                predictor(source=frame, bboxes=[pixels_bbox(seeds[label], frame.shape)],
                          obj_ids=[0], update_memory=True)
                entry["seeded"] = True
            result = predictor(source=frame)[0]
            if result.masks is None or len(result.boxes) == 0:
                continue
            box = result.boxes[0]
            binary = result.masks.data[0].cpu().numpy()
            if binary.sum() < 32 or binary.mean() > 0.6:
                continue
            payload = mask_payload(binary, "sam2_memory", float(box.conf.item()))
            if payload is None:
                continue
            height, width = frame.shape[:2]
            left, top, right, bottom = box.xyxy[0].cpu().tolist()
            observations[label] = {
                "bbox": [left / width, top / height, (right - left) / width, (bottom - top) / height],
                "mask": payload,
            }
        return observations


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
                else:
                    result = runtime.track(request["cam_id"], request["frame"], request["seeds"])
                connection.send({"result": result})
            except Exception as exc:
                connection.send({"error": str(exc)})
    except (EOFError, BrokenPipeError):
        pass
    finally:
        connection.close()


class RegisteredTargetMaskSegmenter:
    def __init__(self):
        self.enabled = os.getenv("IDENTITY_TEMPLATE_MASK_ENABLED", "1").lower() not in {"0", "false", "no"}
        self.model_path = os.getenv("REGISTERED_MASK_MODEL_PATH", str(Path(__file__).resolve().parents[1] / "models/sam2.1_t.pt"))
        self._lock = threading.RLock()
        self._process = None
        self._connection = None
        self._signatures = {}
        self.last_error = None
        self.retry_at = 0.0
        self.last_ms = 0.0
        atexit.register(self.close)

    def available(self):
        return self.enabled and Path(self.model_path).is_file()

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
            return response["result"]

    def preview(self, frame, bbox, points=None, point_labels=None):
        return self._request(action="preview", frame=frame, bbox=bbox, points=points, point_labels=point_labels)

    def track(self, cam_id, frame, targets, seeds):
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
                signature = tuple((target["label"], target.get("mask_revision", 0)) for target in selected)
                if self._process is None or not self._process.is_alive():
                    self._signatures.clear()
                if self._signatures.get(cam_id) != signature:
                    self._request(action="configure", cam_id=cam_id, targets=[
                        {"label": target["label"], "samples": target.get("mask_samples", [])} for target in selected
                    ])
                    self._signatures[cam_id] = signature
                result = self._request(action="track", cam_id=cam_id, frame=frame, seeds=seeds)
                self.last_error = None
                self.last_ms = round((time.monotonic() - started) * 1000, 1)
                return result
            except Exception as exc:
                self.last_error = str(exc)
                self.retry_at = time.monotonic() + 10
                self.close()
                print(f"[RegisteredMask] {exc}", flush=True)
                return {}

    def remove_camera(self, cam_id):
        with self._lock:
            self._signatures.pop(cam_id, None)
            if self._process is not None and self._process.is_alive():
                try:
                    self._request(action="remove", cam_id=cam_id)
                except Exception:
                    self.close()

    def status(self):
        return {"enabled": self.enabled, "available": self.available(), "model": Path(self.model_path).name,
                "backend": "sam2_registered_memory", "last_error": self.last_error, "last_ms": self.last_ms}

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
