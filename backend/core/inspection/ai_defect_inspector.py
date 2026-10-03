"""Tầng 5 – Deep-learning branch: recognise the article and its visible defects.

The detector runs on a rectified crop of the cell (the article's box plus a
margin, or the whole cell when CV found nothing), so every box comes back in
BEV millimetres and can be compared with the CV measurement directly.

Any Ultralytics-loadable YOLO model works – ``.pt`` on CUDA, an exported
TensorRT ``.engine`` or ``.onnx`` – selected with ``INSPECTION_AI_MODEL``.
Classes are mapped to roles:

* **defect** – damaged/torn/dented cargo (``INSPECTION_AI_DEFECT_CLASSES``,
  default: names containing damage/defect/torn/broken/dent/crush/hole);
* **target** – the expected article (``INSPECTION_AI_TARGET_CLASSES``; when
  empty, every non-defect class);
* **other** – anything else the model knows (a person, a robot): never an
  acceptable article.

Without a model the branch reports ``available = False``; the arbitration gate
then refuses to certify anything it would have needed the AI for.
"""

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("Inspection.AI")

ROLE_TARGET = "target"
ROLE_DEFECT = "defect"
ROLE_OTHER = "other"
DEFAULT_DEFECT_KEYWORDS = ("damage", "defect", "torn", "broken", "dent", "crush", "hole", "rach", "mop", "hong")
DEFAULT_CONFIDENCE = 0.50
DEFAULT_LOW_CONFIDENCE = 0.20

RawDetection = Tuple[str, float, Tuple[float, float, float, float]]   # name, confidence, xyxy (image px)


@dataclass
class AIDetection:
    class_name: str
    confidence: float
    bbox_mm: List[float]
    role: str
    # "model": this branch's own detector; "live": the camera's deployed model (live_detections.py).
    source: str = "model"
    label: str = ""
    camera_box: Optional[List[float]] = None   # normalised [x0, y0, x1, y1] on the camera frame (live only)

    def as_dict(self) -> dict:
        return {"class_name": self.class_name, "confidence": round(self.confidence, 4),
                "bbox_mm": [round(v, 1) for v in self.bbox_mm], "role": self.role, "source": self.source,
                "label": self.label, "camera_box": self.camera_box}


@dataclass
class AIResult:
    available: bool
    model: str = ""
    detections: List[AIDetection] = field(default_factory=list)
    confidence_threshold: float = DEFAULT_CONFIDENCE
    region_mm: Optional[List[float]] = None
    latency_ms: float = 0.0
    error: str = ""

    def best(self, role: Optional[str] = None) -> Optional[AIDetection]:
        candidates = [d for d in self.detections if role is None or d.role == role]
        return max(candidates, key=lambda d: d.confidence) if candidates else None

    def as_dict(self) -> dict:
        return {"available": self.available, "model": self.model,
                "detections": [d.as_dict() for d in self.detections],
                "confidence_threshold": self.confidence_threshold,
                "region_mm": self.region_mm, "latency_ms": round(self.latency_ms, 3), "error": self.error}


class CallableBackend:
    """Wraps ``fn(image_bgr, conf, region_mm) -> [(name, confidence, (x1, y1, x2, y2)), ...]``."""

    def __init__(self, fn: Callable[[np.ndarray, float, Sequence[float]], Iterable[RawDetection]],
                 name: str = "callable"):
        self.fn = fn
        self.name = name

    def predict(self, image: np.ndarray, conf: float, region_mm: Optional[Sequence[float]] = None) -> List[RawDetection]:
        return list(self.fn(image, conf, region_mm))


class UltralyticsBackend:
    """Lazy-loaded Ultralytics YOLO (``.pt`` / ``.engine`` TensorRT / ``.onnx``)."""

    def __init__(self, model_path: str, device: Optional[str] = None, imgsz: int = 640):
        self.model_path = model_path
        self.name = os.path.basename(model_path)
        self.device = device
        self.imgsz = imgsz
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path, task="detect")
        return self._model

    def warmup(self) -> None:
        self.predict(np.zeros((self.imgsz, self.imgsz, 3), np.uint8), DEFAULT_CONFIDENCE)

    def predict(self, image: np.ndarray, conf: float, region_mm: Optional[Sequence[float]] = None) -> List[RawDetection]:
        with self._lock:
            model = self._load()
            kwargs = {"conf": conf, "imgsz": self.imgsz, "verbose": False}
            if self.device:
                kwargs["device"] = self.device
            result = model.predict(image, **kwargs)[0]
        names = result.names
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        return [(str(names.get(int(c), c) if isinstance(names, dict) else names[int(c)]), float(s),
                 tuple(float(v) for v in box)) for box, s, c in zip(xyxy, scores, classes)]


class AIDefectInspector:
    def __init__(self, backend=None, target_classes: Optional[Sequence[str]] = None,
                 defect_classes: Optional[Sequence[str]] = None, confidence: float = DEFAULT_CONFIDENCE,
                 low_confidence: float = DEFAULT_LOW_CONFIDENCE):
        self.backend = backend
        self.target_classes = {c.strip().lower() for c in (target_classes or []) if c.strip()}
        self.defect_classes = {c.strip().lower() for c in (defect_classes or []) if c.strip()}
        self.confidence = confidence
        self.low_confidence = low_confidence

    @property
    def available(self) -> bool:
        return self.backend is not None

    @property
    def model_name(self) -> str:
        return getattr(self.backend, "name", "") if self.backend is not None else ""

    def role_of(self, class_name: str) -> str:
        name = class_name.strip().lower()
        if name in self.defect_classes or (not self.defect_classes and any(k in name for k in DEFAULT_DEFECT_KEYWORDS)):
            return ROLE_DEFECT
        if not self.target_classes or name in self.target_classes:
            return ROLE_TARGET
        return ROLE_OTHER

    def inspect(self, image: np.ndarray, region_mm: Sequence[float], low_sensitivity: bool = False) -> AIResult:
        """Detect on ``image``, the rectified crop covering ``region_mm`` = [x0, y0, x1, y1]."""
        conf = self.low_confidence if low_sensitivity else self.confidence
        region = [float(v) for v in region_mm]
        if self.backend is None:
            return AIResult(False, confidence_threshold=conf, region_mm=region,
                            error="Chưa cấu hình model AI (INSPECTION_AI_MODEL)")
        started = time.perf_counter()
        try:
            raw = self.backend.predict(image, conf, region)
        except Exception as exc:  # a broken model must degrade to "unavailable", never crash the station
            logger.exception("AI inference failed")
            return AIResult(False, self.model_name, confidence_threshold=conf, region_mm=region,
                            latency_ms=(time.perf_counter() - started) * 1000.0, error=f"Lỗi suy luận AI: {exc}")
        height, width = image.shape[:2]
        sx = (region[2] - region[0]) / float(width)
        sy = (region[3] - region[1]) / float(height)
        detections = []
        for name, score, (x1, y1, x2, y2) in raw:
            if score < conf:
                continue
            bbox = [region[0] + x1 * sx, region[1] + y1 * sy, region[0] + x2 * sx, region[1] + y2 * sy]
            detections.append(AIDetection(str(name), float(score), bbox, self.role_of(str(name))))
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return AIResult(True, self.model_name, detections, conf, region, (time.perf_counter() - started) * 1000.0)


def _env_list(name: str) -> List[str]:
    return [item for item in os.getenv(name, "").split(",") if item.strip()]


def create_ai_inspector_from_env() -> AIDefectInspector:
    model_path = os.getenv("INSPECTION_AI_MODEL", "").strip()
    backend = None
    if model_path:
        if os.path.exists(model_path):
            backend = UltralyticsBackend(model_path, os.getenv("INSPECTION_AI_DEVICE") or None,
                                         int(os.getenv("INSPECTION_AI_IMGSZ", "640")))
        else:
            logger.error("INSPECTION_AI_MODEL=%s does not exist; AI branch disabled", model_path)
    return AIDefectInspector(backend, _env_list("INSPECTION_AI_TARGET_CLASSES"), _env_list("INSPECTION_AI_DEFECT_CLASSES"),
                             float(os.getenv("INSPECTION_AI_CONF", str(DEFAULT_CONFIDENCE))),
                             float(os.getenv("INSPECTION_AI_LOW_CONF", str(DEFAULT_LOW_CONFIDENCE))))
