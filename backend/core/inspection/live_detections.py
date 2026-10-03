"""AI layer shared with the camera's deployed model - like the "Ô chứa hàng" rule.

The detectors deployed on a camera (DeepStream / custom models) publish their
detections continuously (``main.latest_objects_by_cam``).  An inspection
station uses them as its AI layer: no second model on the GPU, and the
classes that count as goods are picked with the same chips as the occupancy
rule ("Đối Tượng Áp Dụng" → the rule's ``target_objects``).

A detection belongs to the cell when its ground contact point - the
``ground_point`` the tracker reports, else the bottom-centre of its box - lies
on the cell outline (a small tolerance absorbs box jitter).  A tall article
standing in the cell in front can cover this cell on the image, but its feet
are in front: it is not counted here.

``target_matches`` mirrors ``BehaviorAnalyticsEngine._target_matches`` term for
term (a parity test pins it), so "rack" also accepts pallet / shelf / storage,
"robot" accepts agv / amr and "person" accepts human / worker.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import cv2
import numpy as np

from .ai_defect_inspector import (DEFAULT_DEFECT_KEYWORDS, ROLE_DEFECT, ROLE_OTHER, ROLE_TARGET, AIDetection,
                                  AIResult)
from .homography_rectifier import MetricRectifier

LIVE_MAX_AGE_SEC = 3.0          # older metadata means the camera's model is not running
FOOT_TOLERANCE = 0.03           # of the cell outline's diagonal on the image
FOOT_BOX_MM = 50.0              # half-size of the BEV box of a detection known only by its foot
CAMERA_MODEL_NAME = "model camera"


@dataclass
class LiveDetections:
    """The latest detections of the camera's deployed model(s)."""

    objects: List[dict] = field(default_factory=list)
    age_sec: Optional[float] = None      # None: no detector has reported this camera

    @property
    def available(self) -> bool:
        return self.age_sec is not None and self.age_sec <= LIVE_MAX_AGE_SEC


def target_matches(class_name: str, label: str, targets: Optional[Sequence[str]]) -> bool:
    """Same matching as the occupancy rule (``BehaviorAnalyticsEngine._target_matches``)."""
    if not targets:
        return True
    obj_cls = (class_name or "object").lower()
    obj_label = (label or "").lower()
    wanted = [str(t).lower().strip() for t in targets if str(t).strip()]
    terms = {obj_cls}
    if obj_label:
        terms.add(obj_label)
    if "rack" in obj_cls or "rack" in obj_label or "pallet" in obj_cls or "pallet" in obj_label or "storage" in obj_cls:
        terms.add("rack")
    if "robot" in obj_cls or "robot" in obj_label or "agv" in obj_cls or "amr" in obj_cls:
        terms.add("robot")
    if "person" in obj_cls or "human" in obj_cls or "worker" in obj_cls:
        terms.add("person")
    for target in wanted:
        target_terms = {target}
        if target in ("pallet", "storage", "shelf"):
            target_terms.add("rack")
        if target in ("delivery-robot", "agv", "amr"):
            target_terms.add("robot")
        if target in ("human", "worker"):
            target_terms.add("person")
        if terms & target_terms:
            return True
        if any(target in term or term in target for term in terms):
            return True
    return False


def live_role(class_name: str, label: str, targets: Optional[Sequence[str]]) -> str:
    name = (class_name or "").lower()
    if any(keyword in name for keyword in DEFAULT_DEFECT_KEYWORDS):
        return ROLE_DEFECT
    return ROLE_TARGET if target_matches(class_name, label, targets) else ROLE_OTHER


def _finite(*values) -> bool:
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)


def _box(obj: dict) -> Optional[List[float]]:
    try:
        x, y, w, h = (float(obj[key]) for key in ("x", "y", "w", "h"))
    except (KeyError, TypeError, ValueError):
        return None
    if not _finite(x, y, w, h) or w <= 0 or h <= 0:
        return None
    return [x, y, x + w, y + h]


def _foot(obj: dict, box: List[float]) -> List[float]:
    ground = obj.get("ground_point")
    if isinstance(ground, (list, tuple)) and len(ground) >= 2 and _finite(*ground[:2]):
        return [float(ground[0]), float(ground[1])]
    return [(box[0] + box[2]) / 2.0, box[3]]


def cell_detections(live: Optional[LiveDetections], rect: MetricRectifier, frame_shape,
                    targets: Optional[Sequence[str]] = None) -> List[AIDetection]:
    """The camera model's detections standing on the cell, as AIDetections in BEV mm."""
    if live is None or not live.available or not live.objects:
        return []
    height, width = int(frame_shape[0]), int(frame_shape[1])
    scale = np.asarray([width, height], np.float64)
    outline = (rect.points * scale - 0.5).astype(np.float32)          # camera pixel centres, convex
    lo, hi = outline.min(axis=0), outline.max(axis=0)
    tolerance = FOOT_TOLERANCE * float(np.hypot(*(hi - lo)))
    cell_w, cell_h = rect.cell_mm
    found = []
    for obj in live.objects:
        if not isinstance(obj, dict):
            continue
        box = _box(obj)
        if box is None:
            continue
        foot = np.asarray(_foot(obj, box), np.float64) * scale - 0.5
        if cv2.pointPolygonTest(outline, (float(foot[0]), float(foot[1])), True) < -tolerance:
            continue
        corners = (np.asarray([[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]])
                   * scale - 0.5).astype(np.float32)
        area, overlap = cv2.intersectConvexConvex(corners, outline)
        if area > 1.0 and overlap is not None and len(overlap) >= 3:
            mm = rect.camera_px_to_bev_mm(frame_shape, overlap.reshape(-1, 2))
        else:
            centre = rect.camera_px_to_bev_mm(frame_shape, foot.reshape(1, 2))[0]
            mm = np.asarray([centre - FOOT_BOX_MM, centre + FOOT_BOX_MM])
        x0, y0 = np.clip(mm.min(axis=0), 0.0, [cell_w, cell_h])
        x1, y1 = np.clip(mm.max(axis=0), 0.0, [cell_w, cell_h])
        class_name = str(obj.get("class") or obj.get("category") or "object")
        label = str(obj.get("label") or "")
        confidence = obj.get("confidence")
        confidence = float(confidence) if _finite(confidence) else 1.0
        found.append(AIDetection(class_name, confidence, [float(x0), float(y0), float(x1), float(y1)],
                                 live_role(class_name, label, targets), "live", label,
                                 [round(v, 5) for v in box]))
    found.sort(key=lambda d: d.confidence, reverse=True)
    return found


def merge_live(result: AIResult, live: Optional[LiveDetections], detections: Sequence[AIDetection]) -> AIResult:
    """The AI branch result extended with the camera model's detections on the cell."""
    if live is None or not live.available:
        if not result.available:
            result.error = ("Chưa có AI: model camera chưa chạy trên camera này (Model Manager)"
                            + (f" và {result.error[0].lower()}{result.error[1:]}" if result.error else ""))
        return result
    merged = sorted(list(result.detections) + list(detections), key=lambda d: d.confidence, reverse=True)
    model = f"{result.model} + {CAMERA_MODEL_NAME}" if result.available and result.model else CAMERA_MODEL_NAME
    return AIResult(True, model, merged, result.confidence_threshold, result.region_mm, result.latency_ms)
