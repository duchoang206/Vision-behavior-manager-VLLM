"""Có hàng hay không - the station's headline answer (Building view, Monitor).

Decided next to the metrology verdict (arbitration_gate), which is unchanged:

* **CV_ONLY** - image processing against the empty-cell baselines: an object
  → ``CARFULL``, nothing → ``EMPTY``.
* **AI_ONLY** - the AI finds one of the selected classes on the cell →
  ``CARFULL``, otherwise ``EMPTY``.
* **HYBRID** - image processing finds the object and the AI names it: one of
  the selected classes → ``CARFULL``; an object the AI cannot name, or names
  as something not selected → ``UNKNOWN``.  A selected class the AI finds on
  the cell is goods even when CV missed it (camouflage), so HYBRID never
  answers ``EMPTY`` over a recognised article.

Whatever prevents an answer - no frame, blurred or dark image, no baseline,
ROI changed, no AI where the mode needs it, glare over the cell - is
``UNKNOWN`` with the reason.

``OccupancyDebouncer`` confirms the state over consecutive frames: ``EMPTY``
and "something is there" each need ``window`` frames in a row (entering, the
majority of those frames picks ``CARFULL`` or ``UNKNOWN``); afterwards
``CARFULL`` and ``UNKNOWN`` replace each other only when ``window`` frames in
a row agree, so an AI score hovering at its threshold neither flaps the
state nor ever makes the cell look empty.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Sequence

from .ai_defect_inspector import ROLE_DEFECT, ROLE_OTHER, ROLE_TARGET, AIDetection, AIResult
from .config import OCC_CARFULL, OCC_EMPTY, OCC_UNKNOWN, OK_EMPTY

OCCUPANCY_LABEL = {OCC_CARFULL: "CÓ HÀNG", OCC_EMPTY: "TRỐNG", OCC_UNKNOWN: "KHÔNG XÁC ĐỊNH"}
GOODS_ROLES = (ROLE_TARGET, ROLE_DEFECT)


@dataclass
class OccupancyDecision:
    state: str
    reason: str
    source: str = "NONE"                 # CV | AI | CV+AI | NONE
    label: Optional[str] = None          # what the AI recognised
    confidence: Optional[float] = None
    cv_object: bool = False
    cv_size_mm: Optional[List[float]] = None
    ai_objects: List[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"state": self.state, "text": OCCUPANCY_LABEL.get(self.state, self.state), "reason": self.reason,
                "source": self.source, "label": self.label,
                "confidence": None if self.confidence is None else round(float(self.confidence), 4),
                "cv_object": self.cv_object, "cv_size_mm": self.cv_size_mm, "ai_objects": self.ai_objects}


def unknown_decision(reason: str) -> dict:
    return OccupancyDecision(OCC_UNKNOWN, reason).as_dict()


def _detections(results: Sequence[Optional[AIResult]]) -> List[AIDetection]:
    seen, found = set(), []
    for result in results:
        for det in (result.detections if result is not None and result.available else []):
            key = (det.class_name, det.source, tuple(round(v) for v in det.bbox_mm))
            if key not in seen:
                seen.add(key)
                found.append(det)
    return sorted(found, key=lambda d: d.confidence, reverse=True)


def _names(detections: Sequence[AIDetection]) -> str:
    return ", ".join(dict.fromkeys(f"{d.class_name} {d.confidence:.0%}" for d in detections))


def decide_occupancy(mode: str, cv, ai: Optional[AIResult], verdict, extra_ai: Sequence[AIResult] = ()) -> dict:
    """``cv``: CVResult or None, ``verdict``: the arbitration Verdict of the same frame."""
    results = [ai, *extra_ai]
    detections = _detections(results)
    goods = [d for d in detections if d.role in GOODS_ROLES]
    others = [d for d in detections if d.role == ROLE_OTHER]
    ai_available = any(r is not None and r.available for r in results)
    ai_error = next((r.error for r in results if r is not None and r.error), "")
    best = goods[0] if goods else None
    # The shadow test of the arbitration can prove a CV blob to be light only (OK_EMPTY).
    cv_object = bool(cv is not None and cv.object_present and verdict.code != OK_EMPTY)
    size = None
    if cv_object and cv.main is not None:
        x0, y0, x1, y1 = cv.main.bbox_mm
        size = [round(x1 - x0), round(y1 - y0)]
    glare = bool(cv is not None and cv.glare_unresolved and not cv_object)
    objects = [{"class_name": d.class_name, "confidence": round(d.confidence, 4), "role": d.role,
                "label": d.label, "source": d.source} for d in detections]
    seen = f"Xử lý ảnh thấy vật ~{size[0]}×{size[1]} mm" if size else "Xử lý ảnh thấy vật"

    def result(state: str, reason: str, source: str = "NONE", det: Optional[AIDetection] = None) -> dict:
        return OccupancyDecision(state, reason, source, det.class_name if det else None,
                                 det.confidence if det else None, cv_object, size, objects).as_dict()

    glare_reason = "Vùng cháy sáng che khuất ô – không xác nhận được ô trống"
    if mode == "CV_ONLY":
        if cv_object:
            return result(OCC_CARFULL, f"{seen} trong ô (so với ảnh nền ô trống)", "CV")
        if glare:
            return result(OCC_UNKNOWN, glare_reason)
        return result(OCC_EMPTY, "Xử lý ảnh: ô trống so với ảnh nền", "CV")

    if mode == "AI_ONLY":
        if not ai_available:
            return result(OCC_UNKNOWN, f"AI không khả dụng – {ai_error}" if ai_error else "AI không khả dụng")
        if best is not None:
            return result(OCC_CARFULL, f"AI nhận diện {best.class_name} {best.confidence:.0%} trong ô", "AI", best)
        note = f" (thấy {_names(others)} – không thuộc đối tượng áp dụng)" if others else ""
        return result(OCC_EMPTY, f"AI không thấy đối tượng áp dụng nào trong ô{note}", "AI")

    # HYBRID
    if best is not None:
        if cv_object:
            return result(OCC_CARFULL, f"{seen}, AI nhận diện {best.class_name} {best.confidence:.0%}", "CV+AI", best)
        return result(OCC_CARFULL, f"AI nhận diện {best.class_name} {best.confidence:.0%} trong ô "
                      "(xử lý ảnh chưa tách được vật khỏi nền)", "AI", best)
    if cv_object:
        if not ai_available:
            why = f"AI không khả dụng ({ai_error})" if ai_error else "AI không khả dụng"
            return result(OCC_UNKNOWN, f"{seen} – {why} nên chưa biết là gì", "CV")
        if others:
            return result(OCC_UNKNOWN, f"{seen} – AI nhận ra {_names(others)} nhưng không thuộc đối tượng áp dụng", "CV+AI")
        return result(OCC_UNKNOWN, f"{seen} – AI không nhận ra vật này", "CV+AI")
    if glare:
        return result(OCC_UNKNOWN, glare_reason)
    note = f" (AI thấy {_names(others)} – không thuộc đối tượng áp dụng, bỏ qua)" if others else ""
    return result(OCC_EMPTY, f"Xử lý ảnh: ô trống so với ảnh nền{note}", "CV+AI" if ai_available else "CV")


class OccupancyDebouncer:
    """Confirm the occupancy over consecutive frames (see module docstring)."""

    def __init__(self, window: int = 3):
        self.window = max(1, int(window))
        self.recent: Deque[dict] = deque(maxlen=self.window)
        self.confirmed: Optional[dict] = None
        self.confirmed_at: Optional[float] = None

    def update(self, decision: dict, now: float) -> bool:
        """Add one frame decision; ``True`` when the confirmed state changed."""
        self.recent.append(decision)
        last = list(self.recent)
        if len(last) < self.window:
            return False
        states = [d["state"] for d in last]
        current = self.confirmed["state"] if self.confirmed else None
        if any(state == OCC_EMPTY for state in states) and not all(state == OCC_EMPTY for state in states):
            return False                         # entering or leaving: not settled yet
        if len(set(states)) == 1:
            candidate = last[-1]                 # every recent frame agrees
        elif current in states:
            # CARFULL and UNKNOWN alternating: hold the confirmed one, refresh its details.
            candidate = next(d for d in reversed(last) if d["state"] == current)
        else:
            # Entering from EMPTY with mixed frames: the majority, a tie stays UNKNOWN.
            wanted = OCC_CARFULL if states.count(OCC_CARFULL) * 2 > len(states) else OCC_UNKNOWN
            candidate = next(d for d in reversed(last) if d["state"] == wanted)
        changed = self.confirmed is None or candidate["state"] != self.confirmed["state"]
        self.confirmed = candidate
        if changed:
            self.confirmed_at = now
        return changed

    def snapshot(self) -> dict:
        return {"confirmed": self.confirmed, "since": self.confirmed_at,
                "latest": self.recent[-1] if self.recent else None, "window_frames": self.window}
