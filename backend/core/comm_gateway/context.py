"""Event context builders shared by the AI hook, the test API and previews."""

import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .constants import EVENT_SLOT_CARFULL, EVENT_SLOT_EMPTY, STATE_CARFULL, STATE_EMPTY


def now_fields(now: Optional[float] = None) -> Dict[str, Any]:
    now = time.time() if now is None else now
    iso = datetime.fromtimestamp(now, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    return {"timestamp_iso": iso, "timestamp_ms": int(now * 1000)}


def state_from_status(status: str) -> str:
    return STATE_CARFULL if str(status).upper() == "CARFULL" else STATE_EMPTY


def status_from_state(state: str) -> str:
    text = str(state or "").strip().lower().replace("_", " ")
    return "CARFULL" if text in ("car full", "carfull", "full", "occupied", "true", "1") else "EMPTY"


def slot_event_context(*, slot_id: str, status: str, camera_id: str = "", camera_name: str = "",
                       rule_id: str = "", rule_name: str = "", occupant_label: str = "",
                       occupant_id: Any = None, overlap_ratio: float = 0.0, confidence: Any = None,
                       now: Optional[float] = None) -> Dict[str, Any]:
    occupied = status == "CARFULL"
    context = {
        "event": EVENT_SLOT_CARFULL if occupied else EVENT_SLOT_EMPTY,
        "slot_id": str(slot_id),
        "state": STATE_CARFULL if occupied else STATE_EMPTY,
        "status": "CARFULL" if occupied else "EMPTY",
        "is_occupied": occupied,
        "camera_id": camera_id,
        "camera_name": camera_name or camera_id,
        "rule_id": rule_id,
        "rule_name": rule_name,
        "occupant_label": occupant_label if occupied else "",
        "occupant_id": occupant_id if occupied else None,
        "overlap_ratio": round(float(overlap_ratio or 0.0), 2),
        "confidence": confidence,
        "description": "",
    }
    context.update(now_fields(now))
    return context


def sample_context(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Representative data used by Live Preview and Send Test."""
    overrides = {k: v for k, v in dict(overrides or {}).items() if v is not None}
    status = overrides.pop("status", None) or status_from_state(overrides.get("state", STATE_CARFULL))
    context = slot_event_context(
        slot_id=str(overrides.pop("slot_id", "5")), status=status, camera_id="cam_4", camera_name="Cam 4",
        rule_id="rule_demo", rule_name="Ô A1", occupant_label="rack #400", occupant_id=400,
        overlap_ratio=87.5, confidence=0.94)
    overrides.pop("state", None)
    context.update(overrides)
    return context
