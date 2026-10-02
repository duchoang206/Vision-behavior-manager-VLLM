"""Authoritative in-memory view of every storage slot reported to downstream systems.

The registry is mutated only on the gateway's event loop thread; frame
threads hand transitions over with ``loop.call_soon_threadsafe``.
"""

import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .constants import STATE_CARFULL, STATE_EMPTY

_FMS_SLOT_ID = re.compile(r"^-?\d{1,9}$")


def is_fms_slot_id(slot_id: Any) -> bool:
    """FMS WCS parses ``slot_id`` with ``std::stoi`` - only integers are safe."""
    return bool(_FMS_SLOT_ID.match(str(slot_id or "").strip()))


@dataclass
class SlotEntry:
    cam_id: str
    rule_id: str
    rule_name: str
    slot_id: str
    channel_id: str = ""
    enabled: bool = True
    status: str = "UNKNOWN"  # UNKNOWN | CARFULL | EMPTY
    occupant_label: str = ""
    occupant_id: Any = None
    overlap_ratio: float = 0.0
    updated_at: float = 0.0
    transitions: int = 0
    context: Dict[str, Any] = field(default_factory=dict)

    @property
    def effective_slot_id(self) -> str:
        """Explicit Slot ID, or the rule id for template channels (plan §3.3 fallback)."""
        return self.slot_id or self.rule_id

    def bound_to(self, channel_id: str) -> bool:
        return not self.channel_id or self.channel_id == channel_id

    def as_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data.pop("context", None)
        data["effective_slot_id"] = self.effective_slot_id
        data["state"] = STATE_CARFULL if self.status == "CARFULL" else STATE_EMPTY if self.status == "EMPTY" else None
        data["fms_compatible"] = is_fms_slot_id(self.slot_id)
        return data


def _rule_slot_id(rule: Dict[str, Any]) -> str:
    raw = rule.get("fms_slot_id")
    return str(raw).strip() if raw not in (None, "") else ""


class SlotRegistry:
    def __init__(self) -> None:
        self._entries: Dict[Tuple[str, str], SlotEntry] = {}

    # ── configuration ────────────────────────────────────────────────────
    def sync_rules(self, cam_id: str, rules: Iterable[Dict[str, Any]]) -> None:
        """Replace the slot definitions of ``cam_id`` while keeping live state."""
        keep = set()
        for rule in rules or []:
            rule_type = (rule.get("type") or rule.get("rule_type") or "").lower()
            slot_id = _rule_slot_id(rule)
            if rule_type != "occupancy":
                continue
            key = (cam_id, str(rule.get("id")))
            keep.add(key)
            entry = self._entries.get(key)
            enabled = rule.get("enable_fms_dispatch")
            enabled = True if enabled is None else bool(enabled)
            if entry is None:
                entry = SlotEntry(cam_id=cam_id, rule_id=key[1], rule_name=str(rule.get("name") or key[1]), slot_id=slot_id)
                self._entries[key] = entry
            entry.rule_name = str(rule.get("name") or key[1])
            entry.slot_id = slot_id
            entry.channel_id = str(rule.get("comm_channel_id") or "")
            entry.enabled = enabled
        for key in [k for k in self._entries if k[0] == cam_id and k not in keep]:
            del self._entries[key]

    def remove_camera(self, cam_id: str) -> None:
        for key in [k for k in self._entries if k[0] == cam_id]:
            del self._entries[key]

    # ── runtime state ────────────────────────────────────────────────────
    def get(self, cam_id: str, rule_id: str) -> Optional[SlotEntry]:
        return self._entries.get((cam_id, str(rule_id)))

    def update(self, cam_id: str, rule_id: str, status: str, context: Dict[str, Any]) -> Optional[SlotEntry]:
        entry = self._entries.get((cam_id, str(rule_id)))
        if entry is None:
            return None
        if entry.status != status:
            entry.transitions += 1
        entry.status = status
        entry.occupant_label = context.get("occupant_label") or ""
        entry.occupant_id = context.get("occupant_id")
        entry.overlap_ratio = float(context.get("overlap_ratio") or 0.0)
        entry.updated_at = time.time()
        entry.context = dict(context)
        return entry

    def entries(self) -> List[Dict[str, Any]]:
        return [e.as_dict() for e in sorted(self._entries.values(), key=lambda e: (_sort_key(e.slot_id), e.cam_id))]

    def known_entries(self, channel_id: str) -> List[SlotEntry]:
        return [e for e in self._entries.values()
                if e.enabled and e.status in ("CARFULL", "EMPTY") and e.bound_to(channel_id)]

    def occupied_contexts(self, channel_id: str) -> List[Dict[str, Any]]:
        return [dict(e.context) for e in self.known_entries(channel_id) if e.status == "CARFULL" and e.context]

    def fms_slots(self, channel_id: str) -> Tuple[List[Dict[str, str]], List[str]]:
        """Merged ``[{"slot_id", "state"}]`` snapshot for an FMS WCS channel.

        A slot watched by several cameras is ``Car Full`` when any camera sees
        goods.  Slot ids FMS cannot parse are returned separately so callers
        can warn instead of crashing the WCS parser.
        """
        merged: Dict[str, bool] = {}
        invalid: List[str] = []
        for entry in self.known_entries(channel_id):
            slot_id = entry.slot_id.strip()
            if not slot_id:
                continue  # FMS only receives explicitly assigned slots
            if not is_fms_slot_id(slot_id):
                invalid.append(slot_id)
                continue
            merged[slot_id] = merged.get(slot_id, False) or entry.status == "CARFULL"
        slots = [{"slot_id": slot_id, "state": STATE_CARFULL if full else STATE_EMPTY}
                 for slot_id, full in sorted(merged.items(), key=lambda item: _sort_key(item[0]))]
        return slots, sorted(set(invalid))

    def merged_state(self, channel_id: str, slot_id: str) -> Optional[str]:
        states = [e.status for e in self.known_entries(channel_id)
                  if e.slot_id and e.slot_id.strip() == str(slot_id).strip()]
        if not states:
            return None
        return STATE_CARFULL if "CARFULL" in states else STATE_EMPTY


def _sort_key(slot_id: str):
    text = str(slot_id).strip()
    return (0, int(text), "") if is_fms_slot_id(text) else (1, 0, text)
