"""Tầng 7 – Temporal state machine: debounce frame verdicts into a confirmed state.

* ``OK`` / ``NG`` / ``DETECTED`` are confirmed after ``window_frames``
  consecutive frames that agree (status *and* slot state) - a person walking
  through cannot flip the slot for one frame.
* ``UNCERTAIN`` is confirmed once it has lasted ``window_frames`` frames *and*
  ``uncertain_hold_sec`` (target.md Test 5.3: "kéo dài quá 1.5 giây").  The
  same happens when frames keep disagreeing for that long without any state
  settling (``ERR_UNSTABLE_STATE``).
* **Safety interlock**: the moment a frame departs from a confirmed ``OK``
  (empty or occupied - both grant the AGV permission), the interlock is raised
  at once, without waiting for confirmation (plan: "Khóa an toàn lập tức").
  It is released only by a fresh ``window_frames`` confirmation.
* An operator decision on a confirmed ``UNCERTAIN`` (approve / reject) holds
  while the frames stay in that same uncertain state.
"""

import time
from dataclasses import dataclass
from typing import Optional, Tuple

from .config import (ERR_OPERATOR_REJECTED, ERR_UNSTABLE_STATE, SLOT_UNKNOWN, STATUS_NG, STATUS_OK,
                     STATUS_UNCERTAIN)

EVENT_CONFIRMED = "CONFIRMED"
EVENT_INTERLOCK = "INTERLOCK"
DEFAULT_UNCERTAIN_HOLD_SEC = 1.5


def _key(report: dict) -> Tuple[str, str]:
    return str(report.get("status")), str(report.get("slot_state"))


@dataclass
class TemporalUpdate:
    event: Optional[str]
    state: dict


class TemporalInspectionManager:
    def __init__(self, window_frames: int = 3, uncertain_hold_sec: float = DEFAULT_UNCERTAIN_HOLD_SEC):
        self.window_frames = max(1, int(window_frames))
        self.uncertain_hold_sec = float(uncertain_hold_sec)
        self.confirmed: Optional[dict] = None
        self.confirmed_at: Optional[float] = None
        self.latest: Optional[dict] = None
        self.candidate: Optional[Tuple[str, str]] = None
        self.streak = 0
        self.candidate_since: Optional[float] = None
        self.unsettled_since: Optional[float] = None
        self.interlock = False
        self.override_key: Optional[Tuple[str, str]] = None
        self.frames = 0

    @property
    def confirmed_key(self) -> Optional[Tuple[str, str]]:
        return _key(self.confirmed) if self.confirmed else None

    def _confirm(self, report: dict, now: float) -> TemporalUpdate:
        self.confirmed = dict(report)
        self.confirmed_at = now
        self.interlock = False
        self.override_key = None
        self.unsettled_since = None
        return TemporalUpdate(EVENT_CONFIRMED, self.snapshot(now))

    def update(self, report: dict, now: Optional[float] = None) -> TemporalUpdate:
        now = time.time() if now is None else now
        self.frames += 1
        self.latest = report
        key = _key(report)
        if key == self.candidate:
            self.streak += 1
        else:
            self.candidate, self.streak, self.candidate_since = key, 1, now

        held = self.override_key is not None and key == self.override_key
        if key == self.confirmed_key or held:
            self.unsettled_since = None
            if held:
                return TemporalUpdate(None, self.snapshot(now))
            if self.interlock and self.streak >= self.window_frames:
                return self._confirm(report, now)          # release: the OK state is re-established
            if not self.interlock:
                self.confirmed = dict(report)               # keep the latest measurement of a stable state
            return TemporalUpdate(None, self.snapshot(now))

        if self.unsettled_since is None:
            self.unsettled_since = now
        event = None
        if self.confirmed is not None and self.confirmed.get("status") == STATUS_OK and not self.interlock:
            self.interlock = True
            event = EVENT_INTERLOCK

        settled = self.streak >= self.window_frames
        if key[0] == STATUS_UNCERTAIN:
            if settled and now - self.candidate_since >= self.uncertain_hold_sec:
                return self._confirm(report, now)
        elif settled:
            return self._confirm(report, now)
        if (now - self.unsettled_since >= self.uncertain_hold_sec
                and (self.confirmed is None or self.confirmed.get("status") != STATUS_UNCERTAIN)
                and key[0] != STATUS_UNCERTAIN):
            unstable = {"status": STATUS_UNCERTAIN, "slot_state": SLOT_UNKNOWN, "code": ERR_UNSTABLE_STATE,
                        "message": "Kết quả dao động liên tục giữa các khung hình – giữ UNCERTAIN an toàn",
                        "errors": [ERR_UNSTABLE_STATE], "measurement": report.get("measurement")}
            return self._confirm(unstable, now)
        return TemporalUpdate(event, self.snapshot(now))

    def apply_operator_decision(self, approve: bool, operator: str = "", now: Optional[float] = None) -> TemporalUpdate:
        """Supervisor resolution of a confirmed UNCERTAIN (target.md Test 5.3)."""
        now = time.time() if now is None else now
        if self.confirmed is None or self.confirmed.get("status") != STATUS_UNCERTAIN:
            raise ValueError("Chỉ duyệt được khi trạng thái chốt là UNCERTAIN")
        held = _key(self.confirmed)
        base = dict(self.latest or self.confirmed)
        if approve:
            decided = {**base, "status": STATUS_OK, "code": "OK_OPERATOR_APPROVED",
                       "slot_state": base.get("slot_state") if base.get("slot_state") != SLOT_UNKNOWN else "OCCUPIED",
                       "message": "Giám sát viên chấp thuận – cho phép AGV tiếp cận", "errors": []}
        else:
            decided = {**base, "status": STATUS_NG, "code": ERR_OPERATOR_REJECTED,
                       "message": "Giám sát viên từ chối – chặn ô hàng", "errors": [ERR_OPERATOR_REJECTED]}
        decided["operator_decision"] = {"approved": bool(approve), "operator": operator, "at": now}
        update = self._confirm(decided, now)
        self.override_key = held
        return update

    def snapshot(self, now: Optional[float] = None) -> dict:
        now = time.time() if now is None else now
        return {
            "confirmed": self.confirmed,
            "confirmed_status": self.confirmed.get("status") if self.confirmed else "PENDING",
            "confirmed_at": self.confirmed_at,
            "interlock": self.interlock,
            "agv_permission": bool(self.confirmed and self.confirmed.get("status") == STATUS_OK and not self.interlock),
            "candidate": list(self.candidate) if self.candidate else None,
            "streak": self.streak,
            "window_frames": self.window_frames,
            "unsettled_for_sec": round(now - self.unsettled_since, 3) if self.unsettled_since else 0.0,
            "operator_hold": self.override_key is not None,
            "frames": self.frames,
        }
