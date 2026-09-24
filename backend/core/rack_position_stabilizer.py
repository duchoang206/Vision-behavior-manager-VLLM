"""Temporal position filter for racks, which are normally stationary."""

import math
import os


class RackPositionStabilizer:
    def __init__(self):
        self.states = {}
        self.deadband_m = self._env_float("RACK_POSITION_DEADBAND_M", 0.12, 0.02, 1.0)
        self.move_confirm_m = self._env_float("RACK_MOVE_CONFIRM_M", 0.40, 0.10, 3.0)
        self.move_confirm_frames = self._env_int("RACK_MOVE_CONFIRM_FRAMES", 3, 2, 10)
        self.move_window_sec = self._env_float("RACK_MOVE_CONFIRM_WINDOW_SEC", 1.2, 0.2, 5.0)
        self.move_alpha = self._env_float("RACK_MOVE_SMOOTH_ALPHA", 0.30, 0.05, 1.0)

    @staticmethod
    def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
        try:
            value = float(os.getenv(name, str(default)))
        except ValueError:
            value = default
        return max(minimum, min(maximum, value)) if math.isfinite(value) else default

    @staticmethod
    def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
        try:
            value = int(os.getenv(name, str(default)))
        except ValueError:
            value = default
        return max(minimum, min(maximum, value))

    def update(self, rack_id: str, cam_id: str, x: float, z: float, now: float,
               snapped_to_slot: bool = False, is_carried: bool = False):
        """Return a stable (x, z) point and whether the rack is held static."""
        target = (float(x), float(z))
        if snapped_to_slot or is_carried:
            self.states.pop(rack_id, None)
            return target, snapped_to_slot

        state = self.states.get(rack_id)
        if state is None or state.get("cam_id") != cam_id:
            self.states[rack_id] = {
                "cam_id": cam_id, "position": target, "pending": None, "moving": False, "seen_at": now,
            }
            return target, True

        stable_x, stable_z = state["position"]
        distance = math.hypot(target[0] - stable_x, target[1] - stable_z)
        state["seen_at"] = now
        if distance <= self.deadband_m:
            state["pending"] = None
            state["moving"] = False
            return state["position"], True

        if state.get("moving"):
            updated = (stable_x + (target[0] - stable_x) * self.move_alpha,
                       stable_z + (target[1] - stable_z) * self.move_alpha)
            state["position"] = updated
            if math.hypot(target[0] - updated[0], target[1] - updated[1]) <= self.deadband_m:
                state["position"] = target
                state["moving"] = False
            return state["position"], False

        # A persistent offset smaller than the move-confirmation distance is
        # still camera projection noise for a stored rack, not a relocation.
        if distance < self.move_confirm_m:
            state["pending"] = None
            return state["position"], True

        pending = state.get("pending")
        if (pending is None or now - pending["first_seen"] > self.move_window_sec
                or math.hypot(target[0] - pending["position"][0], target[1] - pending["position"][1]) > self.move_confirm_m):
            pending = {"position": target, "count": 1, "first_seen": now}
        else:
            count = pending["count"] + 1
            previous_x, previous_z = pending["position"]
            pending["position"] = ((previous_x * (count - 1) + target[0]) / count,
                                   (previous_z * (count - 1) + target[1]) / count)
            pending["count"] = count
        state["pending"] = pending

        if pending["count"] >= self.move_confirm_frames:
            state["moving"] = True
            target_x, target_z = pending["position"]
            state["position"] = (
                stable_x + (target_x - stable_x) * self.move_alpha,
                stable_z + (target_z - stable_z) * self.move_alpha,
            )
            state["pending"] = None
            return state["position"], False
        return state["position"], True

    def forget(self, rack_id: str):
        self.states.pop(rack_id, None)
