"""Temporal position filter for racks, providing rock-solid stationary locking and prompt relocation tracking."""

import math
import os
from typing import Optional, Tuple


class RackPositionStabilizer:
    def __init__(self,
                 deadband_m: Optional[float] = None,
                 move_confirm_m: Optional[float] = None,
                 move_confirm_frames: Optional[int] = None,
                 min_move_duration_sec: Optional[float] = None,
                 move_window_sec: Optional[float] = None,
                 move_alpha: Optional[float] = None,
                 settle_frames: Optional[int] = None):
        self.states = {}
        self.deadband_m = deadband_m if deadband_m is not None else self._env_float("RACK_POSITION_DEADBAND_M", 0.15, 0.02, 1.0)
        self.move_confirm_m = move_confirm_m if move_confirm_m is not None else self._env_float("RACK_MOVE_CONFIRM_M", 0.35, 0.10, 5.0)
        self.move_confirm_frames = move_confirm_frames if move_confirm_frames is not None else self._env_int("RACK_MOVE_CONFIRM_FRAMES", 5, 2, 100)
        self.min_move_duration_sec = min_move_duration_sec if min_move_duration_sec is not None else self._env_float("RACK_MIN_MOVE_DURATION_SEC", 0.35, 0.05, 10.0)
        self.move_window_sec = move_window_sec if move_window_sec is not None else self._env_float("RACK_MOVE_CONFIRM_WINDOW_SEC", 3.0, 0.2, 10.0)
        self.move_alpha = move_alpha if move_alpha is not None else self._env_float("RACK_MOVE_SMOOTH_ALPHA", 0.65, 0.05, 1.0)
        self.settle_frames_required = settle_frames if settle_frames is not None else self._env_int("RACK_SETTLE_FRAMES", 5, 1, 50)

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
               snapped_to_slot: bool = False, is_carried: bool = False) -> Tuple[Tuple[float, float], bool]:
        """Return a stable (x, z) point and whether the rack is held static."""
        target = (float(x), float(z))
        if snapped_to_slot or is_carried:
            self.states.pop(rack_id, None)
            return target, snapped_to_slot or is_carried

        state = self.states.get(rack_id)
        if state is None:
            self.states[rack_id] = {
                "cam_id": cam_id, "position": target, "moving": False, "seen_at": now,
                "departure_start": None, "departure_frames": 0,
                "settle_frames": 0, "last_target": target,
            }
            return target, True

        stable_x, stable_z = state["position"]
        last_target = state.get("last_target", target)
        state["last_target"] = target
        state["seen_at"] = now

        # Different camera observing the same rack: if close to existing stable position, keep stable position
        if state.get("cam_id") != cam_id:
            if math.hypot(target[0] - stable_x, target[1] - stable_z) <= self.move_confirm_m:
                state["cam_id"] = cam_id
                return state["position"], True
            else:
                state["cam_id"] = cam_id
                state["position"] = target
                state["moving"] = False
                state["departure_start"] = None
                state["departure_frames"] = 0
                state["settle_frames"] = 0
                return target, True

        dist_from_stable = math.hypot(target[0] - stable_x, target[1] - stable_z)

        # 1. Active movement tracking (user dragging rack or forklift repositioning)
        if state.get("moving"):
            step_movement = math.hypot(target[0] - last_target[0], target[1] - last_target[1])
            if step_movement <= self.deadband_m:
                state["settle_frames"] += 1
            else:
                state["settle_frames"] = 0

            # Follow moving rack smoothly
            updated = (stable_x + (target[0] - stable_x) * self.move_alpha,
                       stable_z + (target[1] - stable_z) * self.move_alpha)
            state["position"] = updated

            # When movement ceases for required settle frames or converges within deadband, lock position
            dist_to_target = math.hypot(target[0] - updated[0], target[1] - updated[1])
            if (state["settle_frames"] >= self.settle_frames_required and dist_to_target <= self.deadband_m) or dist_to_target <= 0.04:
                state["position"] = target
                state["moving"] = False
                state["departure_start"] = None
                state["departure_frames"] = 0
                state["settle_frames"] = 0
                return target, True

            return updated, False

        # 2. Stationary mode: check if within static noise deadband or projection tolerance
        if dist_from_stable <= self.deadband_m or dist_from_stable < self.move_confirm_m:
            state["departure_start"] = None
            state["departure_frames"] = 0
            return (stable_x, stable_z), True

        # Candidate departure from static position
        if state.get("departure_start") is None:
            state["departure_start"] = now
            state["departure_frames"] = 1
        else:
            state["departure_frames"] += 1

        duration = now - state["departure_start"]
        if state["departure_frames"] >= self.move_confirm_frames and duration >= self.min_move_duration_sec:
            state["moving"] = True
            state["settle_frames"] = 0
            updated = (stable_x + (target[0] - stable_x) * self.move_alpha,
                       stable_z + (target[1] - stable_z) * self.move_alpha)
            state["position"] = updated
            return updated, False

        return (stable_x, stable_z), True

    def forget(self, rack_id: str):
        self.states.pop(rack_id, None)
