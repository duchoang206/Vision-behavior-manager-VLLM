"""Time-aligned FMS constraints for visual robot identities, without network I/O."""

import math
import os
import time

import numpy as np

from core.camera_calibrator import CameraCalibrator, camera_calibrator
from core.identity_utils import robot_number_from_label


def robot_identity(target):
    label = str(target.get("label") or "")
    if str(target.get("category") or target.get("class") or "").lower() != "robot":
        return None
    number = robot_number_from_label(label)
    explicit = target.get("fms_robot_id")
    if number is None and explicit is not None:
        number = robot_number_from_label(str(explicit))
    return str(number) if number is not None else None

def _epoch_seconds(value, default=None):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed) or parsed <= 0:
        return default
    return parsed / 1000.0 if parsed >= 1e11 else parsed


def spatial_verification(context, label, bbox):
    context = context or {}
    labels = context.get("labels", {})
    robot_id = labels.get(label)
    if robot_id is None:
        normalized_label = str(label or "").casefold()
        robot_id = next((value for key, value in labels.items() if str(key).casefold() == normalized_label), None)
    unchecked = dict(checked=False, accepted=None, robot_id=robot_id)
    config = context.get("calibration")
    if robot_id is None:
        return dict(unchecked, reason="not_fms_robot")
    if not config:
        return dict(unchecked, reason="camera_uncalibrated")
    pose = context.get("poses", {}).get(robot_id)
    if not pose:
        return dict(unchecked, reason="fms_not_synchronized")
    try:
        left, top, width, height = map(float, bbox)
        if not all(math.isfinite(value) for value in (left, top, width, height)) or min(width, height) <= 0:
            raise ValueError
        image_x, image_y = CameraCalibrator._project_input_point(config, left + width / 2, top + height)
        projected = np.asarray(config["matrix"], dtype=float) @ [image_x, image_y, 1.0]
        if not np.isfinite(projected).all() or abs(projected[2]) < 1e-8:
            raise ValueError
        floor_x, floor_z = projected[:2] / projected[2]
        position = pose["position"]
        distance = math.hypot(floor_x - position[0], floor_z - position[2])
        if not math.isfinite(distance):
            raise ValueError
        tolerance = context["max_distance_m"] + min(0.5, pose.get("skew_ms", 0) / 1000 * context["max_speed_mps"])
        rivals = [(math.hypot(floor_x - other["position"][0], floor_z - other["position"][2]), other_id)
                  for other_id, other in context["poses"].items() if other_id != robot_id]
        rival_distance, rival_id = min(rivals, default=(math.inf, None))
        reason = None
        if distance > tolerance:
            reason = "fms_position_mismatch"
        elif rival_distance < tolerance and rival_distance + context["rival_margin_m"] < distance:
            reason = "fms_other_robot_closer"
        return dict(checked=True, accepted=reason is None, reason=reason, robot_id=robot_id,
                    distance_m=round(distance, 3), tolerance_m=round(tolerance, 3),
                    floor_position=[round(float(floor_x), 4), round(float(floor_z), 4)],
                    fms_position=list(position), fms_skew_ms=round(pose.get("skew_ms", 0), 1),
                    fms_source=pose.get("source"), calibration_id=config.get("save_id"),
                    rival_robot_id=rival_id, rival_distance_m=round(rival_distance, 3) if rival_id else None)
    except (ValueError, TypeError, KeyError, IndexError, OverflowError):
        return dict(unchecked, accepted=False, reason="invalid_spatial_projection")


class RobotSpatialIdentity:
    def __init__(self, calibrator=None, fms=None):
        self.calibrator = calibrator if calibrator is not None else camera_calibrator
        self.fms = fms

    def context(self, cam_id, targets, observed_at, now=None):
        labels = {target["label"]: robot_identity(target) for target in targets if target.get("label")}
        labels = {label: robot_id for label, robot_id in labels.items() if robot_id is not None}
        config = self.calibrator.get_config(cam_id) if labels else None
        result = dict(labels=labels, calibration=config, poses={},
                      max_distance_m=max(0.1, float(os.getenv("REGISTERED_FMS_MAX_DISTANCE_M", "1.5"))),
                      rival_margin_m=max(0.0, float(os.getenv("REGISTERED_FMS_RIVAL_MARGIN_M", "0.35"))),
                      max_speed_mps=max(0.0, float(os.getenv("REGISTERED_FMS_MAX_SPEED_MPS", "1.8"))))
        frame = config.get("fms_frame") if isinstance(config, dict) else None
        required_frame = ("origin_x", "origin_y", "layout_depth")
        try:
            valid_frame = isinstance(frame, dict) and all(
                key in frame and math.isfinite(float(frame[key])) for key in required_frame
            )
            valid_matrix = isinstance(config, dict) and np.asarray(config.get("matrix"), dtype=float).shape == (3, 3)
        except (TypeError, ValueError):
            valid_frame = False
            valid_matrix = False
        if (not config or not valid_matrix or config.get("map_id", "TT") != "TT"
                or config.get("coordinate_space") != "fms_floor_metric" or not valid_frame):
            result["calibration"] = None
            return result
        if self.fms is None:
            from core.fms_bridge import fms_bridge
            self.fms = fms_bridge
        now = time.time() if now is None else now
        if any(abs(float(frame[key]) - getattr(self.fms, key)) > 1e-4 for key in required_frame):
            result["calibration"] = None
            return result
        observed_seconds = _epoch_seconds(observed_at)
        if observed_seconds is None:
            result["calibration"] = None
            return result
        aligned_at = observed_seconds + float((config.get("online") or {}).get("time_offset_ms", 0)) / 1000
        max_skew = min(1.0, max(0.05, float(os.getenv("REGISTERED_FMS_MAX_SKEW_SEC", "0.5"))))
        robot_ids = set(labels.values())
        robot_ids.update(str(number) for key in list(self.fms.robot_states)
                         if (number := robot_number_from_label(str(key))) is not None)
        for robot_id in robot_ids:
            pose = self.fms.pose_at(robot_id, aligned_at, max_skew=max_skew, now=now)
            if pose and len(pose.get("position", [])) == 3 and all(math.isfinite(float(value)) for value in pose["position"]):
                result["poses"][robot_id] = pose
        return result

    def filter_objects(self, cam_id, objects, now=None):
        now = time.time() if now is None else now
        contexts = {}
        accepted, rejected = [], []
        for obj in objects:
            if robot_identity(obj) is None:
                accepted.append(obj)
                continue
            try:
                observed_at = _epoch_seconds(obj.get("observed_at"), default=now)
                if observed_at is None:
                    raise ValueError
            except (ValueError, TypeError):
                obj["spatial_identity"] = dict(checked=False, accepted=False, reason="invalid_observation_time")
                rejected.append(obj["label"])
                continue
            if observed_at not in contexts:
                contexts[observed_at] = self.context(cam_id, objects, observed_at, now=now)
            bbox = [obj.get(key) for key in ("x", "y", "w", "h")]
            verification = spatial_verification(contexts[observed_at], obj.get("label"), bbox)
            obj["spatial_identity"] = verification
            obj["fms_robot_id"] = int(robot_identity(obj))
            if verification["accepted"] is False:
                rejected.append(obj["label"])
            else:
                accepted.append(obj)
        return accepted, rejected


robot_spatial_identity = RobotSpatialIdentity()
