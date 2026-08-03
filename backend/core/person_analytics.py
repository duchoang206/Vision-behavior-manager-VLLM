import math
import time


COCO_KEYPOINT_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)


def valid_landmarks(keypoints):
    landmarks = {}
    for index, point in enumerate(keypoints or []):
        if index >= len(COCO_KEYPOINT_NAMES) or not isinstance(point, (list, tuple)) or len(point) != 3:
            continue
        if all(isinstance(value, (float, int)) and math.isfinite(value) for value in point):
            if point[2] >= 0.5 and 0 <= point[0] <= 1 and 0 <= point[1] <= 1:
                landmarks[COCO_KEYPOINT_NAMES[index]] = point
    return landmarks


def person_ground_point(bbox, keypoints):
    landmarks = valid_landmarks(keypoints)
    ankles = [landmarks[name] for name in ("left_ankle", "right_ankle") if name in landmarks]
    if ankles:
        return [sum(point[axis] for point in ankles) / len(ankles) for axis in (0, 1)], "ankles"
    return [bbox[0] + bbox[2] / 2, bbox[1] + bbox[3]], "bbox_bottom"


class PersonLogic:
    """Temporal pose-based fall state, without guessing falls from missing pose."""

    def __init__(self, confirm_seconds=1.0, recover_seconds=1.5, max_gap=0.75):
        self.confirm_seconds = confirm_seconds
        self.recover_seconds = recover_seconds
        self.max_gap = max_gap
        self.states = {}

    def process(self, det, floor_pos, cam_id):
        now = float(det.get("observed_at", time.time()))
        bbox = det["bbox"]
        frame_width = max(1.0, float(det.get("frame_width", 1)))
        frame_height = max(1.0, float(det.get("frame_height", 1)))
        aspect_ratio = bbox[2] * frame_width / max(bbox[3] * frame_height, 1e-6)
        landmarks = valid_landmarks(det.get("keypoints"))
        ground, ground_source = person_ground_point(bbox, det.get("keypoints"))
        torso_names = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
        pose_valid = all(name in landmarks for name in torso_names)
        torso_angle = None
        if pose_valid:
            shoulder = [(landmarks["left_shoulder"][axis] + landmarks["right_shoulder"][axis]) / 2 for axis in (0, 1)]
            hip = [(landmarks["left_hip"][axis] + landmarks["right_hip"][axis]) / 2 for axis in (0, 1)]
            delta_x = abs(shoulder[0] - hip[0]) * frame_width
            delta_y = abs(shoulder[1] - hip[1]) * frame_height
            if math.hypot(delta_x, delta_y) >= 0.05 * bbox[3] * frame_height:
                torso_angle = math.degrees(math.atan2(delta_x, delta_y))
            else:
                pose_valid = False
        observed = det.get("tracking_state", "tracked").lower() != "predicted" and float(det.get("confidence", 0)) >= 0.45
        lying = observed and pose_valid and torso_angle >= 60 and aspect_ratio >= 1.1
        upright = observed and pose_valid and torso_angle <= 30 and aspect_ratio < 1.1
        key = (cam_id, det.get("global_id", det.get("local_id")))
        now_states = {identity: state for identity, state in self.states.items() if now - state["last_seen"] < 30}
        self.states = now_states
        state = self.states.setdefault(key, {"last_seen": now, "candidate_since": None, "recovery_since": None, "fallen": False, "samples": 0, "output": {}})
        if now - state["last_seen"] > self.max_gap:
            state.update(candidate_since=None, recovery_since=None, samples=0)
        state["last_seen"] = now
        fall_event = False
        if lying:
            if state["candidate_since"] is None:
                state["candidate_since"] = now
                state["samples"] = 0
            state["samples"] += 1
            state["recovery_since"] = None
            if not state["fallen"] and state["samples"] >= 3 and now - state["candidate_since"] >= self.confirm_seconds:
                state["fallen"] = True
                fall_event = True
        else:
            state["candidate_since"] = None
            state["samples"] = 0
            if upright:
                if state["recovery_since"] is None:
                    state["recovery_since"] = now
                if now - state["recovery_since"] >= self.recover_seconds:
                    state["fallen"] = False
            else:
                state["recovery_since"] = None
        posture = "fallen" if state["fallen"] else "fall_candidate" if lying else "standing" if upright else "unknown"
        output = {
            "ground_point": ground, "ground_point_source": ground_source,
            "keypoints": det.get("keypoints") or [], "keypoint_format": "coco17_xyc_normalized",
            "pose_available": pose_valid, "posture": posture, "fall_detected": state["fallen"],
            "fall_event": fall_event, "fall_detection_source": "pose_temporal" if pose_valid else "unavailable",
            "border_color": "red" if state["fallen"] else "orange" if lying else "green",
            "aspect_ratio": round(aspect_ratio, 3), "torso_angle_deg": round(torso_angle, 2) if torso_angle is not None else None,
        }
        state["output"] = output
        return output
