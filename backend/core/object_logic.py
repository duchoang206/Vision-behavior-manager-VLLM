"""
Object Logic Module – Logic tách biệt theo loại đối tượng.

Kiến trúc:
  PersonLogic  → Fall detection, ground point, pose fallback
  RackLogic    → Static footprint polygon, occupancy check (IoU sàn)
  RobotLogic   → Ground point, Vision+FMS fusion, carried rack binding
"""

import math
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from core.person_analytics import PersonLogic as TemporalPersonLogic


# ---------------------------------------------------------------------------
# PERSON LOGIC
# ---------------------------------------------------------------------------

class LegacyPersonLogic:
    """
    Logic cho đối tượng Person (người).
    - Ground point: center bottom bbox (trung điểm 2 mắt cá chân nếu có keypoints)
    - Fall detection: dựa vào aspect ratio bbox + trạng thái track
    - Output bổ sung: fall_detected, posture, ground_point (pixel normalized)
    """

    FALL_ASPECT_RATIO_THRESHOLD = 1.8   # w/h > 1.8 → có thể đang nằm/té
    FALL_CONFIDENCE_THRESHOLD   = 0.65  # confidence tối thiểu để báo ngã

    def process(
        self,
        det: dict,
        floor_pos: Tuple[float, float],
        cam_id: str
    ) -> dict:
        """
        Args:
            det: detection dict với keys: bbox [x,y,w,h], confidence, track_state, velocity
            floor_pos: (floor_x, floor_y) từ homography
            cam_id: camera ID để log

        Returns:
            dict với: ground_point, fall_detected, posture, border_color
        """
        bbox = det.get("bbox", [0, 0, 0, 0])
        x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        conf = float(det.get("confidence", 0.9))

        # Ground point: trung điểm đáy bbox (center bottom)
        ground_u = round(x + w / 2.0, 4)  # normalized [0,1]
        ground_v = round(y + h, 4)

        # Fall detection: aspect ratio method
        # Nếu người nằm → bbox ngang (w >> h)
        aspect_ratio = (w / h) if h > 1e-4 else 0.0
        is_fallen = (
            aspect_ratio >= self.FALL_ASPECT_RATIO_THRESHOLD
            and conf >= self.FALL_CONFIDENCE_THRESHOLD
            and h > 0.05   # bbox không quá nhỏ
        )

        # Posture classification
        if is_fallen:
            posture = "fallen"
            border_color = "red"
        elif h > 0 and (w / h) < 0.6:
            posture = "standing"
            border_color = "green"
        else:
            posture = "walking"
            border_color = "blue"

        return {
            "ground_point": [ground_u, ground_v],
            "fall_detected": is_fallen,
            "posture": posture,
            "border_color": border_color,
            "aspect_ratio": round(aspect_ratio, 3),
        }


# ---------------------------------------------------------------------------
# RACK LOGIC
# ---------------------------------------------------------------------------

class RackLogic:
    """
    Logic cho đối tượng Rack (kệ hàng) – vật thể tĩnh/bán tĩnh.

    - Footprint polygon: 4 góc chân kệ trên mặt sàn (X_w, Y_w)
    - Kích thước thực tế default: 1.2m × 0.8m (chiều sâu x chiều rộng)
    - Occupancy check: IoU giữa footprint kệ và vị trí robot trên sàn
    """

    DEFAULT_RACK_WIDTH_M  = 1.2   # chiều rộng kệ (m) theo X
    DEFAULT_RACK_DEPTH_M  = 0.8   # chiều sâu kệ (m) theo Y
    DEFAULT_RACK_HEIGHT_M = 1.5   # chiều cao kệ (m)
    OCCUPANCY_RADIUS_M    = 0.8   # Khoảng cách robot → rack < Rm → "occupied"

    def process(
        self,
        det: dict,
        floor_pos: Tuple[float, float],
        robot_floor_positions: List[Tuple[float, float, str]],  # [(fx, fy, label), ...]
        rack_3d_size: Optional[Tuple[float, float, float]] = None
    ) -> dict:
        """
        Args:
            det: detection dict
            floor_pos: (floor_x, floor_y) tọa độ trung tâm rack trên sàn
            robot_floor_positions: list [(fx, fy, robot_label)] các robot hiện tại
            rack_3d_size: (width_m, depth_m, height_m) kích thước thực kệ

        Returns:
            dict với: footprint_polygon, occupancy_status, occupying_id, floor_x, floor_y
        """
        fx, fy = float(floor_pos[0]), float(floor_pos[1])

        # Kích thước kệ
        if rack_3d_size and len(rack_3d_size) >= 2:
            half_w = rack_3d_size[0] / 2.0
            half_d = rack_3d_size[1] / 2.0
        else:
            half_w = self.DEFAULT_RACK_WIDTH_M / 2.0
            half_d = self.DEFAULT_RACK_DEPTH_M / 2.0

        # Footprint: 4 góc chân kệ trên mặt sàn (counter-clockwise)
        footprint_polygon = [
            [round(fx - half_w, 3), round(fy - half_d, 3)],  # bottom-left
            [round(fx + half_w, 3), round(fy - half_d, 3)],  # bottom-right
            [round(fx + half_w, 3), round(fy + half_d, 3)],  # top-right
            [round(fx - half_w, 3), round(fy + half_d, 3)],  # top-left
        ]

        # Occupancy check: robot nào nằm trong bán kính OCCUPANCY_RADIUS_M?
        occupying_id = None
        min_dist = float("inf")

        for (rx, ry, rlabel) in robot_floor_positions:
            dist = math.hypot(float(rx) - fx, float(ry) - fy)
            if dist < min_dist:
                min_dist = dist
                if dist <= self.OCCUPANCY_RADIUS_M:
                    occupying_id = rlabel

        occupancy_status = "occupied" if occupying_id else "empty"

        return {
            "footprint_polygon": footprint_polygon,
            "occupancy_status": occupancy_status,
            "occupying_id": occupying_id,
            "floor_x": round(fx, 3),
            "floor_y": round(fy, 3),
            "nearest_robot_dist_m": round(min_dist, 2) if min_dist != float("inf") else None,
        }


# ---------------------------------------------------------------------------
# ROBOT LOGIC
# ---------------------------------------------------------------------------

class RobotLogic:
    """
    Logic cho đối tượng Robot/AMR (xe tự hành).

    - Ground point: tâm đáy bbox → floor homography
    - Vision+FMS fusion: EMA-weighted giữa camera homography và FMS odometry
    - Carried rack detection: dựa vào rack_association
    """

    # Kalman-like EMA weights cho Vision+FMS fusion
    # Khi FMS valid: 60% vision + 40% FMS
    # Khi không có FMS: 100% vision
    VISION_WEIGHT = 0.60
    FMS_WEIGHT    = 0.40
    FMS_STALE_SEC = 5.0   # FMS data bị coi stale nếu quá 5 giây

    def __init__(self):
        # Lưu trạng thái fused position trước đó cho EMA
        self._fused_state: Dict[str, dict] = {}

    def process(
        self,
        det: dict,
        floor_pos: Tuple[float, float],
        robot_label: Optional[str],
        fms_robot_states: Optional[Dict[str, Any]] = None,
    ) -> dict:
        """
        Args:
            det: detection dict với bbox, velocity, track_state
            floor_pos: (floor_x, floor_y) từ camera homography
            robot_label: label đã match (ví dụ "Robot_2001"), hoặc None
            fms_robot_states: dict từ fms_bridge.robot_states

        Returns:
            dict với: ground_point, fused_pos, vision_pos, fms_pos,
                      fusion_mode, carried_rack, battery, speed
        """
        vis_x, vis_y = float(floor_pos[0]), float(floor_pos[1])

        # Ground point (center bottom bbox)
        bbox = det.get("bbox", [0, 0, 0, 0])
        ground_u = round(bbox[0] + bbox[2] / 2.0, 4)
        ground_v = round(bbox[1] + bbox[3], 4)

        # FMS data lookup
        fms_x, fms_y = None, None
        fms_data = {}
        fusion_mode = "vision_only"

        if robot_label and fms_robot_states:
            robot_key = robot_label.replace("Robot_", "")
            for key in (robot_label, robot_key, f"Robot_{robot_key}"):
                if key in fms_robot_states:
                    fms_data = fms_robot_states[key]
                    break

            if fms_data:
                fms_ts = fms_data.get("timestamp", 0)
                is_fresh = (time.time() - float(fms_ts)) < self.FMS_STALE_SEC if fms_ts else False
                fms_pos = fms_data.get("position")
                if is_fresh and fms_pos and len(fms_pos) >= 3:
                    try:
                        fms_x = float(fms_pos[0])
                        fms_y = float(fms_pos[2])  # [X, Y_height, Z] → dùng Z làm Y sàn
                        fusion_mode = "vision_fms_fusion"
                    except (TypeError, ValueError):
                        pass

        # Vision+FMS Fusion (Kalman-like EMA)
        if fms_x is not None and fms_y is not None:
            fused_x = self.VISION_WEIGHT * vis_x + self.FMS_WEIGHT * fms_x
            fused_y = self.VISION_WEIGHT * vis_y + self.FMS_WEIGHT * fms_y
        else:
            fused_x, fused_y = vis_x, vis_y

        # Lưu state để dùng cho EMA frame sau
        key = robot_label or f"robot_{det.get('local_id', 0)}"
        self._fused_state[key] = {
            "fused_x": round(fused_x, 3),
            "fused_y": round(fused_y, 3),
            "updated_at": time.time(),
        }

        return {
            "ground_point": [ground_u, ground_v],
            "fused_pos": [round(fused_x, 3), round(fused_y, 3)],
            "vision_pos": [round(vis_x, 3), round(vis_y, 3)],
            "fms_pos": [round(fms_x, 3), round(fms_y, 3)] if fms_x is not None else None,
            "fusion_mode": fusion_mode,
            "battery": fms_data.get("battery"),
            "fms_speed": fms_data.get("velocity") or fms_data.get("speed"),
            "carried_rack": fms_data.get("carried_rack_id"),
            "fms_status": fms_data.get("status"),
        }


# ---------------------------------------------------------------------------
# Helper: infer category từ class name / label
# ---------------------------------------------------------------------------

def infer_category(label: Optional[str], class_name: str) -> str:
    """
    Xác định category của detection dựa vào matched_label hoặc class_name từ YOLO.

    Priority:
      1. Label prefix: Robot_* → robot, Rack_* → rack, Person_* → person
      2. Class name từ YOLO detector
    """
    label_lower = (label or "").lower()
    class_lower  = class_name.lower()

    # Từ label đã được đặt tên bởi người dùng
    if label_lower.startswith("robot_") or label_lower.startswith("amr_") or label_lower.startswith("agv_"):
        return "robot"
    if label_lower.startswith("rack_") or label_lower.startswith("kệ_") or label_lower.startswith("shelf_"):
        return "rack"
    if label_lower.startswith("person_") or label_lower.startswith("worker_"):
        return "person"

    # Từ YOLO class name
    if any(k in class_lower for k in ("robot", "amr", "agv", "forklift")):
        return "robot"
    if any(k in class_lower for k in ("rack", "shelf", "kệ", "pallet")):
        return "rack"
    if any(k in class_lower for k in ("person", "human", "worker", "pedestrian")):
        return "person"

    return "object"


# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------

person_logic = TemporalPersonLogic()
rack_logic   = RackLogic()
robot_logic  = RobotLogic()
