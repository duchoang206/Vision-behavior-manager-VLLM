"""
Session 3: Rack Association and Storage Slot Placement Engine.
Manages:
1. Static Racks in Storage Slots (Slot_A1 -> Rack_A1)
2. Mobile Racks carried on AMR Robot backs (Robot_9001 -> has_rack: True, carried_rack_id: "Rack_A1")
3. Geometric Vertical Containment and IoU Association
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any


def compute_iou(boxA: List[float], boxB: List[float]) -> float:
    """Computes standard Intersection-over-Union (IoU) between two bounding boxes [x1, y1, x2, y2]"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    if interArea == 0:
        return 0.0
        
    boxAArea = max(1.0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(1.0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return float(iou)


def check_point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
    """Ray casting point in polygon test"""
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


class RackAssociationEngine:
    """
    Associates Racks with Storage Slots and AMR Robots.
    Determines whether a robot is carrying a rack, and whether a rack is parked in a slot.
    """
    def __init__(self):
        # Slot definitions: { slot_id: { "name": "Kệ A · Ô 01", "polygon": [[u1, v1], ...] } }
        self.storage_slots: Dict[str, Dict[str, Any]] = {}
        # Active robot-rack bindings: { "Robot_9001": "Rack_A1" }
        self.active_carrier_bindings: Dict[str, str] = {}

    def register_storage_slot(self, slot_id: str, name: str, polygon: List[List[float]], cam_id: str):
        """Registers a storage rack bay/slot ROI"""
        self.storage_slots[slot_id] = {
            "slot_id": slot_id,
            "name": name,
            "polygon": polygon,
            "cam_id": cam_id
        }

    def check_robot_rack_mounting(
        self,
        robot_bbox: List[float],
        rack_bbox: List[float]
    ) -> Tuple[bool, float]:
        """
        Determines if a rack is mounted/carried on top of a robot.
        Condition:
        1. Horizontal center alignment: Rack center_x is within robot [x1, x2].
        2. Vertical stacking: Bottom of rack sits on/near the top half or center of the robot.
        """
        rx1, ry1, rx2, ry2 = robot_bbox
        kx1, ky1, kx2, ky2 = rack_bbox
        
        r_w = rx2 - rx1
        r_h = ry2 - ry1
        k_w = kx2 - kx1
        k_h = ky2 - ky1
        
        r_cx = (rx1 + rx2) / 2.0
        k_cx = (kx1 + kx2) / 2.0
        
        # 1. Horizontal center distance relative to robot width
        h_offset = abs(r_cx - k_cx) / max(1.0, r_w)
        if h_offset > 0.45:  # Too far horizontally
            return False, 0.0
            
        # 2. Vertical containment: bottom of rack must be near robot body
        # Rack bottom ky2 should be between ry1 - 0.2*r_h and ry2
        is_vertically_aligned = (ry1 - 0.3 * r_h) <= ky2 <= (ry2 + 0.1 * r_h)
        
        # 3. Compute vertical overlap score
        iou = compute_iou(robot_bbox, rack_bbox)
        
        confidence = 0.5 * (1.0 - min(1.0, h_offset)) + 0.5 * min(1.0, iou * 2.0)
        
        is_mounted = is_vertically_aligned and (h_offset <= 0.35 or iou >= 0.15)
        return is_mounted, round(confidence, 3)

    def process_frame_associations(
        self,
        tracked_objects: List[Dict[str, Any]],
        cam_id: str
    ) -> List[Dict[str, Any]]:
        """
        Takes detected robots and racks from a frame and updates:
        - robot["has_rack"]: bool
        - robot["carried_rack_id"]: Optional[str]
        - rack["is_carried"]: bool
        - rack["carrier_robot"]: Optional[str]
        - rack["stored_slot_id"]: Optional[str]
        """
        robots = [o for o in tracked_objects if "robot" in o.get("class_name", "").lower() or o.get("class_id") == 0]
        racks = [o for o in tracked_objects if "rack" in o.get("class_name", "").lower() or "car" in o.get("class_name", "").lower() or o.get("class_id") == 1]
        
        # Default flags
        for r in robots:
            r["has_rack"] = False
            r["carried_rack_id"] = None
            
        for k in racks:
            k["is_carried"] = False
            k["carrier_robot"] = None
            k["stored_slot_id"] = None
            
        # 1. Match mobile racks on robot backs
        for r in robots:
            r_box = r["bbox"]
            best_rack = None
            best_score = 0.0
            
            for k in racks:
                if k["is_carried"]:
                    continue
                k_box = k["bbox"]
                is_mounted, score = self.check_robot_rack_mounting(r_box, k_box)
                if is_mounted and score > best_score:
                    best_score = score
                    best_rack = k
                    
            if best_rack is not None:
                r_id = r.get("matched_label") or f"Robot_{r['track_id']}"
                rack_id = best_rack.get("matched_label") or f"Rack_{best_rack['track_id']}"
                
                r["has_rack"] = True
                r["carried_rack_id"] = rack_id
                best_rack["is_carried"] = True
                best_rack["carrier_robot"] = r_id
                self.active_carrier_bindings[r_id] = rack_id
                
        # 2. Match static racks in storage slot ROIs
        cam_slots = [s for s in self.storage_slots.values() if s["cam_id"] == cam_id]
        for k in racks:
            if k["is_carried"]:
                continue
            ground_pt = k["ground_point_norm"]  # (u_norm, v_norm)
            for s in cam_slots:
                if check_point_in_polygon(ground_pt, s["polygon"]):
                    k["stored_slot_id"] = s["slot_id"]
                    k["slot_name"] = s["name"]
                    break
                    
        return tracked_objects


# Singleton association engine
rack_association_engine = RackAssociationEngine()
