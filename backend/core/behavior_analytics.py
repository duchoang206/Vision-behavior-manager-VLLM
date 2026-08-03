import time
import numpy as np
from typing import Any, Dict, List, Tuple, Optional
from shapely.geometry import Point, Polygon, LineString

class BehaviorAnalyticsEngine:
    """
    Real-Time Behavior Analytics & ROI Overlap Monitoring Engine.
    Features:
    1. ROI Area Overlap Ratio (Default threshold = 5.0% overlap or contact point inside)
    2. Temporal Hysteresis Filter (Chống nhiễu rung frame: cần >= 5 frame liên tiếp để chuyển CARFULL, >= 8 frame liên tiếp để về EMPTY)
    3. Trạng thái vị trí: CARFULL (Có hàng/xe) | EMPTY (Trống)
    4. Tripwire Line Crossing (Directional crossing counter: In/Out)
    5. Dwell Time / Loitering
    """
    def __init__(self):
        # cam_id -> list of parsed rules
        self.rules: Dict[str, List[dict]] = {}
        
        # Tripwire cumulative counters: rule_id -> { "in": int, "out": int }
        self.tripwire_counts: Dict[str, dict] = {}
        
        # Track history for line crossing: (cam_id, global_id) -> list of (timestamp, (x, y))
        self.track_positions: Dict[Tuple[str, int], List[Tuple[float, Tuple[float, float]]]] = {}
        
        # Dwell tracking: (cam_id, rule_id, global_id) -> first_seen_timestamp
        self.zone_occupancy: Dict[Tuple[str, str, int], float] = {}
        
        # Hysteresis state filter for ROI noise cancellation: (cam_id, rule_id) -> dict
        # { "status": "EMPTY"|"CARFULL", "occ_frames": int, "empty_frames": int, "occupant_ids": list }
        self.roi_states_filter: Dict[Tuple[str, str], dict] = {}
        
        # Active alert cooldown to avoid duplicate alert flooding: (cam_id, rule_id, global_id) -> last_alert_time
        self.alert_cooldowns: Dict[Tuple[str, str, int], float] = {}

    def _target_matches(self, obj: dict, target_objects: List[str]) -> bool:
        if not target_objects:
            return True
        obj_cls = (obj.get("class") or "object").lower()
        obj_label = (obj.get("label") or "").lower()
        targets = [str(t).lower().strip() for t in target_objects if str(t).strip()]

        obj_terms = {obj_cls}
        if obj_label:
            obj_terms.add(obj_label)
        if "rack" in obj_cls or "rack" in obj_label or "pallet" in obj_cls or "pallet" in obj_label or "storage" in obj_cls:
            obj_terms.add("rack")
        if "robot" in obj_cls or "robot" in obj_label or "agv" in obj_cls or "amr" in obj_cls:
            obj_terms.add("robot")
        if "person" in obj_cls or "human" in obj_cls or "worker" in obj_cls:
            obj_terms.add("person")

        for target in targets:
            target_terms = {target}
            if target in ("pallet", "storage", "shelf"):
                target_terms.add("rack")
            if target in ("delivery-robot", "agv", "amr"):
                target_terms.add("robot")
            if target in ("human", "worker"):
                target_terms.add("person")

            if obj_terms.intersection(target_terms):
                return True
            if any(target in term or term in target for term in obj_terms):
                return True
        return False

    def _numeric_id(self, value: Any) -> int:
        try:
            return int(value)
        except Exception:
            digits = "".join(ch for ch in str(value) if ch.isdigit())
            return int(digits) if digits else 0

    def _get_fms_floor_objects(self) -> List[dict]:
        try:
            from core.fms_bridge import fms_bridge
        except Exception:
            return []

        objects = []
        for robot_id, robot in getattr(fms_bridge, "robot_states", {}).items():
            if robot.get("status") == "OFFLINE":
                continue
            pos = robot.get("position")
            if not pos or len(pos) < 3:
                continue
            try:
                fx = float(pos[0])
                fz = float(pos[2])
            except (TypeError, ValueError):
                continue
            canonical = str(robot_id)
            label = canonical if canonical.startswith("Robot_") else f"Robot_{canonical}"
            objects.append({
                "id": self._numeric_id(robot_id),
                "class": "robot",
                "label": label,
                "pt_geom": Point(fx, fz),
                "floor_point": (fx, fz),
            })
        return objects

    def _build_camera_to_fms_transform(self, camera_points: List[list], fms_points: List[list]) -> Optional[np.ndarray]:
        pair_count = min(len(camera_points), len(fms_points))
        if pair_count < 3:
            return None

        src = np.asarray(camera_points[:pair_count], dtype=np.float64)
        dst = np.asarray(fms_points[:pair_count], dtype=np.float64)
        if src.shape[1] < 2 or dst.shape[1] < 2:
            return None
        src = src[:, :2]
        dst = dst[:, :2]

        if pair_count >= 4:
            try:
                import cv2
                h, _ = cv2.findHomography(src, dst, 0)
                if h is not None and np.isfinite(h).all():
                    return h
            except Exception:
                pass

        try:
            a = []
            b = []
            for (x, y), (fx, fy) in zip(src, dst):
                a.append([x, y, 1.0, 0.0, 0.0, 0.0])
                a.append([0.0, 0.0, 0.0, x, y, 1.0])
                b.extend([fx, fy])
            params, *_ = np.linalg.lstsq(np.asarray(a), np.asarray(b), rcond=None)
            return np.asarray([
                [params[0], params[1], params[2]],
                [params[3], params[4], params[5]],
                [0.0, 0.0, 1.0],
            ], dtype=np.float64)
        except Exception:
            return None

    def _transform_camera_point_to_fms(self, matrix: Optional[np.ndarray], point: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        if matrix is None:
            return None
        try:
            out = matrix @ np.asarray([float(point[0]), float(point[1]), 1.0], dtype=np.float64)
            denom = out[2] if abs(out[2]) > 1e-9 else 1.0
            return float(out[0] / denom), float(out[1] / denom)
        except Exception:
            return None

    def set_rules(self, cam_id: str, rules_list: List[dict]):
        """
        Update rule configurations for a camera.
        """
        parsed_rules = []
        for r in rules_list:
            rule_type = r.get("type") or r.get("rule_type") or "intrusion"
            rule_id = r.get("id", f"rule_{len(parsed_rules)+1}")
            points = r.get("points", [])
            coordinate_space = (r.get("coordinate_space") or "camera").lower()
            camera_points = r.get("camera_points") or (points if coordinate_space == "camera" else [])
            fms_points = r.get("fms_points") or (points if coordinate_space == "fms" else [])
            target_objects = [str(t).lower() for t in r.get("target_objects", ["rack", "delivery-robot", "robot", "person"])]
            threshold = float(r.get("threshold", 5.0))  # Default 5.0% overlap
            
            rule_obj = {
                "id": rule_id,
                "type": rule_type,
                "name": r.get("name", rule_id),
                "points": points,
                "camera_points": camera_points,
                "fms_points": fms_points,
                "target_objects": target_objects,
                "threshold": threshold,
                "direction": r.get("direction", "both"),
                "coordinate_space": coordinate_space,
            }
            
            if rule_type in ("intrusion", "dwell_time", "density", "occupancy") and len(points) >= 3:
                try:
                    poly = Polygon(points)
                    if not poly.is_valid:
                        poly = poly.buffer(0)
                    rule_obj["polygon"] = poly
                except Exception:
                    rule_obj["polygon"] = None
            if len(camera_points) >= 3:
                try:
                    camera_poly = Polygon(camera_points)
                    if not camera_poly.is_valid:
                        camera_poly = camera_poly.buffer(0)
                    rule_obj["camera_polygon"] = camera_poly
                except Exception:
                    rule_obj["camera_polygon"] = None
            if len(fms_points) >= 3:
                try:
                    fms_poly = Polygon(fms_points)
                    if not fms_poly.is_valid:
                        fms_poly = fms_poly.buffer(0)
                    rule_obj["fms_polygon"] = fms_poly
                except Exception:
                    rule_obj["fms_polygon"] = None

            transform = self._build_camera_to_fms_transform(camera_points, fms_points)
            if transform is not None:
                rule_obj["camera_to_fms_matrix"] = transform

            if rule_type == "tripwire" and len(points) >= 2:
                try:
                    rule_obj["line"] = LineString(points[:2])
                except Exception:
                    rule_obj["line"] = None
                    
            parsed_rules.append(rule_obj)
            if rule_id not in self.tripwire_counts:
                self.tripwire_counts[rule_id] = {"in": 0, "out": 0}
                
        self.rules[cam_id] = parsed_rules

    def process_frame(self, cam_id: str, objects: List[dict]) -> Tuple[List[dict], Dict[str, dict], List[dict]]:
        """
        Evaluates current detections against active camera rules.
        Returns:
          - triggered_events: list of alert event dicts
          - tripwire_stats: updated tripwire counters for the camera
          - roi_states: list of ROI status dicts (status: CARFULL / EMPTY)
        """
        now = time.time()
        triggered_events = []
        cam_rules = self.rules.get(cam_id, [])
        
        roi_states: List[dict] = []
        if not cam_rules:
            return triggered_events, self.get_tripwire_stats(cam_id), roi_states
            
        current_gids_in_frame = set()
        
        # Prepare Bounding Box Polygons for all detected objects
        obj_polygons = []
        floor_objects = []
        for obj in objects:
            gid = obj["id"]
            current_gids_in_frame.add(gid)
            x, y, w, h = obj["x"], obj["y"], obj["w"], obj["h"]
            obj_cls = (obj.get("class") or "object").lower()
            
            # Bottom-center ground contact point
            bottom_center = (round(x + w / 2.0, 4), round(y + h, 4))
            pt_geom = Point(bottom_center[0], bottom_center[1])
            
            # 2D Bounding Box Polygon in normalized space
            bbox_poly = Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])
            if not bbox_poly.is_valid:
                bbox_poly = bbox_poly.buffer(0)
                
            obj_polygons.append({
                "id": gid,
                "class": obj_cls,
                "label": obj.get("label"),
                "bbox": [x, y, w, h],
                "bottom_center": bottom_center,
                "pt_geom": pt_geom,
                "bbox_poly": bbox_poly
            })

            try:
                fx = float(obj.get("floor_x"))
                fy = float(obj.get("floor_y"))
                floor_objects.append({
                    "id": gid,
                    "class": obj_cls,
                    "label": obj.get("label"),
                    "pt_geom": Point(fx, fy),
                    "floor_point": (fx, fy),
                })
            except (TypeError, ValueError):
                pass
            
            # Update trajectory for line crossing
            pos_key = (cam_id, gid)
            if pos_key not in self.track_positions:
                self.track_positions[pos_key] = []
            self.track_positions[pos_key].append((now, bottom_center))
            if len(self.track_positions[pos_key]) > 20:
                self.track_positions[pos_key].pop(0)

        # Pre-compute exclusive Best ROI association for Occupancy rules
        # Prevents a single object from falsely occupying multiple adjacent slots due to 3D perspective overlap
        best_roi_for_object = {}
        for o in obj_polygons:
            best_rid = None
            best_score = -1.0
            best_overlap = 0.0
            
            for rule in cam_rules:
                if rule.get("type") != "occupancy":
                    continue
                if not self._target_matches(o, rule.get("target_objects", [])):
                    continue
                    
                c_poly = rule.get("camera_polygon") or (rule.get("polygon") if rule.get("coordinate_space") != "fms" else None)
                f_poly = rule.get("fms_polygon")
                
                ov_ratio = 0.0
                dist_foot = 999.0
                has_foot = False
                
                if c_poly is not None and c_poly.area > 0:
                    if c_poly.intersects(o["bbox_poly"]):
                        inter_a = c_poly.intersection(o["bbox_poly"]).area
                        ov_ratio = (inter_a / c_poly.area) * 100.0
                    dist_foot = c_poly.distance(o["pt_geom"])
                    if c_poly.contains(o["pt_geom"]) or c_poly.touches(o["pt_geom"]) or dist_foot < 0.025:
                        has_foot = True
                        
                fms_match = False
                if f_poly is not None and f_poly.area > 0:
                    tr = rule.get("camera_to_fms_matrix")
                    if tr is not None:
                        mapped = self._transform_camera_point_to_fms(tr, o["bottom_center"])
                        if mapped is not None:
                            pt = Point(mapped[0], mapped[1])
                            if f_poly.buffer(0.35).contains(pt):
                                fms_match = True
                                
                thresh = float(rule.get("threshold", 15.0))
                if ov_ratio >= thresh or has_foot or fms_match:
                    score = ov_ratio + (60.0 if has_foot else 0.0) + (40.0 if fms_match else 0.0) - min(dist_foot * 100.0, 50.0)
                    if score > best_score:
                        best_score = score
                        best_rid = rule["id"]
                        best_overlap = ov_ratio
                        
            if best_rid is not None:
                best_roi_for_object[o["id"]] = (best_rid, best_score, best_overlap)

        # Process each rule for this camera
        for rule in cam_rules:
            rule_id = rule["id"]
            rule_type = rule["type"]
            target_objects = rule.get("target_objects", [])
            threshold = float(rule.get("threshold", 5.0))
            coordinate_space = rule.get("coordinate_space", "camera")
            use_fms_space = rule_type == "occupancy" and rule.get("fms_polygon") is not None
            # --- 1. ROI OCCUPANCY MONITORING (CARFULL / EMPTY with Multi-Modal Vision + FMS Fusion) ---
            if rule_type in ("intrusion", "dwell_time", "density", "occupancy"):
                raw_occupant_ids = []
                raw_occupant_labels = []
                max_overlap_ratio = 0.0
                seen_candidates = set()

                def remember_occupant(o: dict):
                    candidate_id = self._numeric_id(o["id"])
                    candidate_label = o.get("label") or f"#{o['id']}"
                    candidate_key = candidate_label or str(candidate_id)
                    if candidate_key in seen_candidates:
                        return
                    seen_candidates.add(candidate_key)
                    raw_occupant_ids.append(candidate_id)
                    raw_occupant_labels.append(candidate_label)

                camera_eval_poly = rule.get("camera_polygon") or (rule.get("polygon") if coordinate_space != "fms" else None)
                fms_eval_poly = rule.get("fms_polygon") or (rule.get("polygon") if coordinate_space == "fms" else None)

                # 1.1 Direct Visual Perception (2D Camera Space Bounding Box Overlap)
                if camera_eval_poly is not None and camera_eval_poly.area > 0:
                    camera_roi_area = camera_eval_poly.area
                    for o in obj_polygons:
                        if not self._target_matches(o, target_objects):
                            continue

                        # If occupancy rule, only consider the object if this ROI is its dominant/primary slot
                        if rule_type == "occupancy":
                            best_match = best_roi_for_object.get(o["id"])
                            if not (best_match is not None and best_match[0] == rule_id):
                                continue

                        overlap_ratio = 0.0
                        try:
                            if camera_eval_poly.intersects(o["bbox_poly"]):
                                inter_area = camera_eval_poly.intersection(o["bbox_poly"]).area
                                overlap_ratio = (inter_area / camera_roi_area) * 100.0
                        except Exception:
                            overlap_ratio = 0.0

                        if overlap_ratio > max_overlap_ratio:
                            max_overlap_ratio = overlap_ratio

                        # Point containment checks (bottom-center contact point)
                        dist_to_contact = camera_eval_poly.distance(o["pt_geom"])
                        is_point_inside = (
                            camera_eval_poly.contains(o["pt_geom"])
                            or camera_eval_poly.touches(o["pt_geom"])
                            or dist_to_contact < 0.025
                        )

                        if overlap_ratio >= threshold or is_point_inside:
                            remember_occupant(o)

                # 1.2 Floor / FMS Telemetry Fusion (AMR state & projected coordinates with 0.35m tolerance buffer)
                if fms_eval_poly is not None and fms_eval_poly.area > 0:
                    buffered_fms_poly = fms_eval_poly.buffer(0.35)
                    candidates = []
                    transform = rule.get("camera_to_fms_matrix")
                    if transform is not None:
                        for o in obj_polygons:
                            if not self._target_matches(o, target_objects):
                                continue
                            if rule_type == "occupancy":
                                best_match = best_roi_for_object.get(o["id"])
                                if not (best_match is not None and best_match[0] == rule_id):
                                    continue
                            mapped = self._transform_camera_point_to_fms(transform, o["bottom_center"])
                            if mapped is not None:
                                candidates.append({
                                    **o,
                                    "pt_geom": Point(mapped[0], mapped[1]),
                                    "floor_point": mapped,
                                })
                    else:
                        candidates.extend(floor_objects)

                    candidates.extend(self._get_fms_floor_objects())
                    for o in candidates:
                        if not self._target_matches(o, target_objects):
                            continue
                        if buffered_fms_poly.contains(o["pt_geom"]) or buffered_fms_poly.touches(o["pt_geom"]):
                            remember_occupant(o)
                            if max_overlap_ratio == 0.0:
                                max_overlap_ratio = 100.0
                
                # Instantaneous frame occupancy condition
                instant_occupied = len(raw_occupant_ids) > 0 or max_overlap_ratio >= threshold
                
                # Temporal Hysteresis Filter (Chống nhiễu rung lắc)
                state_key = (cam_id, rule_id)
                if state_key not in self.roi_states_filter:
                    self.roi_states_filter[state_key] = {
                        "status": "EMPTY",
                        "occ_frames": 0,
                        "empty_frames": 0,
                        "occupant_ids": [],
                        "occupant_labels": []
                    }
                filter_state = self.roi_states_filter[state_key]
                
                if instant_occupied:
                    filter_state["occ_frames"] += 1
                    filter_state["empty_frames"] = 0
                    filter_state["occupant_ids"] = raw_occupant_ids
                    filter_state["occupant_labels"] = raw_occupant_labels
                    # Cần >= 2 frame liên tiếp để chuyển sang CARFULL (chống noise 1-frame, phản hồi nhanh)
                    if filter_state["occ_frames"] >= 2 and filter_state["status"] != "CARFULL":
                        filter_state["status"] = "CARFULL"
                        
                        # Trigger Alarm Event
                        cooldown_key = (cam_id, rule_id, raw_occupant_ids[0] if raw_occupant_ids else 0)
                        if now - self.alert_cooldowns.get(cooldown_key, 0) > 3.0:
                            self.alert_cooldowns[cooldown_key] = now
                            occ_str = ", ".join(raw_occupant_labels or [f"#{i}" for i in raw_occupant_ids])
                            triggered_events.append({
                                "cam_id": cam_id,
                                "global_id": raw_occupant_ids[0] if raw_occupant_ids else 0,
                                "rule_id": rule_id,
                                "rule_type": rule_type,
                                "severity": "critical",
                                "description": f"🚨 Ô vị trí '{rule['name']}' chuyển sang CÓ HÀNG (CARFULL) bởi {occ_str}",
                                "timestamp": int(now * 1000)
                            })
                else:
                    filter_state["empty_frames"] += 1
                    filter_state["occ_frames"] = 0
                    # Cần >= 5 frame liên tiếp trống để chuyển về EMPTY (chống mất nhận diện tạm thời khi bị che khuất)
                    if filter_state["empty_frames"] >= 5 and filter_state["status"] != "EMPTY":
                        filter_state["status"] = "EMPTY"
                        filter_state["occupant_ids"] = []
                        filter_state["occupant_labels"] = []

                roi_states.append({
                    "roi_id": rule_id,
                    "name": rule["name"],
                    "status": filter_state["status"],  # "CARFULL" | "EMPTY"
                    "occupant_ids": filter_state["occupant_ids"],
                    "occupant_labels": filter_state.get("occupant_labels", raw_occupant_labels),
                    "polygon": rule.get("camera_points") or ([] if coordinate_space == "fms" else rule.get("points", [])),
                    "fms_polygon": rule.get("fms_points") or ([] if coordinate_space != "fms" else rule.get("points", [])),
                    "rule_type": rule_type,
                    "coordinate_space": "hybrid" if rule.get("camera_points") and rule.get("fms_points") else coordinate_space,
                    "overlap_ratio": round(max_overlap_ratio, 2),
                })

            # --- 2. TRIPWIRE / LINE CROSSING ---
            elif rule_type == "tripwire" and rule.get("line"):
                line_geom = rule["line"]
                for o in obj_polygons:
                    pos_key = (cam_id, o["id"])
                    if len(self.track_positions.get(pos_key, [])) >= 2:
                        prev_pos = self.track_positions[pos_key][-2][1]
                        motion_seg = LineString([prev_pos, o["bottom_center"]])
                        
                        if motion_seg.intersects(line_geom):
                            p1 = rule["points"][0]
                            p2 = rule["points"][1]
                            line_vec = (p2[0] - p1[0], p2[1] - p1[1])
                            motion_vec = (o["bottom_center"][0] - prev_pos[0], o["bottom_center"][1] - prev_pos[1])
                            
                            cross_prod = line_vec[0] * motion_vec[1] - line_vec[1] * motion_vec[0]
                            direction = "in" if cross_prod > 0 else "out"
                            
                            cooldown_key = (cam_id, rule_id, o["id"])
                            if now - self.alert_cooldowns.get(cooldown_key, 0) > 2.0:
                                self.alert_cooldowns[cooldown_key] = now
                                self.tripwire_counts[rule_id][direction] = self.tripwire_counts[rule_id].get(direction, 0) + 1
                                
                                triggered_events.append({
                                    "cam_id": cam_id,
                                    "global_id": o["id"],
                                    "rule_id": rule_id,
                                    "rule_type": "tripwire",
                                    "severity": "info",
                                    "direction": direction,
                                    "description": f"Vượt vạch ảo '{rule['name']}' ({direction.upper()}) bởi đối tượng #{o['id']}",
                                    "counts": dict(self.tripwire_counts[rule_id]),
                                    "bbox": o["bbox"],
                                    "timestamp": int(now * 1000)
                                })

        # Cleanup expired track positions
        self._cleanup(now, current_gids_in_frame, cam_id)
        
        return triggered_events, self.get_tripwire_stats(cam_id), roi_states

    def _cleanup(self, now: float, current_gids: set, cam_id: str):
        keys_to_del = [k for k in self.track_positions.keys() if k[0] == cam_id and k[1] not in current_gids]
        for k in keys_to_del:
            self.track_positions.pop(k, None)
            
        zone_keys_to_del = [k for k in self.zone_occupancy.keys() if k[0] == cam_id and k[2] not in current_gids]
        for k in zone_keys_to_del:
            self.zone_occupancy.pop(k, None)

    def get_tripwire_stats(self, cam_id: str) -> Dict[str, dict]:
        cam_rules = self.rules.get(cam_id, [])
        stats = {}
        for r in cam_rules:
            if r["type"] == "tripwire":
                rid = r["id"]
                stats[rid] = {
                    "name": r["name"],
                    "in": self.tripwire_counts.get(rid, {}).get("in", 0),
                    "out": self.tripwire_counts.get(rid, {}).get("out", 0)
                }
        return stats

# Global singleton
behavior_engine = BehaviorAnalyticsEngine()
