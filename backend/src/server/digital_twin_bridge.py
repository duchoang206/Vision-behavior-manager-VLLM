"""
Session 4: Digital Twin Bridge & Real-Time Multi-Entity Telemetry Broadcaster.
Aggregates and streams Robots, Persons (Workers), and Racks to Three.js 3D View at 15-20Hz.
Fuses FMS Authoritative Fleet Telemetry (State, Battery, Charging, FSM) with Vision AI Ground-Truth Cross-Check.
"""

import asyncio
import json
import math
import time
from typing import Dict, List, Optional, Set, Any, Tuple
from fastapi import WebSocket

from src.controller.registry import target_registry
from src.controller.rack_association import RackAssociationEngine
from core.camera_calibrator import camera_calibrator

try:
    from core.fms_bridge import fms_bridge
except ImportError:
    fms_bridge = None


class DigitalTwinBridge:
    """
    Real-time Multi-Entity 3D Digital Twin Broadcaster.
    Synchronizes AI Vision tracking (Robots, Humans, Racks) + FMS authoritative fleet telemetry.
    """
    def __init__(self):
        self.active_websockets: Set[WebSocket] = set()
        self.rack_engine = RackAssociationEngine()
        
        # Vision-tracked state storages
        self.vision_robots: Dict[str, Dict[str, Any]] = {}
        self.persons: Dict[str, Dict[str, Any]] = {}
        self.racks: Dict[str, Dict[str, Any]] = {}
        
        # Track history for velocity & heading estimation: { entity_id: [(time, x, y), ...] }
        self.position_history: Dict[str, List[Tuple[float, float, float]]] = {}
        
        # Background streaming task
        self._streaming_task: Optional[asyncio.Task] = None
        self._is_running = False

    def _canonical_robot_id(self, fms_id: str) -> str:
        raw = str(fms_id or "").strip()
        if not raw:
            return "Robot_unknown"
        if raw.startswith("Robot_"):
            return raw
        return f"Robot_{raw}"

    async def register_client(self, websocket: WebSocket):
        await websocket.accept()
        self.active_websockets.add(websocket)
        print(f"🔗 [DigitalTwin] New 3D client connected. Total clients: {len(self.active_websockets)}")

    def unregister_client(self, websocket: WebSocket):
        if websocket in self.active_websockets:
            self.active_websockets.remove(websocket)
            print(f"🔌 [DigitalTwin] Client disconnected. Remaining: {len(self.active_websockets)}")

    def update_vision_track(
        self,
        cam_id: str,
        track_id: int,
        raw_class: str,
        norm_u: float,
        norm_v: float,
        bbox: List[float],
        reid_vector: Optional[List[float]] = None,
        object_data: Optional[dict] = None,
    ):
        """
        Updates an entity state from camera vision detection + Homography projection.
        """
        now = time.time()
        
        # 1. Coordinate projection (Camera pixel -> Metric World 3D (X, Z))
        spatial = camera_calibrator.project_ground_point(cam_id, norm_u, norm_v)
        if not spatial["valid"]:
            return
        world_x, world_z = spatial["x"], spatial["z"]
        
        # 2. Re-ID matching against Registered Fleet (e.g. Robot_9001)
        matched_label = None
        if reid_vector and len(reid_vector) == 512:
            matched_label, score = target_registry.match_reid(reid_vector, threshold=0.65)
            
        cls_lower = raw_class.lower()
        identity_label = raw_class if raw_class.startswith(("Robot_", "Rack_", "Person_")) else None
        
        # 3. Categorize entity
        if 'person' in cls_lower or 'human' in cls_lower or 'worker' in cls_lower:
            # --- HUMAN WORKER ---
            person_id = matched_label or identity_label or f"Person_{track_id}"
            heading, velocity = self._calc_motion(person_id, world_x, world_z, now)
            
            self.persons[person_id] = {
                "id": person_id,
                "class": "person",
                "position": [round(world_x, 2), 0.0, round(world_z, 2)],
                "heading": round(heading, 3),
                "velocity": round(velocity, 2),
                "status": "WORKING" if velocity < 0.2 else "WALKING",
                "cam_id": cam_id,
                "track_id": track_id,
                "reid_matched": bool(matched_label),
                "last_seen": now
            }
            if object_data:
                self.persons[person_id].update({
                    "posture": object_data.get("posture"),
                    "fall_detected": bool(object_data.get("fall_detected")),
                    "status": "ALERT" if object_data.get("fall_detected") else ("WORKING" if velocity < 0.2 else "WALKING"),
                    "pose_available": bool(object_data.get("pose_available")),
                    "world_position": object_data.get("world_position"),
                })
            
        elif 'rack' in cls_lower or 'pallet' in cls_lower or 'storage' in cls_lower:
            # --- RACK / STORAGE UNIT ---
            rack_id = matched_label or identity_label or f"Rack_{track_id}"
            
            # Check if this rack is positioned in an ROI Storage Slot with FMS coordinates
            snapped_pos = None
            slot_name = None
            try:
                from core.behavior_analytics import behavior_engine
                from shapely.geometry import Point, Polygon
                
                cam_rules = behavior_engine.rules.get(cam_id, [])
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                bbox_poly = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
                if not bbox_poly.is_valid:
                    bbox_poly = bbox_poly.buffer(0)
                center_pt = Point((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                foot_pt = Point(norm_u, norm_v)
                
                for rule in cam_rules:
                    if rule.get("type") == "occupancy":
                        cam_poly = rule.get("camera_polygon") or rule.get("polygon")
                        fms_pts = rule.get("fms_points")
                        
                        is_match = False
                        if cam_poly is not None and cam_poly.area > 0:
                            if cam_poly.intersects(bbox_poly):
                                inter_area = cam_poly.intersection(bbox_poly).area
                                if (inter_area / cam_poly.area) * 100.0 >= float(rule.get("threshold", 10.0)) or (inter_area / max(bbox_poly.area, 1e-9)) * 100.0 >= 10.0:
                                    is_match = True
                            if not is_match and (cam_poly.contains(foot_pt) or cam_poly.contains(center_pt) or cam_poly.touches(foot_pt)):
                                is_match = True
                                
                        if is_match and fms_pts and len(fms_pts) >= 3:
                            # Exact FMS centroid of this ROI slot
                            fms_cx = sum(p[0] for p in fms_pts) / len(fms_pts)
                            fms_cz = sum(p[1] for p in fms_pts) / len(fms_pts)
                            snapped_pos = (fms_cx, fms_cz)
                            slot_name = rule.get("name")
                            break
            except Exception:
                pass
                
            final_x = snapped_pos[0] if snapped_pos is not None else world_x
            final_z = snapped_pos[1] if snapped_pos is not None else world_z
            
            self.racks[rack_id] = {
                "id": rack_id,
                "class": "rack",
                "position": [round(final_x, 2), 0.0, round(final_z, 2)],
                "status": "STORED",
                "carried_by": None,
                "roi_slot": slot_name,
                "snapped_to_fms": bool(snapped_pos),
                "cam_id": cam_id,
                "track_id": track_id,
                "last_seen": now
            }
            
        else:
            # --- ROBOT (AMR / AGV) ---
            robot_id = matched_label or identity_label or f"Robot_{track_id}"
            heading, velocity = self._calc_motion(robot_id, world_x, world_z, now)
            
            # Check if this robot is carrying a rack physically
            has_rack = False
            carried_rack_id = None
            
            for r_id, r_data in self.racks.items():
                rx, rz = r_data["position"][0], r_data["position"][2]
                dist = math.hypot(world_x - rx, world_z - rz)
                if dist < 0.85:  # Within robot footprint radius
                    has_rack = True
                    carried_rack_id = r_id
                    r_data["status"] = "CARRIED"
                    r_data["carried_by"] = robot_id
                    r_data["position"] = [round(world_x, 2), 0.45, round(world_z, 2)]
                    break
                    
            self.vision_robots[robot_id] = {
                "id": robot_id,
                "class": "robot",
                "position": [round(world_x, 2), 0.0, round(world_z, 2)],
                "heading": round(heading, 3),
                "velocity": round(velocity, 2),
                "has_rack": has_rack,
                "carried_rack_id": carried_rack_id,
                "cam_id": cam_id,
                "track_id": track_id,
                "reid_matched": bool(matched_label),
                "last_seen": now
            }

    def _calc_motion(self, entity_id: str, x: float, z: float, now: float) -> Tuple[float, float]:
        """Calculates smoothed heading (radians) and velocity (m/s) from trajectory history"""
        if entity_id not in self.position_history:
            self.position_history[entity_id] = []
            
        hist = self.position_history[entity_id]
        hist.append((now, x, z))
        
        while len(hist) > 5 or (hist and now - hist[0][0] > 1.5):
            hist.pop(0)
            
        if len(hist) < 2:
            return 0.0, 0.0
            
        dt = hist[-1][0] - hist[0][0]
        if dt < 0.05:
            return 0.0, 0.0
            
        dx = hist[-1][1] - hist[0][1]
        dz = hist[-1][2] - hist[0][2]
        
        dist = math.hypot(dx, dz)
        velocity = dist / dt
        heading = math.atan2(dx, dz) if dist > 0.05 else 0.0
        return heading, velocity

    def clean_stale_tracks(self, timeout_sec: float = 3.5):
        """Removes camera tracks that have not been observed recently"""
        now = time.time()
        for d in [self.vision_robots, self.persons, self.racks]:
            stale_keys = [k for k, v in d.items() if now - v.get("last_seen", 0) > timeout_sec]
            for k in stale_keys:
                del d[k]

    def build_telemetry_payload(self) -> dict:
        """
        Constructs high-frequency real-time 3D payload.
        Fuses FMS Authoritative States (Battery, Status, FSM, Charging, Odometry) with Vision AI Ground Truth.
        """
        self.clean_stale_tracks(timeout_sec=4.0)
        now = time.time()
        
        # 1. Fetch live FMS robots
        fms_dict = {}
        if fms_bridge and hasattr(fms_bridge, 'robot_states'):
            fms_dict = fms_bridge.robot_states
            
        fused_robots: List[Dict[str, Any]] = []
        matched_vision_keys = set()
        
        # Process all FMS robots
        for fms_id, fms_r in fms_dict.items():
            vis_track = None
            vis_key = None
            
            canonical_fms_id = self._canonical_robot_id(fms_id)
            candidates = [canonical_fms_id, fms_id, fms_id.replace("Robot_", "")]
            for cand in candidates:
                if cand in self.vision_robots:
                    vis_track = self.vision_robots[cand]
                    vis_key = cand
                    break
                    
            if not vis_track and fms_r.get("position"):
                fx, fz = fms_r["position"][0], fms_r["position"][2]
                for vk, vt in self.vision_robots.items():
                    if vk not in matched_vision_keys:
                        vx, vz = vt["position"][0], vt["position"][2]
                        if math.hypot(fx - vx, fz - vz) < 1.3:
                            vis_track = vt
                            vis_key = vk
                            break
                            
            if vis_key:
                matched_vision_keys.add(vis_key)
                
            fms_pos = fms_r.get("position", [0.0, 0.0, 0.0])
            fms_heading = fms_r.get("heading", 0.0)
            fms_status = fms_r.get("status", "OFFLINE")
            fms_battery = fms_r.get("battery", 100)
            fms_fsm = fms_r.get("fsm", fms_status)
            
            # Cross-check status
            if fms_status == "OFFLINE":
                cross_status = "OFFLINE"
                cross_msg = "Mất kết nối FMS (Offline)"
            elif vis_track:
                vis_pos = vis_track["position"]
                delta_d = round(math.hypot(fms_pos[0] - vis_pos[0], fms_pos[2] - vis_pos[2]), 2)
                if delta_d > 0.6:
                    cross_status = "DEVIATED"
                    cross_msg = f"Cảnh báo lệch vị trí thực {delta_d}m so với FMS"
                else:
                    cross_status = "SYNCED"
                    cross_msg = f"Đồng bộ chuẩn xác (Lệch {delta_d}m)"
            else:
                cross_status = "FMS_ONLY"
                cross_msg = "Ngoài vùng quan sát camera"
                
            fused_robots.append({
                "id": canonical_fms_id,
                "fms_id": fms_id,
                "class": "robot",
                "model": fms_r.get("model", "AMR-Vision"),
                "floor": fms_r.get("floor", 1),
                "position": fms_pos,
                "heading": fms_heading,
                "velocity": 0.0 if fms_status == "OFFLINE" else fms_r.get("velocity", 0.0),
                "battery": fms_battery,
                "status": fms_status,
                "fsm": fms_fsm,
                "destination": fms_r.get("destination"),
                "fms_position": fms_pos,
                "vision_position": vis_track["position"] if vis_track else None,
                "delta_distance_m": round(math.hypot(fms_pos[0] - vis_track["position"][0], fms_pos[2] - vis_track["position"][2]), 2) if vis_track else None,
                "cross_check_status": cross_status,
                "cross_check_msg": cross_msg,
                "cam_id": vis_track.get("cam_id") if vis_track else None,
                "has_rack": vis_track.get("has_rack", False) if vis_track else False,
                "carried_rack_id": vis_track.get("carried_rack_id") if vis_track else None,
                "raw_fms": fms_r.get("raw_fms"),
                "source": "FMS_AUTHORITATIVE",
                "last_seen": now
            })

        # 2. Worker / Person Proximity Safety Cross-Check against active robots (Real camera tracks only)
        persons_list: List[Dict[str, Any]] = []
        if self.persons:
            for pid, p in self.persons.items():
                px, pz = p["position"][0], p["position"][2]
                min_dist = 999.0
                near_robot_id = None
                
                for r in fused_robots:
                    rx, rz = r["position"][0], r["position"][2]
                    d = math.hypot(px - rx, pz - rz)
                    if d < min_dist:
                        min_dist = d
                        near_robot_id = r["id"]
                        
                is_alert = min_dist < 1.8 and any(r.get("velocity", 0) > 0.05 for r in fused_robots if r["id"] == near_robot_id)
                
                p_data = dict(p)
                p_data["safety_alert"] = is_alert
                p_data["near_robot_id"] = near_robot_id if min_dist < 5.0 else None
                p_data["near_robot_dist"] = round(min_dist, 2) if min_dist < 5.0 else None
                p_data["safety_msg"] = f"⚠ Nguy cơ va chạm với {near_robot_id} ({min_dist:.1f}m)" if is_alert else "Khu vực an toàn"
                persons_list.append(p_data)

        # 3. Racks State Cross-Check (Real camera tracks only)
        racks_list: List[Dict[str, Any]] = []
        if self.racks:
            for rkid, rk in self.racks.items():
                rk_data = dict(rk)
                carried_by = None
                for r in fused_robots:
                    if r.get("carried_rack_id") == rkid:
                        carried_by = r["id"]
                        break
                if carried_by:
                    rk_data["status"] = "CARRIED"
                    rk_data["carried_by"] = carried_by
                    rk_data["cross_check_msg"] = f"Đang cõng trên robot {carried_by}"
                else:
                    rk_data["status"] = "STORED"
                    rk_data["carried_by"] = None
                    if rk_data.get("roi_slot"):
                        rk_data["cross_check_msg"] = f"Đặt chuẩn xác tại vị trí {rk_data['roi_slot']} (FMS Synced)"
                    else:
                        rk_data["cross_check_msg"] = "Đặt tại vị trí ô lưu trữ"
                racks_list.append(rk_data)

        return {
            "type": "DIGITAL_TWIN_TELEMETRY",
            "timestamp": now,
            "fleet_kpi": {
                "total_robots": len(fused_robots),
                "total_persons": len(persons_list),
                "total_racks": len(racks_list),
                "active_robots": len([r for r in fused_robots if r.get("status") in ["RUNNING", "ACTIVE", "CARRYING_RACK"]]),
                "idle_robots": len([r for r in fused_robots if r.get("status") == "IDLE"]),
                "charging_robots": len([r for r in fused_robots if r.get("status") == "CHARGING"]),
                "offline_robots": len([r for r in fused_robots if r.get("status") == "OFFLINE"]),
                "deviated_count": len([r for r in fused_robots if r.get("cross_check_status") == "DEVIATED"]),
            },
            "robots": fused_robots,
            "persons": persons_list,
            "racks": racks_list,
        }

    async def broadcast_loop(self, fps: int = 15):
        """Async broadcast loop pushing telemetry to all 3D WebSocket subscribers"""
        interval = 1.0 / fps
        self._is_running = True
        
        while self._is_running:
            start_time = time.time()
            if self.active_websockets:
                payload = self.build_telemetry_payload()
                msg = json.dumps(payload)
                
                dead_sockets = set()
                for ws in self.active_websockets:
                    try:
                        await ws.send_text(msg)
                    except Exception:
                        dead_sockets.add(ws)
                        
                for dead in dead_sockets:
                    self.unregister_client(dead)
                    
            elapsed = time.time() - start_time
            sleep_time = max(0.005, interval - elapsed)
            await asyncio.sleep(sleep_time)

    def start_broadcaster(self, loop: asyncio.AbstractEventLoop, fps: int = 15):
        if not self._streaming_task or self._streaming_task.done():
            self._streaming_task = loop.create_task(self.broadcast_loop(fps=fps))
            print(f"🚀 [DigitalTwinBridge] 3D Telemetry WebSocket Broadcaster started @ {fps} FPS")


# Singleton instance
digital_twin_bridge = DigitalTwinBridge()
