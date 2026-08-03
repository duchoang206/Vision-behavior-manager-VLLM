import time
import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
from scipy.optimize import linear_sum_assignment
from core.camera_calibrator import camera_calibrator


class DeepTrackState:
    NEW = "NEW"
    TRACKED = "TRACKED"
    FAST_MOTION = "FAST_MOTION"
    DORMANT = "DORMANT"      # Lost temporarily (e.g. occlusion, blind spot, high speed turn), appearance preserved
    RECOVERED = "RECOVERED"  # Re-identified & stitched after temporary track loss
    DELETED = "DELETED"


class MultiAngleGallery:
    """
    Learned Multi-Angle Appearance Gallery for deep Re-ID tracking.
    Maintains up to max_angles (16) viewpoint embeddings with angular diversity filtering.
    """
    def __init__(self, max_angles: int = 16, min_diversity_sim: float = 0.94):
        self.max_angles = max_angles
        self.min_diversity_sim = min_diversity_sim
        self.vectors: List[np.ndarray] = []
        self.timestamps: List[float] = []

    def add_feature(self, feat: Any, timestamp: float) -> bool:
        if feat is None:
            return False
        try:
            arr = np.array(feat, dtype=np.float32)
            if arr.ndim != 1 or len(arr) != 512:
                return False
            norm = float(np.linalg.norm(arr))
            if norm < 1e-6:
                return False
            unit_vec = (arr / norm).astype(np.float32)
        except Exception:
            return False

        if not self.vectors:
            self.vectors.append(unit_vec)
            self.timestamps.append(timestamp)
            return True

        # Check max similarity against existing angles
        max_sim = max(float(np.dot(unit_vec, v)) for v in self.vectors)
        if max_sim < self.min_diversity_sim:
            if len(self.vectors) >= self.max_angles:
                self.vectors.pop(0)
                self.timestamps.pop(0)
            self.vectors.append(unit_vec)
            self.timestamps.append(timestamp)
            return True
        else:
            # EMA refine closest angle vector
            best_idx = int(np.argmax([float(np.dot(unit_vec, v)) for v in self.vectors]))
            alpha = 0.85
            refined = alpha * self.vectors[best_idx] + (1.0 - alpha) * unit_vec
            ref_norm = float(np.linalg.norm(refined))
            if ref_norm > 1e-6:
                self.vectors[best_idx] = (refined / ref_norm).astype(np.float32)
                self.timestamps[best_idx] = timestamp
            return False

    def max_cosine_similarity(self, query_feat: Any) -> float:
        if not self.vectors or query_feat is None:
            return 0.0
        try:
            q_arr = np.array(query_feat, dtype=np.float32)
            if q_arr.ndim != 1 or len(q_arr) != 512:
                return 0.0
            q_norm = float(np.linalg.norm(q_arr))
            if q_norm < 1e-6:
                return 0.0
            q_unit = q_arr / q_norm
            return float(max(np.dot(q_unit, v) for v in self.vectors))
        except Exception:
            return 0.0

    def get_representative_vector(self) -> Optional[np.ndarray]:
        if not self.vectors:
            return None
        return self.vectors[-1]


class KinematicFloorTracker:
    """
    Velocity-Aware Kinematic Motion State Estimator.
    Tracks 2D floor position (X, Y) in meters, instantaneous & smoothed velocity (Vx, Vy) in m/s,
    and normalized bounding box [cx, cy, w, h, v_cx, v_cy].
    Enables deep tracking even during fast object motion (speed > 1.5 m/s) and rapid acceleration.
    """
    def __init__(self, initial_bbox: List[float], initial_floor_pos: Tuple[float, float], timestamp: float):
        self.floor_x = float(initial_floor_pos[0])
        self.floor_y = float(initial_floor_pos[1])
        self.vx = 0.0
        self.vy = 0.0
        self.speed = 0.0
        
        self.bbox = list(initial_bbox)
        self.bcx = initial_bbox[0] + initial_bbox[2] / 2.0
        self.bcy = initial_bbox[1] + initial_bbox[3] / 2.0
        self.bw = initial_bbox[2]
        self.bh = initial_bbox[3]
        self.v_bcx = 0.0
        self.v_bcy = 0.0
        
        self.last_update_time = timestamp
        self.time_since_update = 0.0

    def predict(self, now: float) -> Tuple[float, float, List[float]]:
        """
        Extrapolates expected floor position and bounding box based on velocity.
        """
        dt = max(0.01, min(3.0, now - self.last_update_time))
        self.time_since_update = now - self.last_update_time
        
        # Extrapolate floor position
        pred_fx = self.floor_x + self.vx * dt
        pred_fy = self.floor_y + self.vy * dt
        
        # Extrapolate bbox
        pred_bcx = max(0.0, min(1.0, self.bcx + self.v_bcx * dt))
        pred_bcy = max(0.0, min(1.0, self.bcy + self.v_bcy * dt))
        pred_x = max(0.0, pred_bcx - self.bw / 2.0)
        pred_y = max(0.0, pred_bcy - self.bh / 2.0)
        pred_bbox = [round(pred_x, 4), round(pred_y, 4), round(self.bw, 4), round(self.bh, 4)]
        
        return pred_fx, pred_fy, pred_bbox

    def update(self, bbox: List[float], floor_pos: Tuple[float, float], now: float):
        dt = max(0.01, now - self.last_update_time)
        self.last_update_time = now
        self.time_since_update = 0.0
        
        meas_fx, meas_fy = float(floor_pos[0]), float(floor_pos[1])
        
        # Calculate instantaneous metric velocity
        inst_vx = (meas_fx - self.floor_x) / dt
        inst_vy = (meas_fy - self.floor_y) / dt
        inst_speed = math.hypot(inst_vx, inst_vy)
        
        # Clamp velocity to max realistic industrial speed (3.5 m/s)
        if inst_speed > 3.5:
            scale = 3.5 / inst_speed
            inst_vx *= scale
            inst_vy *= scale
            
        # Velocity smoothing
        v_alpha = 0.70
        self.vx = v_alpha * self.vx + (1.0 - v_alpha) * inst_vx
        self.vy = v_alpha * self.vy + (1.0 - v_alpha) * inst_vy
        self.speed = math.hypot(self.vx, self.vy)
        
        # Position smoothing (Kalman-like low pass filter)
        pos_alpha = 0.65 if self.speed > 0.4 else 0.80
        self.floor_x = pos_alpha * self.floor_x + (1.0 - pos_alpha) * meas_fx
        self.floor_y = pos_alpha * self.floor_y + (1.0 - pos_alpha) * meas_fy
        
        # Update bbox kinematics
        new_bcx = bbox[0] + bbox[2] / 2.0
        new_bcy = bbox[1] + bbox[3] / 2.0
        self.v_bcx = (new_bcx - self.bcx) / dt
        self.v_bcy = (new_bcy - self.bcy) / dt
        
        self.bcx = new_bcx
        self.bcy = new_bcy
        self.bw = bbox[2]
        self.bh = bbox[3]
        self.bbox = list(bbox)


class GlobalTrack:
    """
    Complete Multi-Camera Multi-Target Deep Track representation with:
    - Kinematic state estimation (Kalman velocity & position)
    - Learned Multi-Angle Re-ID Appearance Gallery (16 angles)
    - State Machine (NEW, TRACKED, FAST_MOTION, DORMANT, RECOVERED, DELETED)
    - Trajectory & Occlusion History
    """
    def __init__(
        self,
        global_id: int,
        obj_class: str,
        initial_cam_id: str,
        initial_bbox: List[float],
        floor_pos: Tuple[float, float],
        feature_vector: Optional[Any] = None,
        now: Optional[float] = None
    ):
        timestamp = now or time.time()
        self.global_id = global_id
        self.obj_class = obj_class
        self.last_cam_id = initial_cam_id
        self.assigned_label: Optional[str] = None
        
        self.kinematics = KinematicFloorTracker(initial_bbox, floor_pos, timestamp)
        self.gallery = MultiAngleGallery(max_angles=16, min_diversity_sim=0.94)
        if feature_vector is not None:
            self.gallery.add_feature(feature_vector, timestamp)
            
        self.state = DeepTrackState.NEW
        self.first_seen = timestamp
        self.last_seen = timestamp
        self.lost_time = 0.0
        self.total_detections = 1
        self.consecutive_lost_frames = 0
        
        # Trajectory history: (timestamp, cam_id, bbox, floor_pos, speed)
        self.history: List[Tuple[float, str, List[float], Tuple[float, float], float]] = [
            (timestamp, initial_cam_id, list(initial_bbox), floor_pos, 0.0)
        ]

    @property
    def floor_pos(self) -> Tuple[float, float]:
        return (round(self.kinematics.floor_x, 4), round(self.kinematics.floor_y, 4))

    @property
    def last_bbox(self) -> List[float]:
        return self.kinematics.bbox

    @property
    def velocity(self) -> float:
        return round(self.kinematics.speed, 2)

    @property
    def feature_vector(self) -> Optional[np.ndarray]:
        return self.gallery.get_representative_vector()

    def update(
        self,
        cam_id: str,
        bbox: List[float],
        floor_pos: Tuple[float, float],
        feature_vector: Optional[Any] = None,
        now: Optional[float] = None
    ):
        current_time = now or time.time()
        was_dormant = (self.state == DeepTrackState.DORMANT or self.lost_time >= 0.35 or (current_time - self.last_seen) >= 0.35)
        
        self.last_cam_id = cam_id
        self.last_seen = current_time
        self.lost_time = 0.0
        self.consecutive_lost_frames = 0
        self.total_detections += 1
        
        self.kinematics.update(bbox, floor_pos, current_time)
        if feature_vector is not None:
            self.gallery.add_feature(feature_vector, current_time)
            
        if was_dormant:
            self.state = DeepTrackState.RECOVERED
        elif self.kinematics.speed > 0.75:
            self.state = DeepTrackState.FAST_MOTION
        else:
            self.state = DeepTrackState.TRACKED
            
        self.history.append((
            current_time,
            cam_id,
            list(bbox),
            self.floor_pos,
            self.velocity
        ))
        if len(self.history) > 200:
            self.history.pop(0)

    def mark_lost(self, now: float, dormant_threshold_sec: float = 0.4):
        self.lost_time = now - self.last_seen
        self.consecutive_lost_frames += 1
        if self.lost_time >= dormant_threshold_sec and self.state != DeepTrackState.DELETED:
            self.state = DeepTrackState.DORMANT


class GlobalReIDMatcher:
    """
    Industrial Deep Spatial-Appearance MTMC Tracking Engine.
    Combines:
    1. Kinematic Kalman Velocity Estimation & Motion Prediction (handles fast-moving targets up to 3.5 m/s)
    2. Dynamic Velocity-Scaled Gating Radius (expands search up to 8.5m for fast sprints)
    3. Learned Multi-Angle Re-ID Gallery Cosine Similarity Matching (Threshold >= 0.65)
    4. Temporary Track Loss & Occlusion Recovery (re-identifies and stitches tracks across camera blind spots)
    5. Hungarian Global Optimal Assignment
    """
    def __init__(
        self,
        sim_threshold: float = 0.65,
        max_floor_dist: float = 2.5,
        max_time_gap: float = 35.0,   # Memory TTL for dormant/lost tracks (35 seconds)
        dormant_threshold: float = 0.4 # Time after which an unobserved track is marked DORMANT
    ):
        self.sim_threshold = sim_threshold
        self.max_floor_dist = max_floor_dist
        self.max_time_gap = max_time_gap
        self.dormant_threshold = dormant_threshold
        self.next_global_id = 1
        
        self.gallery: Dict[int, GlobalTrack] = {}                  # global_id -> GlobalTrack
        self.local_to_global_map: Dict[Tuple[str, int], int] = {} # (cam_id, local_id) -> global_id
        self.last_cleanup = time.time()

    def _cleanup_old_tracks(self):
        now = time.time()
        if now - self.last_cleanup < 2.0:
            return
        self.last_cleanup = now
        
        expired_ids = [
            gid for gid, track in self.gallery.items()
            if (now - track.last_seen > self.max_time_gap)
        ]
        for gid in expired_ids:
            del self.gallery[gid]
            
        active_gids = set(self.gallery.keys())
        self.local_to_global_map = {
            k: v for k, v in self.local_to_global_map.items() if v in active_gids
        }

    def process_camera_detections(self, cam_id: str, detections: List[dict]) -> List[dict]:
        """
        Deep Tracking Function:
        Tracks objects through fast motion, velocity extrapolation, and recovers
        lost/occluded objects using learned multi-angle Re-ID features.
        
        Input:
          detections: list of dicts { local_id, class, bbox: [x,y,w,h], confidence, feature (512-dim optional) }
        Returns:
          augmented detections with 'global_id', 'floor_x', 'floor_y', 'velocity', 'track_state', 'reid_score', 'recovered_from_loss'.
        """
        self._cleanup_old_tracks()
        now = time.time()
        augmented = []
        unmatched_dets = []
        matched_gids: Set[int] = set()
        
        # Step 1: Compute metric floor coordinates and predict kinematics
        for det in detections:
            bbox = det["bbox"]
            cx = bbox[0] + bbox[2] / 2.0
            cy = bbox[1] + bbox[3]
            floor_x, floor_y = camera_calibrator.camera_to_floor(cam_id, cx, cy)
            det["floor_x"] = floor_x
            det["floor_y"] = floor_y

        # Step 2: High-Confidence Local Track Continuity Check
        for det in detections:
            local_id = det["local_id"]
            key = (cam_id, local_id)
            obj_class = det.get("class", "object")
            floor_x = det["floor_x"]
            floor_y = det["floor_y"]
            bbox = det["bbox"]
            feat = det.get("feature")
            
            if key in self.local_to_global_map:
                gid = self.local_to_global_map[key]
                if gid in self.gallery:
                    track = self.gallery[gid]
                    # Verify class match and reasonable spatial bounds
                    pred_fx, pred_fy, _ = track.kinematics.predict(now)
                    dist_to_pred = math.hypot(floor_x - pred_fx, floor_y - pred_fy)
                    
                    # Dynamic gating radius based on object speed
                    dt = max(0.01, now - track.last_seen)
                    r_gate = min(8.5, max(2.5, 2.5 + 1.25 * track.velocity * dt))
                    
                    # Appearance verification if feature available
                    reid_sim = track.gallery.max_cosine_similarity(feat) if feat is not None else 1.0
                    
                    # Accept continuity if spatial proximity holds OR appearance strongly matches
                    if (dist_to_pred <= r_gate) or (reid_sim >= self.sim_threshold):
                        track.update(cam_id, bbox, (floor_x, floor_y), feat, now)
                        matched_gids.add(gid)
                        
                        det_copy = dict(det)
                        det_copy["global_id"] = gid
                        det_copy["floor_x"] = track.floor_pos[0]
                        det_copy["floor_y"] = track.floor_pos[1]
                        det_copy["velocity"] = track.velocity
                        det_copy["track_state"] = track.state
                        det_copy["reid_score"] = round(reid_sim, 4) if feat is not None else None
                        det_copy["recovered_from_loss"] = False
                        augmented.append(det_copy)
                        continue
                    else:
                        # Dropped local link due to teleportation/mismatch
                        self.local_to_global_map.pop(key, None)

            unmatched_dets.append(det)

        if not unmatched_dets:
            # Mark unobserved tracks as lost
            for gid, track in self.gallery.items():
                if gid not in matched_gids and track.last_cam_id == cam_id:
                    track.mark_lost(now, self.dormant_threshold)
            return augmented

        # Step 3: Fast-Motion Velocity-Compensated Hungarian Association across candidate active/coasting tracks
        candidate_gids = [
            gid for gid, track in self.gallery.items()
            if gid not in matched_gids and (now - track.last_seen <= 4.0)
        ]

        if candidate_gids and unmatched_dets:
            num_dets = len(unmatched_dets)
            num_cands = len(candidate_gids)
            cost_matrix = np.ones((num_dets, num_cands), dtype=np.float32) * 50.0
            
            for i, det in enumerate(unmatched_dets):
                det_fx, det_fy = det["floor_x"], det["floor_y"]
                det_class = det.get("class", "")
                det_feat = det.get("feature")
                
                for j, gid in enumerate(candidate_gids):
                    track = self.gallery[gid]
                    if det_class and track.obj_class and det_class != track.obj_class:
                        cost_matrix[i, j] = 99.0
                        continue
                        
                    dt = max(0.01, now - track.last_seen)
                    pred_fx, pred_fy, _ = track.kinematics.predict(now)
                    floor_dist = math.hypot(det_fx - pred_fx, det_fy - pred_fy)
                    
                    # Dynamic search radius expands with speed: e.g. at 2.0 m/s for 1s => r_gate = 5.0m
                    r_gate = min(8.5, max(2.5, 2.5 + 1.25 * track.velocity * dt))
                    
                    reid_sim = track.gallery.max_cosine_similarity(det_feat) if det_feat is not None else 0.0
                    
                    # Fused cost for fast-motion tracking
                    if det_feat is not None and track.gallery.vectors:
                        if reid_sim >= self.sim_threshold:
                            # Re-ID match locks the track even over large jumps
                            fused_cost = 0.70 * (1.0 - reid_sim) + 0.30 * min(1.0, floor_dist / r_gate)
                        else:
                            fused_cost = 0.35 * (1.0 - reid_sim) + 0.65 * (floor_dist / r_gate)
                    else:
                        fused_cost = floor_dist / r_gate
                        
                    if floor_dist > r_gate and reid_sim < self.sim_threshold:
                        fused_cost = 99.0
                        
                    cost_matrix[i, j] = fused_cost

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            assigned_rows = set()
            
            for r, c in zip(row_ind, col_ind):
                cost = cost_matrix[r, c]
                if cost <= 1.0:
                    det = unmatched_dets[r]
                    gid = candidate_gids[c]
                    track = self.gallery[gid]
                    
                    reid_sim = track.gallery.max_cosine_similarity(det.get("feature")) if det.get("feature") is not None else None
                    track.update(cam_id, det["bbox"], (det["floor_x"], det["floor_y"]), det.get("feature"), now)
                    self.local_to_global_map[(cam_id, det["local_id"])] = gid
                    matched_gids.add(gid)
                    
                    det_copy = dict(det)
                    det_copy["global_id"] = gid
                    det_copy["floor_x"] = track.floor_pos[0]
                    det_copy["floor_y"] = track.floor_pos[1]
                    det_copy["velocity"] = track.velocity
                    det_copy["track_state"] = track.state
                    det_copy["reid_score"] = round(reid_sim, 4) if reid_sim is not None else None
                    det_copy["recovered_from_loss"] = (track.state == DeepTrackState.RECOVERED)
                    augmented.append(det_copy)
                    assigned_rows.add(r)
                    
            unmatched_dets = [d for idx, d in enumerate(unmatched_dets) if idx not in assigned_rows]

        # Step 4: TEMPORARY LOSS & OCCLUSION RECOVERY (Khôi phục mất dấu đối tượng tạm thời qua Re-ID Gallery)
        # For objects that were lost (0.4s ~ 35.0s) or occluded and have now re-appeared
        dormant_candidates = [
            (gid, track) for gid, track in self.gallery.items()
            if gid not in matched_gids and (0.35 <= now - track.last_seen <= self.max_time_gap)
        ]

        if dormant_candidates and unmatched_dets:
            recovered_rows = set()
            for r, det in enumerate(unmatched_dets):
                det_feat = det.get("feature")
                det_class = det.get("class", "")
                det_fx, det_fy = det["floor_x"], det["floor_y"]
                
                if det_feat is None:
                    continue
                    
                best_dormant_gid = None
                best_dormant_sim = -1.0
                
                for gid, track in dormant_candidates:
                    if gid in matched_gids:
                        continue
                    if det_class and track.obj_class and det_class != track.obj_class:
                        continue
                        
                    sim = track.gallery.max_cosine_similarity(det_feat)
                    if sim >= self.sim_threshold and sim > best_dormant_sim:
                        best_dormant_sim = sim
                        best_dormant_gid = gid
                        
                if best_dormant_gid is not None and best_dormant_sim >= self.sim_threshold:
                    # RECOVERED! Re-stitch track seamlessly
                    track = self.gallery[best_dormant_gid]
                    track.update(cam_id, det["bbox"], (det_fx, det_fy), det_feat, now)
                    track.state = DeepTrackState.RECOVERED
                    
                    self.local_to_global_map[(cam_id, det["local_id"])] = best_dormant_gid
                    matched_gids.add(best_dormant_gid)
                    
                    det_copy = dict(det)
                    det_copy["global_id"] = best_dormant_gid
                    det_copy["floor_x"] = track.floor_pos[0]
                    det_copy["floor_y"] = track.floor_pos[1]
                    det_copy["velocity"] = track.velocity
                    det_copy["track_state"] = DeepTrackState.RECOVERED
                    det_copy["reid_score"] = round(best_dormant_sim, 4)
                    det_copy["recovered_from_loss"] = True
                    augmented.append(det_copy)
                    recovered_rows.add(r)
                    
            unmatched_dets = [d for idx, d in enumerate(unmatched_dets) if idx not in recovered_rows]

        # Step 5: Initialize new Global Track for remaining truly new detections
        for det in unmatched_dets:
            gid = self.next_global_id
            self.next_global_id += 1
            
            obj_class = det.get("class", "object")
            feat = det.get("feature")
            track = GlobalTrack(
                global_id=gid,
                obj_class=obj_class,
                initial_cam_id=cam_id,
                initial_bbox=det["bbox"],
                floor_pos=(det["floor_x"], det["floor_y"]),
                feature_vector=feat,
                now=now
            )
            self.gallery[gid] = track
            self.local_to_global_map[(cam_id, det["local_id"])] = gid
            matched_gids.add(gid)
            
            det_copy = dict(det)
            det_copy["global_id"] = gid
            det_copy["velocity"] = 0.0
            det_copy["track_state"] = DeepTrackState.NEW
            det_copy["reid_score"] = 1.0 if feat is not None else None
            det_copy["recovered_from_loss"] = False
            augmented.append(det_copy)

        # Step 6: Mark unobserved tracks as DORMANT / LOST
        for gid, track in self.gallery.items():
            if gid not in matched_gids and track.last_cam_id == cam_id:
                track.mark_lost(now, self.dormant_threshold)

        return augmented

    def get_track_vector(self, track_id: int) -> Optional[List[float]]:
        track = self.gallery.get(track_id)
        if track and track.feature_vector is not None:
            return track.feature_vector.tolist()
        for (cam_id, local_id), gid in self.local_to_global_map.items():
            if local_id == track_id:
                t = self.gallery.get(gid)
                if t and t.feature_vector is not None:
                    return t.feature_vector.tolist()
        return None

    def get_track_history(self, global_id: int) -> Optional[dict]:
        track = self.gallery.get(global_id)
        if not track:
            return None
        return {
            "global_id": track.global_id,
            "class": track.obj_class,
            "state": track.state,
            "last_camera": track.last_cam_id,
            "first_seen": track.first_seen,
            "last_seen": track.last_seen,
            "lost_time": round(track.lost_time, 2),
            "velocity": track.velocity,
            "gallery_angles": len(track.gallery.vectors),
            "current_floor_pos": track.floor_pos,
            "trajectory": [
                {
                    "timestamp": h[0],
                    "cam_id": h[1],
                    "bbox": h[2],
                    "floor_pos": h[3],
                    "speed": h[4]
                }
                for h in track.history
            ]
        }

    def get_active_tracks(self) -> Dict[int, dict]:
        now = time.time()
        return {
            gid: {
                "id": gid,
                "class": t.obj_class,
                "state": t.state,
                "cam_id": t.last_cam_id,
                "pos": t.floor_pos,
                "velocity": t.velocity,
                "gallery_size": len(t.gallery.vectors),
                "last_seen_sec_ago": round(now - t.last_seen, 2)
            }
            for gid, t in self.gallery.items()
            if t.state != DeepTrackState.DELETED
        }

    def get_dormant_tracks(self) -> List[dict]:
        now = time.time()
        for gid, t in self.gallery.items():
            if now - t.last_seen >= self.dormant_threshold and t.state not in [DeepTrackState.DORMANT, DeepTrackState.DELETED]:
                t.mark_lost(now, self.dormant_threshold)
        return [
            {
                "global_id": gid,
                "class": t.obj_class,
                "last_cam": t.last_cam_id,
                "lost_duration_sec": round(now - t.last_seen, 2),
                "gallery_angles": len(t.gallery.vectors),
                "last_floor_pos": t.floor_pos,
                "last_speed": t.velocity
            }
            for gid, t in self.gallery.items()
            if t.state == DeepTrackState.DORMANT
        ]


# Global singleton MTMC Deep Tracker with 65% cosine threshold and 35s memory
global_reid = GlobalReIDMatcher(
    sim_threshold=0.65,
    max_floor_dist=2.5,
    max_time_gap=35.0,
    dormant_threshold=0.4
)
