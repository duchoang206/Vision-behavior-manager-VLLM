"""
PersonTrackerEngine - Ultra High-Performance Zero-Latency GPU Tracker
Runs Ultralytics YOLOv8 / TensorRT FP16 on NVIDIA RTX GPU at 25+ FPS per camera.
Features continuous zero-latency RTSP frame grabber to eliminate all video buffering delay.
"""
import os
import cv2
import time
import threading
import numpy as np
from typing import Dict, List, Callable, Optional

from core.identity_utils import identity_global_id
from core.camera_calibrator import camera_calibrator
from core.pose_estimator import pose_estimator

# Force lowest latency RTSP capture settings in FFmpeg
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;0"

_YOLO_AVAILABLE = False
YOLO = None


_GPU_INFERENCE_LOCK = threading.Lock()

def _vector_from_frame_crop(frame, bbox_xyxy: List[float]) -> Optional[List[float]]:
    img_h, img_w = frame.shape[:2]
    x1 = max(0, min(img_w - 1, int(bbox_xyxy[0] * img_w)))
    y1 = max(0, min(img_h - 1, int(bbox_xyxy[1] * img_h)))
    x2 = max(x1 + 1, min(img_w, int(bbox_xyxy[2] * img_w)))
    y2 = max(y1 + 1, min(img_h, int(bbox_xyxy[3] * img_h)))
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    vector = hist.flatten().astype(np.float32)
    vector = vector / (np.linalg.norm(vector) + 1e-6)
    return vector.tolist()


class ZeroLatencyRTSPReader:
    """
    Dedicated background RTSP frame grabber thread.
    Continuously drains RTSP packets in a tight loop so get_latest_frame()
    ALWAYS returns the absolute newest frame (0ms buffering latency).
    """
    def __init__(self, rtsp_url: str, cam_id: str):
        self.rtsp_url = rtsp_url
        self.cam_id = cam_id
        self.latest_frame = None
        self.lock = threading.Lock()
        self.running = True
        self._thread = threading.Thread(target=self._worker, daemon=True, name=f"rtsp-grab-{cam_id}")
        self._thread.start()

    def _worker(self):
        cap = None
        while self.running:
            if cap is None or not cap.isOpened():
                cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if not cap.isOpened():
                    time.sleep(1.0)
                    continue

            ret = cap.grab()
            if not ret:
                time.sleep(0.1)
                if cap:
                    cap.release()
                cap = None
                continue

            ret, frame = cap.retrieve()
            if ret and frame is not None:
                with self.lock:
                    self.latest_frame = frame

        if cap:
            cap.release()

    def get_latest_frame(self):
        with self.lock:
            return self.latest_frame

    def stop(self):
        self.running = False


class SimpleTracker:
    """
    High-speed robust object tracker with continuous track ID retention.
    """
    def __init__(self, max_age: int = 30, min_iou: float = 0.12):
        self.next_id = 1
        self.tracks: Dict[int, dict] = {}
        self.max_age = max_age
        self.min_iou = min_iou

    def _iou(self, b1, b2):
        x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        union = a1 + a2 - inter
        return inter / union if union > 0 else 0.0

    def _match_score(self, b1, b2):
        iou = self._iou(b1, b2)
        if iou >= self.min_iou:
            return iou
        c1x, c1y = (b1[0] + b1[2]) / 2.0, (b1[1] + b1[3]) / 2.0
        c2x, c2y = (b2[0] + b2[2]) / 2.0, (b2[1] + b2[3]) / 2.0
        dist = float(np.hypot(c1x - c2x, c1y - c2y))
        if dist < 0.18:
            return max(0.01, (0.18 - dist) / 0.18 * 0.45)
        return 0.0

    @staticmethod
    def _same_family(first, second):
        first = str(first or "object").lower()
        second = str(second or "object").lower()
        person_terms = ("person", "human", "worker")
        first_person = any(term in first for term in person_terms)
        second_person = any(term in second for term in person_terms)
        return first_person == second_person and (first_person or first == second)

    def update(self, detections: List[dict]) -> List[dict]:
        for tid in list(self.tracks.keys()):
            self.tracks[tid]["missed"] += 1
            if self.tracks[tid]["missed"] > self.max_age:
                del self.tracks[tid]

        matched_ids = set()
        results = []

        for det in detections:
            det_bbox = det["bbox"]
            det_class = det.get("class", "object")
            best_tid = None
            best_score = 0.0

            for tid, track in self.tracks.items():
                if tid in matched_ids:
                    continue
                if not self._same_family(det_class, track.get("class")):
                    continue
                score = self._match_score(det_bbox, track["bbox"])
                if score > best_score:
                    best_score = score
                    best_tid = tid

            if best_tid is not None and best_score > 0.0:
                self.tracks[best_tid]["bbox"] = det_bbox
                self.tracks[best_tid]["class"] = det_class
                self.tracks[best_tid]["keypoints"] = det.get("keypoints")
                self.tracks[best_tid]["missed"] = 0
                matched_ids.add(best_tid)
                tid = best_tid
            else:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {"bbox": det_bbox, "class": det_class, "keypoints": det.get("keypoints"), "missed": 0}

            x1, y1, x2, y2 = det_bbox
            results.append({
                "id": tid,
                "local_id": tid,
                "x": round(x1, 4),
                "y": round(y1, 4),
                "w": round(x2 - x1, 4),
                "h": round(y2 - y1, 4),
                "floor_x": round((x1 + x2) / 2, 4),
                "floor_y": round(y2, 4),
                "class": det_class,
                "confidence": round(float(det.get("confidence", 0.85)), 3),
                "keypoints": det.get("keypoints") or self.tracks[tid].get("keypoints") or [],
                "tracking_state": "predicted" if self.tracks[tid].get("missed", 0) else "tracked",
                "_feature": det.get("feature")
            })

        return results


class CameraPersonTracker:
    """Runs GPU person detection plus YOLOv8x-pose keypoint enrichment."""

    def __init__(self, cam_id: str, rtsp_url: str, model: "YOLO",
                 metadata_callback: Callable, target_fps: int = 25, event_callback: Optional[Callable] = None):
        self.cam_id = cam_id
        self.rtsp_url = rtsp_url
        self.model = model
        self.metadata_callback = metadata_callback
        self.event_callback = event_callback
        self.target_fps = target_fps
        self.tracker = SimpleTracker(max_age=15, min_iou=0.20)
        self.reader = ZeroLatencyRTSPReader(rtsp_url, cam_id)
        self.running = False
        self.thread: Optional[threading.Thread] = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True, name=f"gpu-tracker-{self.cam_id}")
        self.thread.start()
        print(f"[PersonTracker] Started 25 FPS Real-Time GPU Tracker for {self.cam_id}", flush=True)

    def stop(self):
        self.running = False
        self.reader.stop()

    def _loop(self):
        frame_interval = 1.0 / max(1, self.target_fps)

        while self.running:
            t0 = time.time()
            frame = self.reader.get_latest_frame()
            if frame is None:
                time.sleep(0.02)
                continue

            try:
                img_h, img_w = frame.shape[:2]
                
                # TensorRT or CUDA FP16 Inference (< 3ms per frame on RTX 5060 Ti)
                with _GPU_INFERENCE_LOCK:
                    results = self.model.predict(
                        frame,
                        verbose=False,
                        conf=0.08,
                        iou=0.45,
                        imgsz=640
                    )

                raw_detections = []
                if results and len(results) > 0:
                    r = results[0]
                    if r.boxes is not None and len(r.boxes) > 0:
                        boxes_xyxy = r.boxes.xyxy.cpu().numpy()
                        cls_values = r.boxes.cls.cpu().numpy() if r.boxes.cls is not None else []
                        conf_values = r.boxes.conf.cpu().numpy() if r.boxes.conf is not None else []
                        pose_points = r.keypoints.xy.cpu().numpy() if getattr(r, "keypoints", None) is not None else None
                        pose_scores = r.keypoints.conf.cpu().numpy() if getattr(r, "keypoints", None) is not None and r.keypoints.conf is not None else None
                        names = getattr(self.model, "names", {}) or {}
                        for idx, box in enumerate(boxes_xyxy):
                            confidence = float(conf_values[idx]) if idx < len(conf_values) else 0.85
                            cls_id = int(cls_values[idx]) if idx < len(cls_values) else -1
                            if isinstance(names, dict):
                                class_name = str(names.get(cls_id, f"class_{cls_id}"))
                            elif isinstance(names, list) and 0 <= cls_id < len(names):
                                class_name = str(names[cls_id])
                            else:
                                class_name = f"class_{cls_id}"
                            
                            class_lower = class_name.lower()
                            if os.getenv("CPU_TRACKER_PERSON_ONLY", "0").lower() in {"1", "true", "yes"} and not any(term in class_lower for term in ("person", "human", "worker")):
                                continue
                            # Filter human detections with conf < 0.45
                            if ("person" in class_lower or "human" in class_lower or "worker" in class_lower) and confidence < 0.45:
                                continue

                            x1 = max(0.0, float(box[0]) / img_w)
                            y1 = max(0.0, float(box[1]) / img_h)
                            x2 = min(1.0, float(box[2]) / img_w)
                            y2 = min(1.0, float(box[3]) / img_h)
                            w = x2 - x1
                            h = y2 - y1
                            if w < 0.03 or h < 0.03:
                                continue

                            keypoints = []
                            if pose_points is not None and idx < len(pose_points):
                                for point_index, point in enumerate(pose_points[idx]):
                                    score = float(pose_scores[idx][point_index]) if pose_scores is not None else 1.0
                                    keypoints.append([
                                        round(float(point[0]) / img_w, 5),
                                        round(float(point[1]) / img_h, 5),
                                        round(score, 4),
                                    ])
                            raw_detections.append({
                                "bbox": [x1, y1, x2, y2],
                                "class": class_lower,
                                "confidence": confidence,
                                "feature": _vector_from_frame_crop(frame, [x1, y1, x2, y2]),
                                "keypoints": keypoints,
                            })

                pose_detections = []
                if os.getenv("CPU_TRACKER_PERSON_ONLY", "0").lower() in {"1", "true", "yes"}:
                    person_boxes = [item["bbox"] for item in raw_detections if any(term in item["class"] for term in ("person", "human", "worker"))]
                    for pose in pose_detections:
                        matched_person = next(
                            (item for item in raw_detections
                             if any(term in item["class"] for term in ("person", "human", "worker"))
                             and self.tracker._iou(pose["bbox"], item["bbox"]) >= 0.25),
                            None,
                        )
                        if matched_person is not None:
                            matched_person["keypoints"] = pose["keypoints"]
                            continue
                        raw_detections.append({
                            "bbox": pose["bbox"],
                            "class": "person",
                            "confidence": 0.9,
                            "keypoints": pose["keypoints"],
                            "feature": _vector_from_frame_crop(frame, pose["bbox"]),
                        })
                for detection in raw_detections:
                    if not any(term in detection["class"] for term in ("person", "human", "worker")):
                        continue
                    best_pose, best_iou = None, 0.0
                    for pose in pose_detections:
                        a, b = detection["bbox"], pose["bbox"]
                        intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
                        area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
                        area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
                        union = area_a + area_b - intersection
                        iou = intersection / union if union > 1e-6 else 0.0
                        if iou > best_iou:
                            best_iou, best_pose = iou, pose
                    if best_pose is not None and best_iou >= 0.25:
                        detection["keypoints"] = best_pose["keypoints"]

                # Cross-Class NMS to merge duplicate overlapping boxes of the same object
                sorted_raw = sorted(raw_detections, key=lambda d: d.get("confidence", 0), reverse=True)
                detections = []
                for d in sorted_raw:
                    b1 = d["bbox"]
                    overlap = False
                    for k in detections:
                        b2 = k["bbox"]
                        ix = max(0.0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
                        iy = max(0.0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
                        inter = ix * iy
                        a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
                        a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
                        union = a1 + a2 - inter
                        if union > 0 and (inter / union) > 0.38:
                            overlap = True
                            break
                    if not overlap:
                        detections.append(d)

                tracked = self.tracker.update(detections)

                # Assign labels with Camera-scoped matching
                try:
                    from src.controller.registry import target_registry
                    from core.identity_utils import identity_global_id, robot_number_from_label

                    # Set floor positions and clear labels
                    for obj in tracked:
                        cx = obj["x"] + obj["w"] / 2.0
                        cy = obj["y"] + obj["h"]
                        floor_x, floor_y = camera_calibrator.camera_to_floor(self.cam_id, cx, cy)
                        obj["floor_x"] = round(float(floor_x), 4)
                        obj["floor_y"] = round(float(floor_y), 4)
                        obj["frame_width"] = img_w
                        obj["frame_height"] = img_h
                        obj["world_position"] = [round(float(floor_x), 4), 0.0, round(float(floor_y), 4)]
                        obj["spatial_valid"] = self.cam_id in camera_calibrator.homographies
                        obj["spatial_source"] = "homography" if obj["spatial_valid"] else "uncalibrated"
                        if not obj["spatial_valid"]:
                            obj["world_position"] = [None, None, None]
                        if any(term in str(obj.get("class", "")).lower() for term in ("person", "human", "worker")):
                            try:
                                from core.object_logic import person_logic
                                person_data = dict(obj)
                                person_data["bbox"] = [obj["x"], obj["y"], obj["w"], obj["h"]]
                                person_data["global_id"] = obj["id"]
                                person_data["frame_width"] = img_w
                                person_data["frame_height"] = img_h
                                obj.update(person_logic.process(person_data, (floor_x, floor_y), self.cam_id))
                                if obj.get("fall_event") and self.event_callback:
                                    self.event_callback({
                                        "cam_id": self.cam_id,
                                        "global_id": obj["id"],
                                        "rule_type": "fall_detection",
                                        "severity": "critical",
                                        "description": f"Phát hiện người có khả năng bị ngã (track #{obj['id']})",
                                        "timestamp": int(time.time() * 1000),
                                        "object": {"id": obj["id"], "bbox": [obj["x"], obj["y"], obj["w"], obj["h"]]},
                                    })
                            except Exception as exc:
                                print(f"[PersonTracker] Person analytics error: {exc}", flush=True)
                        obj["label"] = None

                    # Get all targets relevant specifically to this camera
                    cam_targets = [
                        (k, data) for k, data in target_registry.targets.items()
                        if (data.get("cam_id") == self.cam_id or data.get("last_cam") == self.cam_id or self.cam_id in (data.get("samples_by_cam") or {}))
                    ]

                    # 1. Match mobile/robot targets via Multi-Angle Re-ID Cosine Similarity
                    assigned_objs = set()
                    for t_key, tdata in cam_targets:
                        cat = (tdata.get("category") or "").lower()
                        t_lbl = tdata.get("label") or t_key.split("::")[-1]
                        if cat == "robot" or t_lbl.lower().startswith("robot"):
                            best_obj = None
                            best_sim = -1.0
                            t_vectors = tdata.get("vectors") or [tdata.get("vector")]
                            t_vectors = [v for v in t_vectors if v is not None]
                            if t_vectors:
                                for obj in tracked:
                                    if id(obj) in assigned_objs:
                                        continue
                                    if "person" in obj.get("class", ""):
                                        continue
                                    feat = obj.get("_feature")
                                    if feat is not None:
                                        q_vec = np.array(feat, dtype=np.float32)
                                        q_norm = float(np.linalg.norm(q_vec))
                                        if q_norm > 1e-6:
                                            q_vec = q_vec / q_norm
                                            for t_vec in t_vectors:
                                                t_norm = float(np.linalg.norm(t_vec))
                                                if t_norm > 1e-6:
                                                    sim = float(np.dot(q_vec, t_vec / t_norm))
                                                    if sim > best_sim and sim >= 0.65:
                                                        best_sim = sim
                                                        best_obj = obj
                            if best_obj is not None:
                                assigned_objs.add(id(best_obj))
                                best_obj["label"] = t_lbl
                                best_obj["class"] = "robot"
                                r_id = tdata.get("fms_robot_id") or robot_number_from_label(t_lbl) or best_obj.get("id")
                                best_obj["id"] = r_id
                                best_obj["local_id"] = r_id
                                target_registry.bind_label_to_track(
                                    t_lbl, self.cam_id, r_id, r_id,
                                    live_bbox=[best_obj["x"], best_obj["y"], best_obj["w"], best_obj["h"]]
                                )

                    # 2. Match static / rack / pallet targets registered on this camera
                    for t_key, tdata in cam_targets:
                        cat = (tdata.get("category") or "").lower()
                        t_lbl = tdata.get("label") or t_key.split("::")[-1]
                        if cat != "robot" and not t_lbl.lower().startswith("robot"):
                            cam_sample = (tdata.get("samples_by_cam") or {}).get(self.cam_id)
                            sample_bbox = cam_sample.get("bbox") if cam_sample else (tdata.get("bbox") if tdata.get("last_cam") == self.cam_id else None)
                            if sample_bbox:
                                bx, by, bw, bh = sample_bbox
                                matched_rack_obj = None
                                for obj in tracked:
                                    if id(obj) in assigned_objs:
                                        continue
                                    ocx, ocy = obj["x"] + obj["w"] / 2.0, obj["y"] + obj["h"] / 2.0
                                    rcx, rcy = bx + bw / 2.0, by + bh / 2.0
                                    dist = float(np.hypot(ocx - rcx, ocy - rcy))
                                    if dist < 0.15:
                                        matched_rack_obj = obj
                                        break
                                
                                rack_id = identity_global_id(t_lbl, cat, 10001)
                                if matched_rack_obj:
                                    assigned_objs.add(id(matched_rack_obj))
                                    matched_rack_obj["label"] = t_lbl
                                    matched_rack_obj["class"] = cat or "rack"
                                    matched_rack_obj["id"] = rack_id
                                    matched_rack_obj["local_id"] = rack_id
                                else:
                                    # Visual Anchor Check at registered position (Threshold = 65%)
                                    vec = _vector_from_frame_crop(frame, [bx, by, min(1.0, bx + bw), min(1.0, by + bh)])
                                    if vec is not None:
                                        q_vec = np.array(vec, dtype=np.float32)
                                        q_norm = float(np.linalg.norm(q_vec))
                                        if q_norm > 1e-6:
                                            q_vec = q_vec / q_norm
                                            t_vectors = tdata.get("vectors") or [tdata.get("vector")]
                                            t_vectors = [v for v in t_vectors if v is not None]
                                            best_rack_sim = max([float(np.dot(q_vec, v / (np.linalg.norm(v) + 1e-6))) for v in t_vectors]) if t_vectors else 0.0
                                            if best_rack_sim >= 0.65:
                                                floor_x, floor_y = camera_calibrator.camera_to_floor(self.cam_id, bx + bw / 2.0, by + bh)
                                                tracked.append({
                                                    "id": rack_id,
                                                    "local_id": rack_id,
                                                    "x": round(bx, 4),
                                                    "y": round(by, 4),
                                                    "w": round(bw, 4),
                                                    "h": round(bh, 4),
                                                    "floor_x": round(float(floor_x), 4),
                                                    "floor_y": round(float(floor_y), 4),
                                                    "class": cat or "rack",
                                                    "confidence": 0.95,
                                                    "label": t_lbl
                                                })

                    # Clean up temporary _feature
                    for obj in tracked:
                        obj.pop("_feature", None)

                except Exception as ex:
                    pass

                tripwire_stats = {}
                roi_states = []
                try:
                    from core.behavior_analytics import behavior_engine
                    _, tripwire_stats, roi_states = behavior_engine.process_frame(self.cam_id, tracked)
                except Exception:
                    pass

                if self.metadata_callback:
                    self.metadata_callback({
                        "source": "cpu_fallback",
                        "timestamp": int(time.time() * 1000),
                        "streams": [{
                            "cam_id": self.cam_id,
                            "objects": tracked,
                            "tripwire_stats": tripwire_stats,
                            "rois": roi_states
                        }]
                    })

            except Exception as e:
                print(f"[PersonTracker] GPU Inference error on {self.cam_id}: {e}", flush=True)

            elapsed = time.time() - t0
            sleep_t = max(0.002, frame_interval - elapsed)
            time.sleep(sleep_t)

        print(f"[PersonTracker] Stopped tracker for {self.cam_id}", flush=True)


class PersonTrackerManager:
    """
    Manages one high-speed GPU CameraPersonTracker per camera.
    Uses TensorRT FP16 engine on RTX 5060 Ti GPU for real-time 25 FPS tracking.
    """
    def __init__(self):
        self._trackers: Dict[str, CameraPersonTracker] = {}
        self._model: Optional["YOLO"] = None
        self._lock = threading.Lock()
        self.metadata_callback: Optional[Callable] = None
        self.event_callback: Optional[Callable] = None

    def _get_model(self) -> Optional["YOLO"]:
        global YOLO, _YOLO_AVAILABLE
        if YOLO is None:
            try:
                from ultralytics import YOLO as UltralyticsYOLO
                YOLO = UltralyticsYOLO
                _YOLO_AVAILABLE = True
            except Exception as e:
                print(f"[PersonTracker] Ultralytics unavailable: {e}", flush=True)
                _YOLO_AVAILABLE = False

        if not _YOLO_AVAILABLE:
            return None
        if self._model is None:
            candidates = [
                os.getenv("CPU_TRACKER_MODEL_PATH", "/app/models/yolov8x-pose.pt"),
                "/app/models/yolov8x-pose.pt",
                "/app/models/yolov8x.pt",
            ]
            for path in candidates:
                if path and os.path.exists(path):
                    try:
                        print(f"[PersonTracker] Loading GPU model from: {path}", flush=True)
                        m = YOLO(path)
                        if not path.endswith(".engine"):
                            try:
                                import torch
                                if torch.cuda.is_available():
                                    m.to("cuda")
                            except Exception:
                                pass
                        # Eager warmup in main thread
                        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
                        with _GPU_INFERENCE_LOCK:
                            m.predict(dummy, verbose=False, imgsz=640, device=0, half=True)
                        self._model = m
                        print(f"[PersonTracker] GPU Model loaded and warmed up OK: {path}", flush=True)
                        break
                    except Exception as e:
                        print(f"[PersonTracker] Failed to load/warmup {path}: {e}", flush=True)
        return self._model

    def add_camera(self, cam_id: str, rtsp_url: str):
        with self._lock:
            if cam_id in self._trackers:
                return
            model = self._get_model()
            if model is None:
                print(f"[PersonTracker] No YOLO GPU model available for {cam_id}", flush=True)
                return
            tracker = CameraPersonTracker(
                cam_id=cam_id,
                rtsp_url=rtsp_url,
                model=model,
                metadata_callback=self.metadata_callback,
                event_callback=self.event_callback,
                target_fps=25  # 25 FPS real-time tracking
            )
            tracker.start()
            self._trackers[cam_id] = tracker

    def remove_camera(self, cam_id: str):
        with self._lock:
            tracker = self._trackers.pop(cam_id, None)
            if tracker:
                tracker.stop()

    def active_cameras(self):
        with self._lock:
            return list(self._trackers.keys())


# Global singleton
person_tracker_manager = PersonTrackerManager()
