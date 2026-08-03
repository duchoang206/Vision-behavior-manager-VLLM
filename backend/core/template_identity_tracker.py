"""Low-latency identity tracking driven by trusted, multi-angle label crops."""

import base64
import os
import threading
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

from core.identity_utils import identity_global_id, stable_numeric_id
from core.rtsp_reader import RTSPLatestFrameReader
from core.registered_target_mask import advance_mask, eligible_target, registered_target_mask_segmenter


def _normalize_bbox(bbox: Optional[List[float]]) -> Optional[List[float]]:
    if not bbox or len(bbox) != 4:
        return None
    try:
        x, y, w, h = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None
    x = min(1.0, max(0.0, x))
    y = min(1.0, max(0.0, y))
    w = min(1.0 - x, max(0.0, w))
    h = min(1.0 - y, max(0.0, h))
    if w <= 0.0 or h <= 0.0:
        return None
    return [x, y, w, h]


def _decode_crop_image(crop_image: Optional[str]):
    if not crop_image:
        return None
    payload = crop_image.split(",", 1)[1] if "," in crop_image else crop_image
    try:
        raw = base64.b64decode(payload)
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img if img is not None and img.size > 0 else None
    except Exception:
        return None


def _crop_from_bbox(frame, bbox: List[float]):
    img_h, img_w = frame.shape[:2]
    x = max(0, min(img_w - 1, int(bbox[0] * img_w)))
    y = max(0, min(img_h - 1, int(bbox[1] * img_h)))
    w = max(2, int(bbox[2] * img_w))
    h = max(2, int(bbox[3] * img_h))
    x2 = max(x + 1, min(img_w, x + w))
    y2 = max(y + 1, min(img_h, y + h))
    crop = frame[y:y2, x:x2]
    return crop if crop.size > 0 else None


def _bbox_to_pixels(frame, bbox: List[float]):
    img_h, img_w = frame.shape[:2]
    x = max(0, min(img_w - 2, int(round(bbox[0] * img_w))))
    y = max(0, min(img_h - 2, int(round(bbox[1] * img_h))))
    w = max(2, min(img_w - x, int(round(bbox[2] * img_w))))
    h = max(2, min(img_h - y, int(round(bbox[3] * img_h))))
    return (x, y, w, h)


def _bbox_from_pixels(frame, rect) -> Optional[List[float]]:
    img_h, img_w = frame.shape[:2]
    try:
        x, y, w, h = [float(v) for v in rect]
    except (TypeError, ValueError):
        return None
    return _normalize_bbox([x / img_w, y / img_h, w / img_w, h / img_h])


def _create_cv_tracker():
    factories = [
        getattr(cv2, "TrackerCSRT_create", None),
        getattr(cv2, "TrackerKCF_create", None),
    ]
    legacy = getattr(cv2, "legacy", None)
    if legacy is not None:
        factories.extend([
            getattr(legacy, "TrackerCSRT_create", None),
            getattr(legacy, "TrackerKCF_create", None),
            getattr(legacy, "TrackerMOSSE_create", None),
        ])
    for factory in factories:
        if factory is None:
            continue
        try:
            return factory()
        except Exception:
            continue
    return None


def _template_similarity(a, b) -> float:
    if a is None or b is None or a.size == 0 or b.size == 0:
        return 0.0
    try:
        gray_a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
        if gray_a.shape[:2] != gray_b.shape[:2]:
            gray_b = cv2.resize(gray_b, (gray_a.shape[1], gray_a.shape[0]), interpolation=cv2.INTER_AREA)
        if gray_a.std() < 1.0 or gray_b.std() < 1.0:
            return 0.0

        res = cv2.matchTemplate(gray_a, gray_b, cv2.TM_CCOEFF_NORMED)
        shape_score = max(0.0, float(res[0, 0]))

        hsv_a = cv2.cvtColor(a, cv2.COLOR_BGR2HSV)
        hsv_b = cv2.cvtColor(b, cv2.COLOR_BGR2HSV)
        hist_a = cv2.calcHist([hsv_a], [0, 1], None, [16, 16], [0, 180, 0, 256])
        hist_b = cv2.calcHist([hsv_b], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist_a, hist_a, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist_b, hist_b, 0, 1, cv2.NORM_MINMAX)
        color_corr = cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)
        color_score = (max(-1.0, min(1.0, float(color_corr))) + 1.0) / 2.0

        return (shape_score * 0.65) + (color_score * 0.35)
    except Exception:
        return 0.0


def _bbox_center(bbox: Sequence[float]) -> Tuple[float, float]:
    return bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3] / 2.0


class TemplateIdentityCameraTracker:
    def __init__(self, cam_id: str, rtsp_url: str, metadata_callback: Callable, target_fps: Optional[int] = None):
        self.cam_id = cam_id
        self.rtsp_url = rtsp_url
        self.metadata_callback = metadata_callback
        self.target_fps = target_fps or int(os.getenv("IDENTITY_TEMPLATE_TARGET_FPS", "24"))
        self.init_min_score = float(os.getenv("IDENTITY_TEMPLATE_INIT_MIN_SCORE", "0.68"))
        self.min_match_score = float(os.getenv("IDENTITY_TEMPLATE_MIN_SCORE", "0.68"))
        self.verify_min_score = float(os.getenv("IDENTITY_TEMPLATE_VERIFY_MIN_SCORE", "0.65"))
        self.motion_enabled = os.getenv("IDENTITY_TEMPLATE_ENABLE_MOTION", "0") == "1"
        self.reanchor_interval = max(1, int(os.getenv("IDENTITY_TEMPLATE_REANCHOR_INTERVAL", "12")))
        self.max_lost_hold = int(os.getenv("IDENTITY_TEMPLATE_MAX_LOST_HOLD", "24"))
        self.smooth_alpha = float(os.getenv("IDENTITY_TEMPLATE_SMOOTH_ALPHA", "0.42"))
        self.max_templates = int(os.getenv("IDENTITY_TEMPLATE_GALLERY_SIZE", "16"))
        self.template_diversity_score = float(os.getenv("IDENTITY_TEMPLATE_DIVERSITY_SCORE", "0.965"))
        self.full_search_interval = max(1, int(os.getenv("IDENTITY_TEMPLATE_FULL_SEARCH_INTERVAL", "2")))
        self.targets: Dict[str, dict] = {}
        self.lock = threading.RLock()
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.processed_frames = 0
        self.measured_fps = 0.0
        self._fps_window_started_at = time.time()
        self._fps_window_frames = 0

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True, name=f"identity-template-{self.cam_id}")
        self.thread.start()
        print(f"[IdentityTemplate] Started for camera {self.cam_id}", flush=True)

    def stop(self):
        self.running = False

    def add_target(
        self,
        label: str,
        category: str,
        bbox: List[float],
        crop_image: Optional[str] = None,
        search_full_frame: bool = False,
        mask: Optional[dict] = None,
        frame_image: Optional[str] = None,
    ):
        normalized = _normalize_bbox(bbox)
        if normalized is None:
            return False
        template = _decode_crop_image(crop_image)
        now = time.time()
        with self.lock:
            existing = self.targets.get(label)
            if existing is not None:
                gallery = existing.setdefault("templates", [])
                if not gallery and existing.get("template") is not None:
                    gallery.append(existing["template"])
                self._add_template_to_gallery(gallery, template)
                existing.update({
                    "category": category or existing.get("category", "object"),
                    "bbox": normalized,
                    "registered_bbox": normalized,
                    "template": gallery[-1] if gallery else template,
                    "confidence": 1.0,
                    "lost": 0,
                    "cv_tracker": None,
                    "flow_gray": None,
                    "flow_points": None,
                    "flow_bbox": None,
                    "flow_age": 0,
                    "motion_age": 0,
                    "mask": None,
                    "has_matched": False,
                    "velocity": [0.0, 0.0, 0.0, 0.0],
                    "last_observation_at": now,
                    "updated_at": now,
                    "search_full_frame": bool(search_full_frame),
                })
                template_count = len(gallery)
            else:
                gallery = []
                self._add_template_to_gallery(gallery, template)
                self.targets[label] = {
                    "id": stable_numeric_id(label),
                    "label": label,
                    "category": category or "object",
                    "bbox": normalized,
                    "registered_bbox": normalized,
                    "template": template,
                    "templates": gallery,
                    "confidence": 1.0,
                    "lost": 0,
                    "cv_tracker": None,
                    "flow_gray": None,
                    "flow_points": None,
                    "flow_bbox": None,
                    "flow_age": 0,
                    "motion_age": 0,
                    "mask": None,
                    "has_matched": False,
                    "velocity": [0.0, 0.0, 0.0, 0.0],
                    "last_observation_at": now,
                    "updated_at": now,
                    "search_full_frame": bool(search_full_frame),
                }
                template_count = len(gallery)
            target = self.targets[label]
            target["mask_revision"] = time.monotonic_ns()
            samples = target.setdefault("mask_samples", [])
            if mask is not None and frame_image:
                samples.append({"mask": mask, "frame_image": frame_image})
                del samples[:-16]
        print(
            f"[IdentityTemplate] Learned {label} on {self.cam_id} "
            f"({template_count}/{self.max_templates} trusted views)",
            flush=True,
        )
        return True

    def _add_template_to_gallery(self, gallery: List[np.ndarray], template) -> bool:
        if template is None:
            return False
        similarities = [_template_similarity(template, known) for known in gallery]
        if similarities and max(similarities) >= self.template_diversity_score:
            return False
        gallery.append(template.copy())
        if len(gallery) > self.max_templates:
            gallery.pop(0)
        return True

    def remove_target(self, label: str):
        with self.lock:
            self.targets.pop(label, None)

    def _object_from_target(
        self,
        target: dict,
        confidence: Optional[float] = None,
        state: str = "tracked",
        bbox: Optional[List[float]] = None,
    ) -> dict:
        bbox = bbox or target["bbox"]
        score = target.get("confidence", 0.0) if confidence is None else confidence
        object_id = identity_global_id(target.get("label"), target.get("category"), target["id"]) or target["id"]
        return {
            "id": object_id,
            "local_id": target["id"],
            "class": target["category"],
            "category": target["category"],
            "label": target["label"],
            "x": round(bbox[0], 4),
            "y": round(bbox[1], 4),
            "w": round(bbox[2], 4),
            "h": round(bbox[3], 4),
            "floor_x": round(bbox[0] + bbox[2] / 2.0, 4),
            "floor_y": round(bbox[1] + bbox[3], 4),
            "confidence": round(float(score), 3),
            "tracking_state": state,
        }

    def _template_score_at_bbox(self, frame, target: dict, bbox: List[float]) -> float:
        templates = target.get("templates") or ([target.get("template")] if target.get("template") is not None else [])
        if not templates:
            return 1.0
        crop = _crop_from_bbox(frame, bbox)
        return max((_template_similarity(crop, template) for template in templates), default=0.0)

    def _smooth_bbox(self, old_bbox: List[float], new_bbox: List[float], lost: int = 0) -> List[float]:
        old_cx, old_cy = _bbox_center(old_bbox)
        new_cx, new_cy = _bbox_center(new_bbox)
        relative_jump = np.hypot(new_cx - old_cx, new_cy - old_cy) / max(0.02, old_bbox[2], old_bbox[3])
        alpha = self.smooth_alpha
        if lost > 0:
            alpha = 1.0
        elif relative_jump >= 0.35:
            alpha = max(alpha, 0.86)
        alpha = min(1.0, max(0.05, alpha))
        smoothed = [
            old_bbox[i] + (new_bbox[i] - old_bbox[i]) * alpha
            for i in range(4)
        ]
        return _normalize_bbox(smoothed) or new_bbox

    def _predict_bbox(self, target: dict, now: Optional[float] = None) -> List[float]:
        bbox = target["bbox"]
        velocity = target.get("velocity") or [0.0, 0.0, 0.0, 0.0]
        dt = min(1.25, max(0.0, (now or time.time()) - target.get("last_observation_at", time.time())))
        cx, cy = _bbox_center(bbox)
        pred_w = max(0.005, bbox[2] + velocity[2] * dt)
        pred_h = max(0.005, bbox[3] + velocity[3] * dt)
        pred_cx = cx + velocity[0] * dt
        pred_cy = cy + velocity[1] * dt
        return _normalize_bbox([
            pred_cx - pred_w / 2.0,
            pred_cy - pred_h / 2.0,
            pred_w,
            pred_h,
        ]) or list(bbox)

    def _accept_observation(self, target: dict, measured_bbox: List[float], score: float) -> List[float]:
        now = time.time()
        old_bbox = target["bbox"]
        dt = max(1.0 / max(1, self.target_fps), now - target.get("last_observation_at", now))
        old_cx, old_cy = _bbox_center(old_bbox)
        new_cx, new_cy = _bbox_center(measured_bbox)
        instant = [
            (new_cx - old_cx) / dt,
            (new_cy - old_cy) / dt,
            (measured_bbox[2] - old_bbox[2]) / dt,
            (measured_bbox[3] - old_bbox[3]) / dt,
        ]
        previous = target.get("velocity") or [0.0, 0.0, 0.0, 0.0]
        velocity_alpha = 0.35 if target.get("lost", 0) > 0 else 0.58
        velocity = [previous[i] * velocity_alpha + instant[i] * (1.0 - velocity_alpha) for i in range(4)]
        center_speed = float(np.hypot(velocity[0], velocity[1]))
        if center_speed > 4.0:
            scale = 4.0 / center_speed
            velocity[0] *= scale
            velocity[1] *= scale

        reacquired = not target.get("has_matched") or target.get("lost", 0) > 0
        if reacquired:
            velocity = [0.0, 0.0, 0.0, 0.0]
        accepted = self._smooth_bbox(old_bbox, measured_bbox, lost=1 if reacquired else 0)
        target["bbox"] = accepted
        target["velocity"] = velocity
        target["confidence"] = float(score)
        target["lost"] = 0
        target["last_observation_at"] = now
        target["updated_at"] = now
        return accepted

    def _init_cv_tracker(self, frame, target: dict, bbox: List[float]) -> bool:
        tracker = _create_cv_tracker()
        if tracker is None:
            target["cv_tracker"] = None
            return False
        try:
            ok = tracker.init(frame, _bbox_to_pixels(frame, bbox))
        except Exception:
            ok = False
        if ok is not False:
            target["cv_tracker"] = tracker
            return True
        target["cv_tracker"] = None
        return False

    def _init_optical_flow(self, frame, target: dict, bbox: List[float]) -> bool:
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            x, y, w, h = _bbox_to_pixels(frame, bbox)
            mask = np.zeros_like(gray)
            margin_x = max(1, int(w * 0.06))
            margin_y = max(1, int(h * 0.06))
            mask[y + margin_y:y + h - margin_y, x + margin_x:x + w - margin_x] = 255
            points = cv2.goodFeaturesToTrack(
                gray,
                mask=mask,
                maxCorners=80,
                qualityLevel=0.01,
                minDistance=5,
                blockSize=7,
            )
            if points is None or len(points) < 5:
                target["flow_gray"] = None
                target["flow_points"] = None
                return False
            target["flow_gray"] = gray
            target["flow_points"] = points.astype(np.float32)
            target["flow_bbox"] = list(bbox)
            target["flow_age"] = 0
            return True
        except Exception:
            target["flow_gray"] = None
            target["flow_points"] = None
            return False

    def _init_motion_tracker(self, frame, target: dict, bbox: List[float]) -> bool:
        target["motion_age"] = 0
        if not self.motion_enabled:
            target["cv_tracker"] = None
            target["flow_gray"] = None
            target["flow_points"] = None
            target["flow_bbox"] = None
            return False
        cv_active = self._init_cv_tracker(frame, target, bbox)
        flow_active = self._init_optical_flow(frame, target, bbox)
        return cv_active or flow_active

    def _update_optical_flow(self, frame, target: dict) -> Optional[List[float]]:
        previous_gray = target.get("flow_gray")
        previous_points = target.get("flow_points")
        if previous_gray is None or previous_points is None or len(previous_points) < 5:
            return None
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            next_points, status, errors = cv2.calcOpticalFlowPyrLK(
                previous_gray,
                gray,
                previous_points,
                None,
                winSize=(31, 31),
                maxLevel=4,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 24, 0.01),
                flags=0,
                minEigThreshold=1e-4,
            )
            if next_points is None or status is None:
                return None
            valid = status.reshape(-1).astype(bool)
            if errors is not None:
                valid &= errors.reshape(-1) < 35.0
            old = previous_points.reshape(-1, 2)[valid]
            new = next_points.reshape(-1, 2)[valid]
            if len(new) < 5:
                return None

            displacement = np.median(new - old, axis=0)
            old_center = np.median(old, axis=0)
            new_center = np.median(new, axis=0)
            old_radius = np.linalg.norm(old - old_center, axis=1)
            new_radius = np.linalg.norm(new - new_center, axis=1)
            usable = old_radius > 2.0
            scale = float(np.median(new_radius[usable] / old_radius[usable])) if np.any(usable) else 1.0
            scale = min(1.35, max(0.72, scale))

            img_h, img_w = frame.shape[:2]
            bbox = target.get("flow_bbox") or target["bbox"]
            cx, cy = _bbox_center(bbox)
            width = bbox[2] * scale
            height = bbox[3] * scale
            candidate = _normalize_bbox([
                cx + float(displacement[0]) / img_w - width / 2.0,
                cy + float(displacement[1]) / img_h - height / 2.0,
                width,
                height,
            ])
            if candidate is None:
                return None

            target["flow_gray"] = gray
            target["flow_points"] = new.reshape(-1, 1, 2).astype(np.float32)
            target["flow_bbox"] = candidate
            target["flow_age"] = target.get("flow_age", 0) + 1
            return candidate
        except Exception:
            return None

    def _search_template_in_region(self, search_img, template, min_score: float = 0.45) -> Optional[tuple]:
        if search_img is None or template is None or search_img.size == 0 or template.size == 0:
            return None
        try:
            search_gray = cv2.cvtColor(search_img, cv2.COLOR_BGR2GRAY)
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            if template_gray.std() < 1.0:
                return None

            resize_ratio = min(1.0, 720.0 / max(1, search_gray.shape[1]))
            if resize_ratio < 1.0:
                search_gray = cv2.resize(search_gray, None, fx=resize_ratio, fy=resize_ratio, interpolation=cv2.INTER_AREA)
                template_gray = cv2.resize(template_gray, None, fx=resize_ratio, fy=resize_ratio, interpolation=cv2.INTER_AREA)

            sh, sw = search_gray.shape[:2]
            th, tw = template_gray.shape[:2]
            if sw < 10 or sh < 10 or tw < 6 or th < 6:
                return None

            best_score = -1.0
            best_loc = None
            best_size = None

            for scale in (0.62, 0.78, 0.90, 1.0, 1.12, 1.30, 1.52):
                cur_tw = int(round(tw * scale))
                cur_th = int(round(th * scale))
                if cur_tw >= sw or cur_th >= sh or cur_tw < 8 or cur_th < 8:
                    continue

                tmpl = cv2.resize(template_gray, (cur_tw, cur_th), interpolation=cv2.INTER_AREA)
                res = cv2.matchTemplate(search_gray, tmpl, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                score = float(max_val)
                if score > best_score:
                    best_score = score
                    best_loc = max_loc
                    best_size = (cur_tw, cur_th)

            if best_loc is None or best_size is None or best_score < min_score:
                return None

            if resize_ratio < 1.0:
                best_loc = (int(round(best_loc[0] / resize_ratio)), int(round(best_loc[1] / resize_ratio)))
                best_size = (int(round(best_size[0] / resize_ratio)), int(round(best_size[1] / resize_ratio)))
            return best_loc, best_size, best_score
        except Exception:
            return None

    def _search_gallery_in_region(self, search_img, target: dict, min_score: float) -> Optional[tuple]:
        templates = target.get("templates") or ([target.get("template")] if target.get("template") is not None else [])
        best = None
        for template_index, template in enumerate(templates):
            result = self._search_template_in_region(search_img, template, min_score=min_score)
            if result is None:
                continue
            loc, size, score = result
            if best is None or score > best[2]:
                best = (loc, size, score, template_index)
        return best

    def _local_template_search(self, frame, target: dict) -> Optional[tuple]:
        templates = target.get("templates") or ([target.get("template")] if target.get("template") is not None else [])
        if not templates:
            return None

        bbox = self._predict_bbox(target)
        img_h, img_w = frame.shape[:2]
        bx = bbox[0] * img_w
        by = bbox[1] * img_h
        bw = max(12.0, bbox[2] * img_w)
        bh = max(12.0, bbox[3] * img_h)

        velocity = target.get("velocity") or [0.0, 0.0, 0.0, 0.0]
        lost = target.get("lost", 0)
        velocity_px_x = abs(velocity[0]) * img_w / max(1, self.target_fps)
        velocity_px_y = abs(velocity[1]) * img_h / max(1, self.target_fps)
        expansion = min(8.0, 0.65 + 0.55 * lost)
        pad_x = max(bw * expansion + velocity_px_x * 4.0, 96.0)
        pad_y = max(bh * expansion + velocity_px_y * 4.0, 96.0)
        sx1 = max(0, int(bx - pad_x))
        sy1 = max(0, int(by - pad_y))
        sx2 = min(img_w, int(bx + bw + pad_x))
        sy2 = min(img_h, int(by + bh + pad_y))

        search_region = frame[sy1:sy2, sx1:sx2]
        res = self._search_gallery_in_region(search_region, target, min_score=self.min_match_score)
        if res is not None:
            loc, size, score, _ = res
            x = sx1 + loc[0]
            y = sy1 + loc[1]
            tw, th = size
            found_bbox = _normalize_bbox([x / img_w, y / img_h, tw / img_w, th / img_h])
            if found_bbox:
                return found_bbox, score

        should_search_full = target.get("search_full_frame") or lost > 0
        if not should_search_full or (lost > 0 and (lost - 1) % self.full_search_interval != 0):
            return None

        # Coarse full-frame search reacquires a fast target after a jump/occlusion.
        res_full = self._search_gallery_in_region(frame, target, min_score=self.min_match_score + 0.05)
        if res_full is not None:
            loc, size, score, _ = res_full
            x, y = loc[0], loc[1]
            tw, th = size
            found_bbox = _normalize_bbox([x / img_w, y / img_h, tw / img_w, th / img_h])
            if found_bbox:
                return found_bbox, score

        return None

    def _match_target(self, frame, target: dict) -> Optional[dict]:
        bbox = target["bbox"]
        templates = target.get("templates") or []
        if not templates:
            template = _crop_from_bbox(frame, bbox)
            if template is None:
                return None
            target["template"] = template
            target["templates"] = [template]

        if not target.get("has_matched"):
            score = self._template_score_at_bbox(frame, target, bbox)
            candidate_bbox = bbox
            if score < self.init_min_score:
                found = self._local_template_search(frame, target)
                if found:
                    candidate_bbox, score = found
            if score < self.init_min_score:
                target["confidence"] = float(score)
                target["lost"] = target.get("lost", 0) + 1
                return None

            self._accept_observation(target, candidate_bbox, score)
            target["has_matched"] = True
            self._init_motion_tracker(frame, target, target["bbox"])
            return self._object_from_target(target, confidence=score)

        target["motion_age"] = target.get("motion_age", 0) + 1
        use_motion = self.motion_enabled and target["motion_age"] < self.reanchor_interval
        tracker = target.get("cv_tracker") if use_motion else None
        if tracker is not None:
            try:
                ok, rect = tracker.update(frame)
            except Exception:
                ok, rect = False, None
            if ok:
                tracked_bbox = _bbox_from_pixels(frame, rect)
                if tracked_bbox:
                    verify_score = self._template_score_at_bbox(frame, target, tracked_bbox)
                    if verify_score >= self.verify_min_score:
                        self._accept_observation(target, tracked_bbox, verify_score)
                        self._init_optical_flow(frame, target, tracked_bbox)
                        return self._object_from_target(target, confidence=verify_score)

        flow_bbox = self._update_optical_flow(frame, target) if use_motion else None
        if flow_bbox is not None:
            verify_score = self._template_score_at_bbox(frame, target, flow_bbox)
            if verify_score >= self.verify_min_score:
                self._accept_observation(target, flow_bbox, verify_score)
                if target.get("flow_age", 0) >= 12:
                    self._init_optical_flow(frame, target, flow_bbox)
                return self._object_from_target(target, confidence=verify_score)

        found = self._local_template_search(frame, target)
        if found:
            found_bbox, score = found
            if score >= self.min_match_score:
                self._accept_observation(target, found_bbox, score)
                self._init_motion_tracker(frame, target, found_bbox)
                return self._object_from_target(target, confidence=score)

        target["lost"] = target.get("lost", 0) + 1
        target["cv_tracker"] = None
        target["flow_gray"] = None
        target["flow_points"] = None
        target["flow_bbox"] = None
        if target["lost"] <= self.max_lost_hold:
            predicted_bbox = self._predict_bbox(target)
            confidence = max(0.05, float(target.get("confidence", 0.0)) * (0.82 ** target["lost"]))
            return self._object_from_target(target, confidence=confidence, state="predicted", bbox=predicted_bbox)
        target["has_matched"] = False
        return None

    def _loop(self):
        reader = RTSPLatestFrameReader(self.rtsp_url, cam_id=f"identity-{self.cam_id}")
        try:
            self._process_frames(reader)
        finally:
            reader.stop()
            registered_target_mask_segmenter.remove_camera(self.cam_id)
        print(f"[IdentityTemplate] Stopped for camera {self.cam_id}", flush=True)

    def _process_frames(self, reader):
        frame_interval = 1.0 / max(1, self.target_fps)
        while self.running:
            t0 = time.time()
            ret, frame = reader.get_latest_frame()
            if not ret or frame is None:
                time.sleep(min(0.01, frame_interval))
                continue

            with self.lock:
                objects = self._process_registered_frame(frame)

            self.processed_frames += 1
            self._fps_window_frames += 1
            fps_elapsed = time.time() - self._fps_window_started_at
            if fps_elapsed >= 2.0:
                self.measured_fps = self._fps_window_frames / fps_elapsed
                self._fps_window_frames = 0
                self._fps_window_started_at = time.time()

            if self.metadata_callback:
                tripwire_stats = {}
                roi_states = []
                try:
                    from core.behavior_analytics import behavior_engine
                    triggered_events, tripwire_stats, roi_states = behavior_engine.process_frame(self.cam_id, objects)
                except Exception as be_err:
                    print(f"[IdentityTemplate] BehaviorEngine error for {self.cam_id}: {be_err}", flush=True)

                self.metadata_callback({
                    "source": "identity_template",
                    "timestamp": int(time.time() * 1000),
                    "streams": [{
                        "cam_id": self.cam_id,
                        "objects": objects,
                        "tripwire_stats": tripwire_stats,
                        "rois": roi_states,
                    }]
                })

            remaining = frame_interval - (time.time() - t0)
            if remaining > 0:
                time.sleep(remaining)

    def _process_registered_frame(self, frame):
        targets = list(self.targets.values())
        mask_targets = [target for target in targets if eligible_target(target)]
        mask_enabled = registered_target_mask_segmenter.available()
        seeds = {}
        fallback = {}
        for target in targets:
            if not mask_enabled or not eligible_target(target) or (not target.get("mask") and not target.get("mask_samples")):
                obj = self._match_target(frame, target)
                if obj:
                    fallback[target["label"]] = obj
                    if eligible_target(target) and obj.get("tracking_state") == "tracked":
                        seeds[target["label"]] = [obj["x"], obj["y"], obj["w"], obj["h"]]
        observations = registered_target_mask_segmenter.track(self.cam_id, frame, mask_targets, seeds) if mask_enabled else {}
        objects = []
        for target in targets:
            observation = observations.get(target["label"]) if eligible_target(target) else None
            if observation and observation.get("mask"):
                measured = _normalize_bbox(observation["bbox"])
                if measured:
                    self._accept_observation(target, measured, observation["mask"]["confidence"])
                    try:
                        mask_latency_ms = float(registered_target_mask_segmenter.last_ms)
                    except (TypeError, ValueError):
                        mask_latency_ms = 0.0
                    latency_seconds = min(0.25, max(0.0, mask_latency_ms / 1000.0))
                    velocity = target.get("velocity") or [0.0, 0.0, 0.0, 0.0]
                    current_bbox = _normalize_bbox([
                        measured[0] + velocity[0] * latency_seconds,
                        measured[1] + velocity[1] * latency_seconds,
                        measured[2] + velocity[2] * latency_seconds,
                        measured[3] + velocity[3] * latency_seconds,
                    ]) or measured
                    current_mask = advance_mask(observation["mask"], measured, current_bbox)
                    target["bbox"] = current_bbox
                    target["has_matched"] = True
                    target["mask"] = current_mask
                    obj = self._object_from_target(target)
                    obj["mask"] = dict(current_mask, observed_at=int(time.time() * 1000))
                    objects.append(obj)
                    continue
            target["mask"] = None
            target["lost"] = target.get("lost", 0) + (0 if target["label"] in fallback else 1)
            if target["label"] in fallback:
                objects.append(fallback[target["label"]])
        return objects

    def debug_state(self):
        with self.lock:
            return {
                "cam_id": self.cam_id,
                "target_fps": self.target_fps,
                "measured_fps": round(self.measured_fps, 1),
                "processed_frames": self.processed_frames,
                "motion_enabled": self.motion_enabled,
                "mask_tracking": registered_target_mask_segmenter.status(),
                "targets": [
                    {
                        "label": t["label"],
                        "category": t["category"],
                        "bbox": t["bbox"],
                        "confidence": round(float(t.get("confidence", 0.0)), 3),
                        "lost": t.get("lost", 0),
                        "has_matched": bool(t.get("has_matched")),
                        "tracker_active": bool(t.get("mask")) or t.get("cv_tracker") is not None or t.get("flow_points") is not None,
                        "tracker_backend": "sam2_memory" if t.get("mask") else ("opencv" if t.get("cv_tracker") is not None else ("optical_flow" if t.get("flow_points") is not None else "search")),
                        "gallery_size": len(t.get("templates") or []),
                        "velocity": [round(float(v), 4) for v in (t.get("velocity") or [0.0, 0.0, 0.0, 0.0])],
                        "mask_source": (t.get("mask") or {}).get("source") if t.get("mask") else None,
                    }
                    for t in self.targets.values()
                ]
            }


class TemplateIdentityTrackerManager:
    def __init__(self):
        self.trackers: Dict[str, TemplateIdentityCameraTracker] = {}
        self.metadata_callback: Optional[Callable] = None
        self.lock = threading.RLock()

    def add_camera(self, cam_id: str, rtsp_url: str):
        with self.lock:
            if cam_id in self.trackers:
                tracker = self.trackers[cam_id]
                tracker.metadata_callback = self.metadata_callback
                return tracker
            tracker = TemplateIdentityCameraTracker(cam_id, rtsp_url, self.metadata_callback)
            self.trackers[cam_id] = tracker
            tracker.start()
            return tracker

    def remove_camera(self, cam_id: str):
        with self.lock:
            tracker = self.trackers.pop(cam_id, None)
        if tracker:
            tracker.stop()

    def add_target(
        self,
        cam_id: str,
        rtsp_url: str,
        label: str,
        category: str,
        bbox: List[float],
        crop_image: Optional[str] = None,
        search_full_frame: bool = False,
        mask: Optional[dict] = None,
        frame_image: Optional[str] = None,
    ):
        tracker = self.add_camera(cam_id, rtsp_url)
        return tracker.add_target(label, category, bbox, crop_image, search_full_frame=search_full_frame,
                                  mask=mask, frame_image=frame_image)

    def remove_target(self, label: str, cam_id: Optional[str] = None):
        with self.lock:
            if cam_id:
                tracker = self.trackers.get(cam_id)
                if tracker:
                    tracker.remove_target(label)
                return
            trackers = list(self.trackers.values())
        for tracker in trackers:
            tracker.remove_target(label)

    def active_cameras(self):
        with self.lock:
            return list(self.trackers.keys())

    def match_detection(
        self,
        cam_id: str,
        bbox: Optional[List[float]],
        det_class: Optional[str] = None,
    ) -> Optional[str]:
        detection = _normalize_bbox(bbox)
        if detection is None:
            return None
        with self.lock:
            tracker = self.trackers.get(cam_id)
        if tracker is None:
            return None

        dcx, dcy = _bbox_center(detection)
        det_class_lower = (det_class or "").lower()
        best_label = None
        best_score = 0.0
        with tracker.lock:
            for target in tracker.targets.values():
                if not target.get("has_matched") or target.get("lost", 0) > 2:
                    continue
                target_class = (target.get("category") or "").lower()
                if target_class and det_class_lower:
                    aliases = {
                        "robot": ("robot", "agv", "amr", "delivery"),
                        "rack": ("rack", "shelf", "pallet", "refrigerator"),
                        "person": ("person", "human", "worker"),
                    }
                    if not any(term in det_class_lower for term in aliases.get(target_class, (target_class,))):
                        continue

                tracked = target["bbox"]
                tx1, ty1, tw, th = tracked
                dx1, dy1, dw, dh = detection
                ix1, iy1 = max(tx1, dx1), max(ty1, dy1)
                ix2, iy2 = min(tx1 + tw, dx1 + dw), min(ty1 + th, dy1 + dh)
                intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                union = tw * th + dw * dh - intersection
                iou = intersection / union if union > 1e-6 else 0.0
                tcx, tcy = _bbox_center(tracked)
                center_distance = float(np.hypot(dcx - tcx, dcy - tcy))
                center_score = max(0.0, 1.0 - center_distance / max(0.03, tw, th, dw, dh))
                score = 0.72 * iou + 0.28 * center_score
                if (iou >= 0.12 or center_score >= 0.72) and score > best_score:
                    best_score = score
                    best_label = target["label"]
        return best_label

    def debug_state(self):
        with self.lock:
            return {cam_id: tracker.debug_state() for cam_id, tracker in self.trackers.items()}


template_identity_tracker_manager = TemplateIdentityTrackerManager()
