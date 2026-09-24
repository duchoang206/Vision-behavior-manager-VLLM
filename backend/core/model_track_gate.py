import math

from core.model_track_masks import bbox, compatible_boxes, track_key


class ModelTrackGate:
    def __init__(self, confidence_thresholds=None, acquire_confidence=.50, retain_confidence=.30):
        self.confidence_thresholds = confidence_thresholds or {}
        self.acquire_confidence = acquire_confidence
        self.retain_confidence = retain_confidence
        self.entries = {}

    def get_threshold(self, camera_id, label, confirmed):
        cam_thresholds = self.confidence_thresholds.get("cameras", {}).get(camera_id, {})
        default_thresholds = self.confidence_thresholds.get("default", {})

        def find_val(m, k):
            if not isinstance(m, dict):
                return None
            if k in m:
                return m[k]
            k_cf = str(k).casefold()
            for key, val in m.items():
                if str(key).casefold() == k_cf:
                    return val
            return None

        target = find_val(cam_thresholds, label)
        if target is None:
            target = find_val(default_thresholds, label)

        if target is not None:
            try:
                base = float(target)
                return max(0.005, base * 0.75) if confirmed else max(0.01, base)
            except (ValueError, TypeError):
                pass

        return self.retain_confidence if confirmed else self.acquire_confidence

    def filter(self, camera_id, objects, now, verification_categories=()):
        output = []
        reasons = {}
        for obj in objects:
            key = camera_id, track_key(obj)
            previous = self.entries.get(key)
            box = bbox(obj)
            score = float(obj.get("confidence", 0))
            detected_at = float(obj.get("detected_at", 0))
            valid = all(math.isfinite(value) for value in box + [score, detected_at]) and min(box[2:]) > 0
            continuous = previous and now - previous["seen"] <= 300 and compatible_boxes(previous["box"], box)
            confirmed = bool(continuous and previous["confirmed"])

            label = obj.get("class") or obj.get("label") or ""
            threshold = self.get_threshold(camera_id, label, confirmed)
            can_verify = obj.get("category") in verification_categories and score >= min(threshold, .25)
            reason = None
            if not valid or min(box[2] * 1280, box[3] * 720) < 8:
                reason = "invalid_or_tiny_box"
            elif now - detected_at > 300 or detected_at > now + 50:
                reason = "detector_stale"
            elif score < threshold and not can_verify:
                reason = "low_confidence"
            else:
                category = str(obj.get("category", "")).lower()
                raw_class = str(obj.get("class", "")).lower()
                raw_label = str(obj.get("label", "")).lower()
                is_robot = category == "robot" or "robot" in raw_class or "robot" in raw_label or raw_class.startswith("amr") or raw_class.startswith("agv")

                if is_robot and continuous and previous and "smoothed_box" in previous:
                    prev_sb = previous["smoothed_box"]
                    prev_x, prev_y, prev_w, prev_h = prev_sb
                    prev_cx = prev_x + prev_w / 2.0
                    prev_cy = prev_y + prev_h / 2.0

                    cx = box[0] + box[2] / 2.0
                    cy = box[1] + box[3] / 2.0

                    # 1. Center position: fast responsiveness (alpha=0.80)
                    cx_smooth = prev_cx + 0.80 * (cx - prev_cx)
                    cy_smooth = prev_cy + 0.80 * (cy - prev_cy)

                    # 2. Width deadband (< 4% change treated as noise) & max 8% rate limit
                    w_change_ratio = abs(box[2] - prev_w) / max(prev_w, 1e-4)
                    target_w = prev_w if w_change_ratio < 0.04 else box[2]
                    clamped_w = max(prev_w * 0.92, min(prev_w * 1.08, target_w))
                    w_smooth = prev_w + 0.25 * (clamped_w - prev_w)

                    # 3. Height deadband (< 4% change treated as noise) & max 8% rate limit
                    h_change_ratio = abs(box[3] - prev_h) / max(prev_h, 1e-4)
                    target_h = prev_h if h_change_ratio < 0.04 else box[3]
                    clamped_h = max(prev_h * 0.92, min(prev_h * 1.08, target_h))
                    h_smooth = prev_h + 0.25 * (clamped_h - prev_h)

                    # Top-left reconstruction with [0, 1] clipping
                    x_smooth = max(0.0, min(1.0 - w_smooth, cx_smooth - w_smooth / 2.0))
                    y_smooth = max(0.0, min(1.0 - h_smooth, cy_smooth - h_smooth / 2.0))
                    smoothed_box = [x_smooth, y_smooth, w_smooth, h_smooth]
                else:
                    smoothed_box = [box[0], box[1], box[2], box[3]]

                hits = ((previous["hits"] if continuous else 0) + 1) if score >= threshold else 0
                candidate_hits = (previous.get("candidate_hits", 0) if continuous else 0) + 1
                confirmed = hits >= 2 or confirmed
                self.entries[key] = dict(box=box, smoothed_box=smoothed_box, hits=hits, candidate_hits=candidate_hits, confirmed=confirmed, seen=now)
                if confirmed and score >= threshold:
                    if is_robot and continuous and previous and "smoothed_box" in previous:
                        obj_out = dict(obj,
                                       x=round(smoothed_box[0], 5),
                                       y=round(smoothed_box[1], 5),
                                       w=round(smoothed_box[2], 5),
                                       h=round(smoothed_box[3], 5))
                    else:
                        obj_out = obj
                    output.append(obj_out)
                elif can_verify and candidate_hits >= 2:
                    output.append(dict(obj, requires_label_verification=True))
                else:
                    reason = "confirming_track"
            if reason:
                reasons[reason] = reasons.get(reason, 0) + 1
        self.entries = {key: entry for key, entry in self.entries.items() if now - entry["seen"] <= 700}
        return output, reasons
