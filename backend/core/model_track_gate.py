import math

from core.model_track_masks import bbox, compatible_boxes, track_key

# One Euro filter (Casiez et al.): heavy smoothing while a box is nearly still, light smoothing
# (low lag) while it moves. Class-agnostic: applied to the bbox of every model type.
POSITION_MIN_CUTOFF_HZ = 1.5
POSITION_BETA = 45.0
SIZE_MIN_CUTOFF_HZ = 0.6
SIZE_BETA = 10.0
DERIVATIVE_CUTOFF_HZ = 1.0


def _alpha(dt, cutoff):
    return 1.0 / (1.0 + (1.0 / (2.0 * math.pi * cutoff)) / dt)


class _OneEuro:
    def __init__(self, value, min_cutoff, beta):
        self.value, self.speed = value, 0.0
        self.min_cutoff, self.beta = min_cutoff, beta

    def __call__(self, raw, dt):
        self.speed += _alpha(dt, DERIVATIVE_CUTOFF_HZ) * ((raw - self.value) / dt - self.speed)
        cutoff = self.min_cutoff + self.beta * abs(self.speed)
        self.value += _alpha(dt, cutoff) * (raw - self.value)
        return self.value


def _box_filters(box):
    return dict(cx=_OneEuro(box[0] + box[2] / 2.0, POSITION_MIN_CUTOFF_HZ, POSITION_BETA),
                cy=_OneEuro(box[1] + box[3] / 2.0, POSITION_MIN_CUTOFF_HZ, POSITION_BETA),
                w=_OneEuro(box[2], SIZE_MIN_CUTOFF_HZ, SIZE_BETA),
                h=_OneEuro(box[3], SIZE_MIN_CUTOFF_HZ, SIZE_BETA))


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
                smoothed = bool(continuous and previous and previous.get("filters"))
                if smoothed:
                    filters = previous["filters"]
                    dt = min(0.5, max(0.005, (now - previous["seen"]) / 1000.0))
                    cx = filters["cx"](box[0] + box[2] / 2.0, dt)
                    cy = filters["cy"](box[1] + box[3] / 2.0, dt)
                    # Spike guard: a detector size outlier may move the filtered size by <= 8% per frame.
                    w_smooth = filters["w"](max(filters["w"].value * .92, min(filters["w"].value * 1.08, box[2])), dt)
                    h_smooth = filters["h"](max(filters["h"].value * .92, min(filters["h"].value * 1.08, box[3])), dt)
                    smoothed_box = [max(0.0, min(1.0 - w_smooth, cx - w_smooth / 2.0)),
                                    max(0.0, min(1.0 - h_smooth, cy - h_smooth / 2.0)), w_smooth, h_smooth]
                else:
                    filters = _box_filters(box)
                    smoothed_box = list(box)

                hits = ((previous["hits"] if continuous else 0) + 1) if score >= threshold else 0
                candidate_hits = (previous.get("candidate_hits", 0) if continuous else 0) + 1
                confirmed = hits >= 2 or confirmed
                self.entries[key] = dict(box=box, smoothed_box=smoothed_box, filters=filters, hits=hits, candidate_hits=candidate_hits, confirmed=confirmed, seen=now)
                if confirmed and score >= threshold:
                    if smoothed:
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
