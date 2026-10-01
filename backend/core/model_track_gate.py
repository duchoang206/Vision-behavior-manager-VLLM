import math
import os
import statistics
from collections import deque

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


PREDICTED_FRAME_WEIGHT = .10
VELOCITY_TAU_MS = 100.0
VELOCITY_MAX_GAP_MS = 600.0
COAST_DECAY_TAU_MS = 800.0


def _coast_center(motion, cx, cy, tracked, now, dt):
    """DeepStream alternates detector ("tracked") and Kalman ("predicted") frames, and predicted frames
    under-move, so raw motion arrives as ~12 Hz bursts. Keep a velocity from detector frames and let
    predicted frames coast on it instead of trusting their lagging position."""
    if tracked:
        t_at = motion["t_at"]
        if t_at is not None and 0 < now - t_at <= VELOCITY_MAX_GAP_MS:
            elapsed = now - t_at
            k = 1.0 - math.exp(-elapsed / VELOCITY_TAU_MS)
            motion["vx"] += k * ((cx - motion["t_cx"]) / elapsed - motion["vx"])
            motion["vy"] += k * ((cy - motion["t_cy"]) / elapsed - motion["vy"])
        else:
            motion["vx"] = motion["vy"] = 0.0
        motion["t_at"], motion["t_cx"], motion["t_cy"] = now, cx, cy
        motion["cx"], motion["cy"] = cx, cy
    else:
        step = dt * 1000.0
        decay = math.exp(-step / COAST_DECAY_TAU_MS)
        motion["vx"] *= decay
        motion["vy"] *= decay
        est_x, est_y = motion["cx"] + motion["vx"] * step, motion["cy"] + motion["vy"] * step
        motion["cx"] = PREDICTED_FRAME_WEIGHT * cx + (1 - PREDICTED_FRAME_WEIGHT) * est_x
        motion["cy"] = PREDICTED_FRAME_WEIGHT * cy + (1 - PREDICTED_FRAME_WEIGHT) * est_y
    return motion["cx"], motion["cy"]


STILL_WINDOW = 10
STILL_MIN_SAMPLES = 6
STILL_CENTER_PX = 3.0
STILL_SIZE_RATIO = .05


class _StationaryLock:
    """Detector noise keeps a parked robot's box breathing by a few pixels. While the raw box stays inside a
    tiny envelope for ~0.4 s, output its median instead; any real movement leaves the envelope and unlocks."""

    def __init__(self):
        self.samples = deque(maxlen=STILL_WINDOW)
        self.out = None
        self.locked = False

    def __call__(self, cx, cy, w, h, filtered):
        self.samples.append((cx, cy, w, h))
        locked, target = False, filtered
        if len(self.samples) >= STILL_MIN_SAMPLES:
            columns = list(zip(*self.samples))
            spread = [max(column) - min(column) for column in columns]
            median = [statistics.median(column) for column in columns]
            if (spread[0] < STILL_CENTER_PX / 1280 and spread[1] < STILL_CENTER_PX / 720
                    and spread[2] < STILL_SIZE_RATIO * median[2] and spread[3] < STILL_SIZE_RATIO * median[3]):
                locked, target = True, tuple(median)
        if self.out is None or not (locked or self.locked):
            self.out = target
        else:  # ease into / out of the lock so it never snaps
            self.out = tuple(o + .5 * (t - o) for o, t in zip(self.out, target))
        self.locked = locked
        return self.out


def _box_filters(box):
    return dict(lock=_StationaryLock(), motion=dict(cx=box[0] + box[2] / 2.0, cy=box[1] + box[3] / 2.0, vx=0.0, vy=0.0, t_at=None, t_cx=0.0, t_cy=0.0),
                cx=_OneEuro(box[0] + box[2] / 2.0, POSITION_MIN_CUTOFF_HZ, POSITION_BETA),
                cy=_OneEuro(box[1] + box[3] / 2.0, POSITION_MIN_CUTOFF_HZ, POSITION_BETA),
                w=_OneEuro(box[2], SIZE_MIN_CUTOFF_HZ, SIZE_BETA),
                h=_OneEuro(box[3], SIZE_MIN_CUTOFF_HZ, SIZE_BETA))


# A track the detector has not re-confirmed for this long is dropped (the tracker keeps predicting meanwhile).
DETECTOR_STALE_MS = float(os.getenv("DETECTOR_STALE_MS", "500"))


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
            elif now - detected_at > DETECTOR_STALE_MS or detected_at > now + 50:
                reason = "detector_stale"
            elif score < threshold and not can_verify:
                reason = "low_confidence"
            else:
                smoothed = bool(continuous and previous and previous.get("filters"))
                if smoothed:
                    filters = previous["filters"]
                    dt = min(0.5, max(0.005, (now - previous["seen"]) / 1000.0))
                    cx_raw, cy_raw = _coast_center(filters["motion"], box[0] + box[2] / 2.0, box[1] + box[3] / 2.0,
                                                   str(obj.get("tracking_state", "")).lower() != "predicted", now, dt)
                    cx = filters["cx"](cx_raw, dt)
                    cy = filters["cy"](cy_raw, dt)
                    # Spike guard: a detector size outlier may move the filtered size by <= 8% per frame.
                    w_smooth = filters["w"](max(filters["w"].value * .92, min(filters["w"].value * 1.08, box[2])), dt)
                    h_smooth = filters["h"](max(filters["h"].value * .92, min(filters["h"].value * 1.08, box[3])), dt)
                    cx, cy, w_smooth, h_smooth = filters["lock"](cx_raw, cy_raw, box[2], box[3], (cx, cy, w_smooth, h_smooth))
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
