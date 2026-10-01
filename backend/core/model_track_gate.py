import math
import os
import statistics
from collections import deque

from core.model_track_masks import box_iou, bbox, compatible_boxes, track_key

# One Euro filter (Casiez et al.): heavy smoothing while a box is nearly still, light smoothing
# (low lag) while it moves. Class-agnostic: applied to the bbox of every model type.
POSITION_MIN_CUTOFF_HZ = 1.5
POSITION_BETA = 45.0
SIZE_MIN_CUTOFF_HZ = 0.6
SIZE_BETA = 10.0
SIZE_MEDIAN_WINDOW = 5
WARMUP_FRAMES = 10  # a freshly shown track converges on its real size quickly instead of growing from its first box
WARMUP_BOOST = 3.0
SMALL_BOX_HEIGHT = .25  # boxes shorter than this (normalised) get proportionally stronger size smoothing
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
    tiny envelope for ~0.4 s the output is held (a dead-band around the held value); it only follows the
    low-pass filtered value once that has really moved. It never snaps to the raw box, so a size change that
    the filter is still absorbing cannot pop out when the lock re-engages."""

    def __init__(self):
        self.samples = deque(maxlen=STILL_WINDOW)
        self.out = None
        self.locked = False

    def __call__(self, cx, cy, w, h, filtered):
        self.samples.append((cx, cy, w, h))
        if self.out is None:
            self.out = tuple(filtered)
        locked = False
        if len(self.samples) >= STILL_MIN_SAMPLES:
            columns = list(zip(*self.samples))
            spread = [max(column) - min(column) for column in columns]
            median_w, median_h = statistics.median(columns[2]), statistics.median(columns[3])
            locked = (spread[0] < STILL_CENTER_PX / 1280 and spread[1] < STILL_CENTER_PX / 720
                      and spread[2] < STILL_SIZE_RATIO * median_w and spread[3] < STILL_SIZE_RATIO * median_h)
        if locked:
            dead = (STILL_CENTER_PX / 2 / 1280, STILL_CENTER_PX / 2 / 720, STILL_SIZE_RATIO / 2 * self.out[2],
                    STILL_SIZE_RATIO / 2 * self.out[3])
            self.out = tuple(o if abs(f - o) <= d else o + .35 * (f - o) for o, f, d in zip(self.out, filtered, dead))
        elif self.locked:  # leaving the lock: ease towards the filtered value instead of jumping
            self.out = tuple(o + .5 * (f - o) for o, f in zip(self.out, filtered))
        else:
            self.out = tuple(filtered)
        self.locked = locked
        return self.out


def _box_filters(box):
    return dict(frames=0, lock=_StationaryLock(), size_history=deque(maxlen=SIZE_MEDIAN_WINDOW), motion=dict(cx=box[0] + box[2] / 2.0, cy=box[1] + box[3] / 2.0, vx=0.0, vy=0.0, t_at=None, t_cx=0.0, t_cy=0.0),
                cx=_OneEuro(box[0] + box[2] / 2.0, POSITION_MIN_CUTOFF_HZ, POSITION_BETA),
                cy=_OneEuro(box[1] + box[3] / 2.0, POSITION_MIN_CUTOFF_HZ, POSITION_BETA),
                w=_OneEuro(box[2], SIZE_MIN_CUTOFF_HZ, SIZE_BETA),
                h=_OneEuro(box[3], SIZE_MIN_CUTOFF_HZ, SIZE_BETA))


# A track the detector has not re-confirmed for this long is dropped (the tracker keeps predicting meanwhile).
DETECTOR_STALE_MS = float(os.getenv("DETECTOR_STALE_MS", "500"))


CLASS_SCORE_DECAY = .92
CLASS_SWITCH_RATIO = 1.4


# A confirmed track missing from the tracker output (shadow tracking, a low-confidence frame) is carried on its
# last velocity for up to this long, so a one-frame hole never shows up as a blink on the Monitor.
BRIDGE_MS = float(os.getenv("TRACK_BRIDGE_MS", "400"))


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

    def _bridge(self, camera_id, now, output, emitted):
        bridged = []
        for key, entry in self.entries.items():
            if key[0] != camera_id or key in emitted or not entry.get("confirmed") or not entry.get("last_out"):
                continue
            elapsed = now - entry["out_at"]
            if not 0 < elapsed <= BRIDGE_MS:
                continue
            last, motion = entry["last_out"], entry["filters"]["motion"]
            width, height = float(last["w"]), float(last["h"])
            x = max(0.0, min(1.0 - width, float(last["x"]) + motion["vx"] * elapsed * .9))
            y = max(0.0, min(1.0 - height, float(last["y"]) + motion["vy"] * elapsed * .9))
            box = [x, y, width, height]
            if any(box_iou(box, bbox(other)) >= .4 for other in output + bridged
                   if other.get("category") == last.get("category")):
                continue  # the tracker already re-issued this robot under another id
            bridged.append(dict(last, x=round(x, 5), y=round(y, 5), tracking_state="predicted", bridged=True,
                                observed_at=now))
        return bridged

    @staticmethod
    def _stabilize_class(previous, obj, now):
        recent = previous is not None and now - previous["seen"] <= 300
        scores = {key: (value[0] * CLASS_SCORE_DECAY, value[1]) for key, value in
                  (previous.get("class_scores") or {}).items()} if recent else {}
        try:
            confidence = max(0.0, float(obj.get("confidence", 0) or 0))
        except (TypeError, ValueError):
            confidence = 0.0
        identity = dict(**{name: obj.get(name) for name in ("class", "class_id", "label")})
        key = obj.get("class_id", obj.get("class"))
        scores[key] = (scores.get(key, (0.0, None))[0] + confidence, identity)
        leader = previous.get("class_leader") if recent else None
        best = max(scores, key=lambda candidate: scores[candidate][0])
        if leader not in scores or scores[best][0] > scores[leader][0] * CLASS_SWITCH_RATIO:
            leader = best
        if leader != key:
            obj = dict(obj, **scores[leader][1])
        return obj, scores, leader

    def filter(self, camera_id, objects, now, verification_categories=()):
        output = []
        reasons = {}
        emitted = set()
        for obj in objects:
            is_robot = str(obj.get("category", "")).lower() == "robot"
            # Robots look alike, so the detector flips between robot classes; key them by tracker id only so
            # confirmation and smoothing survive a flip, and let accumulated class scores pick the label.
            key = camera_id, ((obj.get("model_id"), obj["id"]) if is_robot else track_key(obj))
            previous = self.entries.get(key)
            class_scores = class_leader = None
            if is_robot:
                obj, class_scores, class_leader = self._stabilize_class(previous, obj, now)
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
                smoothed = bool(continuous and previous and previous.get("filters") and previous.get("emitted"))
                if smoothed:
                    filters = previous["filters"]
                    dt = min(0.5, max(0.005, (now - previous["seen"]) / 1000.0))
                    cx_raw, cy_raw = _coast_center(filters["motion"], box[0] + box[2] / 2.0, box[1] + box[3] / 2.0,
                                                   str(obj.get("tracking_state", "")).lower() != "predicted", now, dt)
                    cx = filters["cx"](cx_raw, dt)
                    cy = filters["cy"](cy_raw, dt)
                    # Size: median of the last few detections (drops one-frame outliers), then a spike guard
                    # (<= 8% per frame) and a heavier low-pass for small boxes, whose pixel noise is a bigger fraction.
                    filters["size_history"].append((box[2], box[3]))
                    median_w = statistics.median(item[0] for item in filters["size_history"])
                    median_h = statistics.median(item[1] for item in filters["size_history"])
                    softness = max(.7, min(1.0, filters["h"].value / SMALL_BOX_HEIGHT))
                    boost = WARMUP_BOOST if filters["frames"] < WARMUP_FRAMES else 1.0
                    filters["frames"] += 1
                    filters["w"].min_cutoff = filters["h"].min_cutoff = SIZE_MIN_CUTOFF_HZ * softness * boost
                    w_smooth = filters["w"](max(filters["w"].value * .92, min(filters["w"].value * 1.08, median_w)), dt)
                    h_smooth = filters["h"](max(filters["h"].value * .92, min(filters["h"].value * 1.08, median_h)), dt)
                    cx, cy, w_smooth, h_smooth = filters["lock"](cx_raw, cy_raw, box[2], box[3], (cx, cy, w_smooth, h_smooth))
                    smoothed_box = [max(0.0, min(1.0 - w_smooth, cx - w_smooth / 2.0)),
                                    max(0.0, min(1.0 - h_smooth, cy - h_smooth / 2.0)), w_smooth, h_smooth]
                else:
                    filters = _box_filters(box)
                    smoothed_box = list(box)

                hits = ((previous["hits"] if continuous else 0) + 1) if score >= threshold else 0
                candidate_hits = (previous.get("candidate_hits", 0) if continuous else 0) + 1
                confirmed = hits >= 2 or confirmed
                self.entries[key] = dict(box=box, smoothed_box=smoothed_box, filters=filters, class_scores=class_scores, class_leader=class_leader, hits=hits, candidate_hits=candidate_hits, confirmed=confirmed, seen=now,
                                         last_out=(previous or {}).get("last_out"), out_at=(previous or {}).get("out_at"),
                                         emitted=(previous or {}).get("emitted", False))
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
                    emitted.add(key)
                    self.entries[key]["last_out"], self.entries[key]["out_at"], self.entries[key]["emitted"] = obj_out, now, True
                elif can_verify and candidate_hits >= 2:
                    output.append(dict(obj, requires_label_verification=True))
                else:
                    reason = "confirming_track"
            if reason:
                reasons[reason] = reasons.get(reason, 0) + 1
        output.extend(self._bridge(camera_id, now, output, emitted))
        self.entries = {key: entry for key, entry in self.entries.items() if now - entry["seen"] <= 700}
        return output, reasons
