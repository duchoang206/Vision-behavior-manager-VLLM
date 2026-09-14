"""Camera-local continuity and bounded, motion-compensated COCO17 smoothing."""

import math
import time

import numpy as np
from scipy.optimize import linear_sum_assignment


def bbox_iou(first, second):
    intersection = max(0, min(first[0] + first[2], second[0] + second[2]) - max(first[0], second[0])) * max(
        0, min(first[1] + first[3], second[1] + second[3]) - max(first[1], second[1]))
    union = first[2] * first[3] + second[2] * second[3] - intersection
    return intersection / union if union > 1e-9 else 0.0


class TemporalDetectionStabilizer:
    def __init__(self, max_missed=20, track_ttl=1.0, pose_ttl=0.45, detection_hold_sec=0.2):
        self.max_missed = max_missed
        self.track_ttl = track_ttl
        self.pose_ttl = pose_ttl
        self.detection_hold_sec = detection_hold_sec
        self.next_id = 1
        self.tracks = {}

    def _score(self, detection, track, now):
        if detection.get("class") != track["class"]:
            return 0.0
        bbox = np.asarray(detection["bbox"], dtype=float)
        predicted = track["bbox"].copy()
        predicted[:2] += track["velocity"] * min(0.25, max(0, now - track["last_seen"]))
        ratios = bbox[2:] / np.maximum(predicted[2:], 1e-6)
        if np.any(ratios < 0.45) or np.any(ratios > 2.2):
            return 0.0
        overlap = bbox_iou(bbox, predicted)
        distance = np.linalg.norm((bbox[:2] + bbox[2:] / 2 - predicted[:2] - predicted[2:] / 2) / np.maximum(predicted[2:], 0.03))
        if distance > 1.05 or (overlap < 0.08 and distance > 0.55):
            return 0.0
        feature = detection.get("feature")
        previous = track.get("feature")
        if feature is not None and previous is not None:
            current_vector, old_vector = np.asarray(feature), np.asarray(previous)
            if (current_vector.shape == old_vector.shape and current_vector.ndim == 1
                    and np.isfinite(current_vector).all() and np.isfinite(old_vector).all()
                    and np.linalg.norm(current_vector) > 1e-6 and np.linalg.norm(old_vector) > 1e-6):
                similarity = np.dot(current_vector, old_vector) / max(1e-9, np.linalg.norm(current_vector) * np.linalg.norm(old_vector))
                if similarity < 0.5:
                    return 0.0
        return 0.75 * overlap + 0.25 * max(0, 1 - distance)

    def _pose(self, track, detection, now):
        raw = detection.get("keypoints") or []
        measured = np.zeros((17, 3), dtype=float)
        try:
            candidate = np.asarray(raw, dtype=float)
            if candidate.shape == (17, 3):
                measured = candidate.copy()
        except (ValueError, TypeError):
            pass
        valid = np.isfinite(measured).all(axis=1) & (measured[:, 2] >= 0.35) & (measured[:, :2] >= 0).all(axis=1) & (measured[:, :2] <= 1).all(axis=1)
        measured[~valid] = 0
        bbox = np.asarray(detection["bbox"], dtype=float)
        relative = (measured[:, :2] - bbox[:2]) / np.maximum(bbox[2:], 1e-6)
        old_relative = track["pose_relative"]
        ages = now - track["pose_seen"]
        previous_valid = ages <= self.pose_ttl
        displacement = np.linalg.norm(relative - old_relative, axis=1)
        outlier = valid & previous_valid & (displacement > 0.9) & (track["pose_confidence"] >= 0.5)
        valid[outlier] = False
        measured[outlier] = 0
        weights = np.clip(0.72 + displacement * 2, 0.72, 1.0)
        blend = valid & previous_valid
        relative[blend] = old_relative[blend] + weights[blend, None] * (relative[blend] - old_relative[blend])
        track["pose_relative"][valid] = relative[valid]
        track["pose_confidence"][valid] = measured[valid, 2]
        track["pose_seen"][valid] = now
        held = ~valid & previous_valid
        output = np.zeros((17, 3), dtype=float)
        visible = valid | held
        output[visible, :2] = np.clip(bbox[:2] + track["pose_relative"][visible] * bbox[2:], 0, 1)
        output[valid, 2] = measured[valid, 2]
        output[held, 2] = track["pose_confidence"][held] * (1 - 0.35 * ages[held] / self.pose_ttl)
        detection["keypoints_raw"] = measured.tolist() if valid.any() else []
        detection["keypoints"] = output.round(5).tolist() if visible.any() else []
        detection["keypoints_stale"] = bool(held.any())
        detection["keypoints_predicted_indices"] = np.flatnonzero(held).tolist()
        detection["keypoints_age"] = round(float(ages[held].max()), 3) if held.any() else 0.0
        detection["pose_tracking_state"] = "predicted" if held.any() and not valid.any() else "partial" if held.any() else "measured" if valid.any() else "unavailable"

    def update(self, detections, now=None):
        now = time.time() if now is None else now
        for track_id, track in list(self.tracks.items()):
            track["missed"] += 1
            if track["missed"] > self.max_missed or now - track["last_seen"] > self.track_ttl:
                del self.tracks[track_id]
        matches, used = {}, set()
        for detection_index, detection in enumerate(detections):
            source_id = detection.get("local_id")
            if source_id in (None, -1, 0xFFFFFFFF, 0xFFFFFFFFFFFFFFFF):
                continue
            for track_id, track in self.tracks.items():
                if track_id not in used and source_id == track["source_id"] and self._score(detection, track, now) > 0:
                    matches[detection_index] = track_id
                    used.add(track_id)
                    break
        pending = [index for index in range(len(detections)) if index not in matches]
        candidates = [track_id for track_id in self.tracks if track_id not in used]
        if pending and candidates:
            scores = np.asarray([[self._score(detections[index], self.tracks[track_id], now) for track_id in candidates] for index in pending])
            rows, columns = linear_sum_assignment(-scores)
            for row, column in zip(rows, columns):
                score = scores[row, column]
                alternatives = np.delete(scores[row], column)
                competing = np.delete(scores[:, column], row)
                if score < 0.3 or (alternatives.size and score - alternatives.max() < 0.08) or (competing.size and score - competing.max() < 0.08):
                    continue
                matches[pending[row]] = candidates[column]
        output = []
        for index, original in enumerate(detections):
            detection = dict(original)
            track_id = matches.get(index)
            bbox = np.asarray(detection["bbox"], dtype=float)
            if track_id is None:
                track_id = self.next_id
                self.next_id += 1
                self.tracks[track_id] = {
                    "bbox": bbox, "class": detection.get("class"), "velocity": np.zeros(2),
                    "last_seen": now, "pose_relative": np.zeros((17, 2)),
                    "pose_confidence": np.zeros(17), "pose_seen": np.full(17, -math.inf),
                    "missed": 0, "source_id": detection.get("local_id"),
                }
            track = self.tracks[track_id]
            self._pose(track, detection, now)
            elapsed = now - track["last_seen"]
            if elapsed > 0:
                velocity = (bbox[:2] - track["bbox"][:2]) / elapsed
                track["velocity"] = np.clip(0.5 * track["velocity"] + 0.5 * velocity, -1, 1)
            track.update(bbox=bbox, last_seen=now, missed=0, source_id=original.get("local_id"), feature=detection.get("feature"))
            detection.update(local_id=track_id, source_local_id=original.get("local_id"))
            track["detection"] = dict(detection)
            output.append(detection)
        for track_id, track in self.tracks.items():
            if not track["missed"] or track["class"] != "person" or now - track["last_seen"] > self.detection_hold_sec:
                continue
            predicted = track["bbox"].copy()
            predicted[:2] = np.clip(predicted[:2] + track["velocity"] * (now - track["last_seen"]), 0, 1 - predicted[2:])
            if any(bbox_iou(predicted, detection["bbox"]) > 0.08 for detection in output):
                continue
            detection = dict(track["detection"], bbox=predicted.tolist(), keypoints=[], tracking_state="predicted")
            self._pose(track, detection, now)
            detection["detection_age"] = round(now - track["last_seen"], 3)
            output.append(detection)
        return output
