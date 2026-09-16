"""Bounded optical-flow propagation between measured SAM2 masks, without inference."""

import copy

import cv2
import numpy as np

from core.registered_target_mask import eligible_target, mask_bitmap


class MaskMotionPropagator:
    def __init__(self, max_age=1.8, width=640, max_frame_gap=0.65):
        self.max_age = max_age
        self.max_frame_gap = min(max_age, max_frame_gap)
        self.width = width
        self.gray = None
        self.timestamp = 0.0
        self.tracks = {}
        self.drop_counts = {}
        self.last_drop = None

    def _discard(self, label, reason, timestamp):
        self.tracks.pop(label, None)
        self.drop_counts[reason] = self.drop_counts.get(reason, 0) + 1
        self.last_drop = {"label": label, "reason": reason, "timestamp": timestamp}

    def status(self):
        return {"active_labels": list(self.tracks), "max_age_ms": round(self.max_age * 1000),
                "max_frame_gap_ms": round(self.max_frame_gap * 1000),
                "drop_counts": dict(self.drop_counts), "last_drop": self.last_drop}

    def reset(self):
        self.gray = None
        self.timestamp = 0.0
        self.tracks.clear()

    def _gray_frame(self, frame):
        height, width = frame.shape[:2]
        if width > self.width:
            frame = cv2.resize(frame, (self.width, max(1, round(height * self.width / width))), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def seed(self, frame, objects, timestamp):
        measured = {obj["label"]: obj for obj in objects
                    if eligible_target(obj) and obj.get("tracking_state") == "tracked"
                    and not obj.get("mask_stale") and obj.get("mask")}
        for label in measured:
            self.tracks.pop(label, None)
        propagated = {obj["label"]: obj for obj in self.update(frame, timestamp)} if self.tracks else {}
        if self.gray is None or self.timestamp != timestamp:
            self.gray = self._gray_frame(frame)
        self.timestamp = timestamp
        for label, obj in measured.items():
            region = mask_bitmap(obj["mask"], self.gray.shape) * 255
            region = cv2.erode(region, np.ones((3, 3), dtype=np.uint8))
            points = cv2.goodFeaturesToTrack(self.gray, maxCorners=48, qualityLevel=0.02,
                                           minDistance=4, mask=region, blockSize=3)
            if points is None or len(points) < 6:
                points = cv2.goodFeaturesToTrack(self.gray, maxCorners=48, qualityLevel=0.02,
                                               minDistance=2, mask=region, blockSize=3)
            if points is not None and len(points) >= 6:
                self.tracks[label] = {"object": copy.deepcopy(obj), "points": points, "measured_at": timestamp}
            else:
                self._discard(label, "insufficient_seed_points", timestamp)
        combined = {obj["label"]: obj for obj in objects}
        combined.update({label: obj for label, obj in propagated.items() if label not in measured})
        return list(combined.values())

    def update(self, frame, timestamp):
        if self.gray is None or timestamp <= self.timestamp:
            return []
        current = self._gray_frame(frame)
        if current.shape != self.gray.shape or timestamp - self.timestamp > self.max_frame_gap:
            reason = "frame_shape" if current.shape != self.gray.shape else "frame_gap"
            for label in list(self.tracks):
                self._discard(label, reason, timestamp)
            self.reset()
            return []
        height, width = current.shape
        objects = []
        for label, track in list(self.tracks.items()):
            if timestamp - track["measured_at"] > self.max_age:
                self._discard(label, "sam_refresh_timeout", timestamp)
                continue
            points = track["points"]
            moved, forward_status, _ = cv2.calcOpticalFlowPyrLK(self.gray, current, points, None, winSize=(15, 15), maxLevel=2)
            if moved is None:
                self._discard(label, "forward_flow", timestamp)
                continue
            returned, backward_status, _ = cv2.calcOpticalFlowPyrLK(current, self.gray, moved, None, winSize=(15, 15), maxLevel=2)
            if returned is None:
                self._discard(label, "backward_flow", timestamp)
                continue
            good = (forward_status.ravel() > 0) & (backward_status.ravel() > 0)
            good &= np.isfinite(moved).all(axis=(1, 2)) & (np.linalg.norm(points - returned, axis=2).ravel() < 1.5)
            if good.sum() < 6 or good.mean() < 0.5:
                self._discard(label, "flow_consistency", timestamp)
                continue
            transform, inliers = cv2.estimateAffinePartial2D(points[good], moved[good], method=cv2.RANSAC,
                                                          ransacReprojThreshold=2, maxIters=100)
            if transform is None or not np.isfinite(transform).all() or inliers is None or inliers.mean() < 0.65:
                self._discard(label, "affine_inliers", timestamp)
                continue
            scale = np.linalg.norm(transform[:, 0])
            if not 0.85 <= scale <= 1.18:
                self._discard(label, "scale_jump", timestamp)
                continue
            obj = copy.deepcopy(track["object"])
            rings = []
            for ring in obj["mask"]["polygons"]:
                pixels = np.asarray(ring, dtype=np.float32) * [width, height]
                warped = cv2.transform(pixels[None], transform)[0] / [width, height]
                rings.append(np.clip(warped, 0, 1).round(6).tolist())
            vertices = np.concatenate(rings)
            left, top = vertices.min(axis=0)
            right, bottom = vertices.max(axis=0)
            displacement = np.linalg.norm([(left + right) / 2 - obj["x"] - obj["w"] / 2,
                                           (top + bottom) / 2 - obj["y"] - obj["h"] / 2])
            if right <= left or bottom <= top or displacement > max(0.03, obj["w"], obj["h"]) * 0.6:
                self._discard(label, "geometry_jump", timestamp)
                continue
            obj.update(x=float(left), y=float(top), w=float(right - left), h=float(bottom - top),
                       floor_x=float((left + right) / 2), floor_y=float(bottom),
                       observed_at=int(timestamp * 1000), tracking_state="predicted", mask_stale=True,
                       mask_age=round(timestamp - track["measured_at"], 3), image_velocity=[0.0] * 4)
            obj["mask"] = dict(obj["mask"], polygons=rings, source="sam2_optical_flow")
            obj.pop("observation_bbox", None)
            track.update(object=obj, points=moved[good][inliers.ravel() > 0])
            objects.append(obj)
        self.gray = current
        self.timestamp = timestamp
        return objects
