"""Bounded optical-flow propagation between measured SAM2 masks, without inference."""

import copy

import cv2
import numpy as np

from core.registered_target_mask import eligible_target, mask_bitmap


class MaskMotionPropagator:
    def __init__(self, max_age=0.65, width=640):
        self.max_age = max_age
        self.width = width
        self.gray = None
        self.timestamp = 0.0
        self.tracks = {}

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
        self.reset()
        self.gray = self._gray_frame(frame)
        self.timestamp = timestamp
        for obj in objects:
            if not eligible_target(obj) or obj.get("tracking_state") != "tracked" or obj.get("mask_stale") or not obj.get("mask"):
                continue
            region = mask_bitmap(obj["mask"], self.gray.shape) * 255
            region = cv2.erode(region, np.ones((3, 3), dtype=np.uint8))
            points = cv2.goodFeaturesToTrack(self.gray, maxCorners=48, qualityLevel=0.02,
                                           minDistance=4, mask=region, blockSize=3)
            if points is not None and len(points) >= 6:
                self.tracks[obj["label"]] = {"object": copy.deepcopy(obj), "points": points, "measured_at": timestamp}

    def update(self, frame, timestamp):
        if self.gray is None or timestamp <= self.timestamp:
            return []
        current = self._gray_frame(frame)
        if current.shape != self.gray.shape or timestamp - self.timestamp > self.max_age:
            self.reset()
            return []
        height, width = current.shape
        objects = []
        for label, track in list(self.tracks.items()):
            if timestamp - track["measured_at"] > self.max_age:
                self.tracks.pop(label, None)
                continue
            points = track["points"]
            moved, forward_status, _ = cv2.calcOpticalFlowPyrLK(self.gray, current, points, None, winSize=(15, 15), maxLevel=2)
            if moved is None:
                self.tracks.pop(label, None)
                continue
            returned, backward_status, _ = cv2.calcOpticalFlowPyrLK(current, self.gray, moved, None, winSize=(15, 15), maxLevel=2)
            if returned is None:
                self.tracks.pop(label, None)
                continue
            good = (forward_status.ravel() > 0) & (backward_status.ravel() > 0)
            good &= np.isfinite(moved).all(axis=(1, 2)) & (np.linalg.norm(points - returned, axis=2).ravel() < 1.5)
            if good.sum() < 6 or good.mean() < 0.5:
                self.tracks.pop(label, None)
                continue
            transform, inliers = cv2.estimateAffinePartial2D(points[good], moved[good], method=cv2.RANSAC,
                                                          ransacReprojThreshold=2, maxIters=100)
            if transform is None or not np.isfinite(transform).all() or inliers is None or inliers.mean() < 0.65:
                self.tracks.pop(label, None)
                continue
            scale = np.linalg.norm(transform[:, 0])
            if not 0.85 <= scale <= 1.18:
                self.tracks.pop(label, None)
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
                self.tracks.pop(label, None)
                continue
            obj.update(x=float(left), y=float(top), w=float(right - left), h=float(bottom - top),
                       observed_at=int(timestamp * 1000), tracking_state="predicted", mask_stale=True,
                       mask_age=round(timestamp - track["measured_at"], 3), image_velocity=[0.0] * 4)
            obj["mask"] = dict(obj["mask"], polygons=rings, source="sam2_optical_flow")
            obj.pop("observation_bbox", None)
            track.update(object=obj, points=moved[good][inliers.ravel() > 0])
            objects.append(obj)
        self.gray = current
        self.timestamp = timestamp
        return objects
