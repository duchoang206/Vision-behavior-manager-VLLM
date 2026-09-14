"""Bounded robot/FMS correspondences; fitting runs outside video inference."""

import copy
import os
import threading
import time
from collections import Counter

import cv2
import numpy as np


class OnlineRobotCalibration:
    def __init__(self, calibrator, fms, persist=None, min_samples=16, fit_interval=5.0):
        self.calibrator = calibrator
        self.fms = fms
        self.persist = persist
        self.min_samples = min_samples
        self.fit_interval = fit_interval
        self.min_track_confidence = float(os.getenv("ONLINE_CALIBRATION_MIN_TRACK_CONFIDENCE", "0.5"))
        self.sessions = {}
        self.lock = threading.RLock()

    def start(self, cam_id, auto_apply=False, time_offset_ms=0, reset=False):
        with self.lock:
            if reset or cam_id not in self.sessions:
                self.sessions[cam_id] = {"samples": [], "rejected": Counter(), "revision": 0,
                    "last_fit_revision": -1, "last_fit_at": 0.0, "candidate": None,
                    "applied_at": None, "reason": "waiting_for_robot", "last_sample_at": None}
            session = self.sessions[cam_id]
            session.update(running=True, auto_apply=auto_apply, time_offset_ms=time_offset_ms)
        return self.status(cam_id)

    def stop(self, cam_id):
        with self.lock:
            if cam_id in self.sessions:
                self.sessions[cam_id]["running"] = False
            config = self.calibrator.get_config(cam_id)
            if config and config.get("method") == "auto_robot_fms":
                config["online"] = dict(config.get("online", {}), enabled=False)
                if self.persist:
                    self.persist(cam_id, config)
                self.calibrator.apply_config(cam_id, config)
        return self.status(cam_id)

    def remove_camera(self, cam_id):
        with self.lock:
            self.sessions.pop(cam_id, None)

    def save_manual(self, cam_id, config):
        with self.lock:
            if self.persist:
                self.persist(cam_id, config)
            self.calibrator.apply_config(cam_id, config)
            self.sessions.pop(cam_id, None)

    def observe(self, cam_id, objects, registered_robots, now=None):
        now = time.time() if now is None else now
        session = self.sessions.get(cam_id)
        if not session or not session.get("running") or not self.lock.acquire(blocking=False):
            return
        try:
            if not session.get("running") or self.sessions.get(cam_id) is not session:
                return
            for obj in objects:
                label = str(obj.get("label") or "")
                robot_id = registered_robots.get(label.casefold())
                if robot_id is None or obj.get("category", obj.get("class")) != "robot":
                    continue
                try:
                    track_confidence = float(obj.get("tracking_confidence", obj.get("confidence", 0.0)) or 0.0)
                except (TypeError, ValueError):
                    track_confidence = 0.0
                if not np.isfinite(track_confidence) or obj.get("tracking_state") != "tracked" or track_confidence < self.min_track_confidence or obj.get("mask_stale"):
                    session["rejected"]["untrusted_track"] += 1
                    continue
                try:
                    observed_at = float(obj["observed_at"]) / 1000
                    bbox = np.asarray(obj.get("observation_bbox") or [obj["x"], obj["y"], obj["w"], obj["h"]], dtype=float)
                    if bbox.shape != (4,) or not np.isfinite(bbox).all() or (bbox[2:] <= 0).any():
                        raise ValueError
                    image = bbox[:2] + bbox[2:] * [0.5, 1.0]
                    if not np.isfinite(image).all() or (image < 0).any() or (image > 1).any():
                        raise ValueError
                    if (bbox[:2] <= 0.002).any() or (bbox[:2] + bbox[2:] >= 0.998).any():
                        session["rejected"]["clipped_robot"] += 1
                        continue
                    if not -0.05 <= now - observed_at <= 0.6:
                        session["rejected"]["stale_frame"] += 1
                        continue
                    aligned_at = observed_at - session["time_offset_ms"] / 1000
                    states = getattr(self.fms, "robot_states", {})
                    requested_robot_id = str(robot_id)
                    state_candidates = [requested_robot_id]
                    if requested_robot_id.casefold().startswith("robot_"):
                        state_candidates.append(requested_robot_id[6:])
                    else:
                        state_candidates.append(f"Robot_{requested_robot_id}")
                    robot_state = next(
                        (states[key] for key in state_candidates if key in states),
                        next((state for key, state in states.items()
                              if str(key).casefold() in {candidate.casefold() for candidate in state_candidates}), {}),
                    )
                    if robot_state and str(robot_state.get("status", "OFFLINE")).upper() == "OFFLINE":
                        session["rejected"]["fms_robot_offline"] += 1
                        if not session["samples"]:
                            session["reason"] = "fms_robot_offline"
                        continue
                    pose = self.fms.pose_at(robot_id, aligned_at, now=now)
                    if not pose:
                        session["rejected"]["fms_not_synchronized"] += 1
                        continue
                    destination = np.asarray(pose["position"], dtype=float)[[0, 2]]
                    if not np.isfinite(destination).all():
                        raise ValueError
                    samples = session["samples"]
                    samples[:] = [sample for sample in samples if now - sample["at"] <= 600]
                    if any(sample["robot"] == str(robot_id) and observed_at - sample["at"] < 0.25 for sample in samples[-8:]):
                        continue
                    if any(np.linalg.norm(image - sample["image"]) < 0.008 or np.linalg.norm(destination - sample["floor"]) < 0.10 for sample in samples):
                        session["rejected"]["duplicate_or_stationary"] += 1
                        continue
                    samples[:] = samples[-239:]
                    samples.append({"image": image.tolist(), "floor": destination.tolist(), "at": observed_at,
                                    "robot": str(robot_id), "label": label, "skew_ms": pose["skew_ms"],
                                    "fms_source": pose.get("source", "fms")})
                    session["last_sample_at"] = observed_at
                    session["revision"] += 1
                    session["reason"] = "collecting"
                except (ValueError, TypeError, IndexError, KeyError):
                    session["rejected"]["invalid_observation"] += 1
        finally:
            self.lock.release()

    @staticmethod
    def _coverage(points, minimum_area):
        singular = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
        hull = cv2.convexHull(points.astype(np.float32))
        return singular[-1] / max(singular[0], 1e-9) >= 0.08 and cv2.contourArea(hull) >= minimum_area

    def _fit(self, samples, now):
        if len(samples) < self.min_samples:
            raise ValueError("need_more_samples")
        source = np.asarray([sample["image"] for sample in samples], dtype=float)
        destination = np.asarray([sample["floor"] for sample in samples], dtype=float)
        if not self._coverage(source, 0.002) or not self._coverage(destination, 0.5):
            raise ValueError("need_wider_2d_path")
        validation = np.arange(len(samples)) % 4 == 0
        matrix, _ = cv2.findHomography(source[~validation], destination[~validation], cv2.RANSAC, 0.30)
        if matrix is None:
            raise ValueError("fit_failed")
        errors = np.linalg.norm(cv2.perspectiveTransform(source[None], matrix)[0] - destination, axis=1)
        validation_error = float(np.quantile(errors[validation], 0.75))
        if not np.isfinite(errors).all() or validation_error > 0.30:
            raise ValueError("validation_error_too_high")
        inliers = errors <= 0.30
        if inliers.sum() < 12 or inliers.mean() < 0.75:
            raise ValueError("too_many_outliers")
        if not self._coverage(source[inliers], 0.002) or not self._coverage(destination[inliers], 0.5):
            raise ValueError("need_wider_2d_path")
        matrix, _ = cv2.findHomography(source[inliers], destination[inliers], 0)
        errors = np.linalg.norm(cv2.perspectiveTransform(source[inliers][None], matrix)[0] - destination[inliers], axis=1)
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        if not np.isfinite(errors).all() or rmse > 0.20 or np.quantile(errors, 0.95) > 0.30:
            raise ValueError("residual_too_high")
        return self.calibrator.prepare_config(source[inliers], destination[inliers], matrix, {
            "method": "auto_robot_fms", "coordinate_space": "fms_floor_metric", "map_id": "TT",
            "coverage_polygon": cv2.convexHull(source[inliers].astype(np.float32)).reshape(-1, 2).tolist(),
            "sample_count": len(samples), "inlier_count": int(inliers.sum()), "residual_m": round(rmse, 4),
            "validation_error_m": round(validation_error, 4), "calibrated_at": now,
            "robots": sorted({sample["robot"] for sample in samples}),
            "fms_frame": {"origin_x": self.fms.origin_x, "origin_y": self.fms.origin_y, "layout_depth": self.fms.layout_depth},
            "timebase": "camera_decode_and_fms_receive", "ground_reference": "robot_bbox_bottom_center",
        })

    def fit_pending(self, now=None):
        now = time.time() if now is None else now
        with self.lock:
            jobs = [(cam_id, session, session["revision"], copy.deepcopy(session["samples"]))
                    for cam_id, session in self.sessions.items() if session.get("running")
                    and session["revision"] != session["last_fit_revision"] and now - session["last_fit_at"] >= self.fit_interval]
        for cam_id, session, revision, samples in jobs:
            candidate, reason = None, "ready"
            try:
                candidate = self._fit([sample for sample in samples if now - sample["at"] <= 600], now)
            except ValueError as error:
                reason = str(error)
            except (cv2.error, np.linalg.LinAlgError):
                reason = "fit_failed"
            with self.lock:
                if self.sessions.get(cam_id) is not session or not session.get("running"):
                    continue
                session.update(candidate=candidate, reason=reason, last_fit_at=now, last_fit_revision=revision)
            if candidate and session["auto_apply"]:
                try:
                    self.apply(cam_id)
                except (ValueError, RuntimeError):
                    with self.lock:
                        session["reason"] = "persistence_failed"

    def apply(self, cam_id):
        with self.lock:
            session = self.sessions.get(cam_id)
            if not session or not session["candidate"]:
                raise ValueError("Chưa đủ mẫu đạt kiểm tra sai số.")
            if time.time() - session["candidate"]["calibrated_at"] > 30:
                raise ValueError("Kết quả đã cũ; hãy thu mẫu robot mới.")
            config = dict(self.calibrator.get_config(cam_id) or {}, **copy.deepcopy(session["candidate"]))
            config["online"] = {"enabled": session["running"], "auto_apply": session["auto_apply"], "time_offset_ms": session["time_offset_ms"]}
            if self.persist:
                self.persist(cam_id, config)
            self.calibrator.apply_config(cam_id, config)
            session["applied_at"] = time.time()
            return config

    def status(self, cam_id):
        with self.lock:
            session = self.sessions.get(cam_id, {})
            candidate = session.get("candidate") or {}
            return {"cam_id": cam_id, "running": bool(session.get("running")),
                    "auto_apply": bool(session.get("auto_apply")), "ready": bool(candidate) and time.time() - candidate.get("calibrated_at", 0) <= 30,
                    "sample_count": len(session.get("samples", [])), "minimum_samples": self.min_samples,
                    "inlier_count": candidate.get("inlier_count", 0), "residual_m": candidate.get("residual_m"),
                    "validation_error_m": candidate.get("validation_error_m"), "reason": session.get("reason", "stopped"),
                    "rejected": dict(session.get("rejected", {})), "applied_at": session.get("applied_at"),
                    "last_sample_at": session.get("last_sample_at"), "time_offset_ms": session.get("time_offset_ms", 0),
                    "robots": sorted({sample["label"] for sample in session.get("samples", [])})}
