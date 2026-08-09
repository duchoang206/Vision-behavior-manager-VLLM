import base64
import os
import sys
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np


BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from core import template_identity_tracker as tracker_module
from core.template_identity_tracker import TemplateIdentityCameraTracker, TemplateIdentityTrackerManager
from src.controller.registry import TargetRegistry


def encode_crop(image: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Could not encode test image")
    return "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def make_view(primary: tuple, secondary: tuple, diagonal: bool = False) -> np.ndarray:
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    image[:] = (18, 18, 18)
    cv2.rectangle(image, (3, 3), (60, 44), primary, -1)
    if diagonal:
        cv2.line(image, (6, 40), (57, 7), secondary, 8)
        cv2.circle(image, (17, 14), 6, (255, 255, 255), -1)
    else:
        cv2.rectangle(image, (8, 9), (28, 38), secondary, -1)
        cv2.circle(image, (47, 16), 7, (255, 255, 255), -1)
    return image


def place(frame: np.ndarray, image: np.ndarray, x: int, y: int) -> list:
    height, width = image.shape[:2]
    frame[y:y + height, x:x + width] = image
    frame_h, frame_w = frame.shape[:2]
    return [x / frame_w, y / frame_h, width / frame_w, height / frame_h]


class TemplateIdentityTrackerTests(unittest.TestCase):
    def setUp(self):
        self.tracker = TemplateIdentityCameraTracker("cam-test", "unused", None, target_fps=24)
        self.tracker.min_match_score = 0.42
        self.tracker.init_min_score = 0.40
        self.tracker.verify_min_score = 0.32
        self.tracker.full_search_interval = 1
        self.view_front = make_view((30, 170, 235), (220, 65, 30), diagonal=False)
        self.view_side = make_view((35, 165, 225), (25, 225, 85), diagonal=True)

    def test_same_label_builds_multi_angle_gallery(self):
        bbox = [0.1, 0.2, 0.2, 0.2]
        self.assertTrue(self.tracker.add_target("Robot_42", "robot", bbox, encode_crop(self.view_front)))
        self.assertTrue(self.tracker.add_target("Robot_42", "robot", bbox, encode_crop(self.view_side)))

        target = self.tracker.targets["Robot_42"]
        self.assertEqual(2, len(target["templates"]))
        self.assertEqual("Robot_42", target["label"])

    def test_reacquires_fast_jump_with_another_labeled_view(self):
        frame_a = np.zeros((360, 640, 3), dtype=np.uint8)
        bbox_a = place(frame_a, self.view_front, 35, 130)
        self.tracker.add_target("Robot_42", "robot", bbox_a, encode_crop(self.view_front))
        self.tracker.add_target("Robot_42", "robot", bbox_a, encode_crop(self.view_side))
        target = self.tracker.targets["Robot_42"]

        with mock.patch.object(tracker_module, "_create_cv_tracker", return_value=None):
            acquired = self.tracker._match_target(frame_a, target)
            self.assertIsNotNone(acquired)

            frame_b = np.zeros_like(frame_a)
            expected_bbox = place(frame_b, self.view_side, 500, 75)
            missed = self.tracker._match_target(frame_b, target)
            self.assertEqual("predicted", missed["tracking_state"])

            recovered = self.tracker._match_target(frame_b, target)

        self.assertIsNotNone(recovered)
        self.assertEqual("tracked", recovered["tracking_state"])
        recovered_cx = recovered["x"] + recovered["w"] / 2.0
        expected_cx = expected_bbox[0] + expected_bbox[2] / 2.0
        self.assertLess(abs(recovered_cx - expected_cx), 0.035)
        self.assertEqual(0, target["lost"])

    def test_prediction_keeps_moving_during_short_loss(self):
        bbox = [0.20, 0.30, 0.10, 0.14]
        self.tracker.add_target("Rack_A", "rack", bbox, encode_crop(self.view_front))
        target = self.tracker.targets["Rack_A"]
        target["velocity"] = [0.24, -0.04, 0.0, 0.0]
        target["last_observation_at"] = 100.0

        predicted = self.tracker._predict_bbox(target, now=100.5)

        self.assertGreater(predicted[0], bbox[0] + 0.10)
        self.assertLess(predicted[1], bbox[1])
        self.assertAlmostEqual(bbox[2], predicted[2], places=4)

    def test_default_tracking_does_not_follow_drifting_flow(self):
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        bbox = place(frame, self.view_front, 210, 130)
        self.tracker.add_target("Robot_42", "robot", bbox, encode_crop(self.view_front))
        target = self.tracker.targets["Robot_42"]
        target["has_matched"] = True
        wrong_bbox = [0.02, 0.52, 0.07, 0.13]

        with mock.patch.object(self.tracker, "_update_optical_flow", return_value=wrong_bbox), \
                mock.patch.object(self.tracker, "_template_score_at_bbox", return_value=0.41), \
                mock.patch.object(self.tracker, "_init_motion_tracker"):
            tracked = self.tracker._match_target(frame, target)

        self.assertIsNotNone(tracked)
        self.assertAlmostEqual(bbox[0], tracked["x"], delta=0.005)
        self.assertAlmostEqual(bbox[2], tracked["w"], delta=0.005)

    def test_flow_accumulates_against_raw_not_smoothed_bbox(self):
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        bbox = place(frame, self.view_front, 100, 130)
        self.tracker.add_target("Robot_42", "robot", bbox, encode_crop(self.view_front))
        target = self.tracker.targets["Robot_42"]
        self.assertTrue(self.tracker._init_optical_flow(frame, target, bbox))

        for step in range(1, 21):
            points = target["flow_points"]
            moved = points + np.array([2.0, 0.0], dtype=np.float32)
            status = np.ones((len(points), 1), dtype=np.uint8)
            errors = np.zeros((len(points), 1), dtype=np.float32)
            with mock.patch.object(cv2, "calcOpticalFlowPyrLK", return_value=(moved, status, errors)):
                measured = self.tracker._update_optical_flow(frame, target)
            self.assertIsNotNone(measured)
            self.tracker._accept_observation(target, measured, 0.95)

        self.assertAlmostEqual(bbox[0] + 40 / 640, measured[0], places=4)

    def test_blank_template_cannot_match_every_region(self):
        template = np.full((48, 64, 3), 120, dtype=np.uint8)
        frame = np.full((360, 640, 3), 120, dtype=np.uint8)

        self.assertIsNone(self.tracker._search_template_in_region(frame, template))
        self.assertEqual(0.0, tracker_module._template_similarity(template, template))

    def test_reacquisition_clears_stale_velocity(self):
        bbox = [0.2, 0.3, 0.1, 0.14]
        self.tracker.add_target("Robot_42", "robot", bbox, encode_crop(self.view_front))
        target = self.tracker.targets["Robot_42"]
        target["has_matched"] = True
        target["lost"] = 3
        target["velocity"] = [-1.0, 0.6, 0.1, -0.2]

        self.tracker._accept_observation(target, [0.7, 0.2, 0.1, 0.14], 0.95)

        self.assertEqual([0.0, 0.0, 0.0, 0.0], target["velocity"])

    def test_gallery_identity_can_label_overlapping_gpu_detection(self):
        bbox = [0.20, 0.30, 0.16, 0.20]
        self.tracker.add_target("Robot_42", "robot", bbox, encode_crop(self.view_front))
        self.tracker.targets["Robot_42"]["has_matched"] = True
        manager = TemplateIdentityTrackerManager()
        manager.trackers["cam-test"] = self.tracker

        label = manager.match_detection("cam-test", [0.21, 0.31, 0.16, 0.20], "delivery-robot")

        self.assertEqual("Robot_42", label)


class TargetRegistryTemplateRestoreTests(unittest.TestCase):
    def test_persists_and_restores_every_labeled_view(self):
        front = make_view((30, 170, 235), (220, 65, 30), diagonal=False)
        side = make_view((35, 165, 225), (25, 225, 85), diagonal=True)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "targets.json")
            registry = TargetRegistry(path)
            vector_a = np.zeros(512, dtype=np.float32)
            vector_b = np.zeros(512, dtype=np.float32)
            vector_a[10] = 1.0
            vector_b[20] = 1.0
            bbox = [0.1, 0.2, 0.2, 0.2]

            registry.register_target("Robot_42", vector_a.tolist(), "cam-test", category="robot", bbox=bbox, crop_image=encode_crop(front))
            registry.register_target("Robot_42", vector_b.tolist(), "cam-test", category="robot", bbox=bbox, crop_image=encode_crop(side))

            restored = TargetRegistry(path)
            samples = restored.get_template_samples("Robot_42", "cam-test")
            self.assertEqual(2, len(samples))
            self.assertTrue(all(sample.get("crop_image") for sample in samples))

            restored.bind_label_to_track("Robot_42", "cam-test", 7, 11, live_bbox=bbox)
            label = restored.assign_label_for_detection("cam-test", 7, 11, bbox, det_class="delivery-robot")
            self.assertEqual("Robot_42", label)


if __name__ == "__main__":
    unittest.main()
