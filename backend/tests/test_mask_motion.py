import threading
import unittest
from unittest import mock

import cv2
import numpy as np

from core.mask_motion import MaskMotionPropagator
from core.registered_target_mask import RegisteredTargetMaskSegmenter


class MaskMotionTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((180, 320, 3), dtype=np.uint8)
        self.frame[54:126, 96:160] = np.random.default_rng(42).integers(0, 256, (72, 64, 3), dtype=np.uint8)
        self.obj = {"id": 2001, "label": "Robot_2001", "category": "robot", "tracking_state": "tracked",
                    "x": .3, "y": .3, "w": .2, "h": .4,
                    "mask": {"polygons": [[[.3, .3], [.5, .3], [.5, .7], [.3, .7]]], "observed_at": 1000}}
        self.motion = MaskMotionPropagator(max_age=.65)

    def test_translation_moves_contour_and_preserves_measured_timestamp(self):
        self.motion.seed(self.frame, [self.obj], 1.0)
        shifted = cv2.warpAffine(self.frame, np.asarray([[1., 0., 4.], [0., 1., 2.]]), (320, 180))
        objects = self.motion.update(shifted, 1.04)
        self.assertEqual(1, len(objects))
        self.assertEqual(2001, objects[0]["id"])
        self.assertAlmostEqual(.3 + 4 / 320, objects[0]["x"], delta=.003)
        self.assertAlmostEqual(.3 + 2 / 180, objects[0]["y"], delta=.003)
        self.assertTrue(objects[0]["mask_stale"])
        self.assertEqual("predicted", objects[0]["tracking_state"])
        self.assertEqual(1000, objects[0]["mask"]["observed_at"])
        self.assertEqual(1040, objects[0]["observed_at"])
        self.assertNotIn("observation_bbox", objects[0])
        self.assertEqual(.3, self.obj["mask"]["polygons"][0][0][0])

    def test_expired_or_occluded_masks_stop_propagating(self):
        self.motion.seed(self.frame, [self.obj], 1.0)
        self.assertEqual([], self.motion.update(self.frame, 1.7))
        self.motion.seed(self.frame, [self.obj], 2.0)
        self.assertEqual([], self.motion.update(np.zeros_like(self.frame), 2.04))

    def test_no_person_or_predicted_mask_can_seed_motion(self):
        for obj in [dict(self.obj, category="person"), dict(self.obj, mask_stale=True), dict(self.obj, tracking_state="predicted")]:
            self.motion.seed(self.frame, [obj], 1.0)
            self.assertEqual({}, self.motion.tracks)


class MaskSchedulerTests(unittest.TestCase):
    def test_busy_worker_never_blocks_and_queued_camera_gets_priority(self):
        segmenter = RegisteredTargetMaskSegmenter()
        reserved, release = threading.Event(), threading.Event()

        def occupy():
            with segmenter.frame_slot():
                reserved.set()
                release.wait(2)

        thread = threading.Thread(target=occupy)
        thread.start()
        try:
            self.assertTrue(reserved.wait(2))
            with segmenter.try_frame_slot("first") as acquired:
                self.assertFalse(acquired)
            with segmenter.try_frame_slot("second") as acquired:
                self.assertFalse(acquired)
            release.set()
            thread.join(2)
            with segmenter.try_frame_slot("second") as acquired:
                self.assertFalse(acquired)
            with segmenter.try_frame_slot("first") as acquired:
                self.assertTrue(acquired)
            with segmenter.try_frame_slot("second") as acquired:
                self.assertTrue(acquired)
        finally:
            release.set()
            thread.join(2)
            segmenter.close()

    def test_rate_limit_and_camera_removal_clear_scheduling_state(self):
        segmenter = RegisteredTargetMaskSegmenter()
        try:
            with mock.patch("core.registered_target_mask.time.monotonic", return_value=10):
                with segmenter.try_frame_slot("cam", .1) as acquired:
                    self.assertTrue(acquired)
                with segmenter.try_frame_slot("cam", .1) as acquired:
                    self.assertFalse(acquired)
                segmenter.remove_camera("cam")
                with segmenter.try_frame_slot("cam", .1) as acquired:
                    self.assertTrue(acquired)
        finally:
            segmenter.close()
