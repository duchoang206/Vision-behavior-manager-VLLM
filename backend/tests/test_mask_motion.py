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

    def test_missing_sam_result_keeps_only_valid_live_flow(self):
        self.motion.seed(self.frame, [self.obj], 1.0)
        shifted = cv2.warpAffine(self.frame, np.asarray([[1., 0., 4.], [0., 1., 2.]]), (320, 180))
        objects = self.motion.seed(shifted, [], 1.04)
        self.assertEqual(1, len(objects))
        self.assertAlmostEqual(.3 + 4 / 320, objects[0]['x'], delta=.003)
        self.assertAlmostEqual(.4 + 4 / 320, objects[0]['floor_x'], delta=.003)
        self.assertEqual('sam2_optical_flow', objects[0]['mask']['source'])
        self.assertEqual(1.0, self.motion.tracks['Robot_2001']['measured_at'])
        self.assertEqual([], self.motion.seed(np.zeros_like(shifted), [], 1.08))
        self.assertEqual(1, self.motion.status()['drop_counts']['flow_consistency'])

    def test_repeated_sam_misses_never_extend_measurement_lifetime(self):
        self.motion.seed(self.frame, [self.obj], 1.0)
        for timestamp in [1.1, 1.2, 1.4, 1.6]:
            self.assertEqual(1, len(self.motion.seed(self.frame, [], timestamp)))
        self.assertEqual([], self.motion.seed(self.frame, [], 1.66))
        self.assertEqual('sam_refresh_timeout', self.motion.status()['last_drop']['reason'])

    def test_longer_sam_bridge_still_rejects_camera_frame_stall(self):
        motion = MaskMotionPropagator(max_age=1.8)
        motion.seed(self.frame, [self.obj], 1.0)
        self.assertEqual([], motion.update(self.frame, 1.7))
        self.assertEqual('frame_gap', motion.status()['last_drop']['reason'])

    def test_live_stationary_flow_bridges_slow_sam_without_extending_age(self):
        motion = MaskMotionPropagator(max_age=1.8)
        motion.seed(self.frame, [self.obj], 1.0)
        for timestamp in [1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.79]:
            objects = motion.update(self.frame, timestamp)
            self.assertEqual(1, len(objects))
            self.assertAlmostEqual(.3, objects[0]['x'], places=4)
            self.assertEqual(1000, objects[0]['mask']['observed_at'])
        self.assertEqual([], motion.update(self.frame, 2.81))

    def test_partial_sam_result_does_not_clear_other_labels(self):
        other = dict(self.obj, label='Robot_28100', id=28100)
        self.motion.seed(self.frame, [self.obj, other], 1.0)
        objects = self.motion.seed(self.frame, [self.obj], 1.1)
        self.assertEqual({'Robot_2001', 'Robot_28100'}, {obj['label'] for obj in objects})
        self.assertEqual(1.1, self.motion.tracks['Robot_2001']['measured_at'])
        self.assertEqual(1.0, self.motion.tracks['Robot_28100']['measured_at'])

    def test_predicted_result_does_not_overwrite_live_flow(self):
        self.motion.seed(self.frame, [self.obj], 1.0)
        held = dict(self.obj, tracking_state='predicted', mask_stale=True)
        objects = self.motion.seed(self.frame, [held], 1.1)
        self.assertEqual('sam2_optical_flow', objects[0]['mask']['source'])
        self.assertEqual(1.0, self.motion.tracks['Robot_2001']['measured_at'])

    def test_small_mask_uses_more_interior_points_without_expanding_region(self):
        sparse = np.asarray([[[100., 60.]], [[104., 60.]], [[100., 64.]]], dtype=np.float32)
        dense = np.concatenate([sparse, sparse + [2., 2.]]).astype(np.float32)
        with mock.patch('core.mask_motion.cv2.goodFeaturesToTrack', side_effect=[sparse, dense]) as corners:
            self.motion.seed(self.frame, [self.obj], 1.0)
        self.assertEqual([4, 2], [call.kwargs['minDistance'] for call in corners.call_args_list])
        self.assertIs(corners.call_args_list[0].kwargs['mask'], corners.call_args_list[1].kwargs['mask'])
        self.assertEqual(6, len(self.motion.tracks['Robot_2001']['points']))

    def test_featureless_mask_is_not_kept_by_denser_sampling(self):
        self.motion.seed(np.zeros_like(self.frame), [self.obj], 1.0)
        self.assertEqual({}, self.motion.tracks)
        self.assertEqual('insufficient_seed_points', self.motion.status()['last_drop']['reason'])

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

    def test_oldest_inference_gets_priority_over_earlier_poll(self):
        segmenter = RegisteredTargetMaskSegmenter()
        try:
            with mock.patch('core.registered_target_mask.time.monotonic', return_value=10):
                segmenter._last_slot_at = {'slow': 9.0, 'fast': 9.8}
                segmenter._waiting_cameras = {'fast': (9.81, 9.9), 'slow': (9.9, 9.9)}
                with segmenter.try_frame_slot('fast') as acquired:
                    self.assertFalse(acquired)
                with segmenter.try_frame_slot('slow') as acquired:
                    self.assertTrue(acquired)
                with segmenter.try_frame_slot('fast') as acquired:
                    self.assertTrue(acquired)
        finally:
            segmenter.close()

    def test_camera_without_frame_cannot_block_ready_camera(self):
        segmenter = RegisteredTargetMaskSegmenter()
        try:
            with mock.patch('core.registered_target_mask.time.monotonic', return_value=10):
                segmenter._waiting_cameras = {'empty': (9.5, 9.9)}
                segmenter.cancel_frame_wait('empty')
                with segmenter.try_frame_slot('ready') as acquired:
                    self.assertTrue(acquired)
        finally:
            segmenter.close()
