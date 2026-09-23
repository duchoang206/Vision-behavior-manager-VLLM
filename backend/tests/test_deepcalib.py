import base64
import json
import time
import unittest
from unittest import mock

import cv2
import numpy as np

from core.camera_calibrator import CameraCalibrator
from core.calibration_measurement import measure_calibrated_polyline
from core.online_calibration import OnlineRobotCalibration
from deep_calib.adapter import prepare_config
from deep_calib.geometry import (distort_normalized_points, make_rectification_profile,
                                rectify_image_cuda, undistort_normalized_points,
                                validate_visible_rectified_points)
from deep_calib import pipeline


class DeepCalibTests(unittest.TestCase):
    def setUp(self):
        self.profile = make_rectification_profile(dict(image_width=640, image_height=480,
                                                      focal_length_px=560, distortion_xi=.45))
        self.source = np.array([[.25, .25], [.75, .25], [.75, .75], [.25, .75], [.5, .5]])
        self.destination = self.source * [10, 8] + [2, 3]
        self.frame = dict(origin_x=185., origin_y=194.5, layout_depth=18.)
        self.calibrator = CameraCalibrator()

    def config(self, **kwargs):
        return prepare_config(self.calibrator, self.source, self.destination, "TT", self.frame,
                              intrinsic=self.profile, points_space="rectified", **kwargs)

    def test_roundtrip_matches_spherical_model(self):
        for distortion in (0., .45, 1., 1.2):
            profile = make_rectification_profile(dict(self.profile, distortion_xi=distortion))
            raw = distort_normalized_points(self.source, profile)
            np.testing.assert_allclose(undistort_normalized_points(raw, profile), self.source, atol=1e-10)

    def test_new_profiles_do_not_collapse_outside_points(self):
        profile = dict(self.profile, rectified_focal_px=560, distortion_xi=.8)
        projected = undistort_normalized_points([[.02, .4], [.04, .4]], profile)
        self.assertTrue((projected[:, 0] < 0).all())
        self.assertNotEqual(projected[0, 0], projected[1, 0])

    def test_empty_borders_horizon_and_missing_profiles_rejected(self):
        profile = dict(self.profile, rectified_focal_px=10)
        with self.assertRaises(ValueError):
            validate_visible_rectified_points([[0., 0.]], profile)
        with self.assertRaises(ValueError):
            undistort_normalized_points([[0., 0.]], dict(self.profile, focal_length_px=50, distortion_xi=1.2))
        with self.assertRaises(ValueError):
            prepare_config(self.calibrator, self.source, self.destination, "TT", self.frame)

    def test_legacy_profile_keeps_old_mapping_until_recalibrated(self):
        profile = dict(image_width=640, image_height=480, focal_length_px=560, distortion_xi=.8)
        raw = [[.02, .4], [.04, .4]]
        projected = undistort_normalized_points(raw, profile)
        np.testing.assert_array_equal(projected[:, 0], [0., 0.])
        self.assertTrue((undistort_normalized_points(raw, dict(profile, geometry_version=2))[:, 0] < 0).all())

    def test_rectified_save_reload_live_tracking_and_raw_ruler_agree(self):
        config = self.config(length_constraints=[dict(points=[[.3, .5], [.7, .5]], distance_m=4.)])
        persisted = []
        online = OnlineRobotCalibration(self.calibrator, mock.Mock(), persist=lambda camera, value: persisted.append(json.loads(json.dumps(value))))
        online.start("camera", auto_apply=True)
        online.save_manual("camera", config)
        self.assertNotIn("camera", online.sessions)
        self.assertTrue(self.calibrator.restore_config("restored", persisted[0]))
        np.testing.assert_allclose(config["rectified_src_points"], self.source)
        raw = distort_normalized_points([[.3, .5], [.7, .5]], self.profile)
        raw_measure = measure_calibrated_polyline(config, raw.tolist())
        rectified_measure = measure_calibrated_polyline(config, [[.3, .5], [.7, .5]], "rectified")
        self.assertAlmostEqual(raw_measure["total_distance_m"], 4., places=4)
        self.assertEqual(raw_measure["total_distance_m"], rectified_measure["total_distance_m"])
        for index, point in enumerate(raw):
            live = self.calibrator.project_ground_point("restored", *point)
            np.testing.assert_allclose([live["x"], live["z"]], raw_measure["points"][index]["floor"], atol=1e-4)
            self.assertTrue(live["valid"])
        self.assertAlmostEqual(rectified_measure["points"][0]["fms"][0], 190., places=4)

    def test_many_pairs_and_segments_are_preserved(self):
        source = np.random.default_rng(3).uniform(.3, .7, (300, 2))
        source = np.vstack((self.source, source))
        lengths = [dict(points=[[.3, .4 + index / 1000], [.6, .4 + index / 1000]], distance_m=3.)
                   for index in range(130)]
        config = prepare_config(self.calibrator, source, source * [10, 8] + [2, 3], "TT", self.frame,
                                intrinsic=self.profile, points_space="rectified", length_constraints=lengths)
        self.assertEqual(config["point_count"], 305)
        self.assertEqual(len(config["rectified_length_constraints"]), 130)

    def test_bad_intrinsics_do_not_fall_back_to_raw_tracking(self):
        config = self.config()
        config["intrinsic_profile"] = {}
        self.calibrator.apply_config("camera", config)
        self.assertFalse(self.calibrator.project_ground_point("camera", .5, .5)["valid"])
        self.assertEqual(self.calibrator.camera_to_floor("camera", .5, .5), (None, None))

    def test_preview_receipt_binds_profile_and_camera(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        encoded = base64.b64encode(cv2.imencode(".jpg", image)[1]).decode()
        with mock.patch.object(pipeline, "rectify_image_cuda", return_value=image):
            result = pipeline.preview("camera", encoded, self.profile)
        receipt = result["preview_id"]
        saved = pipeline.preview_profile("camera", receipt)
        saved["focal_length_px"] = 1
        self.assertEqual(pipeline.preview_profile("camera", receipt)["focal_length_px"], 560)
        with self.assertRaises(ValueError):
            pipeline.preview_profile("other", receipt)
        with mock.patch.object(pipeline.time, "monotonic", return_value=time.monotonic() + 86401):
            with self.assertRaises(ValueError):
                pipeline.preview_profile("camera", receipt)

    def test_cuda_preview_pixel_geometry_matches_clicks(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        profile = make_rectification_profile(dict(image_width=160, image_height=120,
                                                 focal_length_px=140, distortion_xi=.45))
        grid_y, grid_x = np.mgrid[:120, :160]
        image = np.stack((grid_x, grid_y, np.full_like(grid_x, 100)), axis=-1).astype(np.uint8)
        preview = rectify_image_cuda(image, profile)
        for horizontal, vertical in [(50, 45), (80, 60), (100, 80)]:
            raw = distort_normalized_points([[horizontal / 160, vertical / 120]], profile)[0] * [160, 120]
            np.testing.assert_allclose(preview[vertical, horizontal, :2], raw, atol=1.)


if __name__ == "__main__":
    unittest.main()
