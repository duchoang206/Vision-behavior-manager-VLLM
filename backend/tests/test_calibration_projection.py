import unittest
import copy

import numpy as np

from core.calibration_projection import calibration_footprint, project_calibration
from core.camera_calibrator import CameraCalibrator
from core.manual_calibration import prepare_manual_calibration


class CalibrationProjectionTests(unittest.TestCase):
    def setUp(self):
        self.frame = {"origin_x": 185.0, "origin_y": 194.5, "layout_depth": 18.0}
        self.config = prepare_manual_calibration(
            CameraCalibrator(), [[0, 0], [1, 0], [1, 1], [0, 1]],
            [[0, 0], [10, 0], [10, 5], [0, 5]], "TT", self.frame,
        )

    def test_image_to_floor_and_back(self):
        forward = project_calibration(self.config, [[.25, .5]], "image", self.frame)
        for value in forward["points"][0]["floor"]:
            self.assertAlmostEqual(value, 2.5, places=6)
        backward = project_calibration(self.config, [[2.5, 2.5]], "floor", self.frame)
        for actual, expected in zip(backward["points"][0]["image"], [.25, .5]):
            self.assertAlmostEqual(actual, expected, places=6)

    def test_fms_conversion_and_footprint(self):
        result = project_calibration(self.config, [[187.5, 210.0]], "fms", self.frame)
        self.assertEqual(result["points"][0]["floor"], [2.5, 2.5])
        footprint = calibration_footprint(self.config, self.frame)
        self.assertEqual(len(footprint["footprint"]), 4)
        self.assertEqual(footprint["anchor_kind"], "coverage_center")

    def test_outside_coverage_is_flagged_not_clamped(self):
        result = project_calibration(self.config, [[20, 2]], "floor", self.frame)["points"][0]
        self.assertFalse(result["inside_image"])
        self.assertFalse(result["inside_calibrated_area"])
        self.assertAlmostEqual(result["image"][0], 2)

    def test_frame_mismatch_degenerate_and_nan_rejected(self):
        with self.assertRaises(ValueError):
            project_calibration(self.config, [[1, 1]], "floor", dict(self.frame, origin_x=0))
        with self.assertRaises(ValueError):
            project_calibration(dict(self.config, matrix=[[0, 0, 0]] * 3), [[1, 1]])
        with self.assertRaises(ValueError):
            project_calibration(self.config, [[float("nan"), 1]])

    def test_deepcalib_roundtrip_and_config_unchanged(self):
        from deep_calib.adapter import prepare_config
        from deep_calib.geometry import make_rectification_profile, distort_normalized_points
        profile = make_rectification_profile(dict(image_width=640, image_height=480, focal_length_px=560, distortion_xi=.45))
        points = np.array([[.25, .25], [.75, .25], [.75, .75], [.25, .75]])
        config = prepare_config(CameraCalibrator(), points, points * [10, 8], "TT", self.frame,
                                intrinsic=profile, points_space="rectified")
        before = copy.deepcopy(config)
        raw = distort_normalized_points([[.4, .6]], profile)
        projected = project_calibration(config, raw, "image", self.frame)["points"][0]
        reversed_point = project_calibration(config, [projected["fms"]], "fms", self.frame)["points"][0]
        np.testing.assert_allclose(reversed_point["image"], raw[0], atol=1e-6)
        self.assertEqual(config, before)


if __name__ == "__main__":
    unittest.main()
