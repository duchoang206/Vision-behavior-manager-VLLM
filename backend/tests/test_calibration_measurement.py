import copy
import unittest

import numpy as np

from core.calibration_measurement import measure_calibrated_polyline
from core.camera_calibrator import CameraCalibrator
from core.manual_calibration import prepare_manual_calibration


class CalibrationMeasurementTests(unittest.TestCase):
    def setUp(self):
        source = [[0, 0], [1, 0], [1, 1], [0, 1]]
        destination = [[0, 0], [10, 0], [10, 5], [0, 5]]
        frame = {"origin_x": 185.0, "origin_y": 194.5, "layout_depth": 18.0}
        self.config = prepare_manual_calibration(CameraCalibrator(), source, destination, "TT", frame)

    def test_polyline_uses_real_floor_meters(self):
        result = measure_calibrated_polyline(self.config, [[0.1, 0.2], [0.5, 0.2], [0.5, 0.6]])
        self.assertAlmostEqual(result["segments"][0]["distance_m"], 4.0, places=5)
        self.assertAlmostEqual(result["segments"][1]["distance_m"], 2.0, places=5)
        self.assertAlmostEqual(result["total_distance_m"], 6.0, places=5)
        self.assertEqual(result["points"][0]["fms"], [186.0, 211.5])
        self.assertAlmostEqual(result["direct_distance_m"], np.sqrt(20), places=5)
        self.assertEqual(result["warnings"], [])
        self.assertEqual(result["calibration"]["updated_at"], self.config["calibration_updated_at"])

    def test_perspective_is_not_constant_pixel_scale(self):
        self.config["matrix"] = [[10, 0, 0], [0, 5, 0], [0, 1, 1]]
        result = measure_calibrated_polyline(self.config, [[0, 0], [1, 0], [1, 1], [0, 1]])
        self.assertAlmostEqual(result["segments"][0]["distance_m"], 10, places=5)
        self.assertAlmostEqual(result["segments"][2]["distance_m"], 5, places=5)

    def test_warns_outside_calibrated_area_without_clamping(self):
        self.config["coverage_polygon"] = [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]
        result = measure_calibrated_polyline(self.config, [[0.1, 0.5], [0.9, 0.5]])
        self.assertAlmostEqual(result["total_distance_m"], 8, places=5)
        self.assertFalse(result["points"][0]["inside_calibrated_area"])
        self.assertTrue(result["warnings"])

    def test_legacy_calibration_uses_source_hull_not_fake_fms(self):
        self.config.pop("coverage_polygon")
        self.config.pop("fms_frame")
        result = measure_calibrated_polyline(self.config, [[0, 0], [1, 0]])
        self.assertTrue(result["points"][0]["inside_calibrated_area"])
        self.assertIsNone(result["points"][0]["fms"])

    def test_does_not_change_calibration(self):
        previous = copy.deepcopy(self.config)
        measure_calibrated_polyline(self.config, [[0, 0], [1, 1]])
        self.assertEqual(self.config, previous)

    def test_rejects_horizon_and_wrong_side_of_ground(self):
        self.config["matrix"] = [[10, 0, 0], [0, 5, 0], [0, 1, -0.5]]
        for points in ([[0.1, 0.5], [0.2, 0.2]], [[0.1, 0.3], [0.2, 0.7]], [[0.1, 0.8], [0.2, 0.9]]):
            with self.subTest(points=points), self.assertRaises(ValueError):
                measure_calibrated_polyline(self.config, points)

    def test_rejects_uncalibrated_or_invalid_points(self):
        with self.assertRaises(ValueError):
            measure_calibrated_polyline({}, [[0, 0], [1, 1]])
        with self.assertRaises(ValueError):
            measure_calibrated_polyline(self.config, [[0, 0]])
        with self.assertRaises(ValueError):
            measure_calibrated_polyline(self.config, [[-0.1, 0], [0.5, 0]])
        for points in ([[0, 0], [float("nan"), 0]], [[0, 0], [float("inf"), 0]], [[0, 0], [1]], [[0, 0]] * 257):
            with self.subTest(points=points), self.assertRaises(ValueError):
                measure_calibrated_polyline(self.config, points)

    def test_rejects_non_metric_space(self):
        self.config["coordinate_space"] = "normalized"
        with self.assertRaises(ValueError):
            measure_calibrated_polyline(self.config, [[0, 0], [1, 1]])


if __name__ == "__main__":
    unittest.main()
