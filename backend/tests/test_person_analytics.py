import os
import sys
import unittest

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from core.camera_calibrator import CameraCalibrator
from core.person_analytics import PersonLogic, person_ground_point


def pose(angle="upright"):
    points = [[0.5, 0.5, 0.9] for _ in range(17)]
    if angle == "upright":
        points[5] = [0.45, 0.30, 0.9]
        points[6] = [0.55, 0.30, 0.9]
        points[11] = [0.45, 0.60, 0.9]
        points[12] = [0.55, 0.60, 0.9]
        points[15] = [0.45, 0.85, 0.9]
        points[16] = [0.55, 0.85, 0.9]
    else:
        points[5] = [0.30, 0.50, 0.9]
        points[6] = [0.30, 0.55, 0.9]
        points[11] = [0.70, 0.50, 0.9]
        points[12] = [0.70, 0.55, 0.9]
        points[15] = [0.90, 0.50, 0.9]
        points[16] = [0.90, 0.55, 0.9]
    return points


class PersonAnalyticsTests(unittest.TestCase):
    def test_ground_point_prefers_ankles(self):
        ground, source = person_ground_point([0.2, 0.2, 0.2, 0.5], pose())
        self.assertEqual("ankles", source)
        self.assertEqual([0.5, 0.85], ground)

    def test_fall_requires_temporal_confirmation(self):
        logic = PersonLogic(confirm_seconds=0.0, recover_seconds=0.0)
        detection = {"bbox": [0.2, 0.4, 0.7, 0.2], "confidence": 0.9, "keypoints": pose("fallen"), "global_id": 7, "frame_width": 1000, "frame_height": 1000}
        first = logic.process(detection, (1, 2), "cam")
        second = logic.process(detection, (1, 2), "cam")
        third = logic.process(detection, (1, 2), "cam")
        self.assertFalse(first["fall_detected"])
        self.assertFalse(second["fall_detected"])
        self.assertTrue(third["fall_detected"])
        self.assertTrue(third["fall_event"])

    def test_homography_reports_valid_metric_projection(self):
        calibrator = CameraCalibrator()
        self.assertTrue(calibrator.set_calibration("cam", [[0, 0], [1, 0], [1, 1], [0, 1]], [[0, 0], [10, 0], [10, 5], [0, 5]]))
        projection = calibrator.project_ground_point("cam", 0.5, 0.5)
        self.assertTrue(projection["valid"])
        self.assertAlmostEqual(projection["x"], 5.0, places=3)
        self.assertAlmostEqual(projection["z"], 2.5, places=3)
        self.assertFalse(calibrator.project_ground_point("unknown", 0.5, 0.5)["valid"])


if __name__ == "__main__":
    unittest.main()
