import time
import unittest
from collections import deque
from unittest import mock

import numpy as np

from core.camera_calibrator import CameraCalibrator
from core.fms_bridge import FMSBridge
from core.robot_spatial_identity import RobotSpatialIdentity, robot_identity, spatial_verification


class RobotSpatialIdentityTests(unittest.TestCase):
    def setUp(self):
        self.now = time.time()
        self.fms = FMSBridge()
        self.calibrator = CameraCalibrator()
        self.calibrator.set_calibration("cam", [[0, 0], [1, 0], [1, 1], [0, 1]],
                                        [[0, 0], [10, 0], [10, 10], [0, 10]])
        self.calibrator.configs["cam"].update(
            method="manual_camera_fms_click", map_id="TT", coordinate_space="fms_floor_metric",
            fms_frame={"origin_x": self.fms.origin_x,
                       "origin_y": self.fms.origin_y,
                       "layout_depth": self.fms.layout_depth})
        self.gate = RobotSpatialIdentity(self.calibrator, self.fms)
        self.targets = [{"label": "Robot_2001", "category": "robot"},
                        {"label": "Robot_6969", "category": "robot"}]
        self.pose("2001", [3, 0, 5])
        self.pose("6969", [8, 0, 5])

    def pose(self, robot_id, position, at=None, status="IDLE"):
        self.fms.robot_states[robot_id] = dict(id=robot_id, status=status)
        self.fms.pose_history[robot_id] = deque([
            dict(at=self.now if at is None else at, position=position, status=status, source="mqtt")
        ], maxlen=300)

    def context(self, observed_at=None):
        return self.gate.context("cam", self.targets, self.now if observed_at is None else observed_at, now=self.now)

    def test_number_is_identity_not_nearest_robot(self):
        self.assertEqual("2001", robot_identity(dict(label="Robot_2001", category="robot", fms_robot_id=6969)))
        self.assertEqual("6969", robot_identity(dict(label="Robot_6969", category="robot")))
        self.assertIsNone(robot_identity(dict(label="Person_2001", category="person")))
        result = spatial_verification(self.context(), "Robot_2001", [.2, .2, .2, .3])
        self.assertTrue(result["accepted"])
        self.assertEqual("2001", result["robot_id"])
        self.assertAlmostEqual(0, result["distance_m"])

    def test_wrong_location_rejects_even_when_label_is_valid(self):
        result = spatial_verification(self.context(), "Robot_2001", [.7, .2, .2, .3])
        self.assertFalse(result["accepted"])
        self.assertEqual("fms_position_mismatch", result["reason"])
        self.assertEqual("6969", result["rival_robot_id"])

    def test_nearby_rival_rejects_within_distance_tolerance(self):
        self.pose("6969", [4, 0, 5])
        result = spatial_verification(self.context(), "Robot_2001", [.3, .2, .2, .3])
        self.assertFalse(result["accepted"])
        self.assertEqual("fms_other_robot_closer", result["reason"])
        self.assertLess(result["distance_m"], result["tolerance_m"])

    def test_time_aligned_history_not_latest_position(self):
        self.pose("2001", [3, 0, 5], at=self.now - .2)
        self.fms.pose_history["2001"].append(dict(at=self.now, position=[4, 0, 5], source="mqtt", status="RUNNING"))
        self.fms.robot_states["2001"]["position"] = [9, 0, 9]
        result = spatial_verification(self.context(self.now - .1), "Robot_2001", [.25, .2, .2, .3])
        self.assertTrue(result["accepted"])
        self.assertAlmostEqual(0, result["distance_m"])

    def test_stale_offline_and_unsynchronized_poses_are_not_used(self):
        for offset, status in [(-10, "IDLE"), (0, "OFFLINE"), (-1, "RUNNING")]:
            with self.subTest(offset=offset, status=status):
                self.pose("2001", [3, 0, 5], at=self.now + offset, status=status)
                result = spatial_verification(self.context(), "Robot_2001", [.7, .2, .2, .3])
                self.assertFalse(result["checked"])
                self.assertIsNone(result["accepted"])
                self.assertEqual("fms_not_synchronized", result["reason"])

    def test_uncalibrated_camera_does_not_use_legacy_map_guess(self):
        context = self.gate.context("unknown", self.targets, self.now, now=self.now)
        result = spatial_verification(context, "Robot_2001", [.7, .2, .2, .3])
        self.assertIsNone(result["accepted"])
        self.assertEqual("camera_uncalibrated", result["reason"])

    def test_latest_saved_calibration_is_used_without_restart(self):
        before = spatial_verification(self.context(), "Robot_2001", [.2, .2, .2, .3])
        self.calibrator.set_calibration("cam", [[0, 0], [1, 0], [1, 1], [0, 1]],
                                        [[20, 0], [30, 0], [30, 10], [20, 10]])
        self.calibrator.configs["cam"].update(
            method="manual_camera_fms_click", map_id="TT", coordinate_space="fms_floor_metric",
            fms_frame={"origin_x": self.fms.origin_x,
                       "origin_y": self.fms.origin_y,
                       "layout_depth": self.fms.layout_depth})
        after = spatial_verification(self.context(), "Robot_2001", [.2, .2, .2, .3])
        self.assertTrue(before["accepted"])
        self.assertFalse(after["accepted"])
        self.assertEqual("fms_position_mismatch", after["reason"])

    def test_non_fms_calibration_is_not_used_as_robot_gate(self):
        self.calibrator.configs["cam"].update(coordinate_space="normalized", fms_frame=None)
        result = spatial_verification(self.context(), "Robot_2001", [.7, .2, .2, .3])
        self.assertIsNone(result["accepted"])
        self.assertEqual("camera_uncalibrated", result["reason"])

    def test_calibration_from_another_fms_frame_is_not_used(self):
        self.calibrator.configs["cam"]["fms_frame"] = {
            "origin_x": self.fms.origin_x + 1.0,
            "origin_y": self.fms.origin_y,
            "layout_depth": self.fms.layout_depth,
        }
        result = spatial_verification(self.context(), "Robot_2001", [.2, .2, .2, .3])
        self.assertIsNone(result["accepted"])
        self.assertEqual("camera_uncalibrated", result["reason"])

    def test_projection_uses_deepcalib_point_transform(self):
        with mock.patch.object(CameraCalibrator, "_project_input_point", return_value=(.3, .5)) as rectify:
            result = spatial_verification(self.context(), "Robot_2001", [.7, .2, .2, .3])
        self.assertTrue(result["accepted"])
        rectify.assert_called_once()

    def test_nan_or_invalid_geometry_cannot_accept_identity(self):
        for bbox in [[np.nan, .2, .2, .3], [.2, .2, -.2, .3], None]:
            with self.subTest(bbox=bbox):
                result = spatial_verification(self.context(), "Robot_2001", bbox)
                self.assertFalse(result["accepted"])

    def test_predicted_masks_are_filtered_without_relabeling(self):
        objects = [dict(label="Robot_2001", category="robot", x=.7, y=.2, w=.2, h=.3,
                        observed_at=self.now * 1000, tracking_state="predicted"),
                   dict(label="Person_1", category="person", x=.7, y=.2, w=.2, h=.3)]
        filtered, rejected = self.gate.filter_objects("cam", objects, now=self.now)
        self.assertEqual(["Robot_2001"], rejected)
        self.assertEqual(["Person_1"], [obj["label"] for obj in filtered])
        self.assertEqual("Robot_2001", objects[0]["label"])


if __name__ == "__main__":
    unittest.main()
