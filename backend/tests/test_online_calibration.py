import time
import unittest
from unittest import mock
from collections import deque

import numpy as np

from core.camera_calibrator import CameraCalibrator
from core.online_calibration import OnlineRobotCalibration
from core.fms_bridge import FMSBridge


class FakeFms:
    origin_x = 0.0
    origin_y = 0.0
    layout_depth = 10.0

    def pose_at(self, robot_id, observed_at, now=None):
        return {"position": [observed_at, 0.0, observed_at], "skew_ms": 0}


class OnlineCalibrationTests(unittest.TestCase):
    def test_rejects_duplicate_and_fits_metric_homography(self):
        calibrator = CameraCalibrator()
        fms = FakeFms()
        online = OnlineRobotCalibration(calibrator, fms, fit_interval=0)
        online.start("cam", auto_apply=False)
        random = np.random.default_rng(25)
        for index, (image_x, image_y) in enumerate(random.uniform(.15, .85, (30, 2))):
            timestamp = float(index + 1)
            fms.pose_at = mock.Mock(return_value={"position": [image_x * 10, 0.0, image_y * 5], "skew_ms": 0})
            object_data = {"label": "Robot_2001", "category": "robot", "tracking_state": "tracked", "confidence": .95,
                           "x": image_x - .02, "y": image_y - .08, "w": .04, "h": .08, "observed_at": int(timestamp * 1000)}
            online.observe("cam", [object_data], {"robot_2001": "2001"}, now=timestamp)
        online.fit_pending(now=31)
        self.assertFalse(calibrator.project_ground_point("cam", .5, .5)["valid"])
        with mock.patch("core.online_calibration.time.time", return_value=31):
            status = online.status("cam")
            self.assertTrue(status["ready"], status)
            self.assertGreaterEqual(status["inlier_count"], 16)
            online.apply("cam")
        projected = calibrator.project_ground_point("cam", .5, .5)
        self.assertTrue(projected["valid"])
        self.assertAlmostEqual(5.0, projected["x"], places=3)
        self.assertAlmostEqual(2.5, projected["z"], places=3)

    def test_fms_jump_is_not_used_by_pose_history_contract(self):
        calibrator = CameraCalibrator()
        fms = FakeFms()
        online = OnlineRobotCalibration(calibrator, fms, min_samples=8)
        online.start("cam")
        fms.pose_at = lambda robot_id, observed_at, now=None: None
        online.observe("cam", [{"label": "Robot_2001", "category": "robot", "tracking_state": "tracked", "confidence": .9,
                                 "x": .4, "y": .4, "w": .1, "h": .1, "observed_at": 1000}], {"robot_2001": "2001"}, now=1)
        self.assertEqual(0, online.status("cam")["sample_count"])

    def test_offline_robot_state_alias_is_rejected(self):
        calibrator = CameraCalibrator()
        fms = FakeFms()
        fms.robot_states = {"Robot_2001": {"id": "Robot_2001", "status": "OFFLINE"}}
        online = OnlineRobotCalibration(calibrator, fms, fit_interval=0)
        online.start("cam")
        online.observe("cam", [{"label": "Robot_2001", "category": "robot", "tracking_state": "tracked",
                                 "confidence": .9, "x": .4, "y": .4, "w": .1, "h": .1,
                                 "observed_at": 1000}], {"robot_2001": "2001"}, now=1)
        status = online.status("cam")
        self.assertEqual(0, status["sample_count"])
        self.assertEqual(1, status["rejected"]["fms_robot_offline"])

    def test_collinear_robot_path_is_not_applied(self):
        calibrator = CameraCalibrator()
        fms = FakeFms()
        online = OnlineRobotCalibration(calibrator, fms, min_samples=8, fit_interval=0)
        online.start("cam")
        for index in range(12):
            image_x = .15 + index * .05
            image_y = .2 + index * .03
            fms.pose_at = mock.Mock(return_value={"position": [image_x * 10, 0.0, image_y * 5], "skew_ms": 0})
            online.observe("cam", [{"label": "Robot_2001", "category": "robot", "tracking_state": "tracked",
                                     "confidence": .95, "x": image_x - .02, "y": image_y - .08,
                                     "w": .04, "h": .08, "observed_at": (index + 1) * 1000}],
                           {"robot_2001": "2001"}, now=index + 1)
        online.fit_pending(now=20)
        self.assertEqual("need_wider_2d_path", online.status("cam")["reason"])
        self.assertFalse(calibrator.project_ground_point("cam", .5, .5)["valid"])

    def test_failed_persistence_does_not_replace_existing_matrix(self):
        calibrator = CameraCalibrator()
        self.assertTrue(calibrator.set_calibration("cam", [[0, 0], [1, 0], [1, 1], [0, 1]],
                                                   [[0, 0], [1, 0], [1, 1], [0, 1]]))
        fms = FakeFms()
        online = OnlineRobotCalibration(calibrator, fms, min_samples=8, fit_interval=0,
                                        persist=mock.Mock(side_effect=RuntimeError("db down")))
        online.start("cam")
        for index, (image_x, image_y) in enumerate(np.random.default_rng(4).uniform(.15, .85, (12, 2))):
            fms.pose_at = mock.Mock(return_value={"position": [image_x * 10, 0.0, image_y * 5], "skew_ms": 0})
            online.observe("cam", [{"label": "Robot_2001", "category": "robot", "tracking_state": "tracked",
                                     "confidence": .95, "x": image_x - .02, "y": image_y - .08,
                                     "w": .04, "h": .08, "observed_at": (index + 1) * 1000}],
                           {"robot_2001": "2001"}, now=index + 1)
        online.fit_pending(now=20)
        with mock.patch("core.online_calibration.time.time", return_value=20):
            with self.assertRaises(RuntimeError):
                online.apply("cam")
        projected = calibrator.project_ground_point("cam", .5, .5)
        self.assertTrue(projected["valid"])
        self.assertAlmostEqual(.5, projected["x"], places=3)

    def test_rejected_pose_does_not_invalidate_last_good_live_pose(self):
        bridge = FMSBridge()
        bridge.robot_states["2001"] = {"id": "2001", "status": "IDLE"}
        bridge.pose_history["2001"] = deque([{
            "at": 9.8, "position": [4.0, 0.0, 5.0], "source": "mqtt"
        }], maxlen=300)
        bridge.pose_rejected_at["2001"] = 10.0
        pose = bridge.pose_at("2001", 9.9, now=10.0)
        self.assertIsNotNone(pose)
        self.assertEqual([4.0, 0.0, 5.0], pose["position"])

    def test_rebroadcast_offline_pose_cannot_be_used_for_calibration(self):
        bridge = FMSBridge()
        bridge.robot_states["2001"] = {"id": "2001", "status": "OFFLINE"}
        bridge._apply_pose_update(
            bridge.robot_states["2001"], [3.0, 0.0, 4.0], 0.0, "TT", "datasocket"
        )
        now = time.time()
        pose = bridge.pose_at("2001", now, now=now)
        self.assertIsNone(pose)


if __name__ == "__main__":
    unittest.main()
