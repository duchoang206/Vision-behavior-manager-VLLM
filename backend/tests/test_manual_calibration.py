import unittest
from unittest import mock

import numpy as np

from core.camera_calibrator import CameraCalibrator
from core.manual_calibration import prepare_manual_calibration
from core.online_calibration import OnlineRobotCalibration


class ManualCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.calibrator = CameraCalibrator()
        self.frame = {'origin_x': 185., 'origin_y': 194.5, 'layout_depth': 18.}
        self.source = np.random.default_rng(7).uniform(.1, .9, (40, 2))
        self.destination = self.source * [20, 12] + [1, 2]

    def test_many_pairs_use_one_persisted_primary_calibration(self):
        config = prepare_manual_calibration(self.calibrator, self.source, self.destination, 'TT', self.frame)
        persist = mock.Mock()
        online = OnlineRobotCalibration(self.calibrator, mock.Mock(), persist=persist)
        online.start('camera', auto_apply=True)
        online.save_manual('camera', config)
        self.assertNotIn('camera', online.sessions)
        persist.assert_called_once_with('camera', config)
        self.assertEqual(40, config['point_count'])
        self.assertLess(config['reprojection_error'], 1e-5)
        point = self.calibrator.project_ground_point('camera', .5, .5)
        self.assertAlmostEqual(11, point['x'], places=3)
        self.assertAlmostEqual(8, point['z'], places=3)
        self.assertTrue(config['primary_map'])

    def test_duplicate_and_unpaired_points_do_not_apply(self):
        for source, destination in [(list(self.source) + [self.source[0]], list(self.destination) + [self.destination[0]]),
                                    (self.source[:-1], self.destination)]:
            with self.assertRaises(ValueError):
                prepare_manual_calibration(self.calibrator, source, destination, 'TT', self.frame)
        self.assertIsNone(self.calibrator.get_config('camera'))

    def test_database_error_keeps_active_calibration(self):
        config = prepare_manual_calibration(self.calibrator, self.source, self.destination, 'TT', self.frame)
        self.calibrator.apply_config('camera', config)
        online = OnlineRobotCalibration(self.calibrator, mock.Mock(), persist=mock.Mock(side_effect=RuntimeError('SSD full')))
        replacement = prepare_manual_calibration(self.calibrator, self.source, self.destination + 2, 'TT', self.frame)
        with self.assertRaises(RuntimeError):
            online.save_manual('camera', replacement)
        self.assertEqual(config, self.calibrator.get_config('camera'))


if __name__ == '__main__':
    unittest.main()
