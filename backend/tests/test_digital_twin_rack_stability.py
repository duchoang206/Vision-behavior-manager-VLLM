import os
import sys
import unittest

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from core.rack_position_stabilizer import RackPositionStabilizer


class DigitalTwinRackStabilityTests(unittest.TestCase):
    def setUp(self):
        self.bridge = RackPositionStabilizer()

    def test_small_detector_jitter_keeps_the_initial_rack_position(self):
        first, stable = self.bridge.update("Rack_1", "cam", 10.0, 5.0, 1.0)
        jittered, stable = self.bridge.update("Rack_1", "cam", 10.07, 5.08, 1.1)
        self.assertTrue(stable)
        self.assertEqual(first, jittered)

    def test_real_rack_move_requires_multiple_consistent_frames(self):
        self.bridge.update("Rack_1", "cam", 10.0, 5.0, 1.0)
        first, first_stable = self.bridge.update("Rack_1", "cam", 11.0, 5.0, 1.1)
        second, second_stable = self.bridge.update("Rack_1", "cam", 11.02, 5.01, 1.2)
        third, moving = self.bridge.update("Rack_1", "cam", 10.98, 5.0, 1.3)
        self.assertTrue(first_stable)
        self.assertTrue(second_stable)
        self.assertEqual((10.0, 5.0), first)
        self.assertEqual((10.0, 5.0), second)
        self.assertFalse(moving)
        self.assertGreater(third[0], 10.0)

    def test_repeated_medium_projection_error_is_not_treated_as_a_move(self):
        self.bridge.update("Rack_1", "cam", 10.0, 5.0, 1.0)
        for now in (1.1, 1.2, 1.3, 1.4):
            position, stable = self.bridge.update("Rack_1", "cam", 10.24, 5.0, now)
            self.assertTrue(stable)
            self.assertEqual((10.0, 5.0), position)

    def test_storage_slot_position_remains_exact(self):
        self.bridge.update("Rack_1", "cam", 10.0, 5.0, 1.0)
        position, stable = self.bridge.update("Rack_1", "cam", 11.0, 5.0, 1.1, snapped_to_slot=True)
        self.assertEqual((11.0, 5.0), position)
        self.assertTrue(stable)


if __name__ == "__main__":
    unittest.main()
