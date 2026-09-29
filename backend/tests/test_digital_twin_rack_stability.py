import os
import sys
import unittest

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from core.rack_position_stabilizer import RackPositionStabilizer


class DigitalTwinRackStabilityTests(unittest.TestCase):
    def setUp(self):
        # Configure test instance with short thresholds for unit testing
        self.bridge = RackPositionStabilizer(
            move_confirm_frames=3,
            min_move_duration_sec=0.2,
            move_confirm_m=0.90,
            deadband_m=0.15
        )
        # Production-standard instance (40 frames, 1.8s)
        self.prod_bridge = RackPositionStabilizer()

    def test_small_detector_jitter_keeps_the_initial_rack_position(self):
        first, stable = self.bridge.update("Rack_1", "cam", 10.0, 5.0, 1.0)
        jittered, stable = self.bridge.update("Rack_1", "cam", 10.07, 5.08, 1.1)
        self.assertTrue(stable)
        self.assertEqual(first, jittered)

    def test_real_rack_move_requires_multiple_consistent_frames(self):
        self.bridge.update("Rack_1", "cam", 10.0, 5.0, 1.0)
        first, first_stable = self.bridge.update("Rack_1", "cam", 12.0, 5.0, 1.1)
        second, second_stable = self.bridge.update("Rack_1", "cam", 12.02, 5.01, 1.2)
        third, moving = self.bridge.update("Rack_1", "cam", 11.98, 5.0, 1.35)
        self.assertTrue(first_stable)
        self.assertTrue(second_stable)
        self.assertEqual((10.0, 5.0), first)
        self.assertEqual((10.0, 5.0), second)
        self.assertFalse(moving)
        self.assertGreater(third[0], 10.0)

    def test_short_detector_glitch_under_1_5s_is_rejected_in_production(self):
        # Initial position
        init_pos, stable = self.prod_bridge.update("Rack_1", "cam", 19.23, 15.08, 1.0)
        self.assertEqual((19.23, 15.08), init_pos)
        
        # Detector glitches to (19.90, 15.49) for 5 frames over 0.2s (Mode B)
        for i in range(1, 6):
            glitch_t = 1.0 + i * 0.04
            pos, stable = self.prod_bridge.update("Rack_1", "cam", 19.90, 15.49, glitch_t)
            self.assertTrue(stable)
            self.assertEqual((19.23, 15.08), pos)  # Must remain rock solid!
            
        # Detector recovers back to Mode A
        recovered, stable = self.prod_bridge.update("Rack_1", "cam", 19.23, 15.08, 1.3)
        self.assertEqual((19.23, 15.08), recovered)

    def test_carried_rack_moves_immediately_without_delay(self):
        self.prod_bridge.update("Rack_1", "cam", 10.0, 5.0, 1.0)
        # Robot carries rack and moves
        carried_pos, stable = self.prod_bridge.update("Rack_1", "cam", 12.5, 7.8, 1.04, is_carried=True)
        self.assertEqual((12.5, 7.8), carried_pos)

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

    def test_continuous_dragging_relocation_updates_subsecond(self):
        # Initial stationary rack at (19.64, 15.53)
        self.prod_bridge.update("Rack_drag", "cam", 19.64, 15.53, 1.0)
        
        # Dragging rack across 2.5 meters at 15 FPS (every 0.067s)
        started_moving_t = None
        for i in range(1, 25):
            t = 1.0 + i * 0.067
            curr_x = 19.64 + min(2.5, (i / 15.0) * 2.5)
            curr_z = 15.53 + min(2.5, (i / 15.0) * 2.5)
            pos, stable = self.prod_bridge.update("Rack_drag", "cam", curr_x, curr_z, t)
            if pos != (19.64, 15.53) and started_moving_t is None:
                started_moving_t = t - 1.0
                
        # Must start moving promptly in sub-second (< 0.6s)
        self.assertIsNotNone(started_moving_t)
        self.assertLess(started_moving_t, 0.60)
        self.assertGreater(pos[0], 21.0)

    def test_settled_at_new_position_locks_rock_solid(self):
        self.prod_bridge.update("Rack_settle", "cam", 10.0, 5.0, 1.0)
        # Move to (14.0, 8.0) across enough frames to confirm movement and settle
        for i in range(1, 20):
            t = 1.0 + i * 0.067
            pos, stable = self.prod_bridge.update("Rack_settle", "cam", 14.0, 8.0, t)
        
        # After settling at (14.0, 8.0)
        final_pos, stable = self.prod_bridge.update("Rack_settle", "cam", 14.0, 8.0, 2.5)
        self.assertTrue(stable)
        self.assertEqual((14.0, 8.0), final_pos)
        
        # Minor detector jitter around (14.0, 8.0) does not move rack
        jitter_pos, stable = self.prod_bridge.update("Rack_settle", "cam", 14.06, 8.05, 2.1)
        self.assertTrue(stable)
        self.assertEqual((14.0, 8.0), jitter_pos)


if __name__ == "__main__":
    unittest.main()

