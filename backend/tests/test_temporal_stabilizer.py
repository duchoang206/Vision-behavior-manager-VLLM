import unittest

from core.temporal_stabilizer import TemporalDetectionStabilizer


def pose(offset=0.0, confidence=0.9):
    return [[0.5 + offset, 0.5, confidence] for _ in range(17)]


class TemporalStabilizerTests(unittest.TestCase):
    def test_local_id_change_keeps_stable_id(self):
        stabilizer = TemporalDetectionStabilizer(track_ttl=2.0)
        first = stabilizer.update([{"local_id": 10, "class": "person", "bbox": [.4, .2, .15, .5], "keypoints": pose()}], now=1.0)
        second = stabilizer.update([{"local_id": 99, "class": "person", "bbox": [.405, .205, .15, .5], "keypoints": pose(.01)}], now=1.05)
        self.assertEqual(first[0]["local_id"], second[0]["local_id"])
        self.assertEqual(99, second[0]["source_local_id"])

    def test_short_pose_gap_is_held_and_marked_stale(self):
        stabilizer = TemporalDetectionStabilizer(track_ttl=2.0, pose_ttl=0.35)
        stabilizer.update([{"local_id": 1, "class": "person", "bbox": [.4, .2, .15, .5], "keypoints": pose()}], now=1.0)
        held = stabilizer.update([{"local_id": 1, "class": "person", "bbox": [.41, .2, .15, .5], "keypoints": []}], now=1.1)
        self.assertEqual(17, len(held[0]["keypoints"]))
        self.assertTrue(held[0]["keypoints_stale"])
        expired = stabilizer.update([{"local_id": 1, "class": "person", "bbox": [.42, .2, .15, .5], "keypoints": []}], now=1.5)
        self.assertEqual([], expired[0]["keypoints"])
        self.assertFalse(expired[0]["keypoints_stale"])

    def test_small_pose_jitter_is_smoothed_without_delaying_large_motion(self):
        stabilizer = TemporalDetectionStabilizer(track_ttl=2.0, pose_ttl=0.45)
        stabilizer.update([{"local_id": 1, "class": "person", "bbox": [.4, .2, .15, .5], "keypoints": pose()}], now=1.0)
        filtered = stabilizer.update([{"local_id": 1, "class": "person", "bbox": [.4, .2, .15, .5], "keypoints": pose(.005)}], now=1.04)
        self.assertLess(filtered[0]["keypoints"][0][0], 0.505)
        self.assertGreater(filtered[0]["keypoints"][0][0], 0.5)
        moved = stabilizer.update([{"local_id": 1, "class": "person", "bbox": [.4, .2, .15, .5], "keypoints": pose(.08)}], now=1.08)
        self.assertAlmostEqual(0.58, moved[0]["keypoints"][0][0])

    def test_large_single_landmark_jump_is_held(self):
        stabilizer = TemporalDetectionStabilizer(track_ttl=2.0, pose_ttl=0.45)
        stabilizer.update([{"local_id": 1, "class": "person", "bbox": [.4, .2, .15, .5], "keypoints": pose()}], now=1.0)
        jumped = pose()
        jumped[0] = [.70, .5, .9]
        result = stabilizer.update([{"local_id": 1, "class": "person", "bbox": [.4, .2, .15, .5], "keypoints": jumped}], now=1.04)
        self.assertAlmostEqual(.5, result[0]["keypoints"][0][0])
        self.assertIn(0, result[0]["keypoints_predicted_indices"])

    def test_short_detection_gap_keeps_person_track_and_pose(self):
        stabilizer = TemporalDetectionStabilizer(track_ttl=2.0, detection_hold_sec=0.2)
        first = stabilizer.update([{"local_id": 1, "class": "person", "bbox": [.4, .2, .15, .5], "keypoints": pose()}], now=1.0)
        held = stabilizer.update([], now=1.1)
        self.assertEqual(first[0]["local_id"], held[0]["local_id"])
        self.assertEqual(17, len(held[0]["keypoints"]))
        self.assertEqual("predicted", held[0]["tracking_state"])
        self.assertEqual([], stabilizer.update([], now=1.25))


if __name__ == "__main__":
    unittest.main()
