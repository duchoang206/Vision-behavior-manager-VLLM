"""Regression: storage-slot ROIs drawn on the camera only must survive restarts."""

import os
import unittest

SQUARE = [[0.1, 0.1], [0.4, 0.1], [0.4, 0.4], [0.1, 0.4]]


@unittest.skipUnless(os.getenv("COMM_TEST_DB_URL"), "set COMM_TEST_DB_URL to a disposable PostgreSQL")
class OccupancyCleanupTests(unittest.TestCase):
    def setUp(self):
        from core.database import DatabaseManager

        self.db = DatabaseManager(os.environ["COMM_TEST_DB_URL"])
        self.db.delete_rules_by_camera("cam_cleanup")

    def tearDown(self):
        self.db.delete_rules_by_camera("cam_cleanup")

    def save(self, rule_id, **fields):
        self.db.save_rule({"id": rule_id, "cam_id": "cam_cleanup", "type": "occupancy", "name": rule_id, **fields})

    def test_camera_only_slot_is_kept_and_empty_slot_removed(self):
        self.save("camera_only", points=SQUARE, camera_points=SQUARE, fms_points=[], coordinate_space="camera")
        self.save("hybrid", points=SQUARE, camera_points=SQUARE, fms_points=SQUARE, coordinate_space="hybrid")
        self.save("no_geometry", points=[], camera_points=[], fms_points=[])

        deleted = self.db.delete_invalid_occupancy_rules("cam_cleanup")

        self.assertEqual(deleted, 1)
        kept = sorted(r["id"] for r in self.db.get_rules_by_camera("cam_cleanup"))
        self.assertEqual(kept, ["camera_only", "hybrid"])


if __name__ == "__main__":
    unittest.main()
