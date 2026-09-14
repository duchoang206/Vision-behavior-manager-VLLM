import unittest
from unittest import mock

from core.reid_matcher import GlobalReIDMatcher


class ReIdContinuityTests(unittest.TestCase):
    def setUp(self):
        self.matcher = GlobalReIDMatcher()

    def detection(self, local_id, left=.4):
        return {"local_id": local_id, "class": "person", "bbox": [left, .2, .15, .5]}

    def update(self, camera, objects, now):
        with mock.patch("core.reid_matcher.time.time", return_value=now):
            return self.matcher.process_camera_detections(camera, objects)

    def test_changed_local_id_survives_short_occlusion_without_reid_model(self):
        first = self.update("cam", [self.detection(1)], 1)[0]
        second = self.update("cam", [self.detection(2, .415)], 1.8)[0]
        self.assertEqual(first["global_id"], second["global_id"])

    def test_no_image_only_matching_across_cameras_or_long_gaps(self):
        first = self.update("first", [self.detection(1)], 1)[0]
        second = self.update("second", [self.detection(1)], 1.1)[0]
        late = self.update("first", [self.detection(9)], 3)[0]
        self.assertNotEqual(first["global_id"], second["global_id"])
        self.assertNotEqual(first["global_id"], late["global_id"])

    def test_local_id_reuse_at_distant_box_does_not_steal_identity(self):
        first = self.update("cam", [self.detection(1)], 1)[0]
        distant = self.update("cam", [self.detection(1, .8)], 1.04)[0]
        self.assertNotEqual(first["global_id"], distant["global_id"])

    def test_overlapping_people_do_not_collapse_into_one_id(self):
        objects = self.update("cam", [self.detection(1), self.detection(2, .46)], 1)
        self.assertEqual(2, len({obj["global_id"] for obj in objects}))
