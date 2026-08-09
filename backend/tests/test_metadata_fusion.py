import unittest

from core.metadata_fusion import MetadataFusion


def robot(label="Robot_2001", identifier=9, confidence=0.9):
    return {"id": identifier, "class": "robot", "label": label, "confidence": confidence, "x": 0.2, "y": 0.3, "w": 0.2, "h": 0.4}


class MetadataFusionTests(unittest.TestCase):
    def setUp(self):
        self.fusion = MetadataFusion(ttl=1.0)
        self.person = {"id": 4, "class": "person", "keypoints": [[0.5, 0.5, 0.9]] * 17}

    def test_detector_only_contributes_people(self):
        objects = self.fusion.update("cam", "deepstream", [self.person, robot(), {"class": "chair"}], now=1)
        self.assertEqual([self.person["id"]], [item["id"] for item in objects])
        self.assertEqual(17, len(objects[0]["keypoints"]))

    def test_one_box_per_label_with_canonical_id(self):
        objects = self.fusion.update("cam", "identity_template", [robot(), robot("robot_2001", 20, 0.4)], now=1)
        self.assertEqual(1, len(objects))
        self.assertEqual(2001, objects[0]["id"])
        self.assertEqual("Robot_2001", objects[0]["label"])

    def test_independent_snapshots_do_not_overwrite_or_retain_missing_people(self):
        self.fusion.update("cam", "deepstream", [self.person], now=1)
        objects = self.fusion.update("cam", "identity_template", [robot()], now=1.1)
        self.assertEqual(2, len(objects))
        objects = self.fusion.update("cam", "deepstream", [], now=1.2)
        self.assertEqual(["Robot_2001"], [item["label"] for item in objects])
        self.assertEqual([], self.fusion.update("cam", "identity_template", [], now=1.3))

    def test_expired_source_is_not_kept_by_other_updates(self):
        self.fusion.update("cam", "identity_template", [robot()], now=1)
        objects = self.fusion.update("cam", "deepstream", [self.person], now=2.1)
        self.assertEqual(1, len(objects))
        self.assertEqual("person", objects[0]["class"])

    def test_gpu_people_take_precedence_over_fallback(self):
        self.fusion.update("cam", "deepstream", [self.person], now=1)
        objects = self.fusion.update("cam", "cpu_fallback", [dict(self.person, id=7)], now=1.1)
        self.assertEqual([4], [item["id"] for item in objects])

    def test_template_does_not_accept_unregistered_or_person_labels(self):
        objects = self.fusion.update("cam", "identity_template", [dict(robot(), label=""), self.person], now=1)
        self.assertEqual([], objects)

    def test_camera_isolation(self):
        self.fusion.update("first", "identity_template", [robot()], now=1)
        self.assertEqual([], self.fusion.update("second", "deepstream", [], now=1.1))


if __name__ == "__main__":
    unittest.main()
