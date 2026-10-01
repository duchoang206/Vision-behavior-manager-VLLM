import os
import sys
import unittest

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from core.custom_deepstream_worker import limit_one_object_per_class


class CameraClassLimitTests(unittest.TestCase):
    def test_keeps_one_highest_confidence_object_for_each_label(self):
        objects = [
            {"class": "Robot_2001", "confidence": .61, "w": .1, "h": .2, "local_id": 1},
            {"class": "robot_2001", "confidence": .89, "w": .05, "h": .1, "local_id": 2},
            {"class": "Robot_6868", "confidence": .77, "w": .1, "h": .2, "local_id": 3},
            {"class": "Rack", "confidence": .42, "w": .2, "h": .2, "local_id": 4},
            {"class": "Rack", "confidence": .42, "w": .3, "h": .2, "local_id": 5},
        ]

        selected, rejected = limit_one_object_per_class(objects)

        self.assertEqual(2, rejected)
        self.assertEqual({"robot_2001": 2, "robot_6868": 3, "rack": 5},
                         {obj["class"].casefold(): obj["local_id"] for obj in selected})

    def test_empty_or_missing_class_is_not_silently_reported_as_a_detection(self):
        selected, rejected = limit_one_object_per_class([
            {"class": "", "confidence": .9, "local_id": 1},
            {"class": "Robot_1", "confidence": .6, "local_id": 2},
        ])
        self.assertEqual(1, rejected)
        self.assertEqual([2], [obj["local_id"] for obj in selected])

    def test_preferred_track_is_not_stolen_by_near_tied_rival(self):
        objects = [dict(**{"class": "Robot_1"}, local_id="1", confidence=.60, w=.1, h=.1),
                   dict(**{"class": "Robot_1"}, local_id="2", confidence=.70, w=.1, h=.1)]
        preferred = {"robot_1": "1"}
        selected, _ = limit_one_object_per_class(objects, preferred)
        self.assertEqual(["1"], [obj["local_id"] for obj in selected])
        objects[0]["confidence"] = .40
        selected, _ = limit_one_object_per_class(objects, preferred)
        self.assertEqual(["2"], [obj["local_id"] for obj in selected])
        self.assertEqual({"robot_1": "2"}, preferred)


if __name__ == "__main__":
    unittest.main()
