import unittest

from fastapi import HTTPException

from core.monitor_model_policy import filter_monitor_metadata
from routers.legacy_labels import reject_legacy_label_registration


class FakeProcess:
    def __init__(self, return_code=None):
        self.return_code = return_code

    def poll(self):
        return self.return_code


class MonitorModelPolicyTests(unittest.TestCase):
    def setUp(self):
        self.accepted = set()

    def accepts_stream(self, camera_id, model_id):
        return (camera_id, model_id) in self.accepted

    def test_legacy_identity_packets_are_rejected(self):
        for source in ("identity_template", "identity_people"):
            payload = {
                "source": source,
                "timestamp": 10,
                "streams": [{
                    "cam_id": "cam-1",
                    "model_id": "model-1",
                    "objects": [{"id": 1, "model_id": "model-1", "label": "Robot_2001", "mask": {"polygons": []}}],
                }],
            }
            self.accepted.add(("cam-1", "model-1"))
            self.assertIsNone(filter_monitor_metadata(payload, self.accepts_stream))

    def test_non_custom_packets_cannot_carry_registered_masks(self):
        payload = {
            "source": "deepstream",
            "timestamp": 10,
            "streams": [{
                "cam_id": "cam-1",
                "objects": [{
                    "id": 1,
                    "class": "person",
                    "label": "Robot_2001",
                    "model_id": "old-model",
                    "mask": {"polygons": [[[0, 0], [1, 1]]]},
                    "identity_registered": True,
                }],
            }],
        }
        result = filter_monitor_metadata(payload, self.accepts_stream)
        obj = result["streams"][0]["objects"][0]
        self.assertEqual("person", obj["class"])
        self.assertEqual(1, obj["id"])
        self.assertIsNone(obj["mask"])
        self.assertTrue(obj["mask_stale"])
        self.assertIsNone(obj["model_id"])
        self.assertIsNone(obj["label"])
        self.assertFalse(obj["identity_registered"])
        self.assertIsNotNone(payload["streams"][0]["objects"][0]["mask"])

    def test_custom_stream_requires_active_model_and_camera(self):
        payload = {
            "source": "custom_deepstream",
            "timestamp": 10,
            "streams": [{
                "cam_id": "cam-1",
                "model_id": "model-1",
                "objects": [
                    {"id": 1, "model_id": "model-1", "mask": {"polygons": []}},
                    {"id": 2, "model_id": "model-2", "mask": {"polygons": []}},
                    {"id": 3, "mask": {"polygons": []}},
                ],
            }],
        }
        self.assertIsNone(filter_monitor_metadata(payload, self.accepts_stream))
        self.accepted.add(("cam-1", "model-1"))
        result = filter_monitor_metadata(payload, self.accepts_stream)
        self.assertEqual([1], [obj["id"] for obj in result["streams"][0]["objects"]])

    def test_empty_custom_frame_is_kept_to_clear_previous_mask(self):
        self.accepted.add(("cam-1", "model-1"))
        payload = {
            "source": "custom_deepstream",
            "timestamp": 11,
            "streams": [{"cam_id": "cam-1", "model_id": "model-1", "objects": []}],
        }
        result = filter_monitor_metadata(payload, self.accepts_stream)
        self.assertEqual([], result["streams"][0]["objects"])

    def test_legacy_registration_dependency_always_returns_gone(self):
        with self.assertRaises(HTTPException) as context:
            reject_legacy_label_registration()
        self.assertEqual(410, context.exception.status_code)


class CustomDetectorAcceptanceTests(unittest.TestCase):
    def test_accepts_only_live_matching_signature(self):
        from core.custom_detector import CustomDetector

        detector = CustomDetector(None, None, None, None, None)
        detector.signature = ("model-1", ("cam-1", "cam-2"))
        detector.model_id = "model-1"
        detector.process = FakeProcess()
        self.assertTrue(detector.accepts_stream("cam-1", "model-1"))
        self.assertFalse(detector.accepts_stream("cam-3", "model-1"))
        self.assertFalse(detector.accepts_stream("cam-1", "model-2"))
        self.assertFalse(detector.accepts_stream("cam-1", None))
        detector.stopping.set()
        self.assertFalse(detector.accepts_stream("cam-1", "model-1"))
        detector.stopping.clear()
        detector.process = FakeProcess(return_code=1)
        self.assertFalse(detector.accepts_stream("cam-1", "model-1"))
        detector.process = None
        self.assertFalse(detector.accepts_stream("cam-1", "model-1"))


if __name__ == "__main__":
    unittest.main()
