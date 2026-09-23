import unittest

from core.monitor_model_policy import apply_monitor_visibility, hidden_monitor_cameras


class MonitorVisibilityTests(unittest.TestCase):
    def test_workflow_can_process_without_showing_monitor(self):
        workflows = [{"definition": {"nodes": [
            {"type": "source", "config": {"camera_ids": ["cam-a"]}},
            {"type": "display", "config": {"monitor": False}},
        ]}}]
        self.assertEqual(hidden_monitor_cameras(None, workflows, {"cam-a"}), {"cam-a"})
        payload = {"streams": [{"cam_id": "cam-a", "objects": [{"id": 1}]}]}
        result = apply_monitor_visibility(payload, {"cam-a"})
        self.assertEqual(result["streams"][0]["objects"], [])
        self.assertTrue(result["streams"][0]["monitor_hidden"])

    def test_direct_deployment_keeps_monitor_visible(self):
        deployment = {"all_cameras": False, "camera_ids": ["cam-a"]}
        workflows = [{"definition": {"nodes": [
            {"type": "source", "config": {"camera_ids": ["cam-a"]}},
            {"type": "display", "config": {"monitor": False}},
        ]}}]
        self.assertEqual(hidden_monitor_cameras(deployment, workflows, {"cam-a"}), set())

    def test_missing_display_is_compatible_and_visible(self):
        workflows = [{"definition": {"nodes": [
            {"type": "source", "config": {"camera_ids": ["cam-a"]}},
            {"type": "event", "config": {}},
        ]}}]
        self.assertEqual(hidden_monitor_cameras(None, workflows, {"cam-a"}), set())


if __name__ == "__main__":
    unittest.main()
