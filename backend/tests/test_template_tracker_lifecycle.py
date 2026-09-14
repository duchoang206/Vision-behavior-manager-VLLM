import os
import sys
import tempfile
import threading
import unittest
from unittest import mock

import numpy as np

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from core.template_identity_tracker import TemplateIdentityCameraTracker, TemplateIdentityTrackerManager
from src.controller.registry import TargetRegistry


class TemplateTrackerLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.manager = TemplateIdentityTrackerManager()
        self.bbox = [.1, .2, .3, .4]

    def test_new_camera_starts_only_after_valid_target(self):
        with mock.patch.object(TemplateIdentityCameraTracker, "start") as start:
            tracker = self.manager.add_camera("new-camera", "rtsp://physical")
            self.assertFalse(tracker.running)
            start.assert_not_called()
            self.assertFalse(self.manager.add_target("new-camera", "rtsp://physical", "Robot_42", "robot", [0, 0, 0, 0]))
            start.assert_not_called()
            self.assertTrue(self.manager.add_target("new-camera", "rtsp://physical", "Robot_42", "robot", self.bbox))
            start.assert_called_once()
            self.assertEqual("rtsp://127.0.0.1:8554/new-camera", tracker.stream_url)

    def test_start_recovers_dead_thread_without_duplicate_live_thread(self):
        tracker = TemplateIdentityCameraTracker("cam", "rtsp://physical", None)
        tracker.running = True
        tracker.thread = mock.Mock()
        tracker.thread.is_alive.return_value = False
        with mock.patch("core.template_identity_tracker.threading.Thread") as thread:
            tracker.start()
            thread.assert_called_once()
            thread.return_value.start.assert_called_once()
            thread.return_value.is_alive.return_value = True
            tracker.start()
            thread.assert_called_once()

    def test_reader_uses_relay_and_recovers_initial_exception(self):
        tracker = TemplateIdentityCameraTracker("new-camera", "rtsp://physical", None)
        tracker.running = True
        reader = mock.Mock()
        with mock.patch("core.template_identity_tracker.RTSPLatestFrameReader", side_effect=[RuntimeError("offline"), reader]) as reader_class, \
                mock.patch("core.template_identity_tracker.registered_target_mask_segmenter") as segmenter, \
                mock.patch.object(tracker.stop_event, "wait") as wait, \
                mock.patch.object(tracker, "_process_frames", side_effect=lambda reader: setattr(tracker, "running", False)):
            tracker._loop()
            self.assertEqual(2, reader_class.call_count)
            self.assertTrue(all(call.args[0] == tracker.stream_url for call in reader_class.call_args_list))
            wait.assert_called_once_with(1.0)
            reader.stop.assert_called_once()
            segmenter.remove_camera.assert_called_once_with("new-camera")
            self.assertFalse(tracker.running)
            self.assertIsNone(tracker.last_error)

    def test_camera_delete_waits_for_old_worker_and_re_registration_restarts(self):
        tracker = self.manager.add_camera("cam", "rtsp://physical")
        stopped = threading.Event()
        tracker.thread = threading.Thread(target=lambda: (tracker.stop_event.wait(), stopped.set()))
        tracker.running = True
        tracker.thread.start()
        self.manager.remove_camera("cam")
        self.assertTrue(stopped.is_set())
        self.assertFalse(tracker.thread.is_alive())
        with mock.patch.object(TemplateIdentityCameraTracker, "start") as start:
            self.manager.add_target("cam", "rtsp://new", "Robot_42", "robot", self.bbox)
            self.assertIsNot(tracker, self.manager.trackers["cam"])
            start.assert_called_once()

    def test_deleting_one_camera_label_keeps_other_view(self):
        with mock.patch.object(TemplateIdentityCameraTracker, "start"), \
                mock.patch.object(TemplateIdentityCameraTracker, "stop") as stop:
            for camera_id in ["first", "second"]:
                self.manager.add_target(camera_id, "rtsp://physical", "Robot_42", "robot", self.bbox)
            self.manager.remove_target("Robot_42", cam_id="first")
            self.assertNotIn("first", self.manager.trackers)
            self.assertIn("Robot_42", self.manager.trackers["second"].targets)
            stop.assert_called_once()

    def test_registry_restores_camera_views_with_same_global_id(self):
        vector = np.zeros(512, dtype=np.float32)
        vector[1] = 1
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "targets.json")
            registry = TargetRegistry(path)
            registry.register_target("Rack_A", vector.tolist(), "first", category="rack", bbox=self.bbox)
            other_bbox = [.6, .2, .3, .4]
            registry.register_target("Rack_A", vector.tolist(), "second", category="rack", bbox=other_bbox)
            restored = TargetRegistry(path)
            first = restored.get_all_targets(cam_id="first")[0]
            second = restored.get_all_targets(cam_id="second")[0]
            self.assertEqual(first["global_id"], second["global_id"])
            self.assertNotEqual(first["key"], second["key"])
            self.assertNotEqual(first["bbox"], second["bbox"])
            restored.remove_target("Rack_A", cam_id="first")
            self.assertEqual([second], restored.get_all_targets())


if __name__ == "__main__":
    unittest.main()
