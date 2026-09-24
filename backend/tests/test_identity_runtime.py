import os
import sys
import unittest
from unittest import mock

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from core.identity_utils import identity_global_id
from core.mediamtx_client import MediaMTXClient, camera_preview_relay_url, camera_relay_url
from core.metadata_fusion import MetadataFusion


class IdentityRuntimeTests(unittest.TestCase):
    def test_explicit_identity_overrides_camera_local_ids(self):
        for label, category in [("Robot_2001", "robot"), ("Rack_A", "rack"), ("Person_A", "person")]:
            with self.subTest(label=label):
                self.assertEqual(identity_global_id(label, category, 7), identity_global_id(label, category, 90))
                self.assertNotEqual(identity_global_id(label, category), identity_global_id(label + "2", category))
        self.assertEqual(2001, identity_global_id("Robot_2001", "robot"))
        self.assertEqual(7, identity_global_id(None, "person", 7))

    def test_person_identity_requires_live_detector_and_preserves_pose(self):
        fusion = MetadataFusion(ttl=1.0)
        identity = {"id": 8, "local_id": 8, "class": "person", "category": "person",
                    "label": "Person_A", "tracking_state": "tracked", "x": .2, "y": .2,
                    "w": .2, "h": .4}
        self.assertEqual([], fusion.update("cam", "identity_template", [identity], now=1.0))
        detector = dict(identity, id=90, label=None, keypoints=[[.3, .3, .9]] * 17, fall_detected=True)
        result = fusion.update("cam", "deepstream", [detector], now=1.1)
        self.assertEqual(["Person_A"], [item.get("label") for item in result])
        self.assertEqual(identity_global_id("Person_A", "person"), result[0]["id"])
        self.assertEqual(detector["keypoints"], result[0]["keypoints"])
        self.assertTrue(result[0]["fall_detected"])
        self.assertEqual([], fusion.update("cam", "deepstream", [], now=1.2))

    def test_same_label_keeps_independent_camera_boxes_and_masks(self):
        fusion = MetadataFusion()
        first = {"id": 7, "class": "robot", "label": "Robot_2001", "x": .2,
                 "mask": {"polygons": [[[.1, .2], [.3, .2], [.3, .5]]]}}
        second = dict(first, id=99, x=.7, mask={"polygons": [[[.6, .2], [.8, .2], [.8, .5]]]})
        first_result = fusion.update("first", "identity_template", [first], now=1)
        second_result = fusion.update("second", "identity_template", [second], now=1)
        self.assertEqual(first_result[0]["id"], second_result[0]["id"])
        self.assertEqual(.2, first_result[0]["x"])
        self.assertEqual(.7, second_result[0]["x"])
        self.assertNotEqual(first_result[0]["mask"], second_result[0]["mask"])
        fusion.remove_camera("first")
        self.assertEqual(second_result, fusion.update("second", "deepstream", [], now=1.1))

    def test_person_label_does_not_leak_between_cameras_or_duplicate_people(self):
        fusion = MetadataFusion()
        person = {"id": 8, "class": "person", "x": .2, "y": .2, "w": .2, "h": .4}
        identity = dict(person, label="Person_A", tracking_state="tracked")
        fusion.update("first", "identity_template", [identity], now=1)
        result = fusion.update("first", "deepstream", [person, dict(person, id=9)], now=1.1)
        self.assertEqual(1, sum(item.get("label") == "Person_A" for item in result))
        result = fusion.update("second", "deepstream", [person], now=1.1)
        self.assertIsNone(result[0].get("label"))
        result = fusion.update("first", "deepstream", [person], now=2.1)
        self.assertIsNone(result[0].get("label"))

    def test_relay_url_is_camera_scoped(self):
        with mock.patch.dict(os.environ, {"MEDIAMTX_RTSP_ORIGIN": "rtsp://127.0.0.1:8554/"}):
            self.assertEqual("rtsp://127.0.0.1:8554/cam%2F1", camera_relay_url("cam/1"))
            self.assertEqual("rtsp://127.0.0.1:8554/cam%2F1_preview", camera_preview_relay_url("cam/1"))

    def test_preview_command_uses_low_resolution_relay_only(self):
        values = {
            "MEDIAMTX_RTSP_ORIGIN": "rtsp://127.0.0.1:8554",
            "DISPLAY_PREVIEW_WIDTH": "640",
            "DISPLAY_PREVIEW_HEIGHT": "360",
            "DISPLAY_PREVIEW_FPS": "12",
            "DISPLAY_PREVIEW_BITRATE_KBPS": "900",
        }
        with mock.patch.dict(os.environ, values, clear=False):
            command = MediaMTXClient()._preview_command("cam1")
        self.assertIn("scale=640:360", command[command.index("-vf") + 1])
        self.assertIn("fps=12", command[command.index("-vf") + 1])
        self.assertEqual("rtsp://127.0.0.1:8554/cam1", command[command.index("-i") + 1])
        self.assertEqual("rtsp://127.0.0.1:8554/cam1_preview", command[-1])

    def test_preview_process_is_not_started_twice(self):
        client = MediaMTXClient()
        process = mock.Mock()
        process.poll.return_value = None
        with mock.patch.dict(os.environ, {"DISPLAY_PREVIEW_ENABLED": "1"}, clear=False), \
                mock.patch("core.mediamtx_client.subprocess.Popen", return_value=process) as popen:
            self.assertTrue(client.ensure_preview("cam1"))
            self.assertTrue(client.ensure_preview("cam1"))
        popen.assert_called_once()

    def test_existing_mediamtx_path_is_not_recreated(self):
        client = MediaMTXClient("http://mediamtx/v3/config/paths")
        config = {"source": "rtsp://camera", "sourceOnDemand": False, "rtspTransport": "tcp"}
        with mock.patch("core.mediamtx_client.requests.get") as get, \
                mock.patch("core.mediamtx_client.requests.post") as post, \
                mock.patch("core.mediamtx_client.requests.patch") as patch:
            get.return_value.status_code = 200
            get.return_value.json.return_value = config
            client.ensure_path("cam1", "rtsp://camera")
            post.assert_not_called()
            patch.assert_not_called()

    def test_changed_source_uses_patch_and_missing_path_uses_post(self):
        client = MediaMTXClient("http://mediamtx/v3/config/paths")
        with mock.patch("core.mediamtx_client.requests.get") as get, \
                mock.patch("core.mediamtx_client.requests.post") as post, \
                mock.patch("core.mediamtx_client.requests.patch") as patch:
            get.return_value.status_code = 200
            get.return_value.json.return_value = {"source": "rtsp://old"}
            client.ensure_path("cam1", "rtsp://new")
            self.assertEqual("http://mediamtx/v3/config/paths/patch/cam1", patch.call_args.args[0])
            self.assertEqual("rtsp://new", patch.call_args.kwargs["json"]["source"])
            post.assert_not_called()
            get.return_value.status_code = 404
            client.ensure_path("cam2", "rtsp://new")
            self.assertEqual("http://mediamtx/v3/config/paths/add/cam2", post.call_args.args[0])

    def test_removing_camera_uses_delete_method(self):
        with mock.patch("core.mediamtx_client.requests.delete") as delete:
            MediaMTXClient("http://mediamtx/v3/config/paths").delete_path("cam1")
            delete.assert_called_once_with("http://mediamtx/v3/config/paths/delete/cam1", timeout=3)


if __name__ == "__main__":
    unittest.main()
