import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import torch

from core.model_label_metric import ModelLabelMetric
from core.model_label_prompts import LabelPromptTracks, latest_prompts
from core.model_label_store import ModelLabelStore
from core.model_sam2 import ModelSAM2
from core.model_track_masks import TrackMaskCache


def saved_sample(identifier="view", **values):
    return dict(id=identifier, camera_id="cam", label="Robot_2001", class_name="Robot_2001", category="robot",
                negative=False, signature="test", vector=torch.eye(1, 512)[0].tolist(), frame_id=5,
                bbox=[.1, .2, .3, .4], mask={"polygons": [[[.1, .2], [.4, .2], [.4, .6]]]}, **values)


class LabelPromptTests(unittest.TestCase):
    def setUp(self):
        self.sample = saved_sample()
        self.tracks = LabelPromptTracks()
        self.frame = dict(model_id="model", generation="generation")
        self.identity = dict(accepted=True, label="Robot_2001", class_name="Robot_2001", revision=1, version=1,
                             score=.99, match_source="user_gallery")

    def candidate(self):
        return self.tracks.candidates("cam", [self.sample], self.frame, 1000)[0]

    def test_latest_coordinates_are_scoped_by_camera_and_label(self):
        newer = dict(self.sample, id="new", bbox=[.4, .2, .2, .3])
        other = dict(self.sample, id="other", camera_id="other")
        result = latest_prompts([self.sample, other, newer])
        self.assertEqual(result["cam"]["robot_2001"]["bbox"], newer["bbox"])
        self.assertEqual(result["other"]["robot_2001"]["id"], "other")
        self.assertEqual(latest_prompts([dict(newer, negative=True)]), {})
        self.assertEqual(latest_prompts([dict(newer, bbox=[float("nan"), 0, 1, 1])]), {})
        self.assertEqual(latest_prompts([dict(newer, mask="bad")]), {})

    def test_coordinates_alone_never_publish_a_track(self):
        target = self.candidate()
        self.assertEqual([target[key] for key in ("x", "y", "w", "h")], self.sample["bbox"])
        self.assertEqual(self.tracks.live("cam", [self.sample], 1001), [])
        self.tracks.observe("cam", target, self.sample["mask"], dict(self.identity, label="Robot_1"), 1000, 20)
        self.assertEqual(self.tracks.live("cam", [self.sample], 1001), [])
        self.assertEqual(self.tracks.candidates("cam", [self.sample], self.frame, 1100), [])

    def test_live_coordinates_move_and_expire_without_timestamp_refresh(self):
        target = dict(self.candidate(), x=.12)
        self.tracks.observe("cam", target, self.sample["mask"], self.identity, 1000, 20)
        live = self.tracks.live("cam", [self.sample], 1050)[0]
        self.assertEqual(live["x"], .12)
        self.assertEqual(live["observed_at"], 1000)
        self.assertEqual(live["frame_id"], 20)
        self.assertEqual(len(self.tracks.live("cam", [self.sample], 1300)), 1)
        self.assertEqual(self.tracks.live("cam", [self.sample], 1351), [])
        self.assertEqual(self.tracks.live("other", [self.sample], 1050), [])

    def test_transient_missing_mask_keeps_moving_memory_not_stale_display(self):
        target = dict(self.candidate(), x=.38)
        self.tracks.observe("cam", target, self.sample["mask"], self.identity, 1000, 20)
        self.tracks.observe("cam", target, None, None, 1100, 21)
        candidates = self.tracks.candidates("cam", [self.sample], self.frame, 1300)
        self.assertEqual(candidates[0]["x"], .38)
        self.assertEqual(candidates[0]["observed_at"], 1000)
        self.assertEqual(self.tracks.live("cam", [self.sample], 1351), [])
        self.tracks.observe("cam", candidates[0], self.sample["mask"], self.identity, 1310, 22)
        self.assertEqual(len(self.tracks.live("cam", [self.sample], 1320)), 1)

    def test_generation_change_never_reuses_old_tracking_memory(self):
        target = dict(self.candidate(), x=.38)
        self.tracks.observe("cam", target, self.sample["mask"], self.identity, 1000, 20)
        self.assertEqual(self.tracks.candidates("cam", [self.sample], dict(self.frame, generation="new"), 1300), [])
        restarted = self.tracks.candidates("cam", [self.sample], dict(self.frame, generation="new"), 2100)
        self.assertEqual(restarted[0]["x"], self.sample["bbox"][0])

    def test_detector_label_blocks_duplicate_coordinate_prompt(self):
        self.assertFalse(self.tracks.needs_frame("cam", [self.sample], 1000, ["Robot_2001"]))
        self.assertEqual(self.tracks.candidates("cam", [self.sample], self.frame, 1000, ["Robot_2001"]), [])

    def test_unverified_or_revoked_yolo_guess_cannot_block_label(self):
        cache = TrackMaskCache()
        target = self.candidate()
        target.pop("label_prompt_id")
        policies = {"robot": dict(revision=1, version=1, error=None)}
        self.assertEqual(cache.verified_labels("cam", [target], 1000, policies), set())
        cache.store("cam", target, self.sample["mask"], 20, 1000, self.identity)
        self.assertEqual(cache.verified_labels("cam", [target], 1050, policies), {"robot_2001"})
        policies["robot"]["revision"] = 2
        self.assertEqual(cache.verified_labels("cam", [target], 1060, policies), set())
        policies["robot"]["revision"] = 1
        self.assertEqual(cache.verified_labels("cam", [target], 1701, policies), set())

    def test_delete_or_rejection_removes_prompt_immediately(self):
        target = self.candidate()
        self.tracks.observe("cam", target, self.sample["mask"], self.identity, 1000, 20)
        self.assertEqual(self.tracks.live("cam", [], 1050), [])
        self.tracks.observe("cam", target, None, dict(self.identity, accepted=False), 1060, 21)
        self.assertEqual(self.tracks.live("cam", [self.sample], 1061), [])
        self.tracks.candidates("cam", [], self.frame, 1062)
        self.assertEqual(self.tracks.attempts, {})

    def test_monitor_only_attaches_verified_fresh_prompt(self):
        owner = ModelSAM2.__new__(ModelSAM2)
        owner.cache, owner.prompt_tracks, owner.lock = TrackMaskCache(), self.tracks, threading.Lock()
        owner.metric = MagicMock()
        owner.metric.prompt_samples.return_value = [self.sample]
        owner.metric.policy.return_value = dict(required=True, revision=1, version=1, error=None)
        target = self.candidate()
        self.assertEqual(owner.attach("cam", [], 20, 1000), [])
        self.tracks.observe("cam", target, self.sample["mask"], self.identity, 1000, 20)
        owner.cache.store("cam", target, self.sample["mask"], 20, 1000, self.identity)
        result = owner.attach("cam", [], 21, 1040)[0]
        self.assertEqual(result["label"], "Robot_2001")
        self.assertEqual(result["mask"]["source"], "sam2_live")
        self.assertEqual(result["mask"]["attached_at"], 1000)
        self.assertEqual(owner.attach("cam", [], 30, 1360), [])

    def test_add_view_keeps_track_identity_position_and_memory(self):
        target = dict(self.candidate(), x=.32)
        self.tracks.observe('cam', target, self.sample['mask'], self.identity, 1000, 20)
        new_view = dict(self.sample, id='second_view', bbox=[.6, .1, .2, .2])
        self.assertEqual(self.tracks.live('cam', [new_view], 1100)[0]['x'], .32)
        continued = self.tracks.candidates('cam', [new_view], self.frame, 1120)[0]
        self.assertEqual(continued['id'], target['id'])
        self.assertEqual(continued['x'], .32)
        self.assertEqual(continued['label_prompt_id'], 'second_view')

    def test_bound_nvdcf_updates_position_not_mask_evidence_age(self):
        target = self.candidate()
        detection = dict(target, id=12, local_id='12', class_id=0, label_prompt_id=None, detected_at=1000)
        self.tracks.observe('cam', target, self.sample['mask'], self.identity, 1000, 20, [detection])
        moved = dict(detection, x=.18, detected_at=1460)
        result = self.tracks.live('cam', [self.sample], 1500, detections=[moved])[0]
        self.assertAlmostEqual(result['x'], .18)
        self.assertEqual(result['positioned_at'], 1500)
        cache = TrackMaskCache()
        cache.store('cam', target, self.sample['mask'], 20, 1000, self.identity)
        cache.attach('cam', [result], 30, 1500)
        self.assertEqual(result['mask']['observed_at'], 1000)
        self.assertEqual(result['mask']['attached_at'], 1500)
        self.assertEqual(self.tracks.live('cam', [self.sample], 1701, detections=[dict(moved, detected_at=1700)]), [])

    def test_bridge_does_not_bind_to_replacement_or_stale_detector(self):
        target = self.candidate()
        detection = dict(target, id=12, class_id=0, label_prompt_id=None, detected_at=1000)
        self.tracks.observe('cam', target, self.sample['mask'], self.identity, 1000, 20, [detection])
        replacement = dict(detection, id=13, detected_at=1360)
        self.assertEqual(self.tracks.live('cam', [self.sample], 1400, detections=[replacement]), [])
        self.assertEqual(self.tracks.live('cam', [self.sample], 1400, detections=[detection]), [])
        self.tracks.observe('cam', target, self.sample['mask'], self.identity, 1410, 30, [replacement])
        self.assertEqual(self.tracks.live('cam', [self.sample], 1780, detections=[dict(replacement, detected_at=1780)]), [])

    def test_recovery_after_display_gap_keeps_mask_cache(self):
        target = self.candidate()
        cache = TrackMaskCache()
        cache.store('cam', target, self.sample['mask'], 20, 1000, self.identity)
        cache.attach('cam', [], 21, 1400)
        cache.attach('cam', [target], 22, 1450)
        self.assertIsNotNone(target['mask'])
        cache.reject('cam', target)
        cache.attach('cam', [target], 23, 1460)
        self.assertIsNone(target['mask'])

    def test_missing_bound_detector_cannot_double_apply_old_motion(self):
        target = self.candidate()
        detection = dict(target, id=12, class_id=0, label_prompt_id=None, detected_at=1000)
        self.tracks.observe('cam', target, self.sample['mask'], self.identity, 1000, 20, [detection])
        self.tracks.observe('cam', dict(target, x=.18), self.sample['mask'], self.identity, 1100, 21, [])
        moved = dict(detection, x=.18, detected_at=1150)
        result = self.tracks.live('cam', [self.sample], 1150, detections=[moved])[0]
        self.assertEqual(result['x'], .18)
        self.assertNotIn('positioned_at', result)

    def test_new_sample_result_replaces_old_sample_state(self):
        target = self.candidate()
        self.tracks.observe('cam', target, self.sample['mask'], self.identity, 1000, 20)
        newer = dict(target, label_prompt_id='second', x=.2)
        self.tracks.observe('cam', newer, self.sample['mask'], self.identity, 1100, 21)
        self.tracks.observe('cam', target, self.sample['mask'], self.identity, 1050, 20)
        self.assertEqual(len(self.tracks.entries), 1)
        self.assertEqual(self.tracks.live('cam', [self.sample], 1120)[0]['x'], .2)

    def test_publish_includes_saved_geometry_without_modifying_it(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ModelLabelStore(MagicMock())
            store.root = Path(directory)
            cursor = MagicMock()
            store.transaction = MagicMock()
            store.transaction.return_value.__enter__.return_value = cursor
            cursor.fetchone.return_value = dict(revision=5, require_labels=False)
            cursor.fetchall.return_value = [self.sample]
            document = json.loads(store.publish("a" * 32).read_text())
            self.assertEqual(document["samples"][0], self.sample)
            query = cursor.execute.call_args_list[-1].args[0]
            self.assertIn("camera_id,frame_id,bbox,mask", query)

    def test_reload_acknowledges_geometry_before_background_training(self):
        with tempfile.TemporaryDirectory() as directory:
            metric = ModelLabelMetric(directory, "test", device="cpu", start=False)
            release = threading.Event()
            metric.fit = lambda: release.wait(2)
            try:
                Path(directory, "gallery.json").write_text(json.dumps(dict(revision=2, samples=[self.sample])))
                metric.start()
                receipt = metric.reload(2, timeout=1)
                self.assertEqual(receipt["mode"], "applied")
                self.assertEqual(receipt["prompt_count"], 1)
                self.assertEqual(metric.prompt_samples("cam")[0]["bbox"], self.sample["bbox"])
                self.assertTrue(metric.verify(self.sample["vector"], "robot")["accepted"])
            finally:
                release.set()
                metric.stop()


if __name__ == "__main__":
    unittest.main()
