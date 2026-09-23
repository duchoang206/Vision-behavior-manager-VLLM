import os
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from core.registered_mask_recovery import appearance_candidates
from core.registered_target_mask import MaskRuntime, RegisteredTargetMaskSegmenter
from core.sam2_live_predictor import LiveSAM2Predictor
from core.template_identity_tracker import TemplateIdentityCameraTracker


class RegisteredMaskRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.runtime = MaskRuntime.__new__(MaskRuntime)
        self.runtime.shared_features = False
        self.runtime.identity = SimpleNamespace(verify=mock.Mock(return_value={"accepted": True, "reason": None}))
        self.frame = np.zeros((40, 40, 3), dtype=np.uint8)
        binary = torch.zeros((1, 40, 40), dtype=torch.bool)
        binary[:, 8:20, 8:20] = True
        box = SimpleNamespace(conf=torch.tensor(.06), xyxy=torch.tensor([[8, 8, 20, 20]]))
        self.result = SimpleNamespace(masks=SimpleNamespace(data=binary), boxes=[box])
        self.predictor = mock.Mock(return_value=[self.result])
        self.predictor.memory_bank = [object()]
        self.predictor.obj_idx_set = {0}
        self.predictor.obj_id_to_idx = {0: 0}
        self.predictor.obj_idx_to_id = {0: 0}
        self.predictor._max_obj_num = 1
        self.predictor.commit_live_memory.return_value = True
        self.entry = {"predictor": self.predictor, "seeded": True, "last_mask_at": 0.0}
        self.runtime.cameras = {"cam": {"objects": {"Robot_2001": self.entry}}}
        self.descriptor = mock.patch.object(MaskRuntime, "_identity_descriptor", return_value=torch.ones(512))
        self.descriptor.start()
        self.addCleanup(self.descriptor.stop)

    def test_verified_live_frame_commits_memory_not_gallery(self):
        observation = self.runtime.track("cam", self.frame, {})["Robot_2001"]
        self.assertIn("mask", observation)
        self.predictor.commit_live_memory.assert_called_once()
        self.assertEqual("tracking", self.entry["diagnostics"]["state"])
        self.assertEqual(1, self.entry["diagnostics"]["memory_updates"])

    def test_tracking_disables_autograd_without_shared_features(self):
        with mock.patch.object(self.runtime, "_track_objects", side_effect=lambda *args: torch.is_inference_mode_enabled()):
            self.assertTrue(self.runtime.track("cam", self.frame, {}))

    def test_repeated_empty_masks_reset_and_reseed_live_frame(self):
        self.predictor.return_value = [SimpleNamespace(masks=None, boxes=[])]
        with mock.patch.dict(os.environ, {"REGISTERED_MASK_RESET_MISSES": "3"}):
            self.assertEqual({}, self.runtime.track("cam", self.frame, {}))
            self.assertTrue(self.entry["seeded"])
            self.runtime.track("cam", self.frame, {})
            lost = self.runtime.track("cam", self.frame, {})
        self.assertTrue(lost["Robot_2001"]["tracking_lost"])
        self.assertFalse(self.entry["seeded"])
        self.assertEqual([], self.predictor.memory_bank)
        self.predictor.commit_live_memory.assert_not_called()
        self.predictor.return_value = [self.result]
        observation = self.runtime.track("cam", self.frame, {"Robot_2001": [.2, .2, .3, .3]})
        self.assertIn("mask", observation["Robot_2001"])
        self.assertTrue(self.predictor.call_args.kwargs["update_memory"])
        self.assertEqual(1, self.entry["diagnostics"]["recoveries"])
        self.assertEqual(0, self.entry["misses"])

    def test_wrong_identity_never_updates_memory_or_publishes_mask(self):
        self.runtime.identity.verify.return_value = {"accepted": False, "reason": "identity_mismatch"}
        observation = self.runtime.track("cam", self.frame, {})["Robot_2001"]
        self.assertTrue(observation["identity_rejected"])
        self.assertNotIn("mask", observation)
        self.assertFalse(self.entry["seeded"])
        self.assertEqual([], self.predictor.memory_bank)
        self.predictor.commit_live_memory.assert_not_called()

    def test_fms_position_mismatch_never_commits_sam_memory(self):
        context = {"labels": {"Robot_2001": "2001"}, "calibration": {"matrix": np.eye(3).tolist()},
                   "poses": {"2001": {"position": [8, 0, 8]}}, "max_distance_m": 1.5,
                   "max_speed_mps": 1.8, "rival_margin_m": .35}
        observation = self.runtime.track("cam", self.frame, {}, context)["Robot_2001"]
        self.assertTrue(observation["identity_rejected"])
        self.assertEqual("fms_position_mismatch", observation["identity"]["reason"])
        self.assertFalse(self.entry["seeded"])
        self.predictor.commit_live_memory.assert_not_called()
        self.runtime.identity.verify.assert_not_called()

    def test_fms_match_does_not_override_wrong_appearance(self):
        context = {"labels": {"Robot_2001": "2001"}, "calibration": {"matrix": np.eye(3).tolist()},
                   "poses": {"2001": {"position": [.35, 0, .5]}}, "max_distance_m": 1.5,
                   "max_speed_mps": 1.8, "rival_margin_m": .35}
        self.runtime.identity.verify.return_value = {"accepted": False, "reason": "identity_mismatch"}
        observation = self.runtime.track("cam", self.frame, {}, context)["Robot_2001"]
        self.assertTrue(observation["identity_rejected"])
        self.assertTrue(self.entry["diagnostics"]["spatial_identity"]["accepted"])
        self.predictor.commit_live_memory.assert_not_called()

    def test_bad_geometry_is_not_committed(self):
        self.result.masks.data[:] = True
        self.assertEqual({}, self.runtime.track("cam", self.frame, {}))
        self.predictor.commit_live_memory.assert_not_called()
        self.assertEqual("mask_area", self.entry["diagnostics"]["last_rejection"])

    def test_recovery_is_bounded_without_pausing_healthy_target(self):
        sleeping = {"predictor": mock.Mock(), "seeded": False, "next_recovery_at": time.monotonic() + 10}
        ready = {"predictor": mock.Mock(return_value=[self.result]), "seeded": False}
        other = {"predictor": mock.Mock(), "seeded": False}
        self.runtime.cameras["cam"]["objects"].update(Sleeping=sleeping, Ready=ready, Other=other)
        with mock.patch.object(self.runtime, "_track_objects", return_value={}) as track:
            self.runtime.track("cam", self.frame, {label: [.1, .1, .2, .2] for label in ("Sleeping", "Ready", "Other")})
        self.assertEqual({"Robot_2001", "Ready"}, set(track.call_args.args[1]))

    def test_search_proposal_must_pass_identity_verification(self):
        self.entry.update(seeded=False, recovery_attempts=1, box_sizes=[[.3, .3]])
        self.runtime.identity.verify.return_value = {"accepted": False, "reason": "identity_ambiguous"}
        with mock.patch("core.registered_mask_recovery.appearance_candidates", return_value=[[.6, .5, .3, .3]]) as search:
            observation = self.runtime.track("cam", self.frame, {"Robot_2001": [.1, .1, .2, .2]})
        search.assert_called_once()
        self.assertEqual([[24., 20., 36., 32.]], self.predictor.call_args.kwargs["bboxes"])
        self.assertTrue(observation["Robot_2001"]["identity_rejected"])
        self.predictor.commit_live_memory.assert_not_called()

    def test_gpu_feature_proposals_locate_new_position_not_old_bbox(self):
        features = torch.zeros((1, 256, 8, 8))
        features[:, 1] = 1
        features[:, 1, 4:6, 5:7] = 0
        features[:, 0, 4:6, 5:7] = 1
        gallery = torch.zeros((1, 512))
        gallery[:, 0] = 1
        identity = SimpleNamespace(lock=threading.RLock(), gallery={"Robot_2001": gallery})
        predictor = SimpleNamespace(feat_sizes=[(8, 8)], imgsz=(40, 40),
                                    vision_feats=[features.flatten(2).permute(2, 0, 1)])
        candidates = appearance_candidates(identity, "Robot_2001", predictor, self.frame.shape, [[.25, .25]])
        self.assertEqual([.625, .5, .25, .25], candidates[0])

    def test_recovery_cycles_all_candidates_between_nearby_attempts(self):
        self.entry.update(seeded=False, box_sizes=[[.2, .2]], diagnostics={})
        candidates = [[position, .5, .15, .2] for position in (.1, .3, .5, .7)]
        nearby = [.2, .2, .3, .3]
        with mock.patch("core.registered_mask_recovery.appearance_candidates", return_value=candidates):
            attempts = [self.runtime._recovery_bbox("Robot_2001", self.entry, self.frame,
                                                   {"Robot_2001": nearby}) for _attempt in range(6)]
        self.assertEqual([nearby, *candidates[:3], nearby, candidates[3]], attempts)

    def test_verified_memory_bank_is_bounded_and_reuses_frame_features(self):
        anchor = object()
        model = SimpleNamespace(num_maskmem=7, _encode_new_memory=mock.Mock(return_value=(torch.ones(1), [torch.ones(1)])))
        predictor = SimpleNamespace(model=model, memory_bank=[anchor], vision_feats=[object()], feat_sizes=[(8, 8)])
        for _step in range(20):
            predictor.live_output = {key: torch.ones(1) for key in ("pred_masks", "pred_masks_high_res", "obj_ptr", "object_score_logits")}
            self.assertTrue(LiveSAM2Predictor.commit_live_memory(predictor, limit=3))
            self.assertLessEqual(len(predictor.memory_bank), 3)
        self.assertIs(anchor, predictor.memory_bank[0])
        self.assertIsNone(predictor.live_output)
        self.assertIs(predictor.vision_feats, model._encode_new_memory.call_args.kwargs["current_vision_feats"])

    def test_tracking_loss_keeps_registration_but_removes_stale_overlay(self):
        tracker = TemplateIdentityCameraTracker("cam", "", None)
        mask = {"polygons": [[[.2, .2], [.5, .2], [.5, .5], [.2, .5]]]}
        tracker.add_target("Robot_2001", "robot", [.2, .2, .3, .3], mask=mask, frame_image="saved-snapshot")
        target = tracker.targets["Robot_2001"]
        target["mask_observed_at"] = time.time()
        tracker.mask_motion.tracks["Robot_2001"] = {"object": {}}
        with mock.patch("core.template_identity_tracker.registered_target_mask_segmenter") as segmenter:
            segmenter.available.return_value = True
            segmenter.track.return_value = {"Robot_2001": {"tracking_lost": True}}
            self.assertEqual([], tracker._process_registered_frame(self.frame))
        self.assertIn("Robot_2001", tracker.targets)
        self.assertEqual(1, len(target["mask_samples"]))
        self.assertIsNone(target["mask"])
        self.assertFalse(target["has_matched"])
        self.assertNotIn("Robot_2001", tracker.mask_motion.tracks)

    def test_rejected_identity_is_not_counted_as_an_observed_mask(self):
        segmenter = RegisteredTargetMaskSegmenter()
        self.addCleanup(segmenter.close)
        targets = [{"label": "Robot_2001", "category": "robot"}]
        with mock.patch.object(segmenter, "available", return_value=True), \
                mock.patch.object(segmenter, "_request", return_value={"Robot_2001": {"identity_rejected": True}}):
            segmenter.track("cam", self.frame, targets, {})
        diagnostics = segmenter.status("cam")["camera"]
        self.assertEqual([], diagnostics["last_observed_labels"])
        self.assertEqual(1, diagnostics["empty_frames"])


if __name__ == "__main__":
    unittest.main()
