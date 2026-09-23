import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from core.model_sam2 import ModelSAM2, reference_identity
from core.sam2_live_predictor import LiveSAM2Predictor


class LivePredictorTests(unittest.TestCase):
    def test_reference_identity_never_changes_on_existing_track(self):
        identity = dict(accepted=True, label='Robot_2', candidate_label='Robot_2', score=.98)
        rejected = reference_identity(identity, 'Robot_1')
        self.assertFalse(rejected['accepted'])
        self.assertEqual(rejected['reason'], 'reference_identity_mismatch')
        self.assertEqual(rejected['candidate_label'], 'Robot_2')
        self.assertEqual(rejected['expected_label'], 'Robot_1')
        self.assertTrue(identity['accepted'])
        self.assertIs(reference_identity(identity, 'robot_2'), identity)

    def test_reference_never_overrides_negative_or_ambiguous_result(self):
        for reason in ('negative_sample', 'ambiguous_identity', 'appearance_mismatch'):
            identity = dict(accepted=False, reason=reason, candidate_label='Robot_1')
            self.assertIs(reference_identity(identity, 'Robot_1'), identity)

    def predictor(self):
        output = {"pred_masks": torch.ones(1, 1, 8, 8), "object_score_logits": torch.tensor([[16.]])}
        return SimpleNamespace(get_im_features=Mock(), imgsz=[640, 640], src_shape=(720, 1280),
                               _prepare_prompts=Mock(return_value=(torch.zeros(1, 2, 2), torch.zeros(1, 2), None)),
                               track_step=Mock(return_value=output), obj_idx_set=set(), memory_bank=[])

    def test_prompt_decodes_once_and_does_not_commit_unverified_memory(self):
        predictor = self.predictor()
        image = object()
        masks, scores = LiveSAM2Predictor.infer_live(predictor, image, [[10, 10, 80, 80]])
        predictor.get_im_features.assert_called_once_with(image)
        predictor.track_step.assert_called_once()
        self.assertEqual(predictor.track_step.call_args.kwargs["obj_idx"], 0)
        self.assertEqual(masks.shape, (1, 8, 8))
        self.assertEqual(float(scores[0]), .5)
        self.assertEqual(float(predictor.live_output["object_score_logits"][0, 0]), 16.)
        self.assertEqual(predictor.memory_bank, [])
        self.assertEqual(predictor.obj_idx_set, {0})

    def test_propagation_requires_verified_memory_and_decodes_once(self):
        predictor = self.predictor()
        with self.assertRaises(RuntimeError):
            LiveSAM2Predictor.infer_live(predictor, object())
        predictor.track_step.assert_not_called()
        predictor.memory_bank.append(object())
        LiveSAM2Predictor.infer_live(predictor, object())
        predictor.track_step.assert_called_once_with()


class CameraSchedulingTests(unittest.TestCase):
    def test_parallel_workers_never_take_other_camera_memory(self):
        owner = ModelSAM2.__new__(ModelSAM2)
        owner.lock = threading.Lock()
        owner.wake = threading.Event()
        owner.camera_workers = {'first': 0, 'second': 1, 'third': 0}
        owner.pending = {'second': 10, 'first': 20, 'third': 30}
        self.assertEqual(owner._take_pending(0), ('first', 20))
        self.assertEqual(owner._take_pending(1), ('second', 10))
        self.assertEqual(owner._take_pending(1), (None, None))
        self.assertEqual(owner._take_pending(0), ('third', 30))

    def test_newest_frame_replacement_keeps_camera_turn(self):
        owner = ModelSAM2.__new__(ModelSAM2)
        owner.lock = threading.Lock()
        owner.wake = threading.Event()
        owner.pending = {"fast": 1, "slow": 2, "third": 3}
        for frame in range(4, 20):
            owner.pending["fast"] = frame
        self.assertEqual(owner._take_pending(), ("fast", 19))
        owner.pending["fast"] = 20
        self.assertEqual(owner._take_pending(), ("slow", 2))
        self.assertEqual(owner._take_pending(), ("third", 3))
        self.assertEqual(owner._take_pending(), ("fast", 20))
        self.assertEqual(owner._take_pending(), (None, None))


if __name__ == "__main__":
    unittest.main()
