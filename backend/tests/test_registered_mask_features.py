import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

BACKEND_DIR = str(Path(__file__).resolve().parents[1])
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from core.registered_target_mask import MaskRuntime


class FramePredictor:
    def __init__(self, model):
        self.model = model
        self.im = None
        self.backbone_out = None
        self.memory_bank = [object()]
        self.inputs = []
        self.fail = False

    def reset_image(self):
        self.im = None

    def setup_source(self, source):
        self.shape = source.shape

    def preprocess(self, images):
        return images[0].copy() if self.im is None else self.im

    def __call__(self, source, **prompts):
        image = self.preprocess([source])
        backbone = self.backbone_out
        if backbone is None:
            backbone = self.model.forward_image(image)
        self.inputs.append((image, backbone))
        if self.fail:
            raise RuntimeError('inference failed')
        if prompts.get('update_memory'):
            self.memory_bank.append(source.copy())
        return [SimpleNamespace(masks=None, boxes=[])]


class RegisteredMaskFeatureTests(unittest.TestCase):
    def setUp(self):
        self.torch_patch = mock.patch.dict(sys.modules, {'torch': SimpleNamespace(inference_mode=nullcontext)})
        self.torch_patch.start()
        self.addCleanup(self.torch_patch.stop)
        self.model = SimpleNamespace(forward_image=mock.Mock(side_effect=lambda image: {'image': image.copy()}))
        self.runtime = MaskRuntime.__new__(MaskRuntime)
        self.runtime.shared_features = True
        self.predictors = [FramePredictor(self.model) for _ in range(3)]
        self.runtime.cameras = {'cam': {'objects': {
            label: {'predictor': predictor, 'seeded': True}
            for label, predictor in zip(['Robot_1', 'Rack_1', 'Robot_2'], self.predictors)
        }}}
        self.frame = np.zeros((30, 50, 3), dtype=np.uint8)

    def assert_cache_cleared(self):
        for predictor in self.predictors:
            self.assertIsNone(predictor.im)
            self.assertIsNone(predictor.backbone_out)

    def test_one_encoding_per_frame_with_separate_label_memory(self):
        memories = [predictor.memory_bank for predictor in self.predictors]
        self.runtime.track('cam', self.frame, {})
        self.assertEqual(1, self.model.forward_image.call_count)
        first_image, first_backbone = self.predictors[0].inputs[0]
        for predictor, memory in zip(self.predictors, memories):
            image, backbone = predictor.inputs[0]
            self.assertIs(first_image, image)
            self.assertIs(first_backbone, backbone)
            self.assertIs(memory, predictor.memory_bank)
            self.assertEqual(1, len(memory))
        self.assertEqual(3, len({id(memory) for memory in memories}))
        self.assert_cache_cleared()

    def test_new_frame_same_buffer_never_reuses_old_features(self):
        self.runtime.track('cam', self.frame, {})
        self.frame[:] = 240
        self.runtime.track('cam', self.frame, {})
        self.assertEqual(2, self.model.forward_image.call_count)
        for predictor in self.predictors:
            first, second = predictor.inputs
            self.assertIsNot(first[1], second[1])
            self.assertEqual(0, int(first[1]['image'][0, 0, 0]))
            self.assertEqual(240, int(second[1]['image'][0, 0, 0]))
        self.assert_cache_cleared()

    def test_different_camera_and_resolution_get_new_features(self):
        other_predictor = FramePredictor(self.model)
        self.runtime.cameras['other'] = {'objects': {
            'Other_1': {'predictor': other_predictor, 'seeded': True},
            'Other_2': {'predictor': FramePredictor(self.model), 'seeded': True},
        }}
        self.runtime.track('cam', self.frame, {})
        other_frame = np.full((50, 30, 3), 80, dtype=np.uint8)
        self.runtime.track('other', other_frame, {})
        self.assertEqual(2, self.model.forward_image.call_count)
        self.assertEqual(other_frame.shape, other_predictor.inputs[0][0].shape)
        self.assertEqual(80, int(other_predictor.inputs[0][1]['image'][0, 0, 0]))
        self.assertIsNot(self.predictors[0].inputs[0][1], other_predictor.inputs[0][1])

    def test_cache_cleared_on_failure_before_next_frame(self):
        self.predictors[1].fail = True
        with self.assertRaisesRegex(RuntimeError, 'inference failed'):
            self.runtime.track('cam', self.frame, {})
        self.assert_cache_cleared()
        self.predictors[1].fail = False
        self.runtime.track('cam', np.ones_like(self.frame), {})
        self.assertEqual(2, self.model.forward_image.call_count)
        self.assert_cache_cleared()

    def test_cache_cleared_when_encoder_fails(self):
        self.model.forward_image.side_effect = RuntimeError('encoder failed')
        with self.assertRaisesRegex(RuntimeError, 'encoder failed'):
            self.runtime.track('cam', self.frame, {})
        self.assert_cache_cleared()
        self.assertTrue(all(not predictor.inputs for predictor in self.predictors))

    def test_unseeded_labels_do_not_trigger_unneeded_encoding(self):
        for entry in self.runtime.cameras['cam']['objects'].values():
            entry['seeded'] = False
        self.assertEqual({}, self.runtime.track('cam', self.frame, {}))
        self.model.forward_image.assert_not_called()
        self.runtime.track('cam', self.frame, {'Robot_1': [.1, .1, .2, .2]})
        self.assertEqual(2, self.model.forward_image.call_count)
        self.assertTrue(self.runtime.cameras['cam']['objects']['Robot_1']['seeded'])
        self.assertFalse(self.predictors[1].inputs)
        self.assertFalse(self.predictors[2].inputs)

    def test_legacy_seeding_shares_frame_without_sharing_memory(self):
        entries = self.runtime.cameras['cam']['objects']
        entries['Robot_1']['seeded'] = False
        self.runtime.track('cam', self.frame, {'Robot_1': [.1, .1, .2, .2]})
        self.assertEqual(1, self.model.forward_image.call_count)
        self.assertEqual([2, 1, 1], [len(predictor.memory_bank) for predictor in self.predictors])
        self.assert_cache_cleared()

    def test_disabling_sharing_uses_original_inference_path(self):
        self.runtime.shared_features = False
        self.runtime.track('cam', self.frame, {})
        self.assertEqual(3, self.model.forward_image.call_count)
        self.assertEqual([1, 1, 1], [len(predictor.inputs) for predictor in self.predictors])
        self.assert_cache_cleared()


if __name__ == '__main__':
    unittest.main()
