import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from core.model_label_metric import ModelLabelMetric
from core.model_reference_memory import LabelReferenceMemory, polygon_bitmap
from core.sam2_live_predictor import LiveSAM2Predictor


class ReferenceMemoryTests(unittest.TestCase):
    def test_polygon_rasterization_preserves_holes(self):
        outer = [[0, 0], [1, 0], [1, 1], [0, 1]]
        hole = [[.25, .25], [.75, .25], [.75, .75], [.25, .75]]
        bitmap = polygon_bitmap([outer, hole], (8, 8), 'cpu')
        self.assertEqual(int(bitmap.sum()), 48)
        self.assertTrue(bool(bitmap[0, 0]))
        self.assertFalse(bool(bitmap[4, 4]))

    def test_reference_mask_commits_user_shape_not_box(self):
        mask = torch.ones(1, 1, 8, 8)
        outputs = (None, None, None, mask, mask, torch.ones(1, 4), torch.ones(1, 1))
        predictor = SimpleNamespace(vision_feats=[torch.ones(4, 1, 4)], feat_sizes=[(2, 2)],
            high_res_features=[], model=SimpleNamespace(_use_mask_as_output=Mock(return_value=outputs)),
            obj_idx_set=set(), commit_live_memory=Mock())
        LiveSAM2Predictor.seed_reference_mask(predictor, mask)
        self.assertIs(predictor.model._use_mask_as_output.call_args.args[0], mask)
        self.assertIs(predictor.live_output['pred_masks'], mask)
        predictor.commit_live_memory.assert_called_once_with(limit=1)

    def test_memory_keeps_all_reference_anchors_and_latest_live(self):
        anchors = [dict(index=index) for index in range(2)]
        predictor = SimpleNamespace(memory_bank=anchors + [dict(index=index) for index in range(2, 5)],
            vision_feats=[], feat_sizes=[], model=SimpleNamespace(num_maskmem=7,
                _encode_new_memory=Mock(return_value=(torch.ones(1), [torch.ones(1)]))),
            live_output={key: torch.ones(1) for key in
                ('pred_masks', 'pred_masks_high_res', 'obj_ptr', 'object_score_logits')})
        LiveSAM2Predictor.commit_live_memory(predictor, limit=5, reference_count=2)
        self.assertEqual(len(predictor.memory_bank), 5)
        self.assertEqual(predictor.memory_bank[:2], anchors)
        self.assertEqual(predictor.memory_bank[2]['index'], 3)

    def test_reference_memory_matches_half_precision_encoder(self):
        mask = torch.ones(1, 1, 8, 8)
        outputs = (None, None, None, mask, mask, torch.ones(1, 4), torch.ones(1, 1))
        predictor = SimpleNamespace(vision_feats=[torch.ones(4, 1, 4).half()], feat_sizes=[(2, 2)],
            high_res_features=[], model=SimpleNamespace(_use_mask_as_output=Mock(return_value=outputs)),
            obj_idx_set=set(), commit_live_memory=Mock())
        LiveSAM2Predictor.seed_reference_mask(predictor, mask)
        self.assertTrue(all(value.dtype == torch.float16 for value in predictor.live_output.values()))

    def test_cache_is_bounded_and_deleted_references_are_pruned(self):
        cache = LabelReferenceMemory('/tmp', capacity=2)
        cache._encode = Mock(side_effect=lambda runtime, sample: dict(id=sample['id']))
        for identifier in ['first', 'second', 'third']:
            cache.get(None, dict(id=identifier))
        self.assertEqual(list(cache.entries), ['second', 'third'])
        cache.get(None, dict(id='third'))
        self.assertEqual(cache._encode.call_count, 3)
        cache.prune({'third'})
        self.assertEqual(list(cache.entries), ['third'])

    def test_reference_selection_uses_distinct_same_camera_views(self):
        with tempfile.TemporaryDirectory() as directory:
            metric = ModelLabelMetric(directory, 'test', device='cpu', start=False)
            vectors = torch.eye(512)
            def sample(identifier, camera, vector, negative=False):
                return dict(id=identifier, label='Robot_1', category='robot', class_name='robot', negative=negative,
                            camera_id=camera, signature='test', vector=vector.tolist(), mask={'polygons': [[[0, 0], [1, 0], [1, 1]]]})
            metric.load(dict(revision=1, samples=[sample('old', 'cam', vectors[1]),
                sample('remote', 'other', vectors[2]), sample('duplicate', 'cam', vectors[0]),
                sample('negative', 'cam', vectors[3], True), sample('new', 'cam', vectors[0])]))
            self.assertEqual([item['id'] for item in metric.reference_samples('Robot_1', 'cam', 2)], ['new', 'old'])
            self.assertNotIn('negative', metric.reference_ids())
            metric.load(dict(revision=2, samples=[]))
            self.assertEqual(metric.reference_samples('Robot_1', 'cam'), [])


if __name__ == '__main__':
    unittest.main()
