import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch

import torch

from core.model_label_metric import ModelLabelMetric
from core.model_label_prompts import LabelPromptTracks
from core.model_label_session import ModelLabelSession
from core.model_sam2 import CudaMaskRuntime, ModelSAM2
from core.model_track_masks import TrackMaskCache


def sample(identifier='view', label='Robot_2001', vector=None, negative=False):
    return dict(id=identifier, label=label, camera_id='cam', category='robot', class_name='robot',
                negative=negative, signature='test', vector=vector or torch.eye(1, 512)[0].tolist(),
                frame_id=20, bbox=[.1, .2, .3, .4], mask={'polygons': [[[.1, .2], [.4, .2], [.4, .6]]]})


class GalleryContinuityTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.metric = ModelLabelMetric(self.directory.name, 'test', device='cpu', start=False)

    def test_reference_identity_wins_when_other_view_is_only_slightly_closer(self):
        query = torch.eye(1, 512)[0]
        expected = query * .80
        expected[1] = (1 - .80 ** 2) ** .5
        rival = query * .83
        rival[2] = (1 - .83 ** 2) ** .5
        self.metric.load(dict(revision=1, samples=[sample(vector=expected.tolist()),
                         sample('other', 'Robot_2', rival.tolist())]))
        result = self.metric.verify(query, 'robot', expected_label='Robot_2001')
        self.assertTrue(result['accepted'])
        self.assertEqual(result['label'], 'Robot_2001')
        self.assertEqual(result['match_source'], 'user_reference')
        self.assertFalse(self.metric.verify(query, 'robot')['accepted'])

    def test_reference_never_overrides_explicit_negative_or_stronger_other_object(self):
        expected = torch.eye(1, 512)[0] * .80
        expected[1] = .60
        for negative in (True, False):
            self.metric.load(dict(revision=1, samples=[sample(vector=expected.tolist()),
                             sample('other', 'Other', negative=negative)]))
            result = self.metric.verify(torch.eye(1, 512)[0], 'robot', expected_label='Robot_2001')
            self.assertFalse(result['accepted'])
            self.assertEqual(result['reason'], 'negative_sample' if negative else 'reference_identity_mismatch')

    def test_added_positive_views_keep_bounded_identity_cache_but_deletion_revokes(self):
        first, second = sample(), sample('second')
        self.metric.load(dict(revision=1, samples=[first]))
        self.metric.load(dict(revision=2, samples=[first, second]))
        self.assertEqual(self.metric.policy('robot')['continuity_revision'], 1)
        self.metric.load(dict(revision=3, samples=[second]))
        self.assertEqual(self.metric.policy('robot')['continuity_revision'], 3)
        self.metric.load(dict(revision=4, samples=[second, sample('bad', 'negative', negative=True)]))
        self.assertEqual(self.metric.policy('robot')['continuity_revision'], 4)
        self.metric.load(dict(revision=5, samples=[second, sample('bad', 'negative', negative=True), sample('other', 'Robot_2')]))
        self.assertEqual(self.metric.policy('robot')['continuity_revision'], 5)

    def test_new_reference_keeps_live_memory_and_deleted_reference_resets(self):
        runtime = CudaMaskRuntime.__new__(CudaMaskRuntime)
        runtime.metric = MagicMock()
        runtime.metric.reference_ids.return_value = {'first', 'second'}
        runtime.reference_ids = {'first'}
        runtime.entries = {'track': object()}
        runtime.reference_choices = {'old': object()}
        runtime.rejected_tracks = {}
        runtime.references = MagicMock()
        runtime._sync_references(2)
        self.assertIn('track', runtime.entries)
        self.assertEqual(runtime.reference_choices, {})
        runtime.metric.reference_ids.return_value = {'second'}
        runtime._sync_references(3)
        self.assertEqual(runtime.entries, {})


class SaveActivationTests(unittest.TestCase):
    def setUp(self):
        self.saved = sample()
        self.owner = ModelSAM2.__new__(ModelSAM2)
        self.owner.metric = MagicMock()
        self.owner.metric.saved_sample.return_value = self.saved
        self.owner.metric.prompt_samples.return_value = [self.saved]
        self.owner.metric.policy.return_value = dict(required=True, revision=1, version=1, error=None)
        self.owner.labels = ModelLabelSession(MagicMock(), clock=lambda: 0)
        self.owner.labels.previews['view'] = dict(camera_id='cam', expires=100, sample=dict(
            captured_at=1000, frame_info=dict(model_id='model', generation='first')))
        self.owner.prompt_tracks = LabelPromptTracks()
        self.owner.cache = TrackMaskCache()
        self.owner.lock = threading.Lock()
        self.owner.wake = threading.Event()
        self.owner.revoked = {}

    @patch('core.model_sam2.time.time', return_value=1.1)
    def test_save_initializes_exact_user_mask_without_yolo(self, clock):
        result = self.owner.activate_label('cam', 'view')
        self.assertEqual(result['mode'], 'mask_initialized')
        self.assertFalse(result['requires_yolo_detection'])
        live = self.owner.attach('cam', [], 21, 1120)
        self.assertEqual(len(live), 1)
        self.assertEqual(live[0]['mask']['polygons'], self.saved['mask']['polygons'])
        self.assertEqual(live[0]['mask']['observed_at'], 1000)
        self.assertEqual(live[0]['label'], 'Robot_2001')

    @patch('core.model_sam2.time.time', return_value=10.)
    def test_old_annotation_starts_tracking_without_faking_fresh_position(self, clock):
        result = self.owner.activate_label('cam', 'view')
        self.assertEqual(result['mode'], 'tracking_queued')
        self.assertTrue(self.owner.wake.is_set())
        self.assertEqual(self.owner.attach('cam', [], 30, 10010), [])

    def test_deleted_or_untrusted_receipt_cannot_activate(self):
        self.owner.metric.saved_sample.return_value = None
        with self.assertRaises(ValueError):
            self.owner.activate_label('cam', 'view')
        self.owner.metric.saved_sample.return_value = self.saved
        with self.assertRaises(ValueError):
            self.owner.activate_label('other', 'view')

    @patch('core.model_sam2.time.time', return_value=1.1)
    def test_mask_can_publish_before_remaining_objects_finish(self, clock):
        target = self.owner.prompt_tracks.candidates('cam', [self.saved],
                 dict(model_id='model', generation='first'), 1000)[0]
        identity = dict(accepted=True, label='Robot_2001', class_name='robot', score=.9,
                        revision=1, version=1, match_source='user_reference')
        self.owner._publish_result('cam', 20, 1000, [], (target, self.saved['mask'], identity))
        self.assertEqual(len(self.owner.attach('cam', [], 21, 1100)), 1)
        self.owner._publish_result('cam', 21, 1050, [], (target, None, dict(identity, accepted=False)))
        self.assertEqual(self.owner.attach('cam', [], 22, 1100), [])
        self.assertIn(('cam', target['id']), self.owner.revoked)

    @patch('core.model_sam2.time.time', return_value=2.)
    def test_late_inference_does_not_refresh_mask_age(self, clock):
        target = self.owner.prompt_tracks.candidates('cam', [self.saved],
                 dict(model_id='model', generation='first'), 1000)[0]
        self.owner._publish_result('cam', 20, 1000, [], (target, self.saved['mask'], None))
        self.assertEqual(self.owner.cache.entries, {})

    @patch('core.model_sam2.time.time', return_value=1.1)
    def test_old_inference_cannot_revoke_new_user_correction(self, clock):
        self.owner.activate_label('cam', 'view')
        target = self.owner.attach('cam', [], 21, 1100)[0]
        self.owner._publish_result('cam', 19, 990, [], (target, None, dict(accepted=False, reason='appearance_mismatch')))
        self.assertEqual(len(self.owner.attach('cam', [], 22, 1110)), 1)
        self.assertNotIn(('cam', target['id']), self.owner.revoked)


if __name__ == '__main__':
    unittest.main()
