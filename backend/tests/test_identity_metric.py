import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from core.identity_metric import IdentityMetric, masked_descriptor, triplet_loss
from core.template_identity_tracker import TemplateIdentityCameraTracker
from src.controller.registry import TargetRegistry


class IdentityMetricTests(unittest.TestCase):
    def test_triplet_penalizes_wrong_identity_and_has_gradients(self):
        features = torch.tensor([[1., 0.], [0., 1.], [1., .01]], requires_grad=True)
        loss = triplet_loss(features, torch.tensor([0, 0, 1]))
        self.assertGreater(float(loss.detach()), .5)
        loss.backward()
        self.assertTrue(torch.isfinite(features.grad).all())

    def test_descriptor_ignores_letterbox_background(self):
        features = torch.ones((1, 256, 8, 8))
        mask = torch.ones((40, 80))
        reference = masked_descriptor(features, mask, (40, 80, 3), (80, 80))
        features[:, :, 4:, :] = 500
        actual = masked_descriptor(features, mask, (40, 80, 3), (80, 80))
        self.assertTrue(torch.allclose(reference, actual))

    def test_metric_trains_and_restores_but_waits_without_negatives(self):
        with tempfile.TemporaryDirectory() as directory:
            metric = IdentityMetric(directory, device='cpu', auto_train=False)
            vector = torch.zeros(512); vector[0] = 1
            side = vector.clone(); side[1] = .4
            rival = vector.clone(); rival[2] = .5
            metric.learn('Robot_A', 'cam1', 'front', vector)
            metric.learn('Robot_A', 'cam2', 'side', side)
            self.assertFalse(metric.fit(steps=3))
            metric.learn('Robot_B', 'cam1', 'front', rival)
            self.assertTrue(metric.fit(steps=30))
            self.assertGreater(metric.revision, 0)
            self.assertTrue(metric.verify('Robot_A', vector)['accepted'])
            self.assertFalse(metric.verify('Robot_A', rival)['accepted'])
            restored = IdentityMetric(directory, device='cpu', auto_train=False)
            self.assertEqual(metric.revision, restored.revision)
            self.assertEqual(3, restored.status()['samples'])
            self.assertFalse(restored.verify('Robot_A', rival)['accepted'])
            restored.forget('Robot_A', 'cam1')
            self.assertEqual(1, restored.status()['labels']['Robot_A'])

    def test_registration_archive_exceeds_sixteen_and_survives_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = TargetRegistry(str(Path(directory) / 'registry.json'))
            vector = [1.] + [0.] * 511
            for index in range(25):
                registry.register_target('Robot_A', vector, 'cam1', category='robot',
                                         crop_image=f'angle-{index}', bbox=[.1, .1, .2, .2])
            restored = TargetRegistry(str(registry.persist_path))
            self.assertEqual(25, restored.get_all_targets()[0]['samples_count'])
            all_samples = restored.get_template_samples('Robot_A', 'cam1', limit=None)
            self.assertEqual(25, len(all_samples))
            self.assertEqual('angle-0', all_samples[0]['crop_image'])
            self.assertEqual(16, len(restored.get_template_samples('Robot_A', 'cam1')))
            self.assertEqual(25, len(list((Path(directory) / 'registered_samples').glob('*/*.json'))))
            persisted = json.loads(registry.persist_path.read_text())
            self.assertTrue(all('frame_image' not in item for item in persisted['cam1::Robot_A']['crops']))

    def test_identity_rejection_clears_hold_and_optical_flow(self):
        tracker = TemplateIdentityCameraTracker('cam1', 'rtsp://test', None)
        mask = {'polygons': [[[.1, .1], [.3, .1], [.3, .3], [.1, .3]]]}
        tracker.add_target('Robot_A', 'robot', [.1, .1, .2, .2], mask=mask, frame_image='test')
        tracker.targets['Robot_A']['mask_observed_at'] = time.time()
        tracker.mask_motion.tracks['Robot_A'] = {'object': {}}
        with mock.patch('core.template_identity_tracker.registered_target_mask_segmenter') as segmenter:
            segmenter.available.return_value = True
            segmenter.track.return_value = {'Robot_A': {'identity_rejected': True, 'identity': {'reason': 'identity_mismatch'}}}
            objects = tracker._process_registered_frame(np.zeros((100, 100, 3), np.uint8))
        self.assertEqual([], objects)
        self.assertNotIn('Robot_A', tracker.mask_motion.tracks)
        self.assertIsNone(tracker.targets['Robot_A']['mask'])

    def test_failed_registry_save_does_not_change_active_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = TargetRegistry(str(Path(directory) / 'registry.json'))
            vector = [1.] + [0.] * 511
            registry.register_target('Robot_A', vector, 'cam', category='robot')
            with mock.patch.object(registry, 'save_to_disk', side_effect=RuntimeError('SSD full')):
                with self.assertRaises(RuntimeError):
                    registry.register_target('Robot_A', vector, 'cam', category='robot')
            self.assertEqual(1, registry.get_all_targets()[0]['samples_count'])


if __name__ == '__main__':
    unittest.main()
