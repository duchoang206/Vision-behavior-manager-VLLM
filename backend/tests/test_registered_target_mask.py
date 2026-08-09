import sys
import unittest
import base64
import tempfile
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

BACKEND_DIR = str(Path(__file__).resolve().parents[1])
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from core.registered_target_mask import RegisteredTargetMaskSegmenter, advance_mask, eligible_target, mask_bitmap, mask_payload, validate_mask
from core.template_identity_tracker import TemplateIdentityCameraTracker
from core.metadata_fusion import MetadataFusion
from src.controller.registry import TargetRegistry

def encode(image):
    return "data:image/png;base64," + base64.b64encode(cv2.imencode('.png', image)[1].tobytes()).decode()


class RegisteredTargetMaskTests(unittest.TestCase):
    def test_only_registered_robot_and_rack_targets_are_eligible(self):
        self.assertTrue(eligible_target({"label": "Robot_2001", "category": "robot"}))
        self.assertTrue(eligible_target({"label": "Rack_1", "category": "rack"}))
        self.assertFalse(eligible_target({"label": "Person_1", "category": "person"}))
        self.assertFalse(eligible_target({"label": "", "category": "robot"}))

    def test_mask_validation_and_rasterization(self):
        mask = validate_mask({"polygons": [[[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]]})
        bitmap = mask_bitmap(mask, (100, 200, 3))
        self.assertGreater(int(bitmap.sum()), 0)
        self.assertRaises(ValueError, validate_mask, {"polygons": [[[0, 0], [2, 0], [0, 1]]]})

    def test_mask_can_be_advanced_with_measured_motion(self):
        mask = {"polygons": [[[0.20, 0.30], [0.30, 0.30], [0.30, 0.50]]], "source": "sam2_memory"}
        moved = advance_mask(mask, [0.20, 0.30, 0.10, 0.20], [0.30, 0.35, 0.10, 0.20])
        self.assertEqual([0.3, 0.35], moved["polygons"][0][0])
        self.assertEqual([0.4, 0.35], moved["polygons"][0][1])
        self.assertEqual("sam2_memory", moved["source"])

    def test_invalid_masks_are_rejected(self):
        for mask in [None, {}, {"polygons": [None]}, {"polygons": []},
                     {"polygons": [[[0, 0], [float('nan'), 0], [0, 1]]]},
                     {"polygons": [[[0, 0], [0, 0], [0, 0]]]}]:
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                validate_mask(mask)

    def test_holes_are_preserved_for_rack_masks(self):
        binary = np.zeros((200, 200), dtype=np.uint8)
        binary[20:180, 20:180] = 1
        binary[60:140, 60:140] = 0
        mask = mask_payload(binary)
        restored = mask_bitmap(mask, binary.shape)
        self.assertEqual(2, len(mask['polygons']))
        self.assertEqual(0, restored[100, 100])
        self.assertEqual(1, restored[30, 30])

    def test_worker_never_receives_people_or_unregistered_objects(self):
        segmenter = RegisteredTargetMaskSegmenter()
        try:
            with mock.patch.object(segmenter, '_request') as request:
                result = segmenter.track('cam', np.zeros((10, 10, 3)), [
                    {'label': 'Person_1', 'category': 'person'}, {'label': '', 'category': 'robot'}], {})
            self.assertEqual({}, result)
            request.assert_not_called()
        finally:
            segmenter.close()

    def test_mask_motion_keeps_identity_without_template_gating(self):
        frame = np.zeros((200, 300, 3), dtype=np.uint8)
        tracker = TemplateIdentityCameraTracker('cam', '', None)
        mask = {'polygons': [[[.4, .3], [.6, .3], [.6, .7]]], 'confidence': .8, 'source': 'sam2_memory'}
        tracker.add_target('Robot_2001', 'robot', [.1, .2, .2, .3], encode(frame[40:100, 30:90]), mask=mask, frame_image=encode(frame))
        observed = {'Robot_2001': {'bbox': [.4, .3, .2, .4], 'mask': mask},
                    'unknown': {'bbox': [.1, .1, .1, .1], 'mask': mask}}
        with mock.patch('core.template_identity_tracker.registered_target_mask_segmenter') as segmenter, \
                mock.patch.object(tracker, '_match_target') as template_match:
            segmenter.available.return_value = True
            segmenter.track.return_value = observed
            objects = tracker._process_registered_frame(frame)
            self.assertEqual(['Robot_2001'], [obj['label'] for obj in objects])
            self.assertEqual(2001, objects[0]['id'])
            self.assertEqual(.4, objects[0]['x'])
            self.assertEqual(mask['polygons'], objects[0]['mask']['polygons'])
            template_match.assert_not_called()
            segmenter.track.return_value = {}
            self.assertEqual([], tracker._process_registered_frame(frame))
            self.assertIsNone(tracker.targets['Robot_2001']['mask'])
            segmenter.track.return_value = observed
            self.assertTrue(tracker._process_registered_frame(frame)[0]['mask'])

    def test_registry_restores_all_mask_views_without_changing_people(self):
        image = encode(np.full((20, 20, 3), 100, dtype=np.uint8))
        masks = [{'polygons': [[[.1, .1], [.3, .1], [.2, .4]]]},
                 {'polygons': [[[.5, .5], [.8, .5], [.6, .8]]]}]
        vector = np.ones(512, dtype=np.float32).tolist()
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'registry.json')
            registry = TargetRegistry(path)
            for mask in masks:
                registry.register_target('Robot_2001', vector, 'cam', category='robot',
                                         bbox=[.1, .1, .8, .8], crop_image=image, mask=mask, frame_image=image)
            restored = TargetRegistry(path)
            samples = restored.get_template_samples('Robot_2001', 'cam')
            self.assertEqual(masks, [sample['mask'] for sample in samples])
            self.assertTrue(all(sample['frame_image'] == image for sample in samples))
            self.assertEqual(2, restored.get_all_targets()[0]['mask_samples_count'])
            with self.assertRaises(ValueError):
                restored.register_target('Person_01', vector, 'cam', category='person', mask=masks[0], frame_image=image)

    def test_person_pose_metadata_is_unchanged(self):
        fusion = MetadataFusion()
        points = [[.5, .5, .9]] * 17
        fusion.update('cam', 'deepstream', [{'id': 1, 'class': 'person', 'keypoints': points}], now=1)
        objects = fusion.update('cam', 'identity_template', [
            {'id': 2, 'class': 'robot', 'category': 'robot', 'label': 'Robot_2001',
             'mask': {'polygons': [[[.1, .1], [.2, .1], [.2, .2]]]}}
        ], now=1.1)
        people = [obj for obj in objects if obj['class'] == 'person']
        self.assertEqual(points, people[0]['keypoints'])
        self.assertNotIn('mask', people[0])


if __name__ == "__main__":
    unittest.main()
