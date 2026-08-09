import unittest

import numpy as np

from core.deepstream_pose import attach_poses, decode_pose_tensor


class DeepStreamPoseTests(unittest.TestCase):
    def tensor(self):
        values = np.zeros((56, 2), dtype=np.float32)
        values[:5, 0] = [320, 320, 128, 180, 0.9]
        values[5:, 0] = np.tile([320, 320, 0.9], 17)
        values[:, 1] = values[:, 0]
        values[4, 1] = 0.8
        return values

    def test_unletterbox_and_nms(self):
        poses = decode_pose_tensor(self.tensor(), 1280, 720)
        self.assertEqual(1, len(poses))
        self.assertTrue(np.allclose(poses[0]["bbox"], [0.4, 0.25, 0.2, 0.5]))
        self.assertEqual(17, len(poses[0]["keypoints"]))
        self.assertTrue(np.allclose(poses[0]["keypoints"][0], [0.5, 0.5, 0.9]))

    def test_one_pose_cannot_attach_to_two_tracks(self):
        poses = decode_pose_tensor(self.tensor(), 1280, 720)
        detections = [{"bbox": poses[0]["bbox"]}, {"bbox": [0.42, 0.25, 0.2, 0.5]}]
        attach_poses(detections, poses)
        self.assertEqual([17, 0], [len(item["keypoints"]) for item in detections])
        self.assertEqual("predicted", detections[1]["tracking_state"])

    def test_invalid_output_rejected_and_no_pose_is_not_fabricated(self):
        with self.assertRaises(ValueError):
            decode_pose_tensor(np.zeros((84, 2)), 1280, 720)
        detections = [{"bbox": [0, 0, 0.2, 0.5], "keypoints": [[0.5, 0.5, 1]] * 17}]
        attach_poses(detections, [])
        self.assertEqual([], detections[0]["keypoints"])

    def test_outside_padding_keypoints_have_no_confidence(self):
        values = self.tensor()
        values[6, :] = 10
        poses = decode_pose_tensor(values, 1280, 720)
        self.assertEqual(0, poses[0]["keypoints"][0][2])


if __name__ == "__main__":
    unittest.main()
