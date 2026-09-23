import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from core.registered_identity_metrics import RegisteredIdentityMetrics
from src.controller.registry import TargetRegistry


class RegisteredIdentityMetricsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.metrics = RegisteredIdentityMetrics(Path(self.directory.name) / "metrics", device="cpu", auto_train=False)
        self.registry = TargetRegistry(str(Path(self.directory.name) / "registry.json"), identity_metrics=self.metrics)

    @staticmethod
    def vector(index):
        vector = np.zeros(512)
        vector[index] = 1
        return vector.tolist()

    def test_every_label_category_learns_and_restores(self):
        categories = ["robot", "rack", "person", "helmet", "object"]
        for index, category in enumerate(categories):
            self.registry.register_target(f"Label_{index}", self.vector(index), "cam", category=category)
        status = self.metrics.status()["encoders"]["reid_512"]
        self.assertEqual(5, status["samples"])
        self.assertEqual(5, len(status["labels"]))
        restored = RegisteredIdentityMetrics(self.metrics.directory, device="cpu", auto_train=False)
        restored.restore(self.registry)
        self.assertEqual(status["samples"], restored.status()["encoders"]["reid_512"]["samples"])

    def test_same_label_across_cameras_trains_shared_identity(self):
        front = self.vector(0)
        side = self.vector(0)
        side[1] = .3
        self.registry.register_target("Person_A", front, "cam1", category="person")
        self.registry.register_target("Person_A", side, "cam2", category="person")
        self.registry.register_target("Helmet_B", self.vector(3), "cam1", category="helmet")
        metric = self.metrics.metrics["reid_512"]
        self.assertTrue(metric.fit(steps=3))
        self.assertEqual(2, metric.status()["labels"]["Person_A"])
        self.assertTrue(self.metrics.verify("Person_A", front)["accepted"])
        self.assertFalse(self.metrics.verify("Person_A", self.vector(3))["accepted"])

    def test_incompatible_encoders_are_kept_separate(self):
        self.metrics.learn("Person_A", "cam", self.vector(0), "reid_512")
        self.metrics.learn("Rack_B", "cam", self.vector(0), "clip_512")
        self.assertIsNone(self.metrics.learn("Robot_2001", "cam", self.vector(0), "sam2_mask_512"))
        self.assertTrue(self.metrics.verify("Person_A", self.vector(0), "reid_512")["accepted"])
        self.assertFalse(self.metrics.verify("Rack_B", self.vector(0), "reid_512")["accepted"])
        self.assertNotEqual(self.metrics.metrics["reid_512"].directory, self.metrics.metrics["clip_512"].directory)

    def test_single_view_without_negative_does_not_claim_training(self):
        self.registry.register_target("Person_A", self.vector(0), "cam", category="person")
        metric = self.metrics.metrics["reid_512"]
        self.assertFalse(metric.fit(steps=1))
        self.assertEqual("waiting_for_positive_and_negative_samples", metric.status()["state"])

    def test_registry_matching_uses_metric_not_raw_similarity(self):
        self.registry.register_target("Person_A", self.vector(0), "cam", category="person")
        with mock.patch.object(self.metrics, "verify", return_value={"accepted": False, "score": .99}):
            label, _score = self.registry.match_reid(self.vector(0))
            self.assertIsNone(label)
            self.assertIsNone(self.registry.assign_label_for_detection("cam", 1, 2, [.1, .1, .2, .2],
                                                                       feature=self.vector(0), det_class="person"))

    def test_failed_registry_save_never_trains_metric(self):
        with mock.patch.object(self.registry, "save_to_disk", side_effect=RuntimeError("SSD full")), \
                mock.patch.object(self.metrics, "learn") as learn:
            with self.assertRaises(RuntimeError):
                self.registry.register_target("Person_A", self.vector(0), "cam", category="person")
        learn.assert_not_called()

    def test_camera_scoped_delete_preserves_other_views(self):
        self.registry.register_target("Person_A", self.vector(0), "cam1", category="person")
        self.registry.register_target("Person_A", self.vector(1), "cam2", category="person")
        self.registry.remove_target("Person_A", "cam1")
        self.assertEqual(1, self.metrics.status()["encoders"]["reid_512"]["labels"]["Person_A"])


if __name__ == "__main__":
    unittest.main()
