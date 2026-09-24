import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.dashboard_auth import require_dashboard_user
from core.identity_metric import triplet_loss
from core.metadata_fusion import MetadataFusion
from core.model_label_metric import ModelLabelMetric
from core.model_label_session import ModelLabelSession
from core.model_label_store import ModelLabelStore, validate_sample
from core.model_sam2 import ModelSAM2
from core.model_track_gate import ModelTrackGate
from core.model_track_masks import TrackMaskCache
from routers.model_labels import create_model_label_router


def sample(label, vector, negative=False, category="robot", signature="signature"):
    return dict(label=label, class_name=label, category=category, negative=negative,
                signature=signature, vector=vector.tolist())


def tracked(identifier=1, **kwargs):
    return dict(id=identifier, local_id=str(identifier), model_id="model", class_id=1,
                **{"category": "robot", "class": "robot", "x": .1, "y": .2, "w": .2, "h": .3,
                                    "confidence": .9, "detected_at": 1000, **kwargs})


class ModelLabelMetricTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.directory = tempfile.TemporaryDirectory()
        self.metric = ModelLabelMetric(self.directory.name, "signature", device="cpu", start=False)
        self.first, self.second = torch.eye(512)[:2]

    def tearDown(self):
        self.metric.stop()
        self.directory.cleanup()

    def test_multiview_negative_and_no_automatic_training_samples(self):
        self.metric.load(dict(revision=1, samples=[sample("Robot_1", self.first),
            sample("Robot_1", self.first + .2 * self.second), sample("wrong", self.second, negative=True)]))
        self.assertTrue(self.metric.verify(self.first, "robot")["accepted"])
        self.assertEqual(self.metric.verify(self.second, "robot")["reason"], "negative_sample")
        self.assertFalse(self.metric.verify(self.first, "rack")["accepted"])
        self.assertEqual(len(self.metric.samples), 3)
        self.assertTrue(self.metric.policy("robot")["required"])
        self.assertFalse(self.metric.policy("rack")["required"])

    def test_ambiguous_different_identities_rejected(self):
        self.metric.load(dict(revision=2, samples=[sample("Robot_1", self.first), sample("Robot_2", self.first)]))
        self.assertEqual(self.metric.verify(self.first, "robot")["reason"], "ambiguous_identity")

    def test_user_gallery_match_has_priority_over_triplet_projection(self):
        self.metric.load(dict(revision=2, samples=[
            sample("Robot_1", self.first), sample("Robot_1", self.first + .01 * self.second),
            sample("Robot_2", self.second),
        ]))
        self.metric.weight = torch.eye(512)
        self.metric.weight[1, 1] = 25
        self.metric.projected = torch.nn.functional.normalize(self.metric.vectors @ self.metric.weight.T, dim=1)
        result = self.metric.verify(self.first + .12 * self.second, "robot")
        self.assertTrue(result["accepted"])
        self.assertEqual(result["label"], "Robot_1")
        self.assertEqual(result["match_source"], "user_gallery")
        self.assertLess(result["projected_score"], .82)
        self.assertGreater(result["raw_score"], .99)

    def test_user_gallery_still_rejects_close_competing_label(self):
        self.metric.load(dict(revision=2, samples=[sample("Robot_1", self.first), sample("Robot_2", self.first + .01 * self.second)]))
        result = self.metric.verify(self.first, "robot")
        self.assertFalse(result["accepted"])
        self.assertEqual(result["reason"], "ambiguous_identity")

    def test_adding_more_than_64_views_keeps_first_view_matchable(self):
        self.metric.load(dict(revision=3, samples=[sample("Robot_1", self.first)] +
            [sample("Robot_1", self.second) for _ in range(70)]))
        result = self.metric.verify(self.first, "robot")
        self.assertTrue(result["accepted"])
        self.assertEqual(result["match_source"], "user_gallery")
        self.assertEqual(self.metric.status()["samples"], 71)

    def test_signature_change_and_strict_mode_fail_closed(self):
        self.metric.load(dict(revision=3, samples=[sample("Robot_1", self.first, signature="old")]))
        self.assertTrue(self.metric.policy("robot")["required"])
        self.assertTrue(self.metric.policy("robot")["error"])
        self.assertFalse(self.metric.verify(self.first, "robot")["accepted"])
        self.metric.load(dict(revision=4, require_labels=True, samples=[]))
        self.assertTrue(self.metric.policy("rack")["required"])
        self.assertFalse(self.metric.verify(self.first, "robot")["accepted"])

    def test_triplet_training_separates_close_negatives(self):
        examples = [sample("Robot_1", self.first + .03 * self.second), sample("Robot_1", self.first - .03 * self.second),
                    sample("Robot_2", self.first + .4 * self.second), sample("Robot_2", self.first + .5 * self.second)]
        self.metric.load(dict(revision=5, samples=examples))
        baseline = float(triplet_loss(self.metric.vectors, torch.tensor([0, 0, 1, 1])))
        self.assertTrue(self.metric.fit(steps=30))
        self.assertLess(self.metric.loss, baseline)
        self.assertTrue((Path(self.directory.name) / "triplet.pt").is_file())
        self.assertFalse(self.metric.status()["detector_weights_trained"])

    def test_one_view_cannot_train_triplet(self):
        self.metric.load(dict(revision=6, samples=[sample("Robot_1", self.first), sample("Robot_2", self.second)]))
        self.assertFalse(self.metric.fit())

    def test_hard_negatives_train_without_becoming_positive_pairs(self):
        examples = [sample("Robot_1", self.first + .03 * self.second), sample("Robot_1", self.first - .03 * self.second),
                    sample("wrong", self.first + .4 * self.second, negative=True),
                    sample("wrong", self.first + .5 * self.second, negative=True)]
        self.metric.load(dict(revision=7, samples=examples))
        baseline = float(triplet_loss(self.metric.vectors, torch.tensor([0, 0, 1, 2])))
        self.assertTrue(self.metric.fit(steps=20))
        self.assertLess(self.metric.loss, baseline)
        self.assertFalse(self.metric.verify(self.first + .4 * self.second, "robot")["accepted"])


class ModelLabelSessionTests(unittest.TestCase):
    def setUp(self):
        self.now = 10
        self.replies = []
        self.session = ModelLabelSession(self.replies.append, clock=lambda: self.now)

    def test_snapshot_waits_for_correct_camera_and_expires(self):
        self.session.request(dict(id="request", action="snapshot", camera_id="cam"))
        self.assertTrue(self.session.needs_frame("cam"))
        self.assertFalse(self.session.needs_frame("other"))
        self.now = 21
        self.session.service(None)
        self.assertIn("error", self.replies[-1])
        self.assertFalse(self.session.waiting)

    def test_sample_cannot_switch_camera_or_reuse_expired_receipt(self):
        self.session.previews["preview"] = dict(camera_id="cam", expires=20, sample={"vector": [1]})
        self.session.request(dict(id="first", action="sample", camera_id="other", preview_id="preview"))
        self.session.service(None)
        self.assertIn("error", self.replies[-1])
        self.session.request(dict(id="second", action="sample", camera_id="cam", preview_id="preview"))
        self.session.service(None)
        self.assertEqual(self.replies[-1]["result"], {"vector": [1]})
        self.now = 21
        self.session.request(dict(id="third", action="sample", camera_id="cam", preview_id="preview"))
        self.session.service(None)
        self.assertIn("error", self.replies[-1])

    def test_gpu_preview_receipt_is_server_owned(self):
        self.session.snapshots["frame"] = dict(camera_id="cam", expires=30, image="gpu_image", jpeg="jpeg", frame_id=8)
        runtime = MagicMock(signature="signature")
        runtime.preview.return_value = ({"polygons": []}, torch.ones(512))
        self.session.request(dict(id="preview", action="preview", camera_id="cam", snapshot_id="frame", bbox=[.1, .2, .3, .4]))
        self.session.service(runtime)
        identifier = self.replies[-1]["result"]["preview_id"]
        self.assertNotIn("vector", self.replies[-1]["result"])
        self.assertEqual(self.session.previews[identifier]["sample"]["signature"], "signature")
        self.assertEqual(self.session.previews[identifier]["sample"]["frame_id"], 8)


class ModelLabelGateTests(unittest.TestCase):
    def test_startup_keeps_configured_label_categories_closed(self):
        owner = ModelSAM2.__new__(ModelSAM2)
        owner.cache = TrackMaskCache()
        owner.metric = None
        owner.initial_categories = {"robot"}
        owner.initial_require_labels = False
        self.assertEqual(owner.attach("cam", [tracked()], 1, 1000), [])
        self.assertEqual(len(owner.attach("cam", [tracked(category="rack")], 2, 1030)), 1)
        owner.initial_require_labels = True
        self.assertEqual(owner.attach("cam", [tracked(category="rack")], 3, 1060), [])

    def test_weak_birth_stale_and_teleport_rejected(self):
        gate = ModelTrackGate()
        self.assertFalse(gate.filter("cam", [tracked(confidence=.35)], 1000)[0])
        self.assertFalse(gate.filter("cam", [tracked()], 1000)[0])
        self.assertTrue(gate.filter("cam", [tracked(detected_at=1030)], 1030)[0])
        self.assertTrue(gate.filter("cam", [tracked(confidence=.35, detected_at=1060)], 1060)[0])
        self.assertFalse(gate.filter("cam", [tracked(detected_at=1000)], 1400)[0])
        self.assertFalse(gate.filter("cam", [tracked(x=.8, detected_at=1430)], 1430)[0])

    def test_robot_bbox_stabilization_deadband_and_rate_limiting(self):
        gate = ModelTrackGate()
        # Frame 1: birth (hits=1)
        gate.filter("cam", [tracked(category="robot", x=0.20, y=0.30, w=0.10, h=0.12, detected_at=1000)], 1000)
        # Frame 2: confirmed (hits=2)
        out2, _ = gate.filter("cam", [tracked(category="robot", x=0.20, y=0.30, w=0.10, h=0.12, detected_at=1030)], 1030)
        self.assertTrue(out2)
        self.assertAlmostEqual(out2[0]["w"], 0.10, places=3)
        self.assertAlmostEqual(out2[0]["h"], 0.12, places=3)

        # Frame 3: Micro-jitter (<4% change in width/height) -> deadband locks dimensions
        out3, _ = gate.filter("cam", [tracked(category="robot", x=0.20, y=0.30, w=0.103, h=0.117, detected_at=1060)], 1060)
        self.assertTrue(out3)
        self.assertAlmostEqual(out3[0]["w"], 0.10, places=3)
        self.assertAlmostEqual(out3[0]["h"], 0.12, places=3)

        # Frame 4: Sudden spike (+30% width) -> rate-limiting clamps to <= 8% change
        out4, _ = gate.filter("cam", [tracked(category="robot", x=0.20, y=0.30, w=0.130, h=0.156, detected_at=1090)], 1090)
        self.assertTrue(out4)
        # Since clamped to 1.08 max and alpha=0.25: 0.10 + 0.25 * (0.108 - 0.10) = 0.102
        self.assertLess(out4[0]["w"], 0.105)
        self.assertLess(out4[0]["h"], 0.126)

    def test_label_candidates_reach_verification_without_relaxing_plain_detection(self):
        gate = ModelTrackGate()
        self.assertFalse(gate.filter("cam", [tracked(confidence=.35)], 1000, {"robot"})[0])
        candidates, reasons = gate.filter("cam", [tracked(confidence=.35, detected_at=1030)], 1030, {"robot"})
        self.assertFalse(reasons)
        self.assertTrue(candidates[0]["requires_label_verification"])
        self.assertFalse(gate.filter("cam", [tracked(confidence=.35, detected_at=1060)], 1060)[0])
        self.assertFalse(gate.filter("cam", [tracked(confidence=.24, detected_at=1090)], 1090, {"robot"})[0])
        self.assertFalse(gate.filter("cam", [tracked(confidence=.35, detected_at=1000)], 1400, {"robot"})[0])
        self.assertFalse(gate.filter("cam", [tracked(confidence=.35, x=.8, detected_at=1430)], 1430, {"robot"})[0])

    def test_label_candidates_fail_closed_when_gallery_disappears(self):
        owner = ModelSAM2.__new__(ModelSAM2)
        owner.cache = TrackMaskCache()
        owner.metric = MagicMock()
        owner.metric.policy.return_value = dict(required=False, error=None, revision=2, version=1)
        target = tracked(requires_label_verification=True)
        self.assertEqual(owner.attach("cam", [dict(target)], 1, 1000), [])
        mask = {"polygons": [[[.1, .2], [.3, .2], [.3, .5]]]}
        identity = dict(accepted=True, revision=1, version=1, label="Robot_2001", class_name="robot", score=.95)
        owner.cache.store("cam", target, mask, 2, 1010, identity)
        self.assertEqual(owner.attach("cam", [dict(target)], 3, 1030), [])
        identity["revision"] = 2
        owner.cache.store("cam", target, mask, 4, 1040, identity)
        result = owner.attach("cam", [dict(target)], 5, 1060)
        self.assertEqual(result[0]["label"], "Robot_2001")
        self.assertNotIn("requires_label_verification", result[0])

    def test_accepted_mask_follows_bbox_but_revocation_is_immediate(self):
        owner = ModelSAM2.__new__(ModelSAM2)
        owner.cache = TrackMaskCache()
        owner.metric = MagicMock()
        owner.metric.policy.return_value = dict(required=True, error=None, revision=2, version=1)
        identity = dict(accepted=True, revision=2, version=1, label="Robot_2001", class_name="robot", score=.95)
        target = tracked()
        mask = {"polygons": [[[.1, .2], [.3, .2], [.3, .5]]]}
        owner.cache.store("cam", target, mask, 10, 1000, identity)
        result = owner.attach("cam", [tracked(x=.12)], 11, 1050)
        self.assertEqual(result[0]["label"], "Robot_2001")
        self.assertTrue(result[0]["identity_verified"])
        self.assertAlmostEqual(result[0]["mask"]["polygons"][0][0][0], .12)
        owner.metric.policy.return_value["revision"] = 3
        self.assertEqual(owner.attach("cam", [tracked()], 12, 1060), [])
        owner.metric.policy.return_value["revision"] = 2
        owner.metric.policy.return_value["version"] = 2
        self.assertEqual(owner.attach("cam", [tracked()], 13, 1070), [])
        owner.metric.policy.return_value["version"] = 1
        owner.cache.reject("cam", target)
        self.assertEqual(owner.attach("cam", [tracked()], 14, 1080), [])

    def test_fusion_keeps_verified_label_but_not_legacy_claim(self):
        fusion = MetadataFusion()
        verified = tracked(label="Robot_2001", identity_verified=True, identity_source="model_label_triplet")
        self.assertEqual(fusion.update("cam", "custom_deepstream", [verified])[0]["label"], "Robot_2001")
        verified["identity_source"] = "identity_template"
        self.assertIsNone(fusion.update("cam", "custom_deepstream", [verified])[0]["label"])

    def test_user_gallery_priority_and_revocation(self):
        owner = ModelSAM2.__new__(ModelSAM2)
        owner.cache = TrackMaskCache()
        owner.metric = MagicMock()
        owner.metric.policy.return_value = dict(required=True, error=None, revision=2, version=8)
        mask = {"polygons": [[[.1, .2], [.3, .2], [.3, .5]]]}
        direct = dict(accepted=True, revision=2, version=1, label="Robot_2001", class_name="robot", score=.94, match_source="user_gallery")
        projected = dict(direct, version=8, score=.99, match_source="triplet")
        owner.cache.store("cam", tracked(1), mask, 1, 1000, projected)
        owner.cache.store("cam", tracked(2), mask, 1, 1000, direct)
        output = owner.attach("cam", [tracked(1), tracked(2)], 2, 1020)
        self.assertEqual([obj["id"] for obj in output], [2])
        self.assertEqual(output[0]["identity_match_source"], "user_gallery")
        owner.metric.policy.return_value["revision"] = 3
        self.assertEqual(owner.attach("cam", [tracked(2)], 3, 1030), [])


class ModelLabelAPITests(unittest.TestCase):
    def setUp(self):
        self.store, self.registry, self.detector = MagicMock(), MagicMock(), MagicMock()
        self.registry.get.return_value = {"labels": ["Robot_2001", "Rack"]}
        self.app = FastAPI()
        self.app.include_router(create_model_label_router(self.store, self.registry, self.detector))
        self.client = TestClient(self.app)
        self.path = "/api/models/" + "a" * 32 + "/labels"

    def test_session_required_and_untrusted_vectors_refused(self):
        self.assertEqual(self.client.post(self.path + "/snapshot", json={"camera_id": "cam"}).status_code, 401)
        self.app.dependency_overrides[require_dashboard_user] = lambda: "admin"
        data = dict(camera_id="cam", preview_id="b" * 32, label="Robot_2001", class_name="Robot_2001", vector=[1])
        self.assertEqual(self.client.post(self.path + "/samples", json=data).status_code, 422)
        self.detector.command.assert_not_called()

    def test_invalid_prompts_and_inactive_worker(self):
        self.app.dependency_overrides[require_dashboard_user] = lambda: "admin"
        data = dict(camera_id="cam", snapshot_id="b" * 32, bbox=[.9, .1, .2, .2])
        self.assertEqual(self.client.post(self.path + "/preview", json=data).status_code, 422)
        data.update(bbox=[.1, .1, .2, .2], points=[[.2, .2]], point_labels=[])
        self.assertEqual(self.client.post(self.path + "/preview", json=data).status_code, 422)
        self.detector.command.side_effect = RuntimeError("not deployed")
        self.assertEqual(self.client.post(self.path + "/snapshot", json={"camera_id": "cam"}).status_code, 409)

    def test_store_receives_only_worker_owned_embedding(self):
        self.app.dependency_overrides[require_dashboard_user] = lambda: "admin"
        gpu_sample = dict(vector=[.1] * 512)
        self.detector.command.return_value = gpu_sample
        self.store.save.return_value = {"saved": True}
        data = dict(camera_id="cam", preview_id="b" * 32, label="Robot_2001", class_name="Robot_2001")
        self.assertEqual(self.client.post(self.path + "/samples", json=data).status_code, 201)
        self.assertIs(self.store.save.call_args.args[-2], gpu_sample)
        self.assertEqual(self.store.save.call_args.args[-1], "admin")
        data["class_name"] = "not_in_model"
        self.assertEqual(self.client.post(self.path + "/samples", json=data).status_code, 422)

    def test_validation_and_model_directory_isolation(self):
        store = ModelLabelStore(MagicMock())
        with self.assertRaises(ValueError):
            store.directory("../../outside")
        for vector in ([0] * 512, [1] * 511, [float("inf")] * 512):
            with self.assertRaises(ValueError):
                validate_sample({"vector": vector})
        validate_sample({"vector": [1] * 512})

    def test_view_routes_require_auth_and_keep_model_scope(self):
        identifier = "c" * 32
        for method, suffix in (("GET", "/samples?label=Robot_2001"), ("GET", f"/samples/{identifier}/image"), ("DELETE", f"/samples/{identifier}")):
            self.assertEqual(self.client.request(method, self.path + suffix).status_code, 401)
        self.app.dependency_overrides[require_dashboard_user] = lambda: "admin"
        self.store.samples.return_value = dict(samples=[], total=0, offset=12, limit=12)
        self.assertEqual(self.client.get(self.path + "/samples?label=Robot_2001&offset=12").status_code, 200)
        self.store.samples.assert_called_once_with("a" * 32, "Robot_2001", 12, 12)
        self.assertEqual(self.client.get(self.path + "/samples?label=Robot_2001&limit=100").status_code, 422)
        self.assertEqual(self.client.delete(self.path + "/samples/not-an-id").status_code, 422)
        self.store.delete_sample.return_value = dict(deleted=1)
        self.assertEqual(self.client.delete(self.path + f"/samples/{identifier}").json(), {"deleted": 1})
        self.store.delete_sample.assert_called_once_with("a" * 32, identifier, "admin")
        self.store.sample_image.side_effect = KeyError("wrong model")
        self.assertEqual(self.client.get(self.path + f"/samples/{identifier}/image").status_code, 404)

class ModelLabelStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.store = ModelLabelStore(MagicMock())
        self.store.root = Path(self.temporary.name)
        self.cursor = MagicMock()
        self.store.transaction = MagicMock()
        self.store.transaction.return_value.__enter__.return_value = self.cursor

    def tearDown(self):
        self.temporary.cleanup()

    def test_publish_includes_all_views(self):
        self.cursor.fetchone.return_value = dict(revision=10, require_labels=False)
        self.cursor.fetchall.return_value = [dict(id=str(index), label="Robot_1") for index in range(71)]
        destination = self.store.publish("a" * 32)
        import json
        self.assertEqual(len(json.loads(destination.read_text())["samples"]), 71)
        query = self.cursor.execute.call_args_list[-1].args[0]
        self.assertNotIn("position <=", query)
        self.assertEqual(self.cursor.execute.call_args_list[-1].args[1], ("a" * 32,))

    def test_delete_one_view_preserves_other_images_and_publishes(self):
        directory = self.store.directory("a" * 32) / "samples"
        directory.mkdir(parents=True)
        target = directory / ("b" * 32 + ".jpg")
        preserved = directory / ("c" * 32 + ".jpg")
        target.write_bytes(b"sample")
        preserved.write_bytes(b"other")
        self.cursor.fetchone.return_value = dict(image_path=str(target.relative_to(self.store.root)), label="Robot_1")
        self.store.publish = MagicMock()
        self.store.list = MagicMock(return_value=dict(labels=[]))
        self.assertEqual(self.store.delete_sample("a" * 32, "b" * 32, "admin")["deleted"], 1)
        self.assertFalse(target.exists())
        self.assertTrue(preserved.exists())
        self.store.publish.assert_called_once_with("a" * 32)
        deletion = next(call for call in self.cursor.execute.call_args_list if call.args[0].startswith("DELETE"))
        self.assertEqual(deletion.args[1], ("a" * 32, "b" * 32))

    def test_image_cannot_escape_model_directory(self):
        external = self.store.root / "other.jpg"
        external.write_bytes(b"private")
        self.cursor.fetchone.return_value = dict(image_path="other.jpg")
        with self.assertRaises(KeyError):
            self.store.sample_image("a" * 32, "b" * 32)


if __name__ == "__main__":
    unittest.main()
