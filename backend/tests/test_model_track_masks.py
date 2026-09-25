import unittest
from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.dashboard_auth import require_dashboard_user
from core.metadata_fusion import MetadataFusion
from core.model_track_masks import TrackMaskCache, boundary_polygons, model_identity
from core.workflow_store import WorkflowConflict
from routers.models import create_model_router


def tracked(identifier=1, class_name="robot", class_id=0):
    return dict(id=identifier, model_id="model", class_id=class_id, category="robot",
                **{"class": class_name, "x": .1, "y": .2, "w": .2, "h": .3})


class ModelMaskTests(unittest.TestCase):
    def setUp(self):
        self.cache = TrackMaskCache()
        self.mask = {"polygons": [[[.12, .22], [.28, .22], [.28, .48], [.12, .48]]]}

    def test_same_class_instances_never_share_masks(self):
        first, second = tracked(1), tracked(2)
        self.cache.store("cam", first, self.mask, 10, 1000)
        self.cache.attach("cam", [first, second], 11, 1020)
        self.assertIsNotNone(first["mask"])
        self.assertIsNone(second["mask"])
        self.assertEqual(first["mask"]["frame_id"], 10)
        other_camera = tracked(1)
        self.cache.attach("other", [other_camera], 11, 1020)
        self.assertIsNone(other_camera["mask"])

    def test_mask_moves_with_box_but_observation_time_does_not_refresh(self):
        target = tracked()
        self.cache.store("cam", target, self.mask, 10, 1000)
        target["x"] += .05
        self.cache.attach("cam", [target], 11, 1200)
        self.assertAlmostEqual(target["mask"]["polygons"][0][0][0], .17)
        self.assertEqual(target["mask"]["observed_at"], 1000)
        self.cache.attach("cam", [target], 12, 1701)
        self.assertIsNone(target["mask"])
        self.assertTrue(target["mask_stale"])

    def test_track_disappearance_and_class_change_invalidate_memory(self):
        target = tracked()
        self.cache.store("cam", target, self.mask, 10, 1000)
        changed = tracked(class_name="Robot_2001", class_id=1)
        self.cache.attach("cam", [changed], 11, 1010)
        self.assertIsNone(changed["mask"])
        self.cache.store("cam", target, self.mask, 12, 1020)
        self.cache.attach("cam", [], 13, 1030)
        self.cache.attach("cam", [target], 14, 1040)
        self.assertIsNone(target["mask"])

    def test_async_result_cannot_resurrect_departed_track(self):
        target = tracked()
        self.cache.attach("cam", [], 20, 1100)
        self.cache.store("cam", target, self.mask, 10, 1000)
        self.cache.attach("cam", [target], 21, 1120)
        self.assertIsNone(target["mask"])

    def test_out_of_order_future_and_discontinuous_masks_rejected(self):
        target = tracked()
        self.cache.store("cam", target, self.mask, 20, 1000)
        self.cache.store("cam", target, {"polygons": []}, 19, 1100)
        self.cache.attach("cam", [target], 18, 1120)
        self.assertIsNone(target["mask"])
        self.cache.attach("cam", [target], 21, 1140)
        self.assertTrue(target["mask"]["polygons"])
        target["x"] = .8
        self.cache.attach("cam", [target], 22, 1160)
        self.assertIsNone(target["mask"])

    def test_identity_requires_trained_robot_identity_class(self):
        self.assertIsNone(model_identity("robot", "robot"))
        self.assertIsNone(model_identity("robot_v11", "robot"))
        self.assertIsNone(model_identity("Robot_2001", "person"))
        self.assertEqual(model_identity("Robot_2001", "robot"), "Robot_2001")

    def test_custom_fusion_preserves_masks_and_track_ids_without_legacy(self):
        fusion = MetadataFusion()
        legacy = dict(tracked(99), label="Robot_2001", mask=self.mask)
        fusion.update("cam", "identity_template", [legacy], now=1)
        first = dict(tracked(1, "Robot_2001"), mask=self.mask)
        second = dict(tracked(2), label="Robot_2001")
        result = fusion.update("cam", "custom_deepstream", [first, second], now=1.1)
        self.assertEqual([obj["id"] for obj in result], [1, 2])
        self.assertEqual(result[0]["label"], "Robot_2001")
        self.assertEqual(result[0]["mask"], self.mask)
        self.assertIsNone(result[1]["label"])
        late = fusion.update("cam", "identity_template", [legacy], now=1.2)
        self.assertEqual([obj["id"] for obj in late], [1, 2])
        self.assertEqual(fusion.update("cam", "custom_deepstream", [], now=1.3), [])

    def test_polygon_assembly_keeps_disconnected_components(self):
        edges = [[0, 0, 3, 0], [3, 0, 3, 3], [3, 3, 0, 3], [0, 3, 0, 0],
                 [5, 5, 8, 5], [8, 5, 8, 8], [8, 8, 5, 8], [5, 8, 5, 5]]
        polygons = boundary_polygons(edges, 10, 10)
        self.assertEqual(len(polygons), 2)
        self.assertTrue(all(len(ring) == 4 for ring in polygons))
        self.assertTrue(all(0 <= coordinate <= 1 for ring in polygons for point in ring for coordinate in point))


class MonitorDeploymentAPITests(unittest.TestCase):
    def setUp(self):
        self.store = MagicMock(chunk_size=2097152, max_size=536870912)
        self.app = FastAPI()
        self.app.include_router(create_model_router(self.store, lambda: {}, lambda: {"cam"}))
        self.client = TestClient(self.app)

    def test_deploy_auth_validation_and_persistence_call(self):
        self.assertEqual(self.client.post("/api/models/model/deploy", json={}).status_code, 401)
        self.app.dependency_overrides[require_dashboard_user] = lambda: "admin"
        self.store.deploy.return_value = {"model_id": "model", "camera_ids": ["cam"], "all_cameras": False}
        response = self.client.post("/api/models/model/deploy", json={"camera_ids": ["cam"], "all_cameras": False})
        self.assertEqual(response.status_code, 200)
        self.store.deploy.assert_called_once_with("model", ["cam"], False, {"cam"}, "admin", {})
        self.assertEqual(self.client.post("/api/models/model/deploy", json={"unexpected": True}).status_code, 422)
        self.assertEqual(self.client.post("/api/models/model/deploy", json={"camera_ids": ["cam"]}).status_code, 422)
        self.store.deploy.side_effect = WorkflowConflict("conflicting model")
        self.assertEqual(self.client.post("/api/models/model/deploy", json={"all_cameras": True}).status_code, 409)

    def test_stop_and_list_deployment(self):
        self.app.dependency_overrides[require_dashboard_user] = lambda: "admin"
        self.store.deployment.return_value = {"model_id": "model", "all_cameras": True, "camera_ids": []}
        self.store.list.return_value = []
        response = self.client.get("/api/models")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["deployment"]["model_id"], "model")
        self.store.stop_deployment.return_value = {"stopped": True}
        self.assertEqual(self.client.delete("/api/models/deployment").status_code, 200)
        self.store.stop_deployment.assert_called_once_with("admin")


if __name__ == "__main__":
    unittest.main()
