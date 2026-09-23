import base64
import csv
import copy
import json
import math
import os
from pathlib import Path
import tempfile
import threading
import unittest
import uuid
from unittest.mock import Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.active_learning_data import atomic_write, detection_lines, quality_gate, validate_annotations
from core.active_learning_dataset import dataset_images, prepare_dataset
from core.active_learning_store import ActiveLearningStore, DEFAULT_SETTINGS
from core.active_learning_worker import ActiveLearningWorker
from core.dashboard_auth import require_dashboard_user
from core.model_registry import ModelRegistry
from core.workflow_store import WorkflowConflict
from routers.active_learning import create_active_learning_router


MODEL_ID = "a" * 32
LABELS = ["Robot_2001", "Rack"]
ANNOTATIONS = [dict(id="object", class_name="Robot_2001", label="Robot_2001",
                    polygons=[[[.1, .1], [.4, .1], [.4, .4], [.1, .4]]])]


class AnnotationTests(unittest.TestCase):
    def test_polygons_to_boxes_keep_class_order(self):
        result = validate_annotations(ANNOTATIONS, LABELS)
        self.assertEqual(result[0]["class_id"], 0)
        self.assertEqual(detection_lines(result), "0 0.2500000 0.2500000 0.3000000 0.3000000\n")

    def test_empty_background_is_explicitly_supported(self):
        self.assertEqual(detection_lines(validate_annotations([], LABELS)), "")

    def test_reject_unknown_class_invalid_coordinates_and_crossed_ring(self):
        for replacement in ({"class_name": "alien"}, {"polygons": [[[0, 0], [1, 1], [0, 1], [1, 0]]]},
                            {"polygons": [[[0, 0], [math.nan, 1], [1, 0]]]},
                            {"polygons": [[[0, 0], [1.1, 0], [1, 1]]]}):
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                validate_annotations([{**ANNOTATIONS[0], **replacement}], LABELS)

    def test_reject_duplicate_object_ids(self):
        with self.assertRaises(ValueError):
            validate_annotations(ANNOTATIONS * 2, LABELS)

    def test_quality_gate_global_improvement_not_enough_if_class_regresses(self):
        baseline = dict(map50_95=.5, map50=.8, per_class={"robot": .6, "rack": .4})
        candidate = dict(map50_95=.52, map50=.81, per_class={"robot": .57, "rack": .47})
        self.assertFalse(quality_gate(baseline, candidate, .001, .02)[0])
        candidate["per_class"]["robot"] = .61
        self.assertTrue(quality_gate(baseline, candidate, .001, .02)[0])
        candidate["map50_95"] = math.nan
        self.assertFalse(quality_gate(baseline, candidate, .001, .02)[0])

    def test_equal_quality_is_not_promoted(self):
        metrics = dict(map50_95=.5, map50=.8, per_class={"robot": .5})
        self.assertFalse(quality_gate(metrics, metrics, 0, .02)[0])

    def test_path_traversal_and_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "base"
            root.mkdir()
            outside = Path(temporary) / "outside.jpg"
            outside.write_bytes(b"jpeg")
            (root / "inside.jpg").symlink_to(outside)
            for entry in ("../outside.jpg", "inside.jpg"):
                with self.assertRaises(ValueError):
                    dataset_images(root, entry)

    def test_atomic_write_replaces_whole_file_without_temp_leaks(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "image.jpg"
            atomic_write(path, b"old")
            atomic_write(path, b"complete new data")
            self.assertEqual(path.read_bytes(), b"complete new data")
            self.assertEqual(list(path.parent.iterdir()), [path])


class RouterTests(unittest.TestCase):
    def setUp(self):
        self.store = Mock(ready=True)
        self.detector = Mock()
        self.registry = Mock()
        self.worker = Mock()
        self.app = FastAPI()
        self.app.include_router(create_active_learning_router(self.store, self.registry, self.detector, self.worker))
        self.client = TestClient(self.app)
        self.url = f"/api/active-learning/models/{MODEL_ID}"

    def authorize(self):
        self.app.dependency_overrides[require_dashboard_user] = lambda: "admin"

    def test_all_routes_require_auth(self):
        for path, method in (("", "get"), ("/snapshots", "post"), ("/samples", "get"), ("/jobs", "post")):
            self.assertEqual(getattr(self.client, method)(self.url + path).status_code, 401)

    def test_snapshot_uses_backend_frame_not_client_image(self):
        self.authorize()
        snapshot = dict(frame_id=42, image="owned by worker")
        self.detector.command.return_value = snapshot
        self.store.capture.return_value = {"id": "sample"}
        response = self.client.post(self.url + "/snapshots", json={"camera_id": "cam"})
        self.assertEqual(response.status_code, 201)
        self.store.capture.assert_called_once_with(MODEL_ID, snapshot, "admin")
        self.assertEqual(self.client.post(self.url + "/snapshots", json={"camera_id": "cam", "image": "forged"}).status_code, 422)

    def test_finite_polygons_and_trusted_checkpoint_required(self):
        self.authorize()
        response = self.client.post(self.url + "/weights", json={"filename": "best.pt", "size_bytes": 99, "trusted": False})
        self.assertEqual(response.status_code, 422)
        response = self.client.post(self.url + "/samples/sample/review", json={"feedback": "corrected", "annotations": [], "complete": True, "image": "forged"})
        self.assertEqual(response.status_code, 422)

    def test_conflict_and_unready_map_to_clear_errors(self):
        self.authorize()
        self.store.enqueue.side_effect = WorkflowConflict("missing checkpoint")
        self.assertEqual(self.client.post(self.url + "/jobs").status_code, 409)
        self.store.ready = False
        self.assertEqual(self.client.get(self.url).status_code, 503)

    def test_upload_chunk_bound(self):
        self.authorize()
        self.store.chunk_size = 4
        response = self.client.put(self.url + "/weights/id/file?offset=0", content=b"12345")
        self.assertEqual(response.status_code, 413)
        self.store.append_weights.assert_not_called()


class WorkerTests(unittest.TestCase):
    def test_idle_worker_status_is_distinct_from_active_job(self):
        worker = ActiveLearningWorker(Mock(), Mock(), Mock())
        worker.thread = Mock()
        worker.thread.is_alive.return_value = True
        status = worker.status()
        self.assertTrue(status["running"])
        self.assertFalse(status["job_active"])
        self.assertEqual(status["state"], "idle")

    def test_no_training_on_live_gpu(self):
        registry = Mock(process=None)
        detector = Mock()
        detector.status.return_value = {"running": True}
        worker = ActiveLearningWorker(Mock(), registry, detector)
        self.assertIn("DeepStream", worker.resource_reason())

    def test_failed_hot_reload_preserves_registry_and_restores_previous_engine(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry = Mock()
            registry.directory.return_value = root
            registry.get.return_value = {"metadata": {}}
            registry.runtime_spec.return_value = {"engine": "previous.engine", "version": "original"}
            detector = Mock(model_update_lock=threading.RLock())
            detector.reload_model.side_effect = [RuntimeError("nvinfer failure"), {"version": "original"}]
            store = Mock(lock=threading.RLock())
            store.inputs_valid.return_value = True
            store.get_job.return_value = {"cancel_requested": False}
            worker = ActiveLearningWorker(store, registry, detector)
            job = dict(id="job", model_id=MODEL_ID, inputs={"baseline": None}, metrics={"accepted": True},
                       version={"id": "b" * 32, "engine_sha256": "hash", "baseline_engine_sha256": "hash"})
            with patch("core.active_learning_worker.checksum", return_value="hash"), self.assertRaises(RuntimeError):
                worker._promote(job)
            registry.activate_version.assert_not_called()
            self.assertEqual(detector.reload_model.call_count, 2)
            detector.reload_model.assert_called_with(MODEL_ID, "previous.engine", "original")


@unittest.skipUnless(os.getenv("ACTIVE_LEARNING_TEST_DSN"), "Use a disposable PostgreSQL database for store integration")
class StoreIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import psycopg2
        from psycopg2 import sql
        cls.admin = psycopg2.connect(os.environ["ACTIVE_LEARNING_TEST_DSN"])
        cls.admin.autocommit = True
        cls.database_name = "al_test_" + uuid.uuid4().hex
        with cls.admin.cursor() as cursor:
            cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(cls.database_name)))
        cls.database = Mock()
        cls.database._get_connection.side_effect = lambda: psycopg2.connect(os.environ["ACTIVE_LEARNING_TEST_DSN"], dbname=cls.database_name)

    @classmethod
    def tearDownClass(cls):
        from psycopg2 import sql
        with cls.admin.cursor() as cursor:
            cursor.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(cls.database_name)))
        cls.admin.close()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.registry = ModelRegistry(self.database)
        self.registry.root = self.root / "models"
        self.registry.root.mkdir()
        self.registry.directory(MODEL_ID).mkdir()
        self.registry.ready = True
        with self.registry.transaction() as cursor:
            cursor.execute("CREATE SCHEMA IF NOT EXISTS vision; CREATE TABLE IF NOT EXISTS vision.models(id TEXT PRIMARY KEY,labels JSONB,state TEXT,metadata JSONB,updated_at TIMESTAMPTZ)")
            cursor.execute("INSERT INTO vision.models VALUES(%s,%s,'ready',%s,NOW()) ON CONFLICT(id) DO UPDATE SET metadata=EXCLUDED.metadata",
                           (MODEL_ID, json.dumps(LABELS), json.dumps(dict(shape=[1, 3, 640, 640]))))
        self.store = ActiveLearningStore(self.database, self.registry)
        self.store.root = self.root / "learning"
        self.store.initialize()
        with self.store.transaction() as cursor:
            cursor.execute("TRUNCATE active_learning.samples,active_learning.jobs,active_learning.settings,active_learning.weights RESTART IDENTITY")

    def tearDown(self):
        self.temporary.cleanup()

    def capture(self, frame=42):
        return self.store.capture(MODEL_ID, dict(image=base64.b64encode(b"\xff\xd8jpeg" + str(frame).encode()).decode(),
            camera_id="cam", frame_id=frame, source_id=2, captured_at=1000, generation="runtime", model_id=MODEL_ID,
            model_version="original", width=1280, height=720, frame_pts_ns=2000, objects=[]), "admin")

    def test_capture_review_idempotence_restart_and_delete(self):
        sample = self.capture()
        self.assertEqual(self.capture()["id"], sample["id"])
        log_path = self.store.directory(MODEL_ID) / "metadata_log.csv"
        with log_path.open(newline="", encoding="utf-8") as source:
            self.assertEqual(len(list(csv.DictReader(source))), 1)
        with self.assertRaises(ValueError):
            self.store.review(MODEL_ID, sample["id"], ANNOTATIONS, "corrected", False, "admin")
        reviewed = self.store.review(MODEL_ID, sample["id"], ANNOTATIONS, "corrected", True, "admin")
        self.assertEqual(reviewed["status"], "approved")
        with log_path.open(newline="", encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
        self.assertEqual([row["event"] for row in rows], ["capture", "review"])
        self.assertEqual(rows[-1]["mask_path"], f"masks/{sample['id']}.json")
        self.assertGreater(reviewed["sequence"], sample["sequence"])
        self.assertEqual(self.store.review(MODEL_ID, sample["id"], ANNOTATIONS, "corrected", True, "admin")["id"], sample["id"])
        self.store.initialize()
        self.assertEqual(self.store.list_samples(MODEL_ID)["counts"]["approved"], 1)
        self.assertTrue(self.store.asset(MODEL_ID, sample["id"], "mask").is_file())
        self.store.delete_sample(MODEL_ID, sample["id"])
        self.assertFalse(self.store.asset(MODEL_ID, sample["id"], "image").exists())
        self.assertEqual(self.store.list_samples(MODEL_ID)["total"], 0)

    def test_weights_upload_offset_and_finalize(self):
        weight = self.store.create_weights(MODEL_ID, "best.pt", 6, "admin")
        with self.assertRaises(WorkflowConflict):
            self.store.append_weights(MODEL_ID, weight["id"], 1, b"abc")
        self.store.append_weights(MODEL_ID, weight["id"], 0, b"abc")
        with self.assertRaises(WorkflowConflict):
            self.store.finish_weights(MODEL_ID, weight["id"])
        self.store.append_weights(MODEL_ID, weight["id"], 3, b"def")
        completed = self.store.finish_weights(MODEL_ID, weight["id"])
        self.assertEqual(completed["state"], "ready")
        self.assertEqual(self.store.finish_weights(MODEL_ID, weight["id"])["sha256"], completed["sha256"])

    def base_dataset(self):
        root = self.store.directory(MODEL_ID) / "base"
        for split in ("train", "val"):
            for index in range(2):
                atomic_write(root / "images" / split / f"{index}.jpg", (split + str(index)).encode())
                atomic_write(root / "labels" / split / f"{index}.txt", f"{index} 0.5 0.5 0.2 0.2\n")
        atomic_write(root / "data.yaml", json.dumps(dict(path=".", train="images/train", val="images/val", names=LABELS)))
        return root

    def enqueue(self):
        self.base_dataset()
        weight = self.store.create_weights(MODEL_ID, "best.pt", 3, "admin")
        self.store.append_weights(MODEL_ID, weight["id"], 0, b"abc")
        self.store.finish_weights(MODEL_ID, weight["id"])
        sample = self.capture()
        self.store.review(MODEL_ID, sample["id"], ANNOTATIONS, "corrected", True, "admin")
        return self.store.enqueue(MODEL_ID, "admin"), sample

    def test_frozen_dataset_replay_split_and_delete_invalidates_job(self):
        job, sample = self.enqueue()
        with self.assertRaises(WorkflowConflict):
            self.store.enqueue(MODEL_ID, "admin")
        output = self.store.directory(MODEL_ID) / "jobs" / job["id"] / "dataset"
        result = prepare_dataset(self.store, job, output)
        self.assertEqual(result["train_images"], 3)
        self.assertEqual(result["validation_images"], 2)
        self.assertTrue((output / "manifest.jsonl").exists())
        self.assertTrue(self.store.inputs_valid(job))
        self.store.delete_sample(MODEL_ID, sample["id"])
        self.assertFalse(self.store.inputs_valid(job))
        self.assertFalse((output / "images/train" / f"correction_{sample['id']}.jpg").exists())
        self.assertTrue(self.store.get_job(MODEL_ID, job["id"])["cancel_requested"])

    def test_training_rejects_leakage_and_missing_validation_class(self):
        job, _ = self.enqueue()
        base = self.store.directory(MODEL_ID) / "base"
        atomic_write(base / "images/val/0.jpg", (base / "images/train/0.jpg").read_bytes())
        with self.assertRaisesRegex(ValueError, "trùng"):
            prepare_dataset(self.store, job, self.root / "leaky")

    def test_settings_reject_external_paths_and_missing_weights(self):
        with self.assertRaises(ValueError):
            self.store.configure(MODEL_ID, dict(DEFAULT_SETTINGS, base_dataset="../../private.yaml"), "admin")
        with self.assertRaises(WorkflowConflict):
            self.store.enqueue(MODEL_ID, "admin")


if __name__ == "__main__":
    unittest.main()
