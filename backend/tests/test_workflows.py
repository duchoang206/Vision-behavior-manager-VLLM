import base64
import copy
import hashlib
import hmac
import json
import os
import time
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from core.dashboard_auth import DEFAULT_PASSWORD_HASH, require_dashboard_user, verify_dashboard_session
from core.workflow_definition import WorkflowDefinition, validate_definition
from core.workflow_runtime import GraphEvaluator, WorkflowRuntime
from core.workflow_store import WorkflowConflict, WorkflowStore
from routers.workflows import create_workflow_router


def definition(*blocks):
    nodes = [{"id": "source", "type": "source", "config": {"camera_ids": ["cam1"]}}]
    nodes.extend({"id": kind, "type": kind, "config": config} for kind, config in blocks)
    nodes.append({"id": "display", "type": "display", "config": {}})
    return WorkflowDefinition(name="Workflow test", nodes=nodes,
                              edges=[{"source": nodes[index]["id"], "target": nodes[index + 1]["id"]} for index in range(len(nodes) - 1)]).model_dump()


def validate(graph):
    return validate_definition(graph, ["cam1", "cam2"], ["Robot_2001"], ["cam1"])


def target(identity=1, horizontal=.4, vertical=.3, **extra):
    return {"id": identity, "class": "person", "category": "person", "confidence": .9,
            "x": horizontal, "y": vertical, "w": .1, "h": .1, **extra}


def evaluator(*blocks):
    graph = definition(*blocks)
    return GraphEvaluator(graph, validate(graph)["order"])


class WorkflowValidationTests(unittest.TestCase):
    def test_branch_merge_and_order(self):
        graph = definition(("detector", {"classes": ["person"]}))
        self.assertTrue(validate(graph)["valid"])
        graph["nodes"].append({"id": "event", "type": "event", "config": {}})
        graph["edges"].append({"source": "detector", "target": "event"})
        self.assertTrue(validate(graph)["valid"])

    def test_cycle_disconnected_duplicate_unknown_rejected(self):
        mutations = [lambda graph: graph["edges"].append({"source": "display", "target": "source"}),
                     lambda graph: graph["edges"].clear(),
                     lambda graph: graph["nodes"].append(graph["nodes"][0]),
                     lambda graph: graph["edges"].append({"source": "missing", "target": "display"}),
                     lambda graph: graph["edges"].append(graph["edges"][0])]
        for change in mutations:
            graph = definition()
            change(graph)
            with self.subTest(change=change):
                self.assertFalse(validate(graph)["valid"])

    def test_missing_camera_and_unavailable_model(self):
        graph = definition(("ppe", {}))
        self.assertFalse(validate(graph)["valid"])
        graph = definition()
        graph["nodes"][0]["config"]["camera_ids"] = ["deleted_camera"]
        self.assertFalse(validate(graph)["valid"])

    def test_config_types_and_geometry(self):
        blocks = [("dwell", {"seconds": "5"}), ("counter", {"count": True}),
                  ("counter", {"count": 1.5}), ("line", {"points": [[.1, .1], [.1, .1]]}),
                  ("roi", {"points": [[0, 0], [1, 1], [1, 0], [0, 1]]}),
                  ("roi", {"points": [[-1, 0], [1, 1], [1, 0]]}),
                  ("detector", {"classes": ["person"], "unknown": 1}),
                  ("matcher", {"labels": ["not_registered"]})]
        for block in blocks:
            with self.subTest(block=block):
                self.assertFalse(validate(definition(block))["valid"])

    def test_nonfinite_and_large_configs_rejected_before_storage(self):
        for config in ({"confidence": float("nan")}, {"data": "x" * 65537}):
            with self.assertRaises(ValidationError):
                definition(("detector", config))

    def test_calibration_and_single_camera_geometry(self):
        graph = definition(("proximity", {"distance_m": 1}))
        self.assertTrue(validate(graph)["valid"])
        graph["nodes"][0]["config"]["camera_ids"] = ["cam2"]
        self.assertFalse(validate(graph)["valid"])
        graph = definition(("line", {"points": [[0, .5], [1, .5]]}))
        graph["nodes"][0]["config"]["camera_ids"] = ["cam1", "cam2"]
        self.assertFalse(validate(graph)["valid"])

    def test_connector_allowlist_and_recording_prerequisite(self):
        graph = definition()
        graph["nodes"][-1] = {"id": "display", "type": "webhook", "config": {"connector": "fms"}}
        with patch.dict(os.environ, {"WORKFLOW_CONNECTORS_JSON": "{}"}):
            self.assertFalse(validate(graph)["valid"])
        with patch.dict(os.environ, {"WORKFLOW_CONNECTORS_JSON": '{"fms":{"url":"https://example.invalid/events"}}'}):
            self.assertTrue(validate(graph)["valid"])
        graph["nodes"][-1] = {"id": "display", "type": "event", "config": {"evidence": True}}
        self.assertFalse(validate_definition(graph, ["cam1"], recording_enabled=False)["valid"])


class WorkflowEvaluatorTests(unittest.TestCase):
    def test_filters_use_existing_class_confidence_identity(self):
        engine = evaluator(("detector", {"classes": ["robot"], "confidence": .6}), ("matcher", {"labels": ["Robot_2001"]}))
        objects = [target(1), target(2, category="robot", label="Robot_2001"), target(3, category="robot", label="Robot_9999")]
        snapshot = copy.deepcopy(objects)
        result, actions = engine.evaluate("cam1", objects, 100)
        self.assertEqual(result["display"]["count"], 1)
        self.assertEqual(result["display"]["objects"][0]["id"], 2)
        self.assertEqual(objects, snapshot)
        self.assertEqual(actions, [])

    def test_polygon_filters_foot_not_bbox_center(self):
        engine = evaluator(("roi", {"points": [[0, .5], [1, .5], [1, 1], [0, 1]]}))
        result, unused = engine.evaluate("cam1", [target(1, vertical=.45), target(2, vertical=.1)], 100)
        self.assertEqual(result["display"]["count"], 1)

    def test_line_finite_segment_and_total(self):
        engine = evaluator(("line", {"points": [[.2, .5], [.8, .5]], "direction": "a_to_b"}))
        engine.evaluate("cam1", [target()], 100)
        result, unused = engine.evaluate("cam1", [target(vertical=.5)], 100.1)
        self.assertEqual(result["line"]["count"], 1)
        self.assertEqual(result["line"]["total_crossings"], 1)
        result, unused = engine.evaluate("cam1", [target(vertical=.3)], 100.2)
        self.assertEqual(result["line"]["count"], 0)
        engine.evaluate("cam1", [target(2, horizontal=.9)], 100.3)
        result, unused = engine.evaluate("cam1", [target(2, horizontal=.9, vertical=.5)], 100.4)
        self.assertEqual(result["line"]["count"], 0)

    def test_dwell_continuous_and_reset_on_missing(self):
        engine = evaluator(("dwell", {"seconds": .5}))
        engine.evaluate("cam1", [target()], 100)
        result, unused = engine.evaluate("cam1", [target()], 100.6)
        self.assertTrue(result["display"]["signal"])
        engine.evaluate("cam1", [], 100.7)
        result, unused = engine.evaluate("cam1", [target()], 100.8)
        self.assertFalse(result["display"]["signal"])
        result, unused = engine.evaluate("cam1", [target()], 104)
        self.assertFalse(result["display"]["signal"])

    def test_zero_counter_but_not_stale_or_predicted_metadata(self):
        engine = evaluator(("detector", {"classes": ["robot"]}), ("counter", {"count": 0, "operator": "eq"}))
        result, unused = engine.evaluate("cam1", [target()], 100)
        self.assertTrue(result["display"]["signal"])
        result, unused = engine.evaluate("cam1", [target(tracking_state="predicted")], 100)
        self.assertFalse(result["display"]["signal"])
        result, unused = engine.evaluate("cam1", [target(observed_at=90)], 100)
        self.assertFalse(result["display"]["signal"])

    def test_counter_does_not_reactivate_blocked_branch(self):
        engine = evaluator(("dwell", {"seconds": 3}), ("counter", {"operator": "eq", "count": 0}))
        result, unused = engine.evaluate("cam1", [target()], 100)
        self.assertFalse(result["display"]["signal"])

    def test_one_stale_track_does_not_hide_other_fresh_tracks(self):
        engine = evaluator(("detector", {"classes": ["person"]}))
        result, unused = engine.evaluate("cam1", [target(1), target(2, tracking_state="predicted")], 100)
        self.assertEqual(result["display"]["count"], 1)
        self.assertTrue(result["display"]["signal"])

    def test_event_empty_does_not_trigger_without_counter(self):
        graph = definition(("detector", {"classes": ["robot"]}))
        graph["nodes"][-1] = {"id": "display", "type": "event", "config": {"cooldown_seconds": 10}}
        engine = GraphEvaluator(graph, validate(graph)["order"])
        result, actions = engine.evaluate("cam1", [target()], 100)
        self.assertEqual(actions, [])
        result, actions = engine.evaluate("cam1", [target(category="robot")], 100.1)
        self.assertEqual(len(actions), 1)
        result, actions = engine.evaluate("cam1", [target(category="robot")], 101)
        self.assertEqual(actions, [])

    def test_camera_state_is_isolated(self):
        engine = evaluator(("line", {"points": [[.2, .5], [.8, .5]]}))
        engine.evaluate("cam1", [target()], 100)
        result, unused = engine.evaluate("cam2", [target(vertical=.5)], 100.1)
        self.assertEqual(result["display"]["count"], 0)

    def test_proximity_requires_valid_calibrated_floor_position(self):
        engine = evaluator(("proximity", {"distance_m": 1.5}))
        objects = [target(1, spatial_valid=True, inside_calibrated_area=True, floor_x=1, floor_y=2),
                   target(2, category="robot", spatial_valid=True, inside_calibrated_area=True, floor_x=2, floor_y=2)]
        result, unused = engine.evaluate("cam1", objects, 100)
        self.assertEqual(result["display"]["count"], 2)
        objects[1]["inside_calibrated_area"] = False
        result, unused = engine.evaluate("cam1", objects, 101)
        self.assertEqual(result["display"]["count"], 0)

    def test_latest_only_queue_and_stale_frames(self):
        runtime = WorkflowRuntime(MagicMock(), validate)
        runtime.camera_ids = {"cam1"}
        now = time.time()
        runtime.publish({"timestamp": now - 3, "streams": [{"cam_id": "cam1", "objects": [target()]}]})
        self.assertEqual(runtime.pending, {})
        runtime.publish({"timestamp": now, "streams": [{"cam_id": "cam1", "objects": [target(1)]}]})
        runtime.publish({"timestamp": now + .01, "streams": [{"cam_id": "cam1", "objects": [target(2)]}]})
        self.assertEqual(len(runtime.pending), 1)
        self.assertEqual(runtime.pending["cam1"][1][0]["id"], 2)
        self.assertEqual(runtime.replaced_frames, 1)
        runtime.publish({"timestamp": now + .01, "streams": [{"cam_id": "cam1", "objects": [target(3)]}]})
        self.assertEqual(runtime.pending["cam1"][1][0]["id"], 2)


class WorkflowAPITests(unittest.TestCase):
    def setUp(self):
        self.store = MagicMock()
        self.runtime = MagicMock()
        self.runtime.status.return_value = {"healthy": True}
        self.app = FastAPI()
        self.app.include_router(create_workflow_router(self.store, self.runtime, lambda: {}, validate))
        self.client = TestClient(self.app)

    def test_authentication_required(self):
        self.assertEqual(self.client.get("/api/workflows").status_code, 401)
        self.assertEqual(self.client.post("/api/workflows", json={"definition": definition()}).status_code, 401)
        self.assertFalse(self.store.save.called)

    def test_backend_rejects_unknown_command_and_revision(self):
        self.app.dependency_overrides[require_dashboard_user] = lambda: "admin"
        self.assertEqual(self.client.post("/api/workflows/deployments/abc/action", json={"action": "execute_shell"}).status_code, 422)
        self.assertEqual(self.client.put("/api/workflows/abc", json={"definition": definition()}).status_code, 422)
        self.store.deploy.side_effect = WorkflowConflict("stale revision")
        self.assertEqual(self.client.post("/api/workflows/abc/deploy", json={"revision": 1}).status_code, 409)

    def test_tampered_and_expired_session(self):
        secret = "workflow-test-secret-only-" * 2
        now = int(time.time())
        payload = {"username": "admin", "issuedAt": now, "expiresAt": now + 28800, "nonce": "a" * 32}
        encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
        key = hmac.new(secret.encode(), f"admin:r-skyview-admin-v1:{DEFAULT_PASSWORD_HASH}".encode(), hashlib.sha256).digest()
        signature = base64.urlsafe_b64encode(hmac.new(key, encoded.encode(), hashlib.sha256).digest()).decode().rstrip("=")
        token = f"{encoded}.{signature}"
        with patch.dict(os.environ, {"RSKYVIEW_SESSION_SECRET": secret}):
            self.assertIsNotNone(verify_dashboard_session(token, now))
            self.assertIsNone(verify_dashboard_session(token + "x", now))
            self.assertIsNone(verify_dashboard_session(token, now + 28800))


@unittest.skipUnless(os.getenv("WORKFLOW_TEST_DATABASE_URL"), "Use a dedicated empty test database, never the production database")
class WorkflowPostgreSQLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import psycopg2
        url = os.environ["WORKFLOW_TEST_DATABASE_URL"]
        if "workflow_test_" not in url:
            raise RuntimeError("Test database name must include workflow_test_")
        database = MagicMock()
        database._get_connection.side_effect = lambda: psycopg2.connect(url)
        cls.store = WorkflowStore(database)
        cls.store.initialize()
        with cls.store.transaction() as cursor:
            cursor.execute("CREATE TABLE IF NOT EXISTS events(cam_id TEXT,rule_id TEXT,rule_type TEXT,severity TEXT,description TEXT,snapshot_bbox JSONB,floor_pos JSONB,timestamp TIMESTAMP,video_file TEXT,evidence_requested BOOLEAN DEFAULT TRUE)")

    def test_revision_immutable_deploy_and_lifecycle(self):
        graph = definition()
        row = self.store.save(graph, "test")
        deployed = self.store.deploy(row["id"], row["revision"], "test", validate)
        graph["name"] = "changed draft"
        updated = self.store.save(graph, "test", row["id"], 1)
        self.assertEqual(self.store.active()[0]["definition"]["name"], "Workflow test")
        with self.assertRaises(WorkflowConflict):
            self.store.save(graph, "test", row["id"], 1)
        with self.assertRaises(WorkflowConflict):
            self.store.deploy(row["id"], 2, "test", validate)
        with self.assertRaises(WorkflowConflict):
            self.store.delete(row["id"], 2, "test")
        self.store.transition(deployed["id"], "pause", "test", validate)
        self.assertEqual(self.store.active(), [])
        self.store.transition(deployed["id"], "resume", "test", validate)
        self.assertEqual(len(self.store.active()), 1)
        self.store.transition(deployed["id"], "stop", "test", validate)
        with self.assertRaises(WorkflowConflict):
            self.store.transition(deployed["id"], "resume", "test", validate)
        self.store.delete(row["id"], updated["revision"], "test")

    def test_event_generation_and_outbox(self):
        row = self.store.save(definition(), "test")
        deployed = self.store.deploy(row["id"], 1, "test", validate)
        event = {"deployment_id": deployed["id"], "deployment_generation": deployed["updated_at"].isoformat(), "pipeline_id": row["id"],
                 "description": "test", "severity": "info", "cam_id": "cam1", "node_id": "event", "timestamp": time.time(),
                 "event_id": "outbox-test", "connector": "test"}
        self.assertTrue(self.store.write_event(event))
        delivery = self.store.claim_delivery()
        self.assertEqual(delivery["id"], "outbox-test")
        self.store.finish_delivery(delivery)
        self.assertIsNone(self.store.claim_delivery())
        self.store.transition(deployed["id"], "pause", "test", validate)
        self.store.transition(deployed["id"], "resume", "test", validate)
        self.assertFalse(self.store.write_event(event))
        self.assertGreater(len(self.store.logs(deployed["id"])["logs"]), 1)
        self.store.transition(deployed["id"], "stop", "test", validate)
        self.store.delete(row["id"], 1, "test")

    def test_evidence_checkbox_controls_recording_link(self):
        from core.recording_store import RecordingStore
        recordings = RecordingStore(self.store.database)
        recordings.initialize()
        started = time.time()
        with self.store.transaction() as cursor:
            for enabled in (True, False):
                cursor.execute("INSERT INTO events(cam_id,timestamp,evidence_requested) VALUES('evidence-camera',to_timestamp(%s) AT TIME ZONE 'UTC',%s)", (started + 5, enabled))
        recordings.save('evidence-camera', 'evidence-camera/test.mp4', started, 'ready', 1000, duration=60, codec='h264')
        with self.store.transaction() as cursor:
            cursor.execute("SELECT evidence_requested,video_file FROM events WHERE cam_id='evidence-camera'")
            result = {row['evidence_requested']: row['video_file'] for row in cursor.fetchall()}
        self.assertEqual(result[True], 'evidence-camera/test.mp4')
        self.assertIsNone(result[False])


if __name__ == "__main__":
    unittest.main()
