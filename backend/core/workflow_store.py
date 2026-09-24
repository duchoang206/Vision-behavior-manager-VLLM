import uuid
from contextlib import contextmanager

from psycopg2.extras import Json, RealDictCursor


class WorkflowConflict(ValueError):
    pass


class WorkflowStore:
    def __init__(self, database):
        self.database = database

    @contextmanager
    def transaction(self):
        connection = self.database._get_connection()
        try:
            with connection:
                with connection.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute("SET LOCAL statement_timeout = '5s'")
                    yield cursor
        finally:
            connection.close()

    def initialize(self):
        with self.transaction() as cursor:
            cursor.execute("""
                CREATE SCHEMA IF NOT EXISTS workflow;
                CREATE TABLE IF NOT EXISTS workflow.pipelines (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, definition JSONB NOT NULL,
                    revision INTEGER NOT NULL DEFAULT 1, actor TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                CREATE TABLE IF NOT EXISTS workflow.deployments (
                    id TEXT PRIMARY KEY, pipeline_id TEXT NOT NULL REFERENCES workflow.pipelines(id),
                    revision INTEGER NOT NULL, definition JSONB NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('running','paused','stopped')),
                    actor TEXT NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                CREATE UNIQUE INDEX IF NOT EXISTS workflow_active_pipeline ON workflow.deployments(pipeline_id) WHERE status <> 'stopped';
                CREATE TABLE IF NOT EXISTS workflow.logs (
                    id BIGSERIAL PRIMARY KEY, deployment_id TEXT, pipeline_id TEXT, level TEXT NOT NULL,
                    message TEXT NOT NULL, details JSONB NOT NULL DEFAULT '{}', created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS workflow_logs_deployment ON workflow.logs(deployment_id, id DESC);
                CREATE TABLE IF NOT EXISTS workflow.deliveries (
                    id TEXT PRIMARY KEY, deployment_id TEXT NOT NULL, connector TEXT NOT NULL, payload JSONB NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0,
                    available_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), error TEXT,
                    expires_at TIMESTAMPTZ NOT NULL DEFAULT NOW() + INTERVAL '60 seconds',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                ALTER TABLE workflow.deliveries ADD COLUMN IF NOT EXISTS expires_at TIMESTAMPTZ NOT NULL DEFAULT NOW() + INTERVAL '60 seconds';
                CREATE INDEX IF NOT EXISTS workflow_delivery_pending ON workflow.deliveries(status, available_at);
            """)

    @staticmethod
    def log(cursor, deployment_id, pipeline_id, message, details=None, level="info"):
        cursor.execute("INSERT INTO workflow.logs(deployment_id,pipeline_id,level,message,details) VALUES(%s,%s,%s,%s,%s)",
                       (deployment_id, pipeline_id, level, message, Json(details or {})))

    def list(self):
        with self.transaction() as cursor:
            cursor.execute("SELECT * FROM workflow.pipelines ORDER BY updated_at DESC LIMIT 200")
            pipelines = [dict(row) for row in cursor.fetchall()]
            cursor.execute("SELECT * FROM workflow.deployments ORDER BY created_at DESC LIMIT 300")
            return {"pipelines": pipelines, "deployments": [dict(row) for row in cursor.fetchall()]}

    def get(self, pipeline_id, cursor=None):
        if cursor is None:
            with self.transaction() as transaction:
                return self.get(pipeline_id, transaction)
        cursor.execute("SELECT * FROM workflow.pipelines WHERE id=%s FOR UPDATE", (pipeline_id,))
        row = cursor.fetchone()
        if not row:
            raise KeyError("Không tìm thấy pipeline.")
        return dict(row)

    def save(self, definition, actor, pipeline_id=None, revision=None):
        with self.transaction() as cursor:
            if pipeline_id:
                row = self.get(pipeline_id, cursor)
                if row["revision"] != revision:
                    raise WorkflowConflict("Bản nháp đã thay đổi ở phiên khác; tải lại trước khi lưu.")
                cursor.execute("UPDATE workflow.pipelines SET definition=%s,name=%s,revision=revision+1,actor=%s,updated_at=NOW() WHERE id=%s RETURNING *",
                               (Json(definition), definition["name"], actor, pipeline_id))
            else:
                pipeline_id = uuid.uuid4().hex
                cursor.execute("INSERT INTO workflow.pipelines(id,name,definition,actor) VALUES(%s,%s,%s,%s) RETURNING *",
                               (pipeline_id, definition["name"], Json(definition), actor))
            result = dict(cursor.fetchone())
            self.log(cursor, None, pipeline_id, "Đã lưu bản nháp", {"actor": actor, "revision": result["revision"]})
            return result

    def deploy(self, pipeline_id, revision, actor, validator, start_paused=False):
        with self.transaction() as cursor:
            cursor.execute("SELECT pg_advisory_xact_lock(724091)")
            row = self.get(pipeline_id, cursor)
            if row["revision"] != revision:
                raise WorkflowConflict("Bản nháp đã thay đổi; không triển khai phiên bản cũ.")
            validation = validator(row["definition"])
            if not validation["valid"]:
                raise WorkflowConflict("; ".join(validation["errors"]))
            self.check_detector_conflict(cursor, row["definition"])
            cursor.execute("SELECT count(*) AS total FROM workflow.deployments WHERE status <> 'stopped'")
            if cursor.fetchone()["total"] >= 32:
                raise WorkflowConflict("Tối đa 32 pipeline hoạt động trên máy biên này.")
            cursor.execute("SELECT id FROM workflow.deployments WHERE pipeline_id=%s AND status <> 'stopped'", (pipeline_id,))
            if cursor.fetchone():
                raise WorkflowConflict("Dừng bản deploy hiện tại trước khi triển khai phiên bản mới.")
            deployment_id = uuid.uuid4().hex
            cursor.execute("INSERT INTO workflow.deployments(id,pipeline_id,revision,definition,status,actor) VALUES(%s,%s,%s,%s,%s,%s) RETURNING *",
                           (deployment_id, pipeline_id, revision, Json(row["definition"]), "paused" if start_paused else "running", actor))
            result = dict(cursor.fetchone())
            self.log(cursor, deployment_id, pipeline_id, "Đã deploy pipeline", {"actor": actor, "revision": revision, "status": result["status"]})
            return result

    def transition(self, deployment_id, action, actor, validator):
        with self.transaction() as cursor:
            cursor.execute("SELECT pg_advisory_xact_lock(724091)")
            cursor.execute("SELECT * FROM workflow.deployments WHERE id=%s FOR UPDATE", (deployment_id,))
            row = cursor.fetchone()
            if not row:
                raise KeyError("Không tìm thấy deployment.")
            allowed = {"pause": ("running", "paused"), "resume": ("paused", "running"), "stop": (("running", "paused"), "stopped")}
            source, target = allowed[action]
            if row["status"] not in (source if isinstance(source, tuple) else (source,)):
                raise WorkflowConflict("Lệnh không hợp lệ với trạng thái hiện tại.")
            if action == "resume":
                self.check_detector_conflict(cursor, row["definition"])
                validation = validator(row["definition"])
                if not validation["valid"]:
                    raise WorkflowConflict("; ".join(validation["errors"]))
            cursor.execute("UPDATE workflow.deployments SET status=%s,updated_at=NOW() WHERE id=%s RETURNING *", (target, deployment_id))
            result = dict(cursor.fetchone())
            if action in {"pause", "stop"}:
                cursor.execute("UPDATE workflow.deliveries SET status='cancelled' WHERE deployment_id=%s AND status IN ('pending','sending')", (deployment_id,))
            self.log(cursor, deployment_id, row["pipeline_id"], f"Pipeline: {target}", {"actor": actor})
            return result

    @staticmethod
    def check_detector_conflict(cursor, definition):
        # A workflow may now use more than one model.  Admission and failure
        # isolation are handled by the per-model DeepStream supervisor, rather
        # than a database singleton lock.
        return None

    def delete(self, pipeline_id, revision, actor):
        with self.transaction() as cursor:
            row = self.get(pipeline_id, cursor)
            if row["revision"] != revision:
                raise WorkflowConflict("Bản nháp đã thay đổi; tải lại trước khi xóa.")
            cursor.execute("SELECT id FROM workflow.deployments WHERE pipeline_id=%s AND status <> 'stopped'", (pipeline_id,))
            if cursor.fetchone():
                raise WorkflowConflict("Dừng tất cả bản deploy trước khi xóa pipeline.")
            cursor.execute("DELETE FROM workflow.deployments WHERE pipeline_id=%s", (pipeline_id,))
            cursor.execute("DELETE FROM workflow.pipelines WHERE id=%s", (pipeline_id,))
            self.log(cursor, None, pipeline_id, "Đã xóa pipeline; giữ nhật ký", {"actor": actor, "name": row["name"]})

    def active(self):
        with self.transaction() as cursor:
            cursor.execute("SELECT * FROM workflow.deployments WHERE status='running'")
            return [dict(row) for row in cursor.fetchall()]

    def logs(self, deployment_id, before=None):
        with self.transaction() as cursor:
            cursor.execute("SELECT * FROM workflow.logs WHERE deployment_id=%s AND (%s IS NULL OR id < %s) ORDER BY id DESC LIMIT 101", (deployment_id, before, before))
            rows = [dict(row) for row in cursor.fetchall()]
            return {"logs": rows[:100], "has_more": len(rows) > 100}

    def write_event(self, event):
        with self.transaction() as cursor:
            cursor.execute("SELECT status,updated_at FROM workflow.deployments WHERE id=%s FOR UPDATE", (event["deployment_id"],))
            row = cursor.fetchone()
            if not row or row["status"] != "running" or row["updated_at"].isoformat() != event["deployment_generation"]:
                return False
            self.log(cursor, event["deployment_id"], event["pipeline_id"], event["description"], event, event["severity"])
            cursor.execute("""INSERT INTO events(cam_id,rule_id,rule_type,severity,description,snapshot_bbox,floor_pos,timestamp,evidence_requested)
                VALUES(%s,%s,'workflow',%s,%s,%s,%s,to_timestamp(%s) AT TIME ZONE 'UTC',%s)""",
                           (event["cam_id"], event["node_id"], event["severity"], event["description"], Json(event.get("bbox", [])), Json(event.get("fms_position")), event["timestamp"], event.get("evidence_requested", False)))
            if event.get("connector"):
                cursor.execute("INSERT INTO workflow.deliveries(id,deployment_id,connector,payload,expires_at) VALUES(%s,%s,%s,%s,to_timestamp(%s)+INTERVAL '60 seconds')",
                               (event["event_id"], event["deployment_id"], event["connector"], Json(event), event["timestamp"]))
            return True

    def claim_delivery(self):
        with self.transaction() as cursor:
            cursor.execute("UPDATE workflow.deliveries SET status='expired',error='Event older than 60 seconds' WHERE status IN ('pending','sending') AND expires_at<=NOW()")
            cursor.execute("""SELECT delivery.* FROM workflow.deliveries delivery
                JOIN workflow.deployments deployment ON deployment.id=delivery.deployment_id
                WHERE deployment.status='running' AND delivery.status IN ('pending','sending')
                AND delivery.available_at <= NOW() AND delivery.expires_at>NOW() AND delivery.attempts<5 ORDER BY delivery.created_at
                FOR UPDATE OF delivery SKIP LOCKED LIMIT 1""")
            row = cursor.fetchone()
            if not row:
                return None
            cursor.execute("UPDATE workflow.deliveries SET status='sending',attempts=attempts+1,available_at=NOW()+INTERVAL '60 seconds' WHERE id=%s", (row["id"],))
            return dict(row, attempts=row["attempts"] + 1)

    def finish_delivery(self, delivery, error=None):
        with self.transaction() as cursor:
            status = "sent" if error is None else "failed" if delivery["attempts"] >= 5 else "pending"
            cursor.execute("UPDATE workflow.deliveries SET status=%s,error=%s,available_at=NOW()+(%s * INTERVAL '1 second') WHERE id=%s AND status='sending'",
                           (status, error, min(300, 2 ** delivery["attempts"]), delivery["id"]))
            self.log(cursor, delivery["deployment_id"], delivery["payload"]["pipeline_id"], "Webhook " + status,
                     {"delivery_id": delivery["id"], "attempts": delivery["attempts"], "error": error}, "error" if error else "info")
