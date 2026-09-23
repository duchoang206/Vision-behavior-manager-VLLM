import base64
import csv
import hashlib
import logging
import os
from pathlib import Path
import re
import shutil
import threading
import uuid

from psycopg2.extras import Json

from core.active_learning_data import atomic_write, checksum, json_bytes, validate_annotations
from core.workflow_store import WorkflowConflict, WorkflowStore


DEFAULT_SETTINGS = dict(enabled=False, threshold=500, daily_hour=0, timezone="Asia/Bangkok",
                        epochs=5, learning_rate=.0001, batch=2, minimum_gain=.001,
                        max_class_drop=.02, auto_promote=False, base_dataset="base/data.yaml")
ACTIVE_STATES = ("queued", "waiting_resources", "preparing", "training", "building", "promotion_queued", "promoting")


class ActiveLearningStore(WorkflowStore):
    chunk_size = 2 * 1024**2
    max_weights_size = 512 * 1024**2

    def __init__(self, database, registry):
        super().__init__(database)
        self.registry = registry
        self.root = Path(os.getenv("ACTIVE_LEARNING_DIR", str(Path(__file__).resolve().parents[1] / "data/active_learning"))).resolve()
        self.lock = threading.RLock()
        self.ready = False

    def initialize(self):
        self.root.mkdir(parents=True, exist_ok=True)
        with self.transaction() as cursor:
            cursor.execute("""CREATE SCHEMA IF NOT EXISTS active_learning;
                CREATE TABLE IF NOT EXISTS active_learning.samples (
                    id TEXT PRIMARY KEY, sequence BIGSERIAL UNIQUE, model_id TEXT NOT NULL REFERENCES vision.models(id),
                    camera_id TEXT NOT NULL, frame_id BIGINT NOT NULL, source_id INTEGER,
                    captured_at DOUBLE PRECISION NOT NULL, generation TEXT NOT NULL,
                    model_version TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'draft',
                    frame JSONB NOT NULL, predictions JSONB NOT NULL, annotations JSONB NOT NULL DEFAULT '[]',
                    annotation_hash TEXT, feedback TEXT, complete BOOLEAN NOT NULL DEFAULT FALSE,
                    actor TEXT NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    UNIQUE(model_id,camera_id,generation,frame_id));
                CREATE INDEX IF NOT EXISTS al_samples_model ON active_learning.samples(model_id,sequence DESC);
                CREATE TABLE IF NOT EXISTS active_learning.settings (
                    model_id TEXT PRIMARY KEY REFERENCES vision.models(id), config JSONB NOT NULL DEFAULT '{}',
                    actor TEXT NOT NULL, updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW());
                CREATE TABLE IF NOT EXISTS active_learning.weights (
                    id TEXT PRIMARY KEY, model_id TEXT NOT NULL REFERENCES vision.models(id), filename TEXT NOT NULL,
                    size_bytes BIGINT NOT NULL, received_bytes BIGINT NOT NULL DEFAULT 0, sha256 TEXT,
                    state TEXT NOT NULL DEFAULT 'uploading', actor TEXT NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW());
                CREATE TABLE IF NOT EXISTS active_learning.jobs (
                    id TEXT PRIMARY KEY, model_id TEXT NOT NULL REFERENCES vision.models(id), state TEXT NOT NULL,
                    reason TEXT, trigger TEXT NOT NULL, config JSONB NOT NULL, inputs JSONB NOT NULL,
                    max_sequence BIGINT NOT NULL, metrics JSONB NOT NULL DEFAULT '{}', version JSONB,
                    cancel_requested BOOLEAN NOT NULL DEFAULT FALSE, actor TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW());
                CREATE UNIQUE INDEX IF NOT EXISTS al_one_active_job ON active_learning.jobs(model_id)
                    WHERE state IN ('queued','waiting_resources','preparing','training','building','promotion_queued','promoting');
                UPDATE active_learning.jobs job SET state='completed',reason='Khôi phục xác nhận version đã lưu.',updated_at=NOW()
                    FROM vision.models model WHERE job.model_id=model.id AND job.state='promoting'
                    AND job.version->>'id'=model.metadata->>'active_learning_version';
                UPDATE active_learning.jobs SET state='failed',reason='Backend đã khởi động lại; không tự promote job bị ngắt.',updated_at=NOW()
                    WHERE state IN ('preparing','training','building','promoting');""")
        self.ready = True
        self.cleanup_drafts()

    def directory(self, model_id):
        self.registry.directory(model_id)
        return self.root / model_id

    def asset(self, model_id, identifier, kind):
        if not re.fullmatch(r"[a-f0-9]{32}", identifier):
            raise KeyError("ID dữ liệu không hợp lệ.")
        folder, suffix = {"image": ("images", ".jpg"), "mask": ("masks", ".json"),
                          "metadata": ("metadata", ".json"), "weights": ("weights", ".pt")}[kind]
        return self.directory(model_id) / folder / (identifier + suffix)

    def _append_metadata_log(self, model_id, values):
        path = self.directory(model_id) / "metadata_log.csv"
        fields = ("event", "sample_id", "model_id", "camera_id", "source_id", "frame_id",
                  "captured_at", "generation", "model_version", "status", "feedback", "complete",
                  "actor", "image_path", "mask_path", "predictions_count", "annotations_count",
                  "annotation_sha256")
        try:
            with self.lock:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", newline="", encoding="utf-8") as output:
                    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
                    if output.tell() == 0:
                        writer.writeheader()
                    writer.writerow({field: values.get(field, "") for field in fields})
                    output.flush()
                    os.fsync(output.fileno())
        except OSError:
            logging.exception("Active Learning metadata log write failed for %s", model_id)

    def capture(self, model_id, snapshot, actor):
        self.registry.get(model_id)
        if not self.ready:
            raise WorkflowConflict("Kho Active Learning chưa sẵn sàng.")
        image = base64.b64decode(snapshot["image"], validate=True)
        if not 4 <= len(image) <= 8 * 1024**2 or not image.startswith(b"\xff\xd8"):
            raise ValueError("Ảnh GPU không phải JPEG hợp lệ hoặc quá lớn.")
        if not snapshot.get("generation") or snapshot.get("model_id") != model_id:
            raise WorkflowConflict("Snapshot thiếu thế hệ runtime/model; cần khởi động worker bản mới.")
        identifier = uuid.uuid4().hex
        frame = {key: value for key, value in snapshot.items() if key not in {"image", "objects", "snapshot_id", "expires_in"}}
        frame["image_sha256"] = hashlib.sha256(image).hexdigest()
        with self.lock, self.transaction() as cursor:
            cursor.execute("SELECT pg_advisory_xact_lock(829401)")
            cursor.execute("SELECT COUNT(*) AS count FROM active_learning.samples WHERE status='draft'")
            if cursor.fetchone()["count"] >= 32 or shutil.disk_usage(self.root).free < len(image) + 2 * 1024**3:
                raise WorkflowConflict("Kho ảnh nháp đầy hoặc SSD còn dưới 2 GiB. Xóa ảnh nháp rồi thử lại.")
            cursor.execute("""SELECT * FROM active_learning.samples WHERE model_id=%s AND camera_id=%s
                AND generation=%s AND frame_id=%s""", (model_id, snapshot["camera_id"], snapshot["generation"], snapshot["frame_id"]))
            previous = cursor.fetchone()
            if previous:
                return dict(previous)
            path = self.asset(model_id, identifier, "image")
            atomic_write(path, image)
            try:
                cursor.execute("""INSERT INTO active_learning.samples
                    (id,model_id,camera_id,frame_id,source_id,captured_at,generation,model_version,frame,predictions,actor)
                    VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING *""",
                    (identifier, model_id, snapshot["camera_id"], snapshot["frame_id"], snapshot.get("source_id"),
                     snapshot["captured_at"], snapshot["generation"], snapshot.get("model_version", "original"),
                     Json(frame), Json(snapshot.get("objects", [])), actor))
                row = dict(cursor.fetchone())
            except Exception:
                path.unlink(missing_ok=True)
                raise
        self._append_metadata_log(model_id, dict(
            event="capture", sample_id=row["id"], model_id=model_id, camera_id=row["camera_id"],
            source_id=row["source_id"], frame_id=row["frame_id"], captured_at=row["captured_at"],
            generation=row["generation"], model_version=row["model_version"], status=row["status"],
            actor=actor, image_path=f"images/{row['id']}.jpg",
            predictions_count=len(row.get("predictions") or [])))
        return row

    def get_sample(self, model_id, identifier, cursor=None):
        self.asset(model_id, identifier, "image")
        if cursor is None:
            with self.transaction() as transaction:
                return self.get_sample(model_id, identifier, transaction)
        cursor.execute("SELECT * FROM active_learning.samples WHERE model_id=%s AND id=%s FOR UPDATE", (model_id, identifier))
        row = cursor.fetchone()
        if not row:
            raise KeyError("Mẫu không tồn tại hoặc đã xóa.")
        return dict(row)

    def review(self, model_id, identifier, annotations, feedback, complete, actor):
        labels = self.registry.get(model_id)["labels"]
        approved = feedback != "reject"
        if approved and not complete:
            raise ValueError("Phải kiểm tra TẤT CẢ vật trong frame, tránh huấn luyện ảnh thiếu nhãn.")
        annotations = validate_annotations(annotations, labels) if approved else []
        digest = hashlib.sha256(json_bytes(annotations)).hexdigest()
        with self.lock, self.transaction() as cursor:
            row = self.get_sample(model_id, identifier, cursor)
            if row["status"] != "draft":
                if row["feedback"] == feedback and row["annotation_hash"] == digest:
                    return row
                raise WorkflowConflict("Mẫu đã chốt, hãy tạo ảnh mới hoặc xóa mẫu này.")
            if not self.asset(model_id, identifier, "image").is_file():
                raise WorkflowConflict("Ảnh không còn trên SSD.")
            status = "approved" if approved else "rejected"
            atomic_write(self.asset(model_id, identifier, "mask"), json_bytes(dict(annotations=annotations, labels=labels)))
            atomic_write(self.asset(model_id, identifier, "metadata"), json_bytes(dict(
                id=identifier, model_id=model_id, camera_id=row["camera_id"], frame=row["frame"],
                image_path=f"images/{identifier}.jpg", mask_path=f"masks/{identifier}.json",
                source_id=row["source_id"], frame_id=row["frame_id"], captured_at=row["captured_at"],
                status=status, feedback=feedback, actor=actor, annotation_hash=digest)))
            cursor.execute("""UPDATE active_learning.samples SET status=%s,annotations=%s,annotation_hash=%s,
                feedback=%s,complete=%s,actor=%s,updated_at=NOW(),
                sequence=nextval(pg_get_serial_sequence('active_learning.samples','sequence')) WHERE id=%s RETURNING *""",
                (status, Json(annotations), digest, feedback, approved, actor, identifier))
            result = dict(cursor.fetchone())
        self._append_metadata_log(model_id, dict(
            event="review", sample_id=result["id"], model_id=model_id, camera_id=result["camera_id"],
            source_id=result["source_id"], frame_id=result["frame_id"], captured_at=result["captured_at"],
            generation=result["generation"], model_version=result["model_version"], status=result["status"],
            feedback=result["feedback"], complete=result["complete"], actor=actor,
            image_path=f"images/{identifier}.jpg", mask_path=f"masks/{identifier}.json",
            predictions_count=len(result.get("predictions") or []),
            annotations_count=len(result.get("annotations") or []), annotation_sha256=digest))
        return result

    def list_samples(self, model_id, status=None, limit=24, offset=0):
        self.registry.get(model_id)
        with self.transaction() as cursor:
            cursor.execute("SELECT status,COUNT(*) AS count FROM active_learning.samples WHERE model_id=%s GROUP BY status", (model_id,))
            counts = {row["status"]: row["count"] for row in cursor.fetchall()}
            cursor.execute("""SELECT * FROM active_learning.samples WHERE model_id=%s AND (%s IS NULL OR status=%s)
                ORDER BY sequence DESC LIMIT %s OFFSET %s""", (model_id, status, status, limit, offset))
            rows = [dict(row) for row in cursor.fetchall()]
        return dict(samples=rows, counts=counts, total=counts.get(status, 0) if status else sum(counts.values()))

    def delete_sample(self, model_id, identifier):
        with self.lock, self.transaction() as cursor:
            self.get_sample(model_id, identifier, cursor)
            cursor.execute("SELECT id,state FROM active_learning.jobs WHERE model_id=%s AND inputs->'samples' @> %s::jsonb",
                           (model_id, Json([dict(id=identifier)])))
            jobs = list(cursor.fetchall())
            if any(job["state"] == "promoting" for job in jobs):
                raise WorkflowConflict("Đang đổi engine dùng mẫu này; chờ hoàn tất trước khi xóa.")
            cursor.execute("""UPDATE active_learning.jobs SET cancel_requested=TRUE WHERE model_id=%s AND inputs->'samples' @> %s::jsonb
                AND state=ANY(%s)""", (model_id, Json([dict(id=identifier)]), list(ACTIVE_STATES)))
            cursor.execute("DELETE FROM active_learning.samples WHERE model_id=%s AND id=%s", (model_id, identifier))
        for kind in ("image", "mask", "metadata"):
            self.asset(model_id, identifier, kind).unlink(missing_ok=True)
        for job in jobs:
            dataset = self.directory(model_id) / "jobs" / job["id"] / "dataset"
            for relative in (f"images/train/correction_{identifier}.jpg", f"labels/train/correction_{identifier}.txt", f"masks/correction_{identifier}.json"):
                (dataset / relative).unlink(missing_ok=True)
        return dict(deleted=True)

    def cleanup_drafts(self):
        with self.transaction() as cursor:
            cursor.execute("DELETE FROM active_learning.samples WHERE status='draft' AND created_at<NOW()-INTERVAL '1 hour' RETURNING id,model_id")
            expired = list(cursor.fetchall())
        for row in expired:
            self.asset(row["model_id"], row["id"], "image").unlink(missing_ok=True)

    def settings(self, model_id):
        self.registry.get(model_id)
        with self.transaction() as cursor:
            cursor.execute("SELECT config FROM active_learning.settings WHERE model_id=%s", (model_id,))
            row = cursor.fetchone()
        return {**DEFAULT_SETTINGS, **(row["config"] if row else {})}

    def configure(self, model_id, config, actor):
        self.registry.get(model_id)
        from zoneinfo import ZoneInfo
        try:
            ZoneInfo(config["timezone"])
        except (KeyError, ValueError) as error:
            raise ValueError("Múi giờ không hợp lệ.") from error
        base = self.directory(model_id) / "base"
        path = (self.directory(model_id) / config["base_dataset"]).resolve()
        if not path.is_relative_to(base.resolve()) or path.suffix not in {".yaml", ".yml"}:
            raise ValueError("Dataset gốc phải nằm trong base/ của model, định dạng YAML.")
        with self.transaction() as cursor:
            cursor.execute("""INSERT INTO active_learning.settings(model_id,config,actor) VALUES(%s,%s,%s)
                ON CONFLICT(model_id) DO UPDATE SET config=EXCLUDED.config,actor=EXCLUDED.actor,updated_at=NOW()""",
                (model_id, Json(config), actor))
        return self.settings(model_id)

    def weights(self, model_id):
        with self.transaction() as cursor:
            cursor.execute("SELECT * FROM active_learning.weights WHERE model_id=%s ORDER BY created_at DESC LIMIT 20", (model_id,))
            return [dict(row) for row in cursor.fetchall()]

    def create_weights(self, model_id, filename, size, actor):
        self.registry.get(model_id)
        if not filename.lower().endswith(".pt") or not 1 <= size <= self.max_weights_size:
            raise ValueError("Chọn checkpoint .pt đáng tin cậy, tối đa 512 MiB.")
        identifier = uuid.uuid4().hex
        with self.lock, self.transaction() as cursor:
            cursor.execute("SELECT pg_advisory_xact_lock(829402)")
            cursor.execute("SELECT COALESCE(SUM(size_bytes-received_bytes),0) AS reserved FROM active_learning.weights WHERE state='uploading'")
            if shutil.disk_usage(self.root).free < int(cursor.fetchone()["reserved"]) + size + 2 * 1024**3:
                raise WorkflowConflict("SSD thiếu dung lượng để nhận checkpoint.")
            atomic_write(self.asset(model_id, identifier, "weights").with_suffix(".part"), b"")
            cursor.execute("""INSERT INTO active_learning.weights(id,model_id,filename,size_bytes,actor)
                VALUES(%s,%s,%s,%s,%s) RETURNING *""", (identifier, model_id, Path(filename).name, size, actor))
            return dict(cursor.fetchone())

    def append_weights(self, model_id, identifier, offset, content):
        path = self.asset(model_id, identifier, "weights")
        if not content or len(content) > self.chunk_size:
            raise ValueError("Mỗi chunk phải từ 1 byte đến 2 MiB.")
        with self.lock, self.transaction() as cursor:
            cursor.execute("SELECT * FROM active_learning.weights WHERE id=%s AND model_id=%s FOR UPDATE", (identifier, model_id))
            row = cursor.fetchone()
            if not row:
                raise KeyError("Không tìm thấy upload.")
            if row["state"] != "uploading" or row["received_bytes"] != offset or offset + len(content) > row["size_bytes"]:
                raise WorkflowConflict("Offset upload không khớp.")
            if shutil.disk_usage(self.root).free < len(content) + 1024**3:
                raise WorkflowConflict("SSD sắp đầy.")
            with path.with_suffix(".part").open("r+b") as target:
                target.seek(offset)
                target.write(content)
                target.truncate()
                target.flush()
                os.fsync(target.fileno())
            cursor.execute("UPDATE active_learning.weights SET received_bytes=%s WHERE id=%s RETURNING *", (offset + len(content), identifier))
            return dict(cursor.fetchone())

    def finish_weights(self, model_id, identifier):
        path = self.asset(model_id, identifier, "weights")
        with self.lock, self.transaction() as cursor:
            cursor.execute("SELECT * FROM active_learning.weights WHERE id=%s AND model_id=%s FOR UPDATE", (identifier, model_id))
            row = cursor.fetchone()
            if not row or row["received_bytes"] != row["size_bytes"]:
                raise WorkflowConflict("Upload checkpoint chưa hoàn tất.")
            if row["state"] == "ready":
                return dict(row)
            partial = path.with_suffix(".part")
            if partial.exists():
                partial.replace(path)
            if not path.is_file() or path.stat().st_size != row["size_bytes"]:
                raise WorkflowConflict("Checkpoint trên SSD không đủ dữ liệu.")
            cursor.execute("UPDATE active_learning.weights SET state='ready',sha256=%s WHERE id=%s RETURNING *", (checksum(path), identifier))
            return dict(cursor.fetchone())

    def prerequisites(self, model_id, config=None):
        config = config or self.settings(model_id)
        problems = []
        current = self.registry.runtime_spec(self.registry.get(model_id))
        if not Path(current["engine"]).with_name("best.pt").is_file() and not any(row["state"] == "ready" for row in self.weights(model_id)):
            problems.append("Cần upload checkpoint YOLO .pt; không fine-tune từ ONNX/engine.")
        if not (self.directory(model_id) / config["base_dataset"]).is_file():
            problems.append("Thiếu dataset gốc + tập validation cố định tại " + str(self.directory(model_id) / config["base_dataset"]))
        return problems

    def enqueue(self, model_id, actor, trigger="manual"):
        config = self.settings(model_id)
        problems = self.prerequisites(model_id, config)
        if problems:
            raise WorkflowConflict(" ".join(problems))
        model = self.registry.get(model_id)
        if model["state"] != "ready":
            raise WorkflowConflict("Model gốc chưa Ready.")
        current = model.get("metadata", {}).get("active_learning_version")
        weights_path = Path(self.registry.runtime_spec(model)["engine"]).with_name("best.pt")
        if not current or not weights_path.is_file():
            weights = next(row for row in self.weights(model_id) if row["state"] == "ready")
            weights_path = self.asset(model_id, weights["id"], "weights")
        identifier = uuid.uuid4().hex
        with self.transaction() as cursor:
            cursor.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", ("active-learning:" + model_id,))
            cursor.execute("SELECT id FROM active_learning.jobs WHERE model_id=%s AND state=ANY(%s)", (model_id, list(ACTIVE_STATES)))
            if cursor.fetchone():
                raise WorkflowConflict("Model đang có job; chờ hoặc hủy job trước.")
            cursor.execute("SELECT * FROM active_learning.samples WHERE model_id=%s AND status='approved' AND complete ORDER BY sequence", (model_id,))
            samples = [dict(row) for row in cursor.fetchall()]
            if not samples:
                raise WorkflowConflict("Chưa có frame đã kiểm tra đầy đủ.")
            inputs = dict(samples=[dict(id=row["id"], hash=row["annotation_hash"]) for row in samples],
                          weights_path=str(weights_path), weights_sha256=checksum(weights_path),
                          baseline=current, labels=model["labels"], contract=model["metadata"],
                          runtime=self.registry.runtime_spec(model))
            cursor.execute("""INSERT INTO active_learning.jobs(id,model_id,state,trigger,config,inputs,max_sequence,actor)
                VALUES(%s,%s,'queued',%s,%s,%s,%s,%s) RETURNING *""",
                (identifier, model_id, trigger, Json(config), Json(inputs), max(row["sequence"] for row in samples), actor))
            return dict(cursor.fetchone())

    def jobs(self, model_id):
        with self.transaction() as cursor:
            cursor.execute("SELECT * FROM active_learning.jobs WHERE model_id=%s ORDER BY created_at DESC LIMIT 50", (model_id,))
            return [dict(row) for row in cursor.fetchall()]

    def get_job(self, model_id, identifier):
        if not re.fullmatch(r"[a-f0-9]{32}", identifier):
            raise KeyError("Job ID không hợp lệ.")
        with self.transaction() as cursor:
            cursor.execute("SELECT * FROM active_learning.jobs WHERE model_id=%s AND id=%s", (model_id, identifier))
            row = cursor.fetchone()
        if not row:
            raise KeyError("Không tìm thấy job.")
        return dict(row)

    def update_job(self, identifier, state, reason=None, metrics=None, version=None):
        with self.transaction() as cursor:
            cursor.execute("""UPDATE active_learning.jobs SET state=%s,reason=%s,metrics=COALESCE(%s,metrics),
                version=COALESCE(%s,version),updated_at=NOW() WHERE id=%s""",
                (state, reason, Json(metrics) if metrics is not None else None, Json(version) if version else None, identifier))

    def cancel(self, model_id, identifier):
        job = self.get_job(model_id, identifier)
        if job["state"] == "promoting":
            raise WorkflowConflict("Đang chuyển engine an toàn, chờ kết quả trước khi hủy.")
        if job["state"] not in ACTIVE_STATES:
            raise WorkflowConflict("Job không còn chạy.")
        with self.transaction() as cursor:
            cursor.execute("UPDATE active_learning.jobs SET cancel_requested=TRUE,updated_at=NOW() WHERE id=%s", (identifier,))
        return dict(cancel_requested=True)

    def inputs_valid(self, job):
        with self.transaction() as cursor:
            cursor.execute("SELECT id,annotation_hash FROM active_learning.samples WHERE model_id=%s AND status='approved'", (job["model_id"],))
            current = {row["id"]: row["annotation_hash"] for row in cursor.fetchall()}
        return all(current.get(sample["id"]) == sample["hash"] for sample in job["inputs"]["samples"])

    def status(self, model_id):
        config = self.settings(model_id)
        return dict(settings=config, weights=self.weights(model_id), jobs=self.jobs(model_id),
                    prerequisites=self.prerequisites(model_id, config), directory=str(self.directory(model_id)),
                    training_target="yolo_detector", segmentation_target="SAM2 unchanged; polygons retained as ground truth")
