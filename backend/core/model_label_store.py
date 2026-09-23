import base64
import json
import math
import os
import re
import threading
from pathlib import Path

from psycopg2.extras import Json

from core.workflow_store import WorkflowStore


def validate_sample(sample):
    vector = sample.get("vector", [])
    if len(vector) != 512 or any(not math.isfinite(float(value)) for value in vector):
        raise ValueError("Mẫu không có vector GPU hợp lệ.")
    if sum(float(value) ** 2 for value in vector) < 1e-8:
        raise ValueError("Vector mẫu rỗng.")


class ModelLabelStore(WorkflowStore):
    def __init__(self, database):
        super().__init__(database)
        self.root = Path(os.getenv("MODEL_LABELS_DIR", str(Path(__file__).resolve().parents[1] / "data/model_labels")))
        self.lock = threading.RLock()

    def directory(self, model_id):
        if not re.fullmatch(r"[a-f0-9]{32}", model_id):
            raise ValueError("Model ID không hợp lệ.")
        return self.root / model_id

    def initialize(self):
        with self.transaction() as cursor:
            cursor.execute("""CREATE TABLE IF NOT EXISTS vision.label_samples (
                id TEXT PRIMARY KEY, model_id TEXT NOT NULL REFERENCES vision.models(id),
                camera_id TEXT NOT NULL, label TEXT NOT NULL, class_name TEXT NOT NULL,
                category TEXT NOT NULL, negative BOOLEAN NOT NULL DEFAULT FALSE,
                signature TEXT NOT NULL, vector JSONB NOT NULL, mask JSONB NOT NULL,
                bbox JSONB NOT NULL, image_path TEXT NOT NULL, frame_id BIGINT NOT NULL,
                actor TEXT NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW());
                CREATE INDEX IF NOT EXISTS label_samples_model ON vision.label_samples(model_id, label);
                CREATE TABLE IF NOT EXISTS vision.label_settings (
                model_id TEXT PRIMARY KEY REFERENCES vision.models(id),
                revision BIGINT NOT NULL DEFAULT 0, require_labels BOOLEAN NOT NULL DEFAULT FALSE)""")
            cursor.execute("SELECT DISTINCT model_id FROM vision.label_samples UNION SELECT model_id FROM vision.label_settings")
            identifiers = [row["model_id"] for row in cursor.fetchall()]
        for model_id in identifiers:
            self.publish(model_id)

    def list(self, model_id):
        self.directory(model_id)
        with self.transaction() as cursor:
            cursor.execute("""SELECT label,class_name,category,negative,COUNT(*) AS samples,
                array_agg(DISTINCT camera_id) AS cameras,MAX(created_at) AS updated_at
                FROM vision.label_samples WHERE model_id=%s
                GROUP BY label,class_name,category,negative ORDER BY label""", (model_id,))
            labels = [dict(row) for row in cursor.fetchall()]
            cursor.execute("SELECT revision,require_labels FROM vision.label_settings WHERE model_id=%s", (model_id,))
            settings = dict(cursor.fetchone() or {"revision": 0, "require_labels": False})
        return dict(labels=labels, **settings)

    def save(self, model_id, label, class_name, category, negative, sample, actor):
        validate_sample(sample)
        label = label.strip()
        if not label or len(label) > 120 or any(ord(char) < 32 for char in label):
            raise ValueError("Tên Label không hợp lệ.")
        identifier = sample.get("preview_id", "")
        if not re.fullmatch(r"[a-f0-9]{32}", identifier):
            raise ValueError("Thiếu biên nhận preview GPU hợp lệ.")
        directory = self.directory(model_id) / "samples"
        directory.mkdir(parents=True, exist_ok=True)
        image_path = directory / f"{identifier}.jpg"
        image = base64.b64decode(sample["image"], validate=True)
        if not image.startswith(b"\xff\xd8") or len(image) > 4 * 1024 * 1024:
            raise ValueError("Ảnh GPU không hợp lệ.")
        with self.lock:
            with self.transaction() as cursor:
                cursor.execute("SELECT model_id,label,class_name,negative FROM vision.label_samples WHERE id=%s", (identifier,))
                previous = cursor.fetchone()
                if previous:
                    if (previous["model_id"], previous["label"].casefold(), previous["class_name"], previous["negative"]) != (model_id, label.casefold(), class_name, negative):
                        raise ValueError("Preview đã được lưu cho Label khác. Lấy frame mới.")
                    return dict(saved=True, sample_id=identifier, **self.list(model_id))
            image_path.write_bytes(image)
            try:
                with self.transaction() as cursor:
                    cursor.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (model_id,))
                    cursor.execute("SELECT label,class_name,category,negative FROM vision.label_samples WHERE model_id=%s AND lower(label)=lower(%s) LIMIT 1", (model_id, label))
                    previous = cursor.fetchone()
                    if previous and (previous["class_name"], previous["category"], previous["negative"]) != (class_name, category, negative):
                        raise ValueError("Label đã thuộc lớp/loại mẫu khác. Dùng tên khác hoặc xóa mẫu cũ trước.")
                    label = previous["label"] if previous else label
                    cursor.execute("""INSERT INTO vision.label_samples
                        (id,model_id,camera_id,label,class_name,category,negative,signature,vector,mask,bbox,image_path,frame_id,actor)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                        (identifier, model_id, sample["camera_id"], label, class_name, category, negative,
                         sample["signature"], Json(sample["vector"]), Json(sample["mask"]), Json(sample["bbox"]),
                         str(image_path.relative_to(self.root)), sample["frame_id"], actor))
                    self._bump(cursor, model_id)
                    self.log(cursor, None, None, "Model Label sample saved", dict(model_id=model_id, label=label, sample_id=identifier, actor=actor))
            except Exception:
                image_path.unlink(missing_ok=True)
                raise
            self.publish(model_id)
        return dict(saved=True, sample_id=identifier, **self.list(model_id))

    def _bump(self, cursor, model_id):
        cursor.execute("""INSERT INTO vision.label_settings(model_id,revision) VALUES(%s,1)
            ON CONFLICT(model_id) DO UPDATE SET revision=vision.label_settings.revision+1""", (model_id,))

    def samples(self, model_id, label, limit=12, offset=0):
        self.directory(model_id)
        limit, offset = max(1, min(int(limit), 48)), max(0, int(offset))
        with self.transaction() as cursor:
            cursor.execute("SELECT COUNT(*) AS total FROM vision.label_samples WHERE model_id=%s AND label=%s", (model_id, label))
            total = cursor.fetchone()["total"]
            cursor.execute("""SELECT id,label,camera_id,created_at,frame_id,bbox,mask FROM vision.label_samples
                WHERE model_id=%s AND label=%s ORDER BY created_at DESC,id DESC LIMIT %s OFFSET %s""", (model_id, label, limit, offset))
            rows = [dict(row) for row in cursor.fetchall()]
        return dict(samples=rows, total=total, offset=offset, limit=limit)

    def sample_image(self, model_id, sample_id):
        directory = (self.directory(model_id) / "samples").resolve()
        with self.transaction() as cursor:
            cursor.execute("SELECT image_path FROM vision.label_samples WHERE model_id=%s AND id=%s", (model_id, sample_id))
            row = cursor.fetchone()
        if not row:
            raise KeyError("Góc nhìn không tồn tại hoặc đã xóa.")
        image = (self.root / row["image_path"]).resolve()
        if not image.is_relative_to(directory) or not image.is_file():
            raise KeyError("Không tìm thấy ảnh của góc nhìn.")
        return image

    def delete_sample(self, model_id, sample_id, actor):
        self.directory(model_id)
        with self.lock:
            with self.transaction() as cursor:
                cursor.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (model_id,))
                cursor.execute("DELETE FROM vision.label_samples WHERE model_id=%s AND id=%s RETURNING image_path,label", (model_id, sample_id))
                row = cursor.fetchone()
                if not row:
                    raise KeyError("Góc nhìn không tồn tại hoặc đã xóa.")
                self._bump(cursor, model_id)
                self.log(cursor, None, None, "Model Label view deleted", dict(model_id=model_id, sample_id=sample_id, label=row["label"], actor=actor))
            self.publish(model_id)
            image = (self.root / row["image_path"]).resolve()
            if image.is_relative_to((self.directory(model_id) / "samples").resolve()):
                image.unlink(missing_ok=True)
        return dict(deleted=1, **self.list(model_id))

    def delete(self, model_id, label, actor):
        self.directory(model_id)
        with self.lock:
            with self.transaction() as cursor:
                cursor.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (model_id,))
                cursor.execute("DELETE FROM vision.label_samples WHERE model_id=%s AND label=%s RETURNING image_path", (model_id, label))
                paths = [row["image_path"] for row in cursor.fetchall()]
                self._bump(cursor, model_id)
                self.log(cursor, None, None, "Model Label deleted", dict(model_id=model_id, label=label, samples=len(paths), actor=actor))
            self.publish(model_id)
            for relative in paths:
                path = (self.root / relative).resolve()
                if path.is_relative_to(self.root.resolve()):
                    path.unlink(missing_ok=True)
        return dict(deleted=len(paths), **self.list(model_id))

    def settings(self, model_id, require_labels, actor="system"):
        self.directory(model_id)
        with self.lock:
            with self.transaction() as cursor:
                self._bump(cursor, model_id)
                cursor.execute("UPDATE vision.label_settings SET require_labels=%s WHERE model_id=%s", (require_labels, model_id))
                self.log(cursor, None, None, "Model Label settings changed", dict(model_id=model_id, require_labels=require_labels, actor=actor))
            self.publish(model_id)
        return self.list(model_id)

    def publish(self, model_id):
        with self.lock:
            with self.transaction() as cursor:
                cursor.execute("SELECT revision,require_labels FROM vision.label_settings WHERE model_id=%s", (model_id,))
                settings = dict(cursor.fetchone() or {"revision": 0, "require_labels": False})
                cursor.execute("""SELECT id,label,class_name,category,negative,signature,vector,
                    camera_id,frame_id,bbox,mask
                    FROM vision.label_samples WHERE model_id=%s ORDER BY created_at,id""", (model_id,))
                samples = [dict(row) for row in cursor.fetchall()]
            directory = self.directory(model_id)
            directory.mkdir(parents=True, exist_ok=True)
            destination = directory / "gallery.json"
            temporary = destination.with_suffix(".tmp")
            temporary.write_text(json.dumps(dict(model_id=model_id, samples=samples, **settings), separators=(",", ":")))
            temporary.replace(destination)
            return destination
