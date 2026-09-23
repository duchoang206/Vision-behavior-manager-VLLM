import json
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid

from psycopg2.extras import Json

from core.model_contract import infer_config, parse_labels
from core.workflow_store import WorkflowConflict, WorkflowStore


class ModelRegistry(WorkflowStore):
    chunk_size = 2 * 1024 * 1024
    max_size = 512 * 1024 * 1024

    def __init__(self, database):
        super().__init__(database)
        self.root = Path(os.getenv("VISION_MODELS_DIR", str(Path(__file__).resolve().parents[1] / "models/uploads"))).resolve()
        self.stop_event = threading.Event()
        self.thread = None
        self.process = None
        self.ready = False
        self.before_gpu_build = None
        self.gpu_build_waiting = False
        self.gpu_build_wait_started = 0.0
        self.parser = os.getenv("VISION_YOLO_PARSER", "/opt/visionmanager/libnvdsinfer_custom_impl_Yolo.so")
        self.trtexec = os.getenv("TRTEXEC_PATH", "/usr/src/tensorrt/bin/trtexec")

    def initialize(self):
        self.root.mkdir(parents=True, exist_ok=True)
        with self.transaction() as cursor:
            cursor.execute("""CREATE SCHEMA IF NOT EXISTS vision;
                CREATE TABLE IF NOT EXISTS vision.models (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, filename TEXT NOT NULL,
                    size_bytes BIGINT NOT NULL, received_bytes BIGINT NOT NULL DEFAULT 0,
                    state TEXT NOT NULL, labels JSONB NOT NULL DEFAULT '[]', metadata JSONB NOT NULL DEFAULT '{}',
                    error TEXT, actor TEXT NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW());
                CREATE TABLE IF NOT EXISTS vision.monitor_model (
                    singleton BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK(singleton),
                    model_id TEXT NOT NULL REFERENCES vision.models(id),
                    camera_ids JSONB NOT NULL DEFAULT '[]', all_cameras BOOLEAN NOT NULL DEFAULT TRUE,
                    actor TEXT NOT NULL, updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW());
                UPDATE vision.models SET state='failed', error='Build bị ngắt bởi lần khởi động trước. Bấm build lại.',
                    updated_at=NOW() WHERE state IN ('building','validating');""")
        self.ready = True
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, name="model-engine-builder", daemon=True)
        self.thread.start()

    def directory(self, model_id):
        if not re.fullmatch(r"[a-f0-9]{32}", model_id):
            raise KeyError("Model ID không hợp lệ.")
        return self.root / model_id

    def list(self):
        if not self.ready:
            return []
        with self.transaction() as cursor:
            cursor.execute("SELECT * FROM vision.models ORDER BY created_at DESC LIMIT 100")
            return [dict(row) for row in cursor.fetchall()]

    def get(self, model_id, cursor=None):
        self.directory(model_id)
        if cursor is None:
            with self.transaction() as transaction:
                return self.get(model_id, transaction)
        cursor.execute("SELECT * FROM vision.models WHERE id=%s FOR UPDATE", (model_id,))
        row = cursor.fetchone()
        if not row:
            raise KeyError("Không tìm thấy model.")
        return dict(row)

    def runtime_spec(self, row):
        directory = self.directory(row["id"])
        version = row.get("metadata", {}).get("active_learning_version")
        if version:
            if not re.fullmatch(r"[a-f0-9]{32}", version):
                raise ValueError("Model version không hợp lệ.")
            directory = directory / "versions" / version
        return {"id": row["id"], "onnx": str(directory / "model.onnx"),
                "engine": str(directory / "detector.engine"), "labels": str(directory / "labels.txt"),
                "config": str(directory / "nvinfer.txt"), "labels_list": list(row.get("labels") or []),
                "version": version or "original"}

    def activate_version(self, model_id, version, expected):
        directory = self.directory(model_id) / "versions" / version
        if not re.fullmatch(r"[a-f0-9]{32}", version) or not all((directory / name).is_file()
                for name in ("detector.engine", "nvinfer.txt", "model.onnx", "best.pt")):
            raise ValueError("Version chưa hoàn tất trên SSD.")
        with self.transaction() as cursor:
            row = self.get(model_id, cursor)
            if row.get("metadata", {}).get("active_learning_version") != expected:
                raise WorkflowConflict("Version đang chạy đã thay đổi, không ghi đè.")
            metadata = dict(row["metadata"], active_learning_version=version)
            cursor.execute("UPDATE vision.models SET metadata=%s,updated_at=NOW() WHERE id=%s", (Json(metadata), model_id))

    def deployment(self):
        if not self.ready:
            return None
        with self.transaction() as cursor:
            cursor.execute("SELECT * FROM vision.monitor_model WHERE singleton=TRUE")
            row = cursor.fetchone()
            return dict(row) if row else None

    def deploy(self, model_id, camera_ids, all_cameras, available, actor):
        if not self.ready:
            raise WorkflowConflict("Kho model chưa sẵn sàng.")
        camera_ids = sorted(set(camera_ids))
        if not all_cameras and not camera_ids:
            raise ValueError("Chọn ít nhất một camera.")
        if set(camera_ids) - set(available):
            raise ValueError("Camera đã chọn không còn tồn tại.")
        with self.transaction() as cursor:
            cursor.execute("SELECT pg_advisory_xact_lock(724091)")
            row = self.get(model_id, cursor)
            directory = self.directory(model_id)
            if row["state"] != "ready" or not all((directory / name).is_file() for name in ("detector.engine", "nvinfer.txt")):
                raise WorkflowConflict("Model chưa có TensorRT engine Ready.")
            from core.custom_detector import requested_models
            cursor.execute("SELECT definition FROM workflow.deployments WHERE status <> 'stopped'")
            active = set().union(*(requested_models(item["definition"]) for item in cursor.fetchall()))
            if active - {model_id}:
                raise WorkflowConflict("Dừng workflow dùng model khác trước khi đổi model Monitor.")
            cursor.execute("""INSERT INTO vision.monitor_model(singleton,model_id,camera_ids,all_cameras,actor)
                VALUES(TRUE,%s,%s,%s,%s) ON CONFLICT(singleton) DO UPDATE SET model_id=EXCLUDED.model_id,
                camera_ids=EXCLUDED.camera_ids,all_cameras=EXCLUDED.all_cameras,actor=EXCLUDED.actor,updated_at=NOW()
                RETURNING *""", (model_id, Json([] if all_cameras else camera_ids), all_cameras, actor))
            result = dict(cursor.fetchone())
            self.log(cursor, None, None, "Model Monitor deployed", {"model_id": model_id, "actor": actor,
                                                                   "all_cameras": all_cameras, "camera_ids": camera_ids})
            return result

    def stop_deployment(self, actor):
        with self.transaction() as cursor:
            cursor.execute("SELECT pg_advisory_xact_lock(724091)")
            cursor.execute("DELETE FROM vision.monitor_model WHERE singleton=TRUE RETURNING model_id")
            previous = cursor.fetchone()
            self.log(cursor, None, None, "Model Monitor stopped", {"actor": actor, "model_id": previous["model_id"] if previous else None})
        return {"stopped": True}

    def create(self, name, filename, size, labels, actor):
        if not self.ready:
            raise WorkflowConflict("Kho model chưa sẵn sàng.")
        if not filename.lower().endswith(".onnx") or not 1 <= size <= self.max_size:
            raise ValueError("Chỉ nhận file .onnx từ 1 byte đến 512 MiB.")
        names = parse_labels(labels) if labels else []
        model_id = uuid.uuid4().hex
        directory = self.directory(model_id)
        with self.transaction() as cursor:
            cursor.execute("SELECT pg_advisory_xact_lock(741927)")
            cursor.execute("SELECT count(*) AS count, COALESCE(SUM(size_bytes-received_bytes) FILTER (WHERE state='uploading'),0) AS reserved FROM vision.models")
            capacity = cursor.fetchone()
            if capacity["count"] >= 100 or shutil.disk_usage(self.root).free < int(capacity["reserved"]) + size + 2 * 1024**3:
                raise WorkflowConflict("Kho model đầy hoặc SSD còn dưới 2 GiB trống.")
            directory.mkdir()
            try:
                (directory / "model.onnx").touch()
                cursor.execute("""INSERT INTO vision.models(id,name,filename,size_bytes,state,labels,actor)
                    VALUES(%s,%s,%s,%s,'uploading',%s,%s) RETURNING *""",
                    (model_id, name.strip() or filename, Path(filename).name, size, Json(names), actor))
                return dict(cursor.fetchone())
            except Exception:
                shutil.rmtree(directory)
                raise

    def append(self, model_id, offset, content):
        if not content or len(content) > self.chunk_size:
            raise ValueError("Mỗi phần upload phải từ 1 byte đến 2 MiB.")
        with self.transaction() as cursor:
            row = self.get(model_id, cursor)
            if row["state"] != "uploading" or offset != row["received_bytes"] or offset + len(content) > row["size_bytes"]:
                raise WorkflowConflict("Offset upload không khớp; tải lại danh sách để tiếp tục.")
            if shutil.disk_usage(self.root).free < len(content) + 1024**3:
                raise WorkflowConflict("SSD không đủ chỗ trống.")
            with (self.directory(model_id) / "model.onnx").open("r+b") as output:
                output.seek(offset)
                output.write(content)
                output.truncate()
            cursor.execute("UPDATE vision.models SET received_bytes=%s,updated_at=NOW() WHERE id=%s", (offset + len(content), model_id))
        return {"received_bytes": offset + len(content)}

    def enqueue(self, model_id, labels=None):
        with self.transaction() as cursor:
            row = self.get(model_id, cursor)
            if row["state"] not in {"uploading", "failed"} or row["received_bytes"] != row["size_bytes"]:
                raise WorkflowConflict("Upload chưa hoàn tất hoặc model đang build/đã sẵn sàng.")
            names = parse_labels(labels) if labels else row["labels"]
            cursor.execute("UPDATE vision.models SET state='queued',labels=%s,error=NULL,updated_at=NOW() WHERE id=%s RETURNING *", (Json(names), model_id))
            return dict(cursor.fetchone())

    def _state(self, model_id, state, metadata=None, error=None):
        with self.transaction() as cursor:
            cursor.execute("UPDATE vision.models SET state=%s,metadata=COALESCE(%s,metadata),labels=COALESCE(%s,labels),error=%s,updated_at=NOW() WHERE id=%s",
                           (state, Json(metadata) if metadata else None, Json(metadata["labels"]) if metadata else None, error, model_id))

    def _execute(self, command, log, timeout, env=None):
        if command[0] == self.trtexec and self.before_gpu_build:
            self.gpu_build_waiting = True
            self.gpu_build_wait_started = time.monotonic()
            try:
                self.before_gpu_build()
            finally:
                self.gpu_build_waiting = False
                self.gpu_build_wait_started = 0.0
        if self.stop_event.is_set():
            raise RuntimeError("Build dừng do backend đang tắt.")
        self.process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            if self.process.wait(timeout=timeout) != 0:
                raise ValueError("Tiến trình kiểm tra/build thất bại; xem chi tiết bên dưới.")
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
            raise ValueError("Build vượt quá thời gian cho phép.")
        finally:
            self.process = None

    def _run(self):
        while not self.stop_event.wait(1):
            model_id = None
            try:
                with self.transaction() as cursor:
                    cursor.execute("SELECT * FROM vision.models WHERE state='queued' ORDER BY created_at LIMIT 1 FOR UPDATE SKIP LOCKED")
                    row = cursor.fetchone()
                    if not row:
                        continue
                    model_id = row["id"]
                    cursor.execute("UPDATE vision.models SET state='validating',updated_at=NOW() WHERE id=%s", (model_id,))
                directory = self.directory(model_id)
                request_file = directory / "inspect.json"
                output_file = directory / "contract.json"
                request_file.write_text(json.dumps({"path": str(directory / "model.onnx"), "labels": row["labels"]}))
                env = dict(os.environ)
                tools = Path(__file__).resolve().parents[1] / ".model-tools"
                env["PYTHONPATH"] = str(tools) + os.pathsep + env.get("PYTHONPATH", "")
                with (directory / "build.log").open("w") as log:
                    self._execute([sys.executable, "-m", "core.model_contract", str(request_file), str(output_file)], log, 90, env)
                    metadata = json.loads(output_file.read_text())
                    if not Path(self.trtexec).is_file() or not Path(self.parser).is_file():
                        raise ValueError("Container thiếu trtexec hoặc custom YOLO parser.")
                    if shutil.disk_usage(self.root).free < row["size_bytes"] * 2 + 1024**3:
                        raise ValueError("SSD thiếu chỗ để build engine.")
                    self._state(model_id, "building", metadata)
                    temporary = directory / "building.engine"
                    self._execute([self.trtexec, f"--onnx={directory / 'model.onnx'}", f"--saveEngine={temporary}",
                                   "--fp16", "--skipInference", "--memPoolSize=workspace:1024", "--builderOptimizationLevel=2", "--maxAuxStreams=0"], log, 1800)
                    if not temporary.is_file() or temporary.stat().st_size < 1024:
                        raise ValueError("Không tạo được TensorRT engine hợp lệ.")
                    temporary.replace(directory / "detector.engine")
                    (directory / "labels.txt").write_text("\n".join(metadata["labels"]) + "\n")
                    (directory / "nvinfer.txt").write_text(infer_config(directory, metadata, self.parser))
                    self._state(model_id, "ready", metadata)
            except Exception as error:
                logging.exception("Model build failed")
                if model_id:
                    log_path = self.directory(model_id) / "build.log"
                    tail = ""
                    if log_path.exists():
                        with log_path.open("rb") as log:
                            log.seek(max(0, log_path.stat().st_size - 4000))
                            tail = log.read().decode(errors="replace")
                    try:
                        self._state(model_id, "failed", error=f"{error}\n{tail}"[-4500:])
                    except Exception:
                        logging.exception("Could not persist model build failure")

    def stop(self):
        self.stop_event.set()
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
        if self.thread:
            self.thread.join(timeout=5)
