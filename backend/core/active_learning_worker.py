import json
import logging
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from core.active_learning_dataset import prepare_dataset
from core.active_learning_data import atomic_write, checksum, json_bytes
from core.model_contract import infer_config
from core.workflow_store import WorkflowConflict


class ActiveLearningWorker:
    def __init__(self, store, registry, detector):
        self.store, self.registry, self.detector = store, registry, detector
        self.stopping = threading.Event()
        self.preempted = threading.Event()
        self.lock = threading.RLock()
        self.thread = self.process = self.job = None
        self.error = None
        self.min_gpu_free = int(os.getenv("ACTIVE_LEARNING_MIN_GPU_FREE_MB", "6000"))

    def start(self):
        with self.lock:
            if self.thread and self.thread.is_alive():
                return
            self.stopping.clear()
            self.thread = threading.Thread(target=self._run, name="active-learning-worker", daemon=True)
            self.thread.start()

    def status(self):
        with self.lock:
            alive = bool(self.thread and self.thread.is_alive())
            job_active = self.job is not None
            return dict(running=alive, job_active=job_active,
                        state="running" if job_active else "idle" if alive else "stopped",
                        job_id=self.job["id"] if self.job else None,
                        error=self.error, gpu_policy="idle_only", min_gpu_free_mb=self.min_gpu_free,
                        note="Không chạy train/build trên GPU khi DeepStream đã deploy. Chỉ thu thập dữ liệu lúc live.")

    def resource_reason(self, starting=True):
        if self.registry.process is not None:
            return "Registry đang build TensorRT."
        if self.detector.status().get("running") or self.registry.deployment() or self.detector.workflows.active():
            return "Chờ dừng deployment Monitor/workflow: ưu tiên GPU cho DeepStream realtime."
        if os.getenv("DISABLE_DEEPSTREAM_GST", "0") != "1":
            return "Person DeepStream có thể đang dùng GPU; không tự tranh tài nguyên."
        if starting:
            try:
                result = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                                        check=True, capture_output=True, text=True, timeout=3)
                if int(result.stdout.splitlines()[0].strip()) < self.min_gpu_free:
                    return f"Chờ GPU còn ít nhất {self.min_gpu_free} MiB trống."
            except Exception:
                return "Không đọc được dung lượng GPU; từ chối chạy ngầm."
        return None

    def _terminate_process(self):
        with self.lock:
            process = self.process
        if process and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=3)
            except ProcessLookupError:
                pass

    def preempt_for_live(self):
        self.preempted.set()
        self._terminate_process()

    def _command(self, command, log, timeout, env=None):
        if self.stopping.is_set() or self.preempted.is_set():
            raise RuntimeError("Job nhường GPU cho runtime/build đang được yêu cầu.")
        reason = self.resource_reason(starting=False)
        if reason:
            raise RuntimeError(reason)
        environment = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", YOLO_AUTOINSTALL="false")
        tools = Path(__file__).resolve().parents[1] / ".model-tools"
        environment["PYTHONPATH"] = str(tools) + os.pathsep + environment.get("PYTHONPATH", "")
        environment.update(env or {})
        with self.lock:
            if self.preempted.is_set():
                raise RuntimeError("Job đã nhường GPU cho runtime.")
            self.process = subprocess.Popen(["nice", "-n", "10", *command], stdout=log, stderr=subprocess.STDOUT,
                                            env=environment, start_new_session=True)
            process = self.process
        deadline, next_check = time.monotonic() + timeout, 0
        try:
            while process.poll() is None:
                if self.stopping.wait(.5) or self.preempted.is_set() or time.monotonic() > deadline:
                    raise RuntimeError("Job bị dừng hoặc vượt thời gian cho phép.")
                if time.monotonic() >= next_check:
                    next_check = time.monotonic() + 2
                    current = self.store.get_job(self.job["model_id"], self.job["id"])
                    if current["cancel_requested"]:
                        raise InterruptedError("Người dùng đã hủy job.")
                    reason = self.resource_reason(starting=False)
                    if reason:
                        raise RuntimeError("Đã ngắt training để ưu tiên runtime: " + reason)
                    if shutil.disk_usage(self.store.root).free < 1024**3:
                        raise RuntimeError("Dừng job: SSD còn dưới 1 GiB.")
            if process.returncode:
                raise RuntimeError(f"Tiến trình thất bại ({process.returncode}); xem train.log.")
        finally:
            self._terminate_process()
            with self.lock:
                self.process = None

    def _build_engine(self, job, version_dir, log):
        request, contract_file = version_dir / "inspect.json", version_dir / "contract.json"
        atomic_write(request, json_bytes(dict(path=str(version_dir / "model.onnx"), labels=job["inputs"]["labels"])))
        tools = Path(__file__).resolve().parents[1] / ".model-tools"
        env = {"PYTHONPATH": str(tools) + os.pathsep + os.environ.get("PYTHONPATH", "")}
        self._command([sys.executable, "-m", "core.model_contract", str(request), str(contract_file)], log, 120, env)
        metadata = json.loads(contract_file.read_text())
        baseline = job["inputs"]["contract"]
        if any(metadata.get(key) != baseline.get(key) for key in ("shape", "output_shape", "input", "output", "labels", "contract")):
            raise RuntimeError("Network resolution/tensor/classes thay đổi; không hot-reload model không tương thích.")
        self._command([sys.executable, "-m", "core.active_learning_network", job["inputs"]["runtime"]["onnx"],
                       str(version_dir / "model.onnx")], log, 120, env)
        if not Path(self.registry.trtexec).is_file() or not Path(self.registry.parser).is_file():
            raise RuntimeError("Thiếu trtexec hoặc YOLO parser.")
        temporary = version_dir / "building.engine"
        self._command([self.registry.trtexec, f"--onnx={version_dir / 'model.onnx'}", f"--saveEngine={temporary}",
                       "--fp16", "--skipInference", "--memPoolSize=workspace:1024",
                       "--builderOptimizationLevel=2", "--maxAuxStreams=0"], log, 1800)
        if not temporary.is_file() or temporary.stat().st_size < 1024:
            raise RuntimeError("TensorRT engine không hợp lệ.")
        temporary.replace(version_dir / "detector.engine")
        atomic_write(version_dir / "labels.txt", "\n".join(metadata["labels"]) + "\n")
        atomic_write(version_dir / "nvinfer.txt", infer_config(version_dir, metadata, self.registry.parser))
        return metadata

    def _run_job(self, job):
        model_id = job["model_id"]
        job_dir = self.store.directory(model_id) / "jobs" / job["id"]
        version_dir = self.registry.directory(model_id) / "versions" / job["id"]
        job_dir.mkdir(parents=True, exist_ok=True)
        version_dir.mkdir(parents=True, exist_ok=False)
        baseline = job["inputs"]["runtime"]["engine"]
        expected_version = job["inputs"].get("baseline")
        if self.registry.get(model_id)["metadata"].get("active_learning_version") != expected_version:
            raise WorkflowConflict("Baseline đã thay đổi; tạo job mới.")
        weights = Path(job["inputs"]["weights_path"])
        if checksum(weights) != job["inputs"]["weights_sha256"]:
            raise RuntimeError("Checkpoint bị thay đổi sau khi tạo job.")
        with (job_dir / "train.log").open("a", buffering=1) as log:
            self.store.update_job(job["id"], "preparing")
            dataset = prepare_dataset(self.store, job, job_dir / "dataset")
            atomic_write(job_dir / "dataset.json", json_bytes(dataset))
            self.store.update_job(job["id"], "training")
            shape = job["inputs"]["contract"]["shape"]
            if shape[2] != shape[3]:
                raise ValueError("Retrain tự động hiện yêu cầu input model vuông; không đổi kích thước model hiện tại.")
            request = dict(dataset=dataset["dataset"], output=str(job_dir / "training"), labels=job["inputs"]["labels"],
                           config=job["config"], imgsz=shape[2], stage="train", weights=str(weights))
            request_file = job_dir / "train.json"
            atomic_write(request_file, json_bytes(request))
            self._command([sys.executable, "-m", "core.active_learning_train", str(request_file)], log, 7200)
            self.store.update_job(job["id"], "building")
            training_dir = job_dir / "training"
            for name in ("best.pt", "model.onnx"):
                shutil.copyfile(training_dir / name, version_dir / name)
            metadata = self._build_engine(job, version_dir, log)
            request.update(stage="engine_validation", engine=str(version_dir / "detector.engine"), baseline_engine=baseline)
            atomic_write(request_file, json_bytes(request))
            self._command([sys.executable, "-m", "core.active_learning_train", str(request_file)], log, 1800)
            metrics = {**json.loads((training_dir / "training_metrics.json").read_text()),
                       **json.loads((training_dir / "engine_metrics.json").read_text()), **dataset}
            version = dict(id=job["id"], metadata=metadata, engine_sha256=checksum(version_dir / "detector.engine"),
                           baseline_engine_sha256=checksum(baseline), weights_sha256=checksum(version_dir / "best.pt"))
            if not self.store.inputs_valid(job):
                raise RuntimeError("Correction đã bị xóa/thay đổi; không promote.")
            if self.store.get_job(model_id, job["id"])["cancel_requested"]:
                raise InterruptedError("Người dùng đã hủy job.")
            if not metrics["accepted"]:
                self.store.update_job(job["id"], "rejected", metrics["reason"], metrics, version)
                return
            self.store.update_job(job["id"], "ready", metrics=metrics, version=version)
            if job["config"]["auto_promote"]:
                self.promote(model_id, job["id"], "scheduler")

    def promote(self, model_id, job_id, actor):
        job = self.store.get_job(model_id, job_id)
        if job["state"] != "ready" or not job.get("metrics", {}).get("accepted") or not self.store.inputs_valid(job):
            raise WorkflowConflict("Job chưa đạt kiểm định hoặc dataset đã thay đổi.")
        with self.store.transaction() as cursor:
            cursor.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", ("active-learning:" + model_id,))
            cursor.execute("""SELECT id FROM active_learning.jobs WHERE model_id=%s AND id<>%s
                AND state IN ('queued','waiting_resources','preparing','training','building','promotion_queued','promoting')""", (model_id, job_id))
            if cursor.fetchone():
                raise WorkflowConflict("Model có job khác đang chạy.")
            cursor.execute("UPDATE active_learning.jobs SET state='promotion_queued',actor=%s,updated_at=NOW() WHERE id=%s AND state='ready'", (actor, job_id))
            if cursor.rowcount != 1:
                raise WorkflowConflict("Job vừa thay đổi; tải lại.")
        return dict(state="promotion_queued")

    def _promote(self, job):
        model_id, version = job["model_id"], job["version"]
        with self.detector.model_update_lock, self.store.lock:
            model = self.registry.get(model_id)
            previous = self.registry.runtime_spec(model)
            destination = self.registry.directory(model_id) / "versions" / version["id"] / "detector.engine"
            if (self.store.get_job(model_id, job["id"])["cancel_requested"]
                    or not job["metrics"].get("accepted") or not self.store.inputs_valid(job)
                    or model["metadata"].get("active_learning_version") != job["inputs"].get("baseline")
                    or checksum(previous["engine"]) != version["baseline_engine_sha256"]
                    or checksum(destination) != version["engine_sha256"]):
                raise WorkflowConflict("Baseline, engine hoặc dataset không còn khớp bản đã kiểm định.")
            self.store.update_job(job["id"], "promoting")
            try:
                response = self.detector.reload_model(model_id, destination, version["id"])
                self.registry.activate_version(model_id, version["id"], job["inputs"].get("baseline"))
            except Exception:
                try:
                    self.detector.reload_model(model_id, previous["engine"], previous["version"])
                except Exception:
                    self.detector._terminate()
                raise
            self.store.update_job(job["id"], "completed", version={**version, "reload": response})

    def _schedule(self):
        self.store.cleanup_drafts()
        with self.store.transaction() as cursor:
            cursor.execute("SELECT model_id FROM active_learning.settings WHERE config->>'enabled'='true'")
            models = [row["model_id"] for row in cursor.fetchall()]
        for model_id in models:
            config = self.store.settings(model_id)
            if self.store.prerequisites(model_id, config):
                continue
            jobs = self.store.jobs(model_id)
            if any(job["state"] in {"queued", "waiting_resources", "preparing", "training", "building", "promotion_queued", "promoting"} for job in jobs):
                continue
            watermark = max((job["max_sequence"] for job in jobs), default=0)
            with self.store.transaction() as cursor:
                cursor.execute("SELECT COUNT(*) AS count FROM active_learning.samples WHERE model_id=%s AND status='approved' AND sequence>%s", (model_id, watermark))
                new = cursor.fetchone()["count"]
            now = datetime.now(ZoneInfo(config["timezone"]))
            daily = config["daily_hour"] is not None and now.hour >= config["daily_hour"] and not any(
                job["created_at"].astimezone(now.tzinfo).date() == now.date() for job in jobs)
            if new >= config["threshold"] or (new > 0 and daily):
                try:
                    self.store.enqueue(model_id, "scheduler", "threshold" if new >= config["threshold"] else "daily")
                except WorkflowConflict:
                    pass

    def _run(self):
        next_schedule = 0
        while not self.stopping.wait(3):
            connection = None
            try:
                connection = self.store.database._get_connection()
                with connection.cursor() as cursor:
                    cursor.execute("SELECT pg_try_advisory_lock(829405)")
                    acquired = cursor.fetchone()[0]
                connection.commit()
                if not acquired:
                    continue
                if time.monotonic() >= next_schedule:
                    self._schedule()
                    next_schedule = time.monotonic() + 60
                with self.store.transaction() as cursor:
                    cursor.execute("""SELECT * FROM active_learning.jobs WHERE state IN ('queued','waiting_resources','promotion_queued')
                        ORDER BY (state='promotion_queued') DESC,created_at LIMIT 1 FOR UPDATE SKIP LOCKED""")
                    row = cursor.fetchone()
                if not row:
                    continue
                job = dict(row)
                if job["cancel_requested"]:
                    self.store.update_job(job["id"], "cancelled", "Người dùng đã hủy.")
                    continue
                if job["state"] != "promotion_queued":
                    reason = self.resource_reason()
                    if reason:
                        if job["state"] != "waiting_resources" or job["reason"] != reason:
                            self.store.update_job(job["id"], "waiting_resources", reason)
                        continue
                with self.lock:
                    self.preempted.clear()
                    self.job, self.error = job, None
                try:
                    if job["state"] == "promotion_queued":
                        self._promote(job)
                    else:
                        self._run_job(job)
                except InterruptedError as error:
                    self.store.update_job(job["id"], "cancelled", str(error))
                except Exception as error:
                    logging.exception("Active Learning job failed")
                    self.error = str(error)
                    self.store.update_job(job["id"], "failed", str(error))
                finally:
                    with self.lock:
                        self.job = None
            except Exception as error:
                self.error = str(error)
                logging.exception("Active Learning supervisor")
                self.stopping.wait(5)
            finally:
                if connection:
                    connection.close()

    def stop(self):
        self.stopping.set()
        self._terminate_process()
        if self.thread:
            self.thread.join(timeout=8)
