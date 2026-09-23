import copy
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import Future
from core.monitor_model_policy import hidden_monitor_cameras, apply_monitor_visibility
from core.deepstream_masktracker import native_masktracker_enabled, prepare_native_tracker_config


def requested_models(definition):
    return {node["config"]["model_id"] for node in definition.get("nodes", [])
            if node.get("type") == "detector" and isinstance(node.get("config", {}).get("model_id"), str)
            and node["config"]["model_id"]}


class CustomDetector:
    def __init__(self, registry, workflows, cameras, callback, clear, claim=None):
        self.registry, self.workflows, self.cameras = registry, workflows, cameras
        self.callback, self.clear = callback, clear
        self.stopping = threading.Event()
        self.lock = threading.RLock()
        self.thread = None
        self.process = None
        self.signature = None
        self.frames = {}
        self.error = None
        self.restarts = 0
        self.started_at = 0
        self.retry_at = 0
        self.model_id = None
        self.claim = claim
        self.commands = {}
        self.command_lock = threading.Lock()
        self.model_update_lock = threading.RLock()
        self.before_launch = None
        self.hidden_monitor_cameras = frozenset()

    def command(self, model_id, action, request, timeout=12):
        camera_id = request.get("camera_id")
        identifier = uuid.uuid4().hex
        future = Future()
        with self.command_lock:
            with self.lock:
                if action in {"reload_model", "reload_labels"}:
                    if not self.process or self.process.poll() is not None or self.model_id != model_id:
                        raise RuntimeError("Model không còn chạy để cập nhật engine.")
                elif not self.accepts_stream(camera_id, model_id):
                    raise RuntimeError("Deploy model cho camera này trước khi đăng ký Label.")
                if len(self.commands) >= 4:
                    raise RuntimeError("GPU đang xử lý yêu cầu Label; hãy chờ lượt hiện tại.")
                process = self.process
                self.commands[identifier] = future
            try:
                process.stdin.write(json.dumps(dict(id=identifier, action=action, **request)) + "\n")
                process.stdin.flush()
            except Exception:
                with self.lock:
                    self.commands.pop(identifier, None)
                raise RuntimeError("Worker GPU đã dừng. Hãy thử lại.")
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            raise TimeoutError("Camera/GPU chưa trả lời. Hãy lấy frame mới và thử lại.")
        finally:
            with self.lock:
                self.commands.pop(identifier, None)

    def status(self):
        with self.lock:
            running = self.process is not None and self.process.poll() is None
            cameras = copy.deepcopy(self.frames)
            for frame in cameras.values():
                frame["age_ms"] = round((time.time() - frame["last_frame_at"]) * 1000)
            return {"running": running, "model_id": self.model_id, "cameras": cameras, "error": self.error,
                    "monitor_hidden_cameras": sorted(self.hidden_monitor_cameras),
                    "restarts": self.restarts, "state": "live" if running and any(row["age_ms"] < 1000 for row in cameras.values()) else "starting" if running else "idle"}

    def reload_model(self, model_id, engine, version):
        with self.lock:
            if self.model_id != model_id or not self.process or self.process.poll() is not None:
                return {"mode": "next_start", "version": version}
        return self.command(model_id, "reload_model", {"engine": str(engine), "version": version}, timeout=75)

    def reload_labels(self, model_id, revision, activation=None):
        with self.lock:
            if self.model_id != model_id or not self.process or self.process.poll() is not None:
                return dict(mode="next_start", revision=revision)
        try:
            return self.command(model_id, "reload_labels", {"revision": revision, "activation": activation}, timeout=4)
        except (RuntimeError, TimeoutError) as error:
            return dict(mode="pending", revision=revision, error=str(error))

    def start(self):
        self.stopping.clear()
        self.thread = threading.Thread(target=self._run, name="custom-detector-supervisor", daemon=True)
        self.thread.start()

    def accepts_stream(self, camera_id, model_id):
        with self.lock:
            return bool(model_id and self.signature and self.process is not None
                        and self.process.poll() is None and self.model_id == model_id
                        and self.signature[0] == model_id and camera_id in self.signature[1]
                        and not self.stopping.is_set())

    def monitor_payload(self, payload):
        return apply_monitor_visibility(payload, self.hidden_monitor_cameras)

    def _terminate(self):
        with self.lock:
            process, self.process = self.process, None
            old_cameras = list(self.frames)
            self.frames = {}
            self.model_id = None
            for future in self.commands.values():
                if not future.done():
                    future.set_exception(RuntimeError("Model đã dừng/thay đổi; lấy lại frame Label."))
            self.commands.clear()
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for camera_id in old_cameras:
            self.clear(camera_id)

    def _read(self, process, model_id, camera_ids):
        pending = {}
        pending_lock = threading.Lock()
        done = threading.Event()

        def publish():
            while not done.wait(.005):
                with pending_lock:
                    newest = list(pending.values())
                    pending.clear()
                for payload in newest:
                    if self.process is not process or time.time() * 1000 - payload["timestamp"] > 300:
                        continue
                    try:
                        if '_phase0' in payload:
                            payload['_phase0']['parent_publish_ns'] = time.perf_counter_ns()
                        self.callback(payload)
                    except Exception:
                        logging.exception("Custom model metadata callback")

        sender = threading.Thread(target=publish, name="custom-latest-metadata", daemon=True)
        sender.start()
        try:
            for line in process.stdout:
                if line.startswith("RSKY_REPLY "):
                    try:
                        reply = json.loads(line[11:])
                        with self.lock:
                            future = self.commands.get(reply["id"]) if self.process is process else None
                            if future and not future.done():
                                if reply.get("error"):
                                    future.set_exception(RuntimeError(reply["error"]))
                                else:
                                    future.set_result(reply["result"])
                    except (ValueError, KeyError):
                        logging.exception("Invalid label reply")
                    continue
                if not line.startswith("RSKY_META "):
                    continue
                try:
                    payload = json.loads(line[10:])
                    if '_phase0' in payload:
                        payload['_phase0']['parent_read_ns'] = time.perf_counter_ns()
                    stream = payload["streams"][0]
                    camera_id = stream["cam_id"]
                    if camera_id not in camera_ids or self.process is not process:
                        continue
                    with self.lock:
                        previous = self.frames.get(camera_id, {})
                        self.frames[camera_id] = {"frame_id": stream["frame_id"], "last_frame_at": payload["timestamp"] / 1000,
                                                 "frames": previous.get("frames", 0) + 1, "model_id": model_id,
                                                 "segmentation": stream.get("segmentation", {}),
                                                 "model_version": stream.get("model_version", "original")}
                        self.error = None
                    with pending_lock:
                        pending[camera_id] = payload
                except (ValueError, KeyError, TypeError):
                    logging.exception("Invalid custom model metadata")
        finally:
            done.set()
            sender.join(timeout=1)

    def _launch(self, model_id, camera_ids):
        if self.before_launch:
            self.before_launch()
        from core.object_logic import infer_category
        row = self.registry.get(model_id)
        if row["state"] != "ready":
            raise ValueError("Model chưa sẵn sàng.")
        if self.claim:
            for camera_id in camera_ids:
                self.claim(camera_id)
        directory = self.registry.directory(model_id)
        runtime = self.registry.runtime_spec(row)
        tracker_source = Path(__file__).resolve().parents[1] / "models_config/tracker_config.yml"
        if native_masktracker_enabled():
            tracker_file = Path(prepare_native_tracker_config())
        else:
            tracker_file = directory / "tracker.yml"
            # Keep NvDCF re-association enabled.  It is required to preserve
            # the prompt track through short detector misses/occlusions.
            tracker_file.write_text(tracker_source.read_text())
        settings = {"model_id": model_id, "generation": uuid.uuid4().hex[:8], "cameras": camera_ids,
                    "config": runtime["config"], "model_version": runtime["version"],
                    "model_directory": str(directory), "engine": runtime["engine"],
                    "tracker": str(tracker_file), "labels": row["labels"],
                    "categories": [infer_category(None, label) for label in row["labels"]],
                    "native_masktracker": native_masktracker_enabled()}
        from core.model_label_store import ModelLabelStore
        settings["label_directory"] = str(ModelLabelStore(self.registry.database).directory(model_id))
        request_file = directory / "runtime.json"
        request_file.write_text(json.dumps(settings))
        with (directory / "runtime.log").open("w") as log:
            process = subprocess.Popen([sys.executable, "-m", "core.custom_deepstream_worker", str(request_file)],
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1, start_new_session=True)
        with self.lock:
            self.process = process
            self.model_id = model_id
            self.started_at = time.time()
        threading.Thread(target=self._read, args=(process, model_id, camera_ids), daemon=True, name="custom-metadata-reader").start()

    def _run(self):
        while not self.stopping.wait(1):
            if os.getenv("DISABLE_CUSTOM_DETECTOR") == "1":
                continue
            try:
                selected = {}
                deployment = self.registry.deployment()
                active_workflows = self.workflows.active()
                registered = self.cameras()
                self.hidden_monitor_cameras = hidden_monitor_cameras(deployment, active_workflows, registered)
                if deployment:
                    camera_ids = registered if deployment["all_cameras"] else set(deployment["camera_ids"]) & registered
                    selected[deployment["model_id"]] = set(camera_ids)
                for deployment in active_workflows:
                    definition = deployment["definition"]
                    for model_id in requested_models(definition):
                        selected.setdefault(model_id, set()).update(camera_id for node in definition["nodes"] if node["type"] == "source"
                            for camera_id in node["config"]["camera_ids"] if camera_id in registered)
                if len(selected) > 1:
                    raise ValueError("GPU hiện hỗ trợ một custom detector dùng chung; dừng model khác trước.")
                signature = next(((model_id, tuple(sorted(ids))) for model_id, ids in selected.items() if ids), None)
                if signature != self.signature:
                    with self.model_update_lock:
                        self._terminate()
                        self.signature = signature
                    self.retry_at = 0
                    self.error = None
                if not signature:
                    continue
                with self.lock:
                    latest_frame_at = max([self.started_at] + [item["last_frame_at"] for item in self.frames.values()])
                if self.process and (self.process.poll() is not None or time.time() - latest_frame_at > 30):
                    log_path = self.registry.directory(signature[0]) / "runtime.log"
                    tail = ""
                    if log_path.exists():
                        with log_path.open("rb") as log:
                            log.seek(max(0, log_path.stat().st_size - 2000))
                            tail = log.read().decode(errors="replace")
                    self.error = "DeepStream ngừng xuất frame; tự khởi động lại. " + tail
                    with self.model_update_lock:
                        self._terminate()
                    self.restarts += 1
                    self.retry_at = time.time() + min(30, 2 ** min(self.restarts, 5))
                if not self.process and time.time() >= self.retry_at:
                    with self.model_update_lock:
                        self._launch(signature[0], list(signature[1]))
            except Exception as error:
                with self.lock:
                    self.error = str(error)
                logging.exception("Custom detector supervisor")
                self.retry_at = time.time() + 10
                self.stopping.wait(3)

    def stop(self):
        self.stopping.set()
        if self.thread:
            self.thread.join(timeout=6)
        self._terminate()
