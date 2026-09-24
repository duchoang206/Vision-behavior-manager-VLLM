"""Per-model DeepStream supervisor used by Monitor and Workflow deployments.

Each enabled model owns an isolated worker process. This keeps a camera or
model failure local: another model observing the same camera keeps running.
"""

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

from core.monitor_model_policy import apply_monitor_visibility, hidden_monitor_cameras


def requested_models(definition):
    return {
        node["config"]["model_id"]
        for node in definition.get("nodes", [])
        if node.get("type") == "detector"
        and isinstance(node.get("config", {}).get("model_id"), str)
        and node["config"]["model_id"]
    }


class CustomDetector:
    def __init__(self, registry, workflows, cameras, callback, clear, claim=None):
        self.registry, self.workflows, self.cameras = registry, workflows, cameras
        self.callback, self.clear, self.claim = callback, clear, claim
        self.stopping = threading.Event()
        self.lock = threading.RLock()
        self.thread = None
        self.workers = {}
        self.commands = {}
        self.command_lock = threading.Lock()
        self.model_update_lock = threading.RLock()
        self.before_launch = None
        self.hidden_monitor_cameras = frozenset()

        # Kept for integrations written against the old one-worker interface.
        self.process = None
        self.signature = None
        self.frames = {}
        self.error = None
        self.restarts = 0
        self.started_at = 0.0
        self.retry_at = 0.0
        self.model_id = None

    def _active_worker(self, model_id, camera_id=None):
        worker = self.workers.get(model_id)
        if not worker or worker["process"] is None or worker["process"].poll() is not None:
            return None
        if camera_id is not None and camera_id not in worker["camera_ids"]:
            return None
        return worker

    def _sync_legacy_state(self):
        active = [worker for worker in self.workers.values() if worker.get("process") and worker["process"].poll() is None]
        primary = active[0] if active else None
        self.process = primary["process"] if primary else None
        self.signature = primary["signature"] if primary else None
        self.model_id = primary["model_id"] if primary else None
        self.frames = {camera_id: frame for worker in self.workers.values() for camera_id, frame in worker["frames"].items()}
        self.error = next((worker.get("error") for worker in self.workers.values() if worker.get("error")), None)
        self.restarts = sum(int(worker.get("restarts", 0)) for worker in self.workers.values())
        self.started_at = primary["started_at"] if primary else 0.0
        self.retry_at = min((worker.get("retry_at", 0.0) for worker in self.workers.values()), default=0.0)

    def command(self, model_id, action, request, timeout=12):
        camera_id = request.get("camera_id")
        identifier, future = uuid.uuid4().hex, Future()
        with self.command_lock:
            with self.lock:
                worker = self._active_worker(model_id, None if action in {"reload_model", "reload_labels"} else camera_id)
                if not worker:
                    message = "Model không còn chạy để cập nhật engine." if action in {"reload_model", "reload_labels"} else "Deploy model cho camera này trước khi đăng ký Label."
                    raise RuntimeError(message)
                if len(self.commands) >= 16:
                    raise RuntimeError("GPU đang xử lý quá nhiều yêu cầu Label; hãy chờ lượt hiện tại.")
                process = worker["process"]
                self.commands[identifier] = {"future": future, "process": process}
            try:
                process.stdin.write(json.dumps(dict(id=identifier, action=action, **request)) + "\n")
                process.stdin.flush()
            except Exception as error:
                with self.lock:
                    self.commands.pop(identifier, None)
                raise RuntimeError("Worker GPU đã dừng. Hãy thử lại.") from error
        try:
            return future.result(timeout=timeout)
        except TimeoutError as error:
            raise TimeoutError("Camera/GPU chưa trả lời. Hãy lấy frame mới và thử lại.") from error
        finally:
            with self.lock:
                self.commands.pop(identifier, None)

    def status(self):
        with self.lock:
            now, workers, cameras = time.time(), {}, {}
            for model_id, worker in self.workers.items():
                running = bool(worker.get("process") and worker["process"].poll() is None)
                frames = copy.deepcopy(worker["frames"])
                for camera_id, frame in frames.items():
                    frame["age_ms"] = round((now - frame["last_frame_at"]) * 1000)
                    old = cameras.get(camera_id)
                    if old is None or frame["last_frame_at"] > old["last_frame_at"]:
                        cameras[camera_id] = frame
                workers[model_id] = {
                    "running": running,
                    "state": "live" if running and any(frame["age_ms"] < 1000 for frame in frames.values()) else "starting" if running else "idle",
                    "cameras": frames,
                    "error": worker.get("error"),
                    "restarts": worker.get("restarts", 0),
                    "model_type": worker.get("model_type", "detect"),
                }
            running = any(item["running"] for item in workers.values())
            return {
                "running": running, "model_id": self.model_id, "model_ids": sorted(workers), "workers": workers,
                "cameras": cameras, "error": next((item["error"] for item in workers.values() if item.get("error")), None),
                "monitor_hidden_cameras": sorted(self.hidden_monitor_cameras),
                "restarts": sum(item["restarts"] for item in workers.values()),
                "state": "live" if running and any(frame["age_ms"] < 1000 for frame in cameras.values()) else "starting" if running else "idle",
            }

    def reload_model(self, model_id, engine, version):
        if not self._active_worker(model_id):
            return {"mode": "next_start", "version": version}
        return self.command(model_id, "reload_model", {"engine": str(engine), "version": version}, timeout=75)

    def reload_labels(self, model_id, revision, activation=None):
        if not self._active_worker(model_id):
            return {"mode": "next_start", "revision": revision}
        try:
            return self.command(model_id, "reload_labels", {"revision": revision, "activation": activation}, timeout=4)
        except (RuntimeError, TimeoutError) as error:
            return {"mode": "pending", "revision": revision, "error": str(error)}

    def start(self):
        self.stopping.clear()
        self.thread = threading.Thread(target=self._run, name="custom-detector-supervisor", daemon=True)
        self.thread.start()

    def accepts_stream(self, camera_id, model_id):
        with self.lock:
            if self.stopping.is_set() or not model_id:
                return False
            if self._active_worker(model_id, camera_id):
                return True
            return bool(not self.workers and self.signature and self.process is not None
                        and self.process.poll() is None and self.model_id == model_id
                        and self.signature[0] == model_id and camera_id in self.signature[1])

    def monitor_payload(self, payload):
        return apply_monitor_visibility(payload, self.hidden_monitor_cameras)

    def _clear(self, camera_id, model_id):
        try:
            self.clear(camera_id, model_id)
        except TypeError:
            self.clear(camera_id)

    def _terminate_worker(self, model_id, message="Model đã dừng/thay đổi; lấy lại frame Label."):
        with self.lock:
            worker = self.workers.pop(model_id, None)
            if not worker:
                return
            process = worker.get("process")
            cameras = list(worker["frames"])
            for identifier, command in list(self.commands.items()):
                if command["process"] is process:
                    if not command["future"].done():
                        command["future"].set_exception(RuntimeError(message))
                    self.commands.pop(identifier, None)
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for camera_id in cameras:
            self._clear(camera_id, model_id)
        with self.lock:
            self._sync_legacy_state()

    def _terminate(self):
        """Compatibility hook used when an engine hot-reload must be aborted."""
        for model_id in list(self.workers):
            self._terminate_worker(model_id)

    def _read(self, process, model_id, camera_ids):
        pending, pending_lock, done = {}, threading.Lock(), threading.Event()

        def publish():
            while not done.wait(.005):
                with pending_lock:
                    newest = list(pending.values())
                    pending.clear()
                for payload in newest:
                    with self.lock:
                        worker = self.workers.get(model_id)
                        current = bool(worker and worker.get("process") is process)
                    if not current or time.time() * 1000 - payload["timestamp"] > 300:
                        continue
                    try:
                        if "_phase0" in payload:
                            payload["_phase0"]["parent_publish_ns"] = time.perf_counter_ns()
                        self.callback(payload)
                    except Exception:
                        logging.exception("Custom model metadata callback")

        sender = threading.Thread(target=publish, name=f"custom-metadata-{model_id[:8]}", daemon=True)
        sender.start()
        try:
            for line in process.stdout:
                if line.startswith("RSKY_REPLY "):
                    try:
                        reply = json.loads(line[11:])
                        with self.lock:
                            command = self.commands.get(reply["id"])
                            future = command["future"] if command and command["process"] is process else None
                        if future and not future.done():
                            if reply.get("error"):
                                future.set_exception(RuntimeError(reply["error"]))
                            else:
                                future.set_result(reply.get("result", {}))
                    except (ValueError, KeyError):
                        logging.exception("Invalid label reply")
                    continue
                if not line.startswith("RSKY_META "):
                    continue
                try:
                    payload = json.loads(line[10:])
                    if "_phase0" in payload:
                        payload["_phase0"]["parent_read_ns"] = time.perf_counter_ns()
                    stream = payload["streams"][0]
                    camera_id = stream["cam_id"]
                    if camera_id not in camera_ids:
                        continue
                    with self.lock:
                        worker = self.workers.get(model_id)
                        if not worker or worker.get("process") is not process:
                            continue
                        previous = worker["frames"].get(camera_id, {})
                        worker["frames"][camera_id] = {
                            "frame_id": stream["frame_id"], "last_frame_at": payload["timestamp"] / 1000,
                            "frames": previous.get("frames", 0) + 1, "model_id": model_id,
                            "segmentation": stream.get("segmentation", {}),
                            "model_version": stream.get("model_version", "original"),
                        }
                        worker["error"] = None
                        self._sync_legacy_state()
                    with pending_lock:
                        pending[camera_id] = payload
                except (ValueError, KeyError, TypeError):
                    logging.exception("Invalid custom model metadata")
        finally:
            done.set()
            sender.join(timeout=1)

    def _launch(self, model_id, camera_ids, confidence_thresholds=None):
        if not camera_ids:
            return
        if self.before_launch:
            self.before_launch()
        from core.object_logic import infer_category

        confidence_thresholds = confidence_thresholds or {}
        row = self.registry.get(model_id)
        if row["state"] != "ready":
            raise ValueError("Model chưa sẵn sàng.")
        if self.claim:
            for camera_id in camera_ids:
                self.claim(camera_id)
        directory, runtime = self.registry.directory(model_id), self.registry.runtime_spec(row)
        tracker_file = directory / "tracker.yml"
        tracker_file.write_text((Path(__file__).resolve().parents[1] / "models_config/tracker_config.yml").read_text())
        generation = uuid.uuid4().hex[:8]
        model_type = str(row.get("metadata", {}).get("model_type") or row.get("metadata", {}).get("task") or "detect")

        base_config_path = Path(runtime["config"])
        base_config_text = base_config_path.read_text()
        lines = []
        for line in base_config_text.splitlines():
            stripped = line.strip()
            if stripped.startswith("[class-attrs-") and not stripped.startswith("[class-attrs-all]"):
                break
            lines.append(line)
        cleaned_config = "\n".join(lines).strip()

        class_sections = []
        for class_id, label in enumerate(row["labels"]):
            min_thresh = None
            def_val = confidence_thresholds.get("default", {}).get(label)
            if def_val is not None:
                try:
                    min_thresh = float(def_val)
                except (ValueError, TypeError):
                    pass
            for cam_cfg in confidence_thresholds.get("cameras", {}).values():
                if isinstance(cam_cfg, dict) and label in cam_cfg:
                    try:
                        c_val = float(cam_cfg[label])
                        min_thresh = c_val if min_thresh is None else min(min_thresh, c_val)
                    except (ValueError, TypeError):
                        pass
            if min_thresh is not None:
                nvinfer_thresh = max(0.005, min(0.25, round(min_thresh, 4)))
                class_sections.append(f"[class-attrs-{class_id}]\npre-cluster-threshold={nvinfer_thresh}")

        runtime_config_file = directory / f"nvinfer-{generation}.txt"
        if class_sections:
            runtime_config_file.write_text(cleaned_config + "\n\n" + "\n\n".join(class_sections) + "\n")
        else:
            runtime_config_file.write_text(cleaned_config + "\n")

        settings = {
            "model_id": model_id, "generation": generation, "cameras": sorted(camera_ids),
            "config": str(runtime_config_file), "model_version": runtime["version"], "model_directory": str(directory),
            "engine": runtime["engine"], "tracker": str(tracker_file), "labels": row["labels"],
            "categories": [infer_category(None, label) for label in row["labels"]], "model_type": model_type,
            "network_shape": row.get("metadata", {}).get("shape", [3, 640, 640]),
            # nvinfer rect_params are stream-space, after DeepStream removes
            # letterboxing. Raw parser integrations can opt into "network".
            "metadata_coordinates": "stream",
            "confidence_thresholds": confidence_thresholds,
        }
        request_file = directory / f"runtime-{generation}.json"
        request_file.write_text(json.dumps(settings))
        log_path = directory / f"runtime-{generation}.log"
        with log_path.open("w") as log:
            process = subprocess.Popen(
                [sys.executable, "-m", "core.custom_deepstream_worker", str(request_file)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1, start_new_session=True,
            )
        signature = (model_id, tuple(sorted(camera_ids)), json.dumps(confidence_thresholds, sort_keys=True))
        worker = {
            "model_id": model_id, "camera_ids": frozenset(camera_ids), "signature": signature,
            "process": process, "frames": {}, "error": None, "restarts": 0, "retry_at": 0.0,
            "started_at": time.time(), "log_path": log_path, "model_type": model_type,
            "confidence_thresholds": confidence_thresholds,
        }
        with self.lock:
            self.workers[model_id] = worker
            self._sync_legacy_state()
        threading.Thread(target=self._read, args=(process, model_id, frozenset(camera_ids)), daemon=True,
                         name=f"custom-metadata-reader-{model_id[:8]}").start()

    @staticmethod
    def _log_tail(path):
        if not path.exists():
            return ""
        with path.open("rb") as log:
            log.seek(max(0, path.stat().st_size - 2000))
            return log.read().decode(errors="replace")

    def _watch_worker(self, model_id, signature, confidence_thresholds=None):
        worker = self.workers.get(model_id)
        if not worker:
            return
        latest = max([worker["started_at"]] + [item["last_frame_at"] for item in worker["frames"].values()])
        process = worker.get("process")
        if process and (process.poll() is not None or time.time() - latest > 30):
            error = "DeepStream ngừng xuất frame; tự khởi động lại. " + self._log_tail(worker["log_path"])
            self._terminate_worker(model_id, "Worker model đã ngừng; model khác vẫn tiếp tục chạy.")
            self.workers[model_id] = {
                **worker, "process": None, "frames": {}, "error": error,
                "restarts": worker["restarts"] + 1, "retry_at": time.time() + min(30, 2 ** min(worker["restarts"] + 1, 5)),
            }
            return
        if not process and time.time() >= worker.get("retry_at", 0):
            self._terminate_worker(model_id)
            self._launch(model_id, signature[1], confidence_thresholds)

    def _run(self):
        while not self.stopping.wait(1):
            if os.getenv("DISABLE_CUSTOM_DETECTOR", "0").lower() in {"1", "true", "yes"}:
                continue
            try:
                registered, selected = set(self.cameras()), {}
                deployments = self.registry.deployments(enabled_only=True)
                workflows = self.workflows.active()
                self.hidden_monitor_cameras = hidden_monitor_cameras(deployments, workflows, registered)
                deployment_thresholds = {}
                for deployment in deployments:
                    camera_ids = registered if deployment["all_cameras"] else set(deployment["camera_ids"]) & registered
                    selected.setdefault(deployment["model_id"], set()).update(camera_ids)
                    if deployment.get("confidence_thresholds"):
                        deployment_thresholds[deployment["model_id"]] = deployment["confidence_thresholds"]
                for deployment in workflows:
                    definition = deployment["definition"]
                    source_cameras = {
                        camera_id for node in definition.get("nodes", []) if node.get("type") == "source"
                        for camera_id in node.get("config", {}).get("camera_ids", []) if camera_id in registered
                    }
                    for model_id in requested_models(definition):
                        selected.setdefault(model_id, set()).update(source_cameras)
                desired = {
                    model_id: (
                        model_id,
                        tuple(sorted(camera_ids)),
                        json.dumps(deployment_thresholds.get(model_id, {}), sort_keys=True)
                    )
                    for model_id, camera_ids in selected.items() if camera_ids
                }
                for model_id, worker in list(self.workers.items()):
                    if desired.get(model_id) != worker["signature"]:
                        self._terminate_worker(model_id)
                for model_id, signature in desired.items():
                    worker = self.workers.get(model_id)
                    conf_thresh = deployment_thresholds.get(model_id, {})
                    if worker is None:
                        try:
                            self._launch(model_id, signature[1], conf_thresh)
                        except Exception as error:
                            with self.lock:
                                self.workers[model_id] = {
                                    "model_id": model_id, "camera_ids": frozenset(signature[1]), "signature": signature,
                                    "process": None, "frames": {}, "error": str(error), "restarts": 1,
                                    "retry_at": time.time() + 10, "started_at": time.time(),
                                    "log_path": self.registry.directory(model_id) / "runtime.log", "model_type": "unknown",
                                }
                            logging.exception("Custom detector launch failed for %s", model_id)
                    else:
                        try:
                            self._watch_worker(model_id, signature, conf_thresh)
                        except Exception:
                            logging.exception("Custom detector worker watch failed for %s", model_id)
                with self.lock:
                    self._sync_legacy_state()
            except Exception:
                logging.exception("Custom detector supervisor")

    def stop(self):
        self.stopping.set()
        if self.thread:
            self.thread.join(timeout=6)
        for model_id in list(self.workers):
            self._terminate_worker(model_id)
