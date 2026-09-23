import json
import logging
import os
import re
import shutil
import signal
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from core.mediamtx_client import camera_relay_url


logger = logging.getLogger(__name__)
SAFE_CAMERA = re.compile(r"^[A-Za-z0-9_-]{1,100}$")
RECORDING_NAME = re.compile(r"^(\d{8}T\d{6}Z)_[a-f0-9]{12}\.mp4$")


def recording_command(source, output_pattern, segment_seconds=3600):
    return ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
            "-rtsp_transport", "tcp", "-timeout", "15000000", "-fflags", "+genpts+discardcorrupt",
            "-i", source, "-map", "0:v:0", "-an", "-sn", "-dn", "-c", "copy",
            "-f", "segment", "-segment_format", "mp4", "-segment_time", str(segment_seconds),
            "-segment_atclocktime", "1", "-reset_timestamps", "1", "-strftime", "1",
            "-segment_format_options", "movflags=+frag_keyframe+empty_moov+default_base_moof", str(output_pattern)]


def probe_recording(path):
    response = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                               "format=duration:stream=codec_name", "-of", "json", str(path)],
                              capture_output=True, text=True, timeout=15, check=True)
    data = json.loads(response.stdout)
    duration = float(data.get("format", {}).get("duration", 0))
    if duration <= 0 or not data.get("streams"):
        raise ValueError("Video chưa có dữ liệu hợp lệ.")
    return duration, data["streams"][0]["codec_name"]


class FFmpegRecorder:
    def __init__(self, store, root=None, segment_seconds=None, retention_hours=None):
        self.store = store
        self.root = Path(root or os.getenv("RECORDINGS_DIR", "/var/vms/recordings")).resolve()
        self.segment_seconds = int(segment_seconds or os.getenv("RECORDING_SEGMENT_SECONDS", "3600"))
        self.retention_hours = float(retention_hours or os.getenv("RECORDING_RETENTION_HOURS", "48"))
        self.min_free_bytes = float(os.getenv("RECORDING_MIN_FREE_GB", "2")) * 1024 ** 3
        self.enabled = os.getenv("ENABLE_EVIDENCE_RECORDING", "1").lower() not in {"0", "false", "no"}
        self.lock = threading.RLock()
        self.desired = set()
        self.jobs = {}
        self.active_paths = set()
        self.finalized = set()
        self.retry_after = {}
        self.errors = {}
        self.stop_event = threading.Event()
        self.thread = None

    def safe_path(self, relative):
        candidate = self.root / relative
        resolved = candidate.resolve()
        if candidate.is_symlink() or not resolved.is_relative_to(self.root) or len(Path(relative).parts) != 2 or resolved.suffix != ".mp4":
            raise ValueError("Đường dẫn video không hợp lệ.")
        return resolved

    def add_camera(self, camera_id):
        if not SAFE_CAMERA.fullmatch(camera_id):
            raise ValueError("Camera ID không hợp lệ.")
        with self.lock:
            self.desired.add(camera_id)

    def remove_camera(self, camera_id):
        with self.lock:
            self.desired.discard(camera_id)

    def start(self, camera_ids):
        self.store.initialize()
        self.root.mkdir(parents=True, exist_ok=True)
        if self.thread and self.thread.is_alive():
            return
        for camera_id in camera_ids:
            self.add_camera(camera_id)
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, name="ffmpeg-recording-supervisor", daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=20)

    def _launch(self, camera_id):
        directory = self.root / camera_id
        directory.mkdir(exist_ok=True)
        if directory.is_symlink():
            raise ValueError("Thư mục camera không được là symbolic link.")
        session = uuid.uuid4().hex[:12]
        pattern = directory / f"%Y%m%dT%H%M%SZ_{session}.mp4"
        process = subprocess.Popen(recording_command(camera_relay_url(camera_id), pattern, self.segment_seconds),
                                   stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   start_new_session=True, env={**os.environ, "TZ": "UTC"})
        self.jobs[camera_id] = {"process": process, "session": session, "started_at": time.time()}
        self.errors.pop(camera_id, None)

    def _finalize(self, camera_id, path, interrupted=False):
        relative = str(path.relative_to(self.root))
        if relative in self.finalized or not path.is_file():
            return
        match = RECORDING_NAME.fullmatch(path.name)
        if not match:
            return
        started = datetime.strptime(match[1], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc).timestamp()
        try:
            duration, codec = probe_recording(path)
            self.store.save(camera_id, relative, started, "interrupted" if interrupted else "ready", path.stat().st_size,
                            duration, codec, "Luồng ngắt/khởi động lại; giữ phần video đã ghi." if interrupted else None)
        except (subprocess.SubprocessError, ValueError, KeyError) as error:
            self.store.save(camera_id, relative, started, "error", path.stat().st_size, reason="MP4 không hoàn chỉnh hoặc không đọc được.")
            logger.warning("Recording probe failed for %s: %s", relative, type(error).__name__)
        self.finalized.add(relative)

    def _stop_job(self, camera_id):
        job = self.jobs.pop(camera_id)
        process = job["process"]
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        for path in sorted((self.root / camera_id).glob(f"*_{job['session']}.mp4")):
            self.active_paths.discard(str(path.relative_to(self.root)))
            self._finalize(camera_id, path, interrupted=True)

    def _scan_job(self, camera_id, job):
        paths = sorted((self.root / camera_id).glob(f"*_{job['session']}.mp4"))
        if not paths:
            return time.time() - job["started_at"] <= 45
        current = paths[-1]
        for previous in paths[:-1]:
            self.active_paths.discard(str(previous.relative_to(self.root)))
            self._finalize(camera_id, previous)
        relative = str(current.relative_to(self.root))
        self.active_paths.add(relative)
        started = datetime.strptime(RECORDING_NAME.fullmatch(current.name)[1], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc).timestamp()
        stats = current.stat()
        self.store.save(camera_id, relative, started, "recording", stats.st_size, max(0, time.time() - started))
        return time.time() - stats.st_mtime < 45

    def delete_recording(self, record_id, actor="retention", reason="user"):
        with self.lock:
            record = self.store.get(record_id)
            if not record:
                raise FileNotFoundError("Không tìm thấy bản ghi.")
            if record["path"] in self.active_paths or record["status"] == "recording":
                raise RuntimeError("Video đang ghi, chưa thể xóa. Hãy đợi segment đóng.")
            path = self.safe_path(record["path"])
            self.store.mark_deleting(record_id)
            path.unlink(missing_ok=True)
            self.store.mark_deleted(record_id, actor, reason)
            self.finalized.add(record["path"])

    def _recover(self):
        for record in self.store.pending_recovery():
            path = self.safe_path(record["path"])
            if record["status"] == "deleting":
                self.delete_recording(record["id"], "recovery", "complete_pending_delete")
            elif path.exists():
                self._finalize(record["camera_id"], path, interrupted=True)
            else:
                self.store.mark_deleted(record["id"], "recovery", "file_missing")
        for path in self.root.glob("*/*.mp4"):
            if RECORDING_NAME.fullmatch(path.name) and not path.is_symlink():
                self._finalize(path.parent.name, self.safe_path(str(path.relative_to(self.root))), interrupted=True)

    def _retention(self):
        cutoff = time.time() - self.retention_hours * 3600
        for path in self.root.glob("*/*.mp4"):
            if path.is_symlink() or not SAFE_CAMERA.fullmatch(path.parent.name):
                continue
            relative = str(path.relative_to(self.root))
            if relative in self.active_paths or path.stat().st_mtime >= cutoff:
                continue
            safe = self.safe_path(relative)
            record_id = self.store.recording_id(relative)
            record = self.store.get(record_id)
            if record:
                self.delete_recording(record_id, "retention", "expired_48h")
            else:
                safe.unlink(missing_ok=True)
                sidecar = safe.with_suffix(".json")
                if not sidecar.is_symlink():
                    sidecar.unlink(missing_ok=True)
            self.finalized.discard(relative)

    def status(self):
        with self.lock:
            storage = shutil.disk_usage(self.root)
            return {"enabled": self.enabled, "engine": "ffmpeg", "mode": "stream_copy", "directory": str(self.root),
                    "segment_seconds": self.segment_seconds, "retention_hours": self.retention_hours,
                    "free_bytes": storage.free, "cameras": [
                        {"camera_id": camera_id, "running": camera_id in self.jobs and self.jobs[camera_id]["process"].poll() is None,
                         "writing": any(path.startswith(f"{camera_id}/") for path in self.active_paths), "error": self.errors.get(camera_id)}
                        for camera_id in sorted(self.desired)]}

    def _run(self):
        last_retention = 0
        try:
            with self.lock:
                self._recover()
            while not self.stop_event.is_set():
                try:
                    with self.lock:
                        if time.time() - last_retention >= 60:
                            self._retention()
                            last_retention = time.time()
                        has_space = shutil.disk_usage(self.root).free >= self.min_free_bytes
                        for camera_id in list(self.jobs):
                            job = self.jobs[camera_id]
                            if (camera_id not in self.desired or not has_space or job["process"].poll() is not None
                                    or not self._scan_job(camera_id, job)):
                                self._stop_job(camera_id)
                                self.retry_after[camera_id] = time.time() + 10
                                self.errors[camera_id] = "SSD gần đầy, tạm dừng ghi." if not has_space else "Luồng ngắt, tự thử nối lại sau 10 giây."
                        if self.enabled and has_space:
                            for camera_id in self.desired:
                                if camera_id not in self.jobs and time.time() >= self.retry_after.get(camera_id, 0):
                                    self._launch(camera_id)
                except Exception:
                    logger.exception("FFmpeg recording supervisor will retry")
                self.stop_event.wait(3)
        except Exception:
            logger.exception("FFmpeg recording recovery failed")
        finally:
            with self.lock:
                for camera_id in list(self.jobs):
                    self._stop_job(camera_id)
