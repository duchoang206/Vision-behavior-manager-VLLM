import os
import signal
import subprocess
import threading
from urllib.parse import quote

import requests


def camera_relay_url(cam_id: str) -> str:
    origin = os.getenv("MEDIAMTX_RTSP_ORIGIN", "rtsp://127.0.0.1:8554").rstrip("/")
    return f"{origin}/{quote(cam_id, safe='')}"


def camera_preview_id(cam_id: str) -> str:
    return f"{cam_id}_preview"


def camera_preview_relay_url(cam_id: str) -> str:
    origin = os.getenv("MEDIAMTX_RTSP_ORIGIN", "rtsp://127.0.0.1:8554").rstrip("/")
    return f"{origin}/{quote(camera_preview_id(cam_id), safe='')}"


def _preview_enabled() -> bool:
    return os.getenv("DISPLAY_PREVIEW_ENABLED", "1").strip().lower() not in {"0", "false", "no"}


def _preview_integer(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


class MediaMTXClient:
    def __init__(self, api=None):
        self.api = (api or os.getenv("MEDIAMTX_API", "http://127.0.0.1:9997/v3/config/paths")).rstrip("/")
        self.preview_processes = {}
        self.preview_desired = set()
        self.preview_failures = {}
        self.preview_retry_timers = {}
        self.preview_lock = threading.RLock()

    def ensure_path(self, cam_id: str, source: str, source_on_demand: bool = False):
        path = quote(cam_id, safe="")
        config = {"source": source, "sourceOnDemand": bool(source_on_demand), "rtspTransport": "tcp"}
        current = requests.get(f"{self.api}/get/{path}", timeout=3)
        if current.status_code == 200:
            if all(current.json().get(key) == value for key, value in config.items()):
                return
            response = requests.patch(f"{self.api}/patch/{path}", json=config, timeout=3)
        elif current.status_code == 404:
            response = requests.post(f"{self.api}/add/{path}", json=config, timeout=3)
            if response.status_code == 409:
                response = requests.patch(f"{self.api}/patch/{path}", json=config, timeout=3)
        else:
            current.raise_for_status()
            return
        response.raise_for_status()

    def delete_path(self, cam_id: str):
        response = requests.delete(f"{self.api}/delete/{quote(cam_id, safe='')}", timeout=3)
        if response.status_code != 404:
            response.raise_for_status()

    def _preview_command(self, cam_id: str):
        width = _preview_integer("DISPLAY_PREVIEW_WIDTH", 640, 160, 1920) // 2 * 2
        height = _preview_integer("DISPLAY_PREVIEW_HEIGHT", 360, 90, 1080) // 2 * 2
        fps = _preview_integer("DISPLAY_PREVIEW_FPS", 25, 1, 30)
        bitrate_kbps = _preview_integer("DISPLAY_PREVIEW_BITRATE_KBPS", 1200, 100, 10000)
        gop = max(1, fps * 2)
        encoder = os.getenv("DISPLAY_PREVIEW_ENCODER", "libx264").strip() or "libx264"
        preset = os.getenv("DISPLAY_PREVIEW_PRESET", "veryfast").strip() or "veryfast"
        return [
            os.getenv("FFMPEG_PATH", "ffmpeg"), "-hide_banner", "-loglevel", "warning", "-nostdin",
            "-rtsp_transport", "tcp", "-fflags", "+genpts+nobuffer+discardcorrupt", "-flags", "low_delay",
            "-i", camera_relay_url(cam_id), "-an",
            "-vf", f"fps={fps},scale={width}:{height}:force_original_aspect_ratio=decrease:flags=lanczos,"
                   f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
            "-c:v", encoder, "-preset", preset, "-tune", "zerolatency",
            "-b:v", f"{bitrate_kbps}k", "-maxrate", f"{bitrate_kbps}k", "-bufsize", f"{bitrate_kbps * 2}k",
            "-g", str(gop), "-pix_fmt", "yuv420p", "-f", "rtsp", "-rtsp_transport", "tcp",
            camera_preview_relay_url(cam_id),
        ]

    @staticmethod
    def _preview_retry_delay(cam_id: str, failures: int) -> float:
        base = min(60.0, 2.0 * (2 ** min(max(0, failures - 1), 5)))
        jitter = (sum(cam_id.encode("utf-8")) % 1000) / 1000.0 * min(5.0, base * 0.25)
        return base + jitter

    def _launch_preview_locked(self, cam_id: str) -> bool:
        try:
            process = subprocess.Popen(
                self._preview_command(cam_id), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError:
            return False
        self.preview_processes[cam_id] = process
        watcher = threading.Thread(
            target=self._watch_preview, args=(cam_id, process),
            name=f"preview-watch-{cam_id}", daemon=True,
        )
        watcher.start()
        return True

    def _watch_preview(self, cam_id: str, process):
        process.wait()
        with self.preview_lock:
            if self.preview_processes.get(cam_id) is not process:
                return
            self.preview_processes.pop(cam_id, None)
            if cam_id not in self.preview_desired:
                return
            failures = self.preview_failures.get(cam_id, 0) + 1
            self.preview_failures[cam_id] = failures
            timer = threading.Timer(self._preview_retry_delay(cam_id, failures), self._restart_preview, args=(cam_id,))
            timer.daemon = True
            self.preview_retry_timers[cam_id] = timer
            timer.start()

    def _restart_preview(self, cam_id: str):
        with self.preview_lock:
            self.preview_retry_timers.pop(cam_id, None)
            if cam_id not in self.preview_desired:
                return
            existing = self.preview_processes.get(cam_id)
            if existing and existing.poll() is None:
                return
            self._launch_preview_locked(cam_id)

    def ensure_preview(self, cam_id: str):
        """Keep one supervised low-bitrate Monitor relay per online camera."""
        if not _preview_enabled():
            return False
        with self.preview_lock:
            self.preview_desired.add(cam_id)
            existing = self.preview_processes.get(cam_id)
            if existing and existing.poll() is None:
                return True
            timer = self.preview_retry_timers.pop(cam_id, None)
            if timer:
                timer.cancel()
            if existing:
                existing.poll()
                self.preview_processes.pop(cam_id, None)
            return self._launch_preview_locked(cam_id)

    def stop_preview(self, cam_id: str):
        with self.preview_lock:
            self.preview_desired.discard(cam_id)
            self.preview_failures.pop(cam_id, None)
            timer = self.preview_retry_timers.pop(cam_id, None)
            if timer:
                timer.cancel()
            process = self.preview_processes.pop(cam_id, None)
        if not process or process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=3)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def stop_previews(self):
        with self.preview_lock:
            camera_ids = list(self.preview_desired | set(self.preview_processes) | set(self.preview_retry_timers))
        for cam_id in camera_ids:
            self.stop_preview(cam_id)


mediamtx_client = MediaMTXClient()
