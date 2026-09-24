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
        self.preview_lock = threading.RLock()

    def ensure_path(self, cam_id: str, source: str):
        path = quote(cam_id, safe="")
        config = {"source": source, "sourceOnDemand": False, "rtspTransport": "tcp"}
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

    def ensure_preview(self, cam_id: str):
        """Start the low-bitrate Monitor relay without changing the AI camera relay."""
        if not _preview_enabled():
            return False
        with self.preview_lock:
            existing = self.preview_processes.get(cam_id)
            if existing and existing.poll() is None:
                return True
            if existing:
                self.preview_processes.pop(cam_id, None)
            try:
                process = subprocess.Popen(
                    self._preview_command(cam_id), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
            except OSError:
                return False
            self.preview_processes[cam_id] = process
        return True

    def stop_preview(self, cam_id: str):
        with self.preview_lock:
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
            camera_ids = list(self.preview_processes)
        for cam_id in camera_ids:
            self.stop_preview(cam_id)


mediamtx_client = MediaMTXClient()
