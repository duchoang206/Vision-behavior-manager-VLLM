import asyncio
import json
import logging
import threading
import time
from dataclasses import dataclass, field


async def close_metadata_socket(websocket, code=1013, reason="Metadata send stalled"):
    try:
        await asyncio.wait_for(websocket.close(code=code, reason=reason), timeout=0.25)
    except Exception:
        pass


@dataclass(frozen=True)
class MetadataFrame:
    message: str
    timestamp: float
    queued_at: float


@dataclass
class MetadataClient:
    pending: dict = field(default_factory=dict)
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    task: object = None


class LatestMetadataBroadcaster:
    def __init__(self, send_timeout=0.10, max_age=0.5, heartbeat_interval=1.0):
        self.send_timeout = send_timeout
        self.max_age = max_age
        self.heartbeat_interval = heartbeat_interval
        self._loop = None
        self._lock = threading.Lock()
        self._pending = {}
        self._latest_timestamp = {}
        self._scheduled = False
        self._clients = {}
        self._counts = dict(sent_frames=0, replaced_frames=0, expired_frames=0,
                            out_of_order_frames=0, send_timeouts=0, send_errors=0)

    def start(self, loop):
        with self._lock:
            self._loop = loop

    def register(self, websocket):
        client = MetadataClient()
        self._clients[websocket] = client
        client.task = asyncio.create_task(self._send_loop(websocket, client))

    async def unregister(self, websocket):
        client = self._clients.pop(websocket, None)
        if client is not None:
            client.pending.clear()
            client.task.cancel()
            await asyncio.gather(client.task, return_exceptions=True)

    async def stop(self):
        with self._lock:
            self._loop = None
            self._pending.clear()
            self._latest_timestamp.clear()
            self._scheduled = False
        await asyncio.gather(*(self.unregister(websocket) for websocket in list(self._clients)))

    def publish(self, payload):
        if not self._clients:
            return
        queued_at = time.monotonic()
        timestamp = payload.get("timestamp", time.time() * 1000)
        frames = {}
        for stream in payload.get("streams", []):
            camera = str(stream.get("cam_id", ""))
            if not camera:
                continue
            message = json.dumps({**payload, "streams": [stream]}, separators=(",", ":"))
            frames[camera] = MetadataFrame(message, timestamp, queued_at)
        with self._lock:
            if self._loop is None or not frames:
                return
            for camera, frame in frames.items():
                if frame.timestamp < self._latest_timestamp.get(camera, float("-inf")):
                    self._counts["out_of_order_frames"] += 1
                    continue
                self._latest_timestamp[camera] = frame.timestamp
                if camera in self._pending:
                    self._counts["replaced_frames"] += 1
                self._pending[camera] = frame
            if self._pending and not self._scheduled:
                self._scheduled = True
                try:
                    self._loop.call_soon_threadsafe(self._deliver_pending)
                except RuntimeError:
                    self._pending.clear()
                    self._scheduled = False

    def _deliver_pending(self):
        with self._lock:
            frames = self._pending
            self._pending = {}
            self._scheduled = False
        for client in list(self._clients.values()):
            for camera, frame in frames.items():
                if camera in client.pending:
                    self._counts["replaced_frames"] += 1
                client.pending[camera] = frame
            client.ready.set()

    async def _send_loop(self, websocket, client):
        next_heartbeat = 0.0
        close_code = 1001
        close_reason = "Metadata connection closed"
        try:
            while self._clients.get(websocket) is client:
                now = time.monotonic()
                is_frame = False
                if now >= next_heartbeat:
                    message = json.dumps({"type": "metadata_heartbeat", "timestamp": int(time.time() * 1000)})
                    next_heartbeat = now + self.heartbeat_interval
                elif client.pending:
                    camera = next(iter(client.pending))
                    frame = client.pending.pop(camera)
                    if now - frame.queued_at > self.max_age:
                        self._counts["expired_frames"] += 1
                        continue
                    message = frame.message
                    is_frame = True
                else:
                    client.ready.clear()
                    try:
                        await asyncio.wait_for(client.ready.wait(), timeout=max(0.001, next_heartbeat - now))
                    except asyncio.TimeoutError:
                        pass
                    continue
                await asyncio.wait_for(websocket.send_text(message), timeout=self.send_timeout)
                if is_frame:
                    self._counts["sent_frames"] += 1
        except asyncio.TimeoutError:
            self._counts["send_timeouts"] += 1
            close_code, close_reason = 1013, "Metadata send timeout; reconnect"
            logging.warning("Metadata WebSocket send timed out; closing slow client")
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self._counts["send_errors"] += 1
            close_code, close_reason = 1013, "Metadata send failed; reconnect"
            logging.warning("Metadata WebSocket send failed: %s", error)
        finally:
            self._clients.pop(websocket, None)
            client.pending.clear()
            await close_metadata_socket(websocket, close_code, close_reason)

    def status(self):
        with self._lock:
            pending = len(self._pending)
            counts = dict(self._counts)
        return {**counts, "clients": len(self._clients), "pending_cameras": pending,
                "queued_frames": sum(len(client.pending) for client in self._clients.values()),
                "max_queue_age_ms": round(self.max_age * 1000),
                "send_timeout_ms": round(self.send_timeout * 1000)}
