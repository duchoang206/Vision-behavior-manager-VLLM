"""WebSocket server/client drivers.

Server mode is what the FMS WCS expects: an FMS device of type ``CAMERA_AI``
with protocol ``WebSocket`` and mode ``client`` runs ``WsCameraClient``, which
connects to ``ws://<Device Ip>:<Device Port><config.path>`` (path defaults to
``/``) and reconnects every 3 s.  Its hand-written client does not answer
ping frames reliably, so keep-alive pings are disabled here.

When the channel port equals the backend port the socket is served by the
FastAPI routes (``/ws/wcs_camera`` and ``/ws/comm/{channel_id}``); any other
port gets a dedicated listener that accepts every request path.
"""

import asyncio
import logging
from typing import Any, Dict, Optional

from .base_driver import BaseDriver, Peer, SendResult, ServerDriverMixin, describe_error

logger = logging.getLogger("CommGateway")

try:  # websockets >= 13 (new asyncio implementation)
    from websockets.asyncio.server import serve as ws_serve
    from websockets.asyncio.client import connect as ws_connect
    _LEGACY = False
except ImportError:  # pragma: no cover - legacy fallback
    try:
        from websockets import serve as ws_serve, connect as ws_connect
        _LEGACY = True
    except ImportError:
        ws_serve = ws_connect = None
        _LEGACY = False


def _peer_label(address: Any) -> str:
    if isinstance(address, (tuple, list)) and len(address) >= 2:
        return f"{address[0]}:{address[1]}"
    return str(address or "unknown")


class WebSocketServerDriver(ServerDriverMixin, BaseDriver):
    protocol = "WEBSOCKET"
    is_server = True

    def __init__(self, channel: Dict[str, Any], hooks=None):
        super().__init__(channel, hooks)
        self._init_peers()
        self._server = None

    @property
    def embedded(self) -> bool:
        return int(self.channel.get("port") or 0) == int(self.hooks.app_port)

    @property
    def target(self) -> str:
        ch = self.channel
        return f"ws://{ch.get('host') or '0.0.0.0'}:{ch.get('port')}{ch.get('endpoint_path') or '/'}"

    async def _open(self) -> None:
        if self.embedded:
            self.state = "listening"
            return
        if ws_serve is None:
            raise RuntimeError("Thư viện websockets chưa được cài đặt")
        host = self.channel.get("host") or "0.0.0.0"
        port = int(self.channel["port"])
        options = dict(ping_interval=None, compression=None, max_size=2 ** 20)
        self._server = await ws_serve(self._handle_ws, host, port, **options)
        self.state = "listening"
        logger.info("[%s] WebSocket server listening on %s", self.id, self.target)

    async def _close(self) -> None:
        await self._close_peers()
        if self._server is not None:
            self._server.close()
            try:
                await asyncio.wait_for(self._server.wait_closed(), 3.0)
            except Exception:
                pass
            self._server = None

    def _path_allowed(self, path: str) -> bool:
        if not self.config.get("strict_path"):
            return True
        expected = (self.channel.get("endpoint_path") or "/").rstrip("/") or "/"
        return (path.split("?")[0].rstrip("/") or "/") == expected

    async def _handle_ws(self, connection, legacy_path: Optional[str] = None) -> None:
        request = getattr(connection, "request", None)
        path = legacy_path or getattr(request, "path", None) or getattr(connection, "path", "/") or "/"
        label = _peer_label(getattr(connection, "remote_address", None))
        if not self._path_allowed(path):
            await connection.close(code=1008, reason="unknown path")
            return
        peer = Peer(label, connection.send, connection.close, path=path)
        self._admit(peer)
        try:
            async for message in connection:
                self._record_inbound(label, message if isinstance(message, str) else message.decode("utf-8", "replace"))
        except Exception:
            pass
        finally:
            self._forget(peer)
            await peer.stop()

    async def attach_starlette(self, websocket, path: str) -> None:
        """Serve one FastAPI/Starlette WebSocket (embedded mode)."""
        await websocket.accept()
        client = getattr(websocket, "client", None)
        label = f"{client.host}:{client.port}" if client else "unknown"
        peer = Peer(label, websocket.send_text, websocket.close, path=path)
        self._admit(peer)
        try:
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                text = message.get("text")
                if text is None and message.get("bytes") is not None:
                    text = message["bytes"].decode("utf-8", "replace")
                if text:
                    self._record_inbound(label, text)
        except Exception:
            pass
        finally:
            self._forget(peer)
            await peer.stop()

    async def send(self, payload: str) -> SendResult:
        return await self._broadcast(payload)

    def get_status(self) -> Dict[str, Any]:
        status = super().get_status()
        status["embedded"] = self.embedded
        return status


class WebSocketClientDriver(BaseDriver):
    protocol = "WEBSOCKET"

    def __init__(self, channel: Dict[str, Any], hooks=None):
        super().__init__(channel, hooks)
        self._ws = None
        self._connector: Optional[asyncio.Task] = None
        self._connected = asyncio.Event()
        self._send_lock = asyncio.Lock()

    @property
    def target(self) -> str:
        ch = self.channel
        scheme = "wss" if self.config.get("use_tls") else "ws"
        path = ch.get("endpoint_path") or "/"
        return f"{scheme}://{ch.get('host')}:{ch.get('port')}{path if path.startswith('/') else '/' + path}"

    def _headers(self) -> Dict[str, str]:
        auth = self.channel.get("auth_config") or {}
        headers = {str(k): str(v) for k, v in (auth.get("headers") or {}).items()}
        if auth.get("bearer_token"):
            headers["Authorization"] = f"Bearer {auth['bearer_token']}"
        return headers

    async def _open(self) -> None:
        if ws_connect is None:
            raise RuntimeError("Thư viện websockets chưa được cài đặt")
        self.state = "connecting"
        self._connector = asyncio.create_task(self._connect_loop(), name=f"comm-wsclient-{self.id}")

    async def _connect_loop(self) -> None:
        backoff = 1.0
        timeout = float(self.config.get("connect_timeout_sec") or 3.0)
        while self._running:
            try:
                kwargs = dict(open_timeout=timeout, ping_interval=None, compression=None)
                headers = self._headers()
                if headers:
                    kwargs["additional_headers" if not _LEGACY else "extra_headers"] = headers
                self._ws = await ws_connect(self.target, **kwargs)
                self.state = "online"
                backoff = 1.0
                self._connected.set()
                logger.info("[%s] connected to %s", self.id, self.target)
                for payload in self.hooks.resync_payloads(self):
                    await self._ws.send(payload)
                async for message in self._ws:
                    self._record_inbound(self.target, message if isinstance(message, str) else message.decode("utf-8", "replace"))
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._record_error(describe_error(exc, self.target))
            finally:
                self._connected.clear()
                if self._running:
                    self.state = "connecting"
            if not self._running:
                break
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2.0, 30.0)

    async def _close(self) -> None:
        if self._connector:
            self._connector.cancel()
            try:
                await self._connector
            except (asyncio.CancelledError, Exception):
                pass
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

    async def wait_ready(self, timeout: float) -> bool:
        try:
            await asyncio.wait_for(self._connected.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def send(self, payload: str) -> SendResult:
        if not self._connected.is_set() or self._ws is None:
            detail = self.stats.get("last_error") or "chưa thiết lập kết nối"
            return SendResult(False, f"Chưa kết nối tới {self.target} ({detail})", self.target)
        async with self._send_lock:
            await self._ws.send(payload)
        return SendResult(True, f"Đã gửi tới {self.target}", self.target, 1)

    def peers(self):
        return [{"peer": self.target}] if self._connected.is_set() else []
