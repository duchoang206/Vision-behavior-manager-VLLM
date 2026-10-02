"""Protocol driver contract.

Every driver owns an ``asyncio.Queue`` drained by a single worker task, so
``enqueue`` is O(1) and never touches the network.  Server-mode drivers give
each connected peer its own ordered outbound queue: resync payloads are
queued synchronously before the peer joins the broadcast set, which keeps a
late snapshot from overtaking a newer transition.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger("CommGateway")

QUEUE_MAX = 1000
PEER_QUEUE_MAX = 500


@dataclass
class SendResult:
    ok: bool
    detail: str
    target: str = ""
    delivered: int = 0
    latency_ms: float = 0.0
    response: Any = None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DriverHooks:
    """Callbacks the gateway provides to drivers (all run on the loop thread)."""
    resync_payloads: Callable[["BaseDriver"], List[str]] = lambda driver: []
    on_result: Callable[["BaseDriver", Dict[str, Any], SendResult], None] = lambda driver, meta, result: None
    on_inbound: Callable[["BaseDriver", str, str], None] = lambda driver, peer, text: None
    app_port: int = 8000


@dataclass
class _Outbound:
    payload: str
    meta: Dict[str, Any] = field(default_factory=dict)
    queued_at: float = field(default_factory=time.perf_counter)


def describe_error(exc: BaseException, target: str = "") -> str:
    """Operator-readable network error (Gate 6: explain *why* a send failed)."""
    if isinstance(exc, ConnectionRefusedError):
        return f"Connection Refused: {target} đang đóng hoặc thiết bị chưa bật"
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return f"Connection Timeout: {target} không phản hồi"
    if isinstance(exc, OSError) and exc.errno in (101, 113):
        return f"Host Unreachable: không có đường mạng tới {target}"
    if isinstance(exc, OSError) and exc.errno == -2:
        return f"DNS Error: không phân giải được địa chỉ {target}"
    text = str(exc) or exc.__class__.__name__
    return f"{exc.__class__.__name__}: {text}"


class Peer:
    """One accepted connection with an ordered, bounded outbound queue."""

    def __init__(self, label: str, send: Callable[[str], Awaitable[None]], close: Callable[[], Awaitable[None]],
                 path: str = "", send_timeout: float = 3.0):
        self.label = label
        self.path = path
        self.connected_at = time.time()
        self.sent = 0
        self._send = send
        self._close = close
        self._timeout = send_timeout
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=PEER_QUEUE_MAX)
        self._task: Optional[asyncio.Task] = None
        self.closed = asyncio.Event()

    def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name=f"comm-peer-{self.label}")

    def push(self, text: str) -> "asyncio.Future":
        future = asyncio.get_running_loop().create_future()
        if self.closed.is_set():
            future.set_exception(ConnectionError(f"{self.label} đã ngắt kết nối"))
            return future
        try:
            self._queue.put_nowait((text, future))
        except asyncio.QueueFull:
            future.set_exception(BufferError(f"{self.label}: hàng đợi đầy, peer quá chậm"))
        return future

    async def _run(self) -> None:
        try:
            while True:
                text, future = await self._queue.get()
                try:
                    await asyncio.wait_for(self._send(text), self._timeout)
                    self.sent += 1
                    if not future.done():
                        future.set_result(True)
                except Exception as exc:
                    if not future.done():
                        future.set_exception(exc)
                    break
        except asyncio.CancelledError:
            pass
        finally:
            self.closed.set()
            while not self._queue.empty():
                _, future = self._queue.get_nowait()
                if not future.done():
                    future.set_exception(ConnectionError(f"{self.label} đã ngắt kết nối"))
            try:
                await self._close()
            except Exception:
                pass

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        self.closed.set()

    def as_dict(self) -> Dict[str, Any]:
        return {"peer": self.label, "path": self.path, "connected_at": self.connected_at, "sent": self.sent}


class BaseDriver(ABC):
    protocol = ""
    is_server = False

    def __init__(self, channel: Dict[str, Any], hooks: Optional[DriverHooks] = None):
        self.channel = channel
        self.id = channel["id"]
        self.config = channel.get("config") or {}
        self.hooks = hooks or DriverHooks()
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_MAX)
        self._worker: Optional[asyncio.Task] = None
        self._running = False
        self.state = "stopped"
        self.started_at = 0.0
        self.stats: Dict[str, Any] = {"sent": 0, "failed": 0, "dropped": 0, "last_error": "",
                                      "last_error_at": 0.0, "last_sent_at": 0.0, "last_latency_ms": 0.0}
        self.inbound = deque(maxlen=20)

    # ── lifecycle ────────────────────────────────────────────────────────
    async def start(self) -> None:
        self._running = True
        self.started_at = time.time()
        self._worker = asyncio.create_task(self._drain(), name=f"comm-driver-{self.id}")
        try:
            await self._open()
        except Exception as exc:
            self._record_error(describe_error(exc, self.target))
            self.state = "error"

    async def stop(self) -> None:
        self._running = False
        if self._worker:
            self._worker.cancel()
            try:
                await self._worker
            except (asyncio.CancelledError, Exception):
                pass
        try:
            await self._close()
        finally:
            self.state = "stopped"

    async def _open(self) -> None:
        """Bind/connect resources.  Must not block for long."""

    async def _close(self) -> None:
        """Release resources."""

    # ── sending ──────────────────────────────────────────────────────────
    def enqueue(self, payload: str, meta: Optional[Dict[str, Any]] = None) -> bool:
        """Non-blocking hand-off from the dispatcher.  Drops the oldest item when full."""
        item = _Outbound(payload, dict(meta or {}))
        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull:
            try:
                self._queue.get_nowait()
                self.stats["dropped"] += 1
            except asyncio.QueueEmpty:
                pass
            self._queue.put_nowait(item)
        return True

    async def _drain(self) -> None:
        while True:
            item: _Outbound = await self._queue.get()
            started = time.perf_counter()
            try:
                result = await self.send(item.payload)
            except Exception as exc:
                result = SendResult(False, describe_error(exc, self.target), self.target)
            result.latency_ms = round((time.perf_counter() - item.queued_at) * 1000.0, 3)
            self._account(result, started)
            try:
                self.hooks.on_result(self, item.meta, result)
            except Exception:
                logger.exception("on_result hook failed")

    def _account(self, result: SendResult, started: float) -> None:
        if result.ok:
            self.stats["sent"] += 1
            self.stats["last_sent_at"] = time.time()
            self.stats["last_latency_ms"] = result.latency_ms or round((time.perf_counter() - started) * 1000.0, 3)
        else:
            self.stats["failed"] += 1
            self._record_error(result.detail)

    def _record_error(self, message: str) -> None:
        self.stats["last_error"] = message
        self.stats["last_error_at"] = time.time()

    def _record_inbound(self, peer: str, text: str) -> None:
        self.inbound.append({"at": time.time(), "peer": peer, "text": text[:500]})
        try:
            self.hooks.on_inbound(self, peer, text)
        except Exception:
            logger.exception("on_inbound hook failed")

    @abstractmethod
    async def send(self, payload: str) -> SendResult:
        """Deliver ``payload`` now and report what happened."""

    async def send_now(self, payload: str, timeout: float = 5.0) -> SendResult:
        """Send immediately and wait for the outcome (used by Send Test)."""
        started = time.perf_counter()
        try:
            result = await asyncio.wait_for(self.send(payload), timeout)
        except Exception as exc:
            result = SendResult(False, describe_error(exc, self.target), self.target)
        result.latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
        self._account(result, started)
        return result

    async def wait_ready(self, timeout: float) -> bool:
        """Wait until a connection-oriented client is usable (no-op for others)."""
        return True

    async def execute(self, command: Dict[str, Any], rendered_payload: Optional[str] = None) -> SendResult:
        """Run a peripheral command.  Text protocols only understand ``send``."""
        if command.get("kind", "send") != "send":
            return SendResult(False, f"Lệnh {command.get('kind')} chỉ hỗ trợ trên giao thức MODBUS_TCP", self.target)
        return await self.send_now(rendered_payload if rendered_payload is not None else str(command.get("payload", "")))

    # ── status ───────────────────────────────────────────────────────────
    @property
    def target(self) -> str:
        ch = self.channel
        return f"{ch.get('host')}:{ch.get('port')}"

    def peers(self) -> List[Dict[str, Any]]:
        return []

    def get_status(self) -> Dict[str, Any]:
        peers = self.peers()
        return {
            "state": self.state,
            "target": self.target,
            "is_server": self.is_server,
            "clients": len(peers),
            "peers": peers,
            "queue": self._queue.qsize(),
            "started_at": self.started_at,
            "inbound": list(self.inbound)[-5:],
            **self.stats,
        }


class ServerDriverMixin:
    """Shared peer bookkeeping for WebSocket/TCP servers."""

    def _init_peers(self) -> None:
        self._peers: Dict[int, Peer] = {}

    def _admit(self, peer: Peer) -> None:
        # Queue the resync snapshot *before* the peer becomes visible to
        # broadcasts - no await in between keeps ordering strict.
        try:
            payloads = self.hooks.resync_payloads(self)
        except Exception:
            logger.exception("resync payload build failed for %s", self.id)
            payloads = []
        peer.start()
        for payload in payloads:
            peer.push(payload).add_done_callback(_swallow)
        self._peers[id(peer)] = peer
        logger.info("[%s] client %s connected (%d active)", self.id, peer.label, len(self._peers))

    def _forget(self, peer: Peer) -> None:
        if self._peers.pop(id(peer), None) is not None:
            logger.info("[%s] client %s disconnected (%d active)", self.id, peer.label, len(self._peers))

    async def _broadcast(self, text: str, timeout: float = 3.0) -> SendResult:
        peers = list(self._peers.values())
        if not peers:
            return SendResult(False, f"Không có client nào đang kết nối tới {self.target}", self.target, 0)
        futures = [peer.push(text) for peer in peers]
        done = await asyncio.gather(*(asyncio.wait_for(asyncio.shield(f), timeout) for f in futures),
                                    return_exceptions=True)
        delivered = 0
        errors = []
        for peer, outcome in zip(peers, done):
            if isinstance(outcome, BaseException):
                errors.append(f"{peer.label}: {describe_error(outcome, peer.label)}")
            else:
                delivered += 1
        if delivered:
            detail = f"Đã gửi tới {delivered}/{len(peers)} client tại {self.target}"
            if errors:
                detail += " · lỗi: " + "; ".join(errors)
            return SendResult(True, detail, self.target, delivered)
        return SendResult(False, "; ".join(errors) or "Gửi thất bại", self.target, 0)

    async def _close_peers(self) -> None:
        peers = list(self._peers.values())
        self._peers.clear()
        await asyncio.gather(*(peer.stop() for peer in peers), return_exceptions=True)

    def peers(self) -> List[Dict[str, Any]]:
        return [peer.as_dict() for peer in self._peers.values()]


def _swallow(future: "asyncio.Future") -> None:
    if not future.cancelled():
        future.exception()
