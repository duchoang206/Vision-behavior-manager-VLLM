"""Raw TCP socket drivers (PLC, call boxes, light towers, legacy WCS)."""

import asyncio
import codecs
import logging
from typing import Any, Dict, Optional

from .base_driver import BaseDriver, Peer, SendResult, ServerDriverMixin, describe_error

logger = logging.getLogger("CommGateway")

TERMINATORS = {"lf": "\n", "crlf": "\r\n", "none": "", "cr": "\r"}


def frame(payload: str, config: Dict[str, Any]) -> bytes:
    terminator = TERMINATORS.get(str(config.get("line_terminator", "lf")).lower(), "\n")
    encoding = config.get("encoding") or "utf-8"
    if config.get("hex_payload"):
        return bytes.fromhex(payload.replace(" ", "")) + terminator.encode(encoding)
    return (payload + terminator).encode(encoding)


def _decode(data: bytes, config: Dict[str, Any]) -> str:
    if config.get("hex_payload"):
        return data.hex(" ")
    return codecs.decode(data, config.get("encoding") or "utf-8", "replace")


class TcpServerDriver(ServerDriverMixin, BaseDriver):
    protocol = "TCPIP"
    is_server = True

    def __init__(self, channel: Dict[str, Any], hooks=None):
        super().__init__(channel, hooks)
        self._init_peers()
        self._server: Optional[asyncio.base_events.Server] = None

    @property
    def target(self) -> str:
        return f"tcp://{self.channel.get('host') or '0.0.0.0'}:{self.channel.get('port')}"

    async def _open(self) -> None:
        self._server = await asyncio.start_server(self._handle, self.channel.get("host") or "0.0.0.0",
                                                  int(self.channel["port"]))
        self.state = "listening"
        logger.info("[%s] TCP server listening on %s", self.id, self.target)

    async def _close(self) -> None:
        await self._close_peers()
        if self._server is not None:
            self._server.close()
            try:
                await asyncio.wait_for(self._server.wait_closed(), 3.0)
            except Exception:
                pass
            self._server = None

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        address = writer.get_extra_info("peername")
        label = f"{address[0]}:{address[1]}" if address else "unknown"

        async def send_bytes(text: str) -> None:
            writer.write(frame(text, self.config))
            await writer.drain()

        async def close() -> None:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

        peer = Peer(label, send_bytes, close)
        self._admit(peer)
        try:
            while True:
                data = await reader.read(4096)
                if not data:
                    break
                self._record_inbound(label, _decode(data, self.config))
        except Exception:
            pass
        finally:
            self._forget(peer)
            await peer.stop()

    async def send(self, payload: str) -> SendResult:
        return await self._broadcast(payload)


class TcpClientDriver(BaseDriver):
    protocol = "TCPIP"

    def __init__(self, channel: Dict[str, Any], hooks=None):
        super().__init__(channel, hooks)
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._reader_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        self._connect_lock = asyncio.Lock()
        self._reconnect_task: Optional[asyncio.Task] = None

    @property
    def target(self) -> str:
        return f"tcp://{self.channel.get('host')}:{self.channel.get('port')}"

    async def _open(self) -> None:
        self.state = "connecting"
        self._reconnect_task = asyncio.create_task(self._keep_connected(), name=f"comm-tcpclient-{self.id}")

    async def _keep_connected(self) -> None:
        backoff = 1.0
        while self._running:
            if self._writer is None:
                try:
                    await self._connect()
                    backoff = 1.0
                    for payload in self.hooks.resync_payloads(self):
                        await self._write(payload)
                except asyncio.CancelledError:
                    break
                except Exception as exc:
                    self._record_error(describe_error(exc, self.target))
                    self.state = "connecting"
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2.0, 30.0)
                    continue
            await asyncio.sleep(1.0)

    async def _connect(self) -> None:
        # The reconnect loop and an on-demand send may race; only one may dial.
        async with self._connect_lock:
            if self._writer is not None:
                return
            timeout = float(self.config.get("connect_timeout_sec") or 3.0)
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(self.channel.get("host"), int(self.channel["port"])), timeout)
            self.state = "online"
            self._reader_task = asyncio.create_task(self._read_loop(self._reader, self._writer),
                                                    name=f"comm-tcpread-{self.id}")
            logger.info("[%s] connected to %s", self.id, self.target)

    async def _read_loop(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                data = await reader.read(4096)
                if not data:
                    break
                self._record_inbound(self.target, _decode(data, self.config))
        except Exception:
            pass
        if self._writer is writer:  # do not tear down a newer connection
            await self._drop_connection()

    async def _drop_connection(self) -> None:
        writer, self._writer, self._reader = self._writer, None, None
        if self._running:
            self.state = "connecting"
        if writer is not None:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def _write(self, payload: str) -> None:
        async with self._lock:
            if self._writer is None:
                raise ConnectionError("chưa kết nối")
            self._writer.write(frame(payload, self.config))
            await self._writer.drain()

    async def send(self, payload: str) -> SendResult:
        try:
            if self._writer is None:
                await self._connect()
            await self._write(payload)
            return SendResult(True, f"Đã gửi tới {self.target}", self.target, 1)
        except Exception as exc:
            await self._drop_connection()
            return SendResult(False, describe_error(exc, self.target), self.target)

    async def _close(self) -> None:
        for task in (self._reconnect_task, self._reader_task):
            if task:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        await self._drop_connection()

    def peers(self):
        return [{"peer": self.target}] if self._writer is not None else []
