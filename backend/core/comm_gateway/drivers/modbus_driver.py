"""Minimal asyncio Modbus TCP master for peripheral control.

Covers the function codes field devices in this plant need (PLC coils for
light towers / doors, box sensors, caller LEDs): FC01 read coils, FC03 read
holding registers, FC05 write single coil, FC06 write single register.
Implemented on raw sockets so no extra dependency is required in the image.
"""

import asyncio
import json
import logging
import struct
import time
from typing import Any, Dict, List, Optional

from .base_driver import BaseDriver, SendResult, describe_error

logger = logging.getLogger("CommGateway")

EXCEPTION_NAMES = {
    1: "ILLEGAL FUNCTION", 2: "ILLEGAL DATA ADDRESS", 3: "ILLEGAL DATA VALUE",
    4: "SERVER DEVICE FAILURE", 5: "ACKNOWLEDGE", 6: "SERVER DEVICE BUSY",
    10: "GATEWAY PATH UNAVAILABLE", 11: "GATEWAY TARGET FAILED TO RESPOND",
}


class ModbusError(Exception):
    pass


def _bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "on", "yes")
    return bool(value)


class ModbusTcpDriver(BaseDriver):
    protocol = "MODBUS_TCP"

    def __init__(self, channel: Dict[str, Any], hooks=None):
        super().__init__(channel, hooks)
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._lock = asyncio.Lock()
        self._tid = 0
        self._poll_task: Optional[asyncio.Task] = None
        self.poll_values: Dict[str, Any] = {}

    @property
    def unit_id(self) -> int:
        return int(self.config.get("unit_id") or 1) & 0xFF

    @property
    def target(self) -> str:
        return f"modbus://{self.channel.get('host')}:{self.channel.get('port')}#unit{self.unit_id}"

    # ── connection ───────────────────────────────────────────────────────
    async def _open(self) -> None:
        self.state = "connecting"
        poll = self.config.get("poll") or {}
        if poll.get("enabled"):
            self._poll_task = asyncio.create_task(self._poll_loop(poll), name=f"comm-modbus-poll-{self.id}")
        else:
            # Probe once so the status column reflects reachability.
            self._poll_task = asyncio.create_task(self._probe(), name=f"comm-modbus-probe-{self.id}")

    async def _probe(self) -> None:
        try:
            async with self._lock:
                await self._ensure_connected()
        except Exception as exc:
            self._record_error(describe_error(exc, self.target))
            self.state = "offline"

    async def _ensure_connected(self) -> None:
        if self._writer is not None:
            return
        timeout = float(self.config.get("connect_timeout_sec") or 3.0)
        self._reader, self._writer = await asyncio.wait_for(
            asyncio.open_connection(self.channel.get("host"), int(self.channel["port"])), timeout)
        self.state = "online"

    async def _disconnect(self) -> None:
        writer, self._writer, self._reader = self._writer, None, None
        if writer is not None:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def _close(self) -> None:
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except (asyncio.CancelledError, Exception):
                pass
        await self._disconnect()

    # ── protocol ─────────────────────────────────────────────────────────
    async def _request(self, function: int, data: bytes) -> bytes:
        timeout = float(self.config.get("timeout_sec") or 3.0)
        async with self._lock:
            try:
                await self._ensure_connected()
                self._tid = (self._tid + 1) & 0xFFFF
                pdu = bytes([function]) + data
                self._writer.write(struct.pack(">HHHB", self._tid, 0, len(pdu) + 1, self.unit_id) + pdu)
                await self._writer.drain()
                header = await asyncio.wait_for(self._reader.readexactly(7), timeout)
                tid, _, length, _ = struct.unpack(">HHHB", header)
                body = await asyncio.wait_for(self._reader.readexactly(length - 1), timeout)
            except Exception:
                await self._disconnect()
                self.state = "offline"
                raise
        if tid != self._tid:
            await self._disconnect()
            raise ModbusError(f"Transaction id lệch ({tid} != {self._tid})")
        if body[0] & 0x80:
            code = body[1] if len(body) > 1 else 0
            raise ModbusError(f"Modbus exception {code}: {EXCEPTION_NAMES.get(code, 'UNKNOWN')}")
        if body[0] != function:
            raise ModbusError(f"Phản hồi sai function code {body[0]}")
        return body[1:]

    async def write_coil(self, address: int, value: bool) -> None:
        await self._request(5, struct.pack(">HH", address, 0xFF00 if value else 0x0000))

    async def write_register(self, address: int, value: int) -> None:
        await self._request(6, struct.pack(">HH", address, int(value) & 0xFFFF))

    async def read_coils(self, address: int, count: int) -> List[bool]:
        data = await self._request(1, struct.pack(">HH", address, count))
        bits = data[1:1 + data[0]]
        return [bool(bits[i // 8] >> (i % 8) & 1) for i in range(count)]

    async def read_registers(self, address: int, count: int) -> List[int]:
        data = await self._request(3, struct.pack(">HH", address, count))
        return list(struct.unpack(f">{data[0] // 2}H", data[1:1 + data[0]]))

    async def run_command(self, command: Dict[str, Any]) -> SendResult:
        kind = command.get("kind")
        address = int(command.get("address") or 0)
        started = time.perf_counter()
        try:
            if kind == "modbus_coil":
                value = _bool(command.get("value"))
                await self.write_coil(address, value)
                detail, response = f"FC05 coil[{address}] = {'ON' if value else 'OFF'}", None
            elif kind == "modbus_register":
                value = int(command.get("value") or 0)
                await self.write_register(address, value)
                detail, response = f"FC06 register[{address}] = {value}", None
            elif kind == "modbus_read_coils":
                response = await self.read_coils(address, max(1, int(command.get("count") or 1)))
                detail = f"FC01 coils[{address}..] = {[int(v) for v in response]}"
            elif kind == "modbus_read_registers":
                response = await self.read_registers(address, max(1, int(command.get("count") or 1)))
                detail = f"FC03 registers[{address}..] = {response}"
            else:
                return SendResult(False, f"Lệnh '{kind}' không hợp lệ cho Modbus TCP", self.target)
        except Exception as exc:
            detail = str(exc) if isinstance(exc, ModbusError) else describe_error(exc, self.target)
            return SendResult(False, detail, self.target,
                              latency_ms=round((time.perf_counter() - started) * 1000, 3))
        return SendResult(True, f"{detail} @ {self.target}", self.target, 1,
                          round((time.perf_counter() - started) * 1000, 3), response)

    async def send(self, payload: str) -> SendResult:
        try:
            command = json.loads(payload)
        except (TypeError, ValueError):
            return SendResult(False, "Payload Modbus phải là lệnh JSON, ví dụ "
                                     '{"kind":"modbus_coil","address":0,"value":true}', self.target)
        return await self.run_command(command)

    async def execute(self, command: Dict[str, Any], rendered_payload: Optional[str] = None) -> SendResult:
        if command.get("kind", "send") == "send":
            return SendResult(False, "Modbus TCP không hỗ trợ lệnh gửi text; dùng modbus_coil/modbus_register", self.target)
        started = time.perf_counter()
        result = await self.run_command(command)
        self._account(result, started)
        return result

    # ── optional polling (box sensors, caller buttons) ───────────────────
    async def _poll_loop(self, poll: Dict[str, Any]) -> None:
        interval = max(0.2, float(poll.get("interval_sec") or 1.0))
        kind = "modbus_read_registers" if poll.get("kind") == "registers" else "modbus_read_coils"
        command = {"kind": kind, "address": int(poll.get("address") or 0), "count": int(poll.get("count") or 8)}
        while self._running:
            result = await self.run_command(command)
            if result.ok:
                self.poll_values = {"at": time.time(), "kind": kind, "address": command["address"], "values": result.response}
            else:
                self._record_error(result.detail)
            await asyncio.sleep(interval)

    def peers(self):
        return [{"peer": self.target}] if self._writer is not None else []

    def get_status(self) -> Dict[str, Any]:
        status = super().get_status()
        status["poll"] = self.poll_values
        return status
