"""Communication gateway: channel lifecycle, event routing and FMS slot sync.

Threading model
---------------
AI frame threads (DeepStream / trackers) call :meth:`on_slot_transition`,
:meth:`on_roi_alert` and :meth:`dispatch_event`.  Those only build a small
context dict and ``call_soon_threadsafe`` into the asyncio loop, so network
I/O can never stall a video pipeline (Gate 7).  Everything else - registry
mutation, rendering, driver queues - runs on the loop thread.
"""

import asyncio
import copy
import json
import logging
import os
import time
from collections import deque
from typing import Any, Callable, Dict, List, Optional

from .channel_store import ChannelStore, ChannelValidationError, normalize_channel
from .constants import (DEFAULT_FMS_CHANNEL_ID, EVENT_ROI_ALERT, EVENT_SLOT_CARFULL, EVENT_SLOT_EMPTY,
                        EVENT_TEST, FORMAT_FMS_SLOTS, STATE_CARFULL, STATE_EMPTY)
from .context import now_fields, sample_context, slot_event_context, status_from_state
from .drivers import BaseDriver, DriverHooks, SendResult, create_driver
from .slot_registry import SlotRegistry, is_fms_slot_id
from .template_renderer import TemplateError, render_template

logger = logging.getLogger("CommGateway")

LOG_LIMIT = 300
HEARTBEAT_TICK_SEC = 0.5


def default_fms_channel() -> Dict[str, Any]:
    return {
        "id": DEFAULT_FMS_CHANNEL_ID,
        "name": "FMS WCS Camera Gateway",
        "description": "Thiết bị CAMERA_AI trên FMS (WebSocket, mode client) kết nối vào đây để nhận trạng thái ô hàng.",
        "device_type": "FMS_WCS",
        "protocol": "WEBSOCKET",
        "mode": "SERVER",
        "host": "0.0.0.0",
        "port": int(os.getenv("BACKEND_PORT", "8000")),
        "endpoint_path": "/ws/wcs_camera",
        "trigger_events": [EVENT_SLOT_CARFULL, EVENT_SLOT_EMPTY],
        "is_enabled": True,
        "priority": 10,
        "config": {"payload_format": FORMAT_FMS_SLOTS, "heartbeat_sec": 2.0, "resync_on_connect": True},
    }


class CommGatewayManager:
    def __init__(self, app_port: Optional[int] = None):
        self.app_port = int(app_port or os.getenv("BACKEND_PORT", "8000"))
        self.registry = SlotRegistry()
        self.channels: Dict[str, Dict[str, Any]] = {}
        self.drivers: Dict[str, BaseDriver] = {}
        self.store: Optional[ChannelStore] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.started = False
        self.store_error = ""
        self.log: deque = deque(maxlen=LOG_LIMIT)
        self._log_seq = 0
        self._camera_name: Callable[[str], str] = lambda cam_id: cam_id
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._last_heartbeat: Dict[str, float] = {}
        self._warned_invalid: set = set()
        self._last_warning: Dict[str, float] = {}
        self._mutation_lock: Optional[asyncio.Lock] = None
        self.stats = {"events": 0, "transitions": 0, "dispatched": 0, "render_errors": 0}

    # ── wiring ───────────────────────────────────────────────────────────
    def bind_store(self, store: ChannelStore) -> None:
        self.store = store

    def set_camera_name_resolver(self, resolver: Callable[[str], str]) -> None:
        self._camera_name = resolver

    def camera_name(self, cam_id: str) -> str:
        try:
            return self._camera_name(cam_id) or cam_id
        except Exception:
            return cam_id

    def _hooks(self) -> DriverHooks:
        return DriverHooks(resync_payloads=self._resync_payloads, on_result=self._on_driver_result,
                           on_inbound=self._on_inbound, app_port=self.app_port)

    def _in_loop(self) -> bool:
        try:
            return asyncio.get_running_loop() is self.loop
        except RuntimeError:
            return False

    def _call(self, fn: Callable, *args) -> None:
        """Run ``fn`` on the gateway loop from any thread, without blocking."""
        if self.loop is None or self.loop.is_closed():
            fn(*args)
        elif self._in_loop():
            fn(*args)
        else:
            self.loop.call_soon_threadsafe(fn, *args)

    # ── lifecycle ────────────────────────────────────────────────────────
    async def start(self) -> None:
        self.loop = asyncio.get_running_loop()
        self._mutation_lock = asyncio.Lock()
        channels: List[Dict[str, Any]] = []
        if self.store is not None:
            try:
                channels = await asyncio.to_thread(self._load_or_seed)
                self.store_error = ""
            except Exception as exc:
                self.store_error = f"Không đọc được cấu hình kênh từ PostgreSQL: {exc}"
                logger.error(self.store_error)
        for channel in channels:
            try:
                channel = normalize_channel(channel, None)
            except ChannelValidationError as exc:
                logger.error("Skipping invalid channel %s: %s", channel.get("id"), exc)
                continue
            self.channels[channel["id"]] = channel
            if channel["is_enabled"]:
                await self._start_driver(channel)
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop(), name="comm-heartbeat")
        self.started = True
        logger.info("Communication gateway started with %d channel(s)", len(self.channels))

    def _load_or_seed(self) -> List[Dict[str, Any]]:
        self.store.ensure_schema()
        rows = self.store.list()
        if not rows and os.getenv("COMM_GATEWAY_SEED_DEFAULT", "1") == "1":
            rows = [self.store.upsert(normalize_channel(default_fms_channel()))]
            logger.info("Seeded default FMS WCS channel '%s'", DEFAULT_FMS_CHANNEL_ID)
        return rows

    async def stop(self) -> None:
        self.started = False
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except (asyncio.CancelledError, Exception):
                pass
        drivers = list(self.drivers.values())
        self.drivers.clear()
        await asyncio.gather(*(d.stop() for d in drivers), return_exceptions=True)

    async def _start_driver(self, channel: Dict[str, Any]) -> None:
        driver = create_driver(channel, self._hooks())
        self.drivers[channel["id"]] = driver
        await driver.start()

    async def _stop_driver(self, channel_id: str) -> None:
        driver = self.drivers.pop(channel_id, None)
        if driver is not None:
            await driver.stop()

    # ── channel CRUD ─────────────────────────────────────────────────────
    def _check_conflicts(self, channel: Dict[str, Any]) -> None:
        if not channel["is_enabled"] or channel["mode"] != "SERVER":
            return
        for other in self.channels.values():
            if other["id"] == channel["id"] or not other["is_enabled"] or other["mode"] != "SERVER":
                continue
            if int(other["port"]) != int(channel["port"]):
                continue
            embedded = int(channel["port"]) == self.app_port
            both_ws = other["protocol"] == channel["protocol"] == "WEBSOCKET"
            if embedded and both_ws and other["endpoint_path"] != channel["endpoint_path"]:
                continue
            raise ChannelValidationError([f"Cổng {channel['port']} đã được kênh '{other['name']}' sử dụng"])
        if int(channel["port"]) == self.app_port and channel["protocol"] != "WEBSOCKET":
            raise ChannelValidationError([f"Cổng {self.app_port} là cổng API backend; chỉ WebSocket được dùng chung cổng này"])

    @property
    def _lock(self) -> asyncio.Lock:
        if self._mutation_lock is None:
            self._mutation_lock = asyncio.Lock()
        return self._mutation_lock

    async def save_channel(self, data: Dict[str, Any], channel_id: Optional[str] = None) -> Dict[str, Any]:
        async with self._lock:
            existing = self.channels.get(channel_id) if channel_id else None
            if channel_id and existing is None:
                raise KeyError(channel_id)
            payload = dict(data)
            auth = payload.get("auth_config")
            if existing and isinstance(auth, dict) and auth.get("password") == "********":
                payload["auth_config"] = {**auth, "password": (existing.get("auth_config") or {}).get("password", "")}
            if channel_id:
                payload["id"] = channel_id
            elif payload.get("id") in self.channels:
                raise ChannelValidationError([f"Mã kênh '{payload['id']}' đã tồn tại"])
            channel = normalize_channel(payload, existing)
            if not channel_id and channel["id"] in self.channels:
                raise ChannelValidationError([f"Mã kênh '{channel['id']}' đã tồn tại"])
            self._check_conflicts(channel)
            if self.store is not None:
                channel = normalize_channel(await asyncio.to_thread(self.store.upsert, channel))
            await self._stop_driver(channel["id"])
            self.channels[channel["id"]] = channel
            if channel["is_enabled"]:
                await self._start_driver(channel)
            self._record({"channel_id": channel["id"], "event": "CONFIG", "ok": True,
                          "detail": f"Đã lưu cấu hình '{channel['name']}' ({'bật' if channel['is_enabled'] else 'tắt'})"})
            return self.describe_channel(channel["id"])

    async def delete_channel(self, channel_id: str) -> None:
        async with self._lock:
            if channel_id not in self.channels:
                raise KeyError(channel_id)
            if self.store is not None:
                await asyncio.to_thread(self.store.delete, channel_id)
            await self._stop_driver(channel_id)
            channel = self.channels.pop(channel_id)
            self._record({"channel_id": channel_id, "event": "CONFIG", "ok": True, "detail": f"Đã xóa '{channel['name']}'"})

    async def restart_channel(self, channel_id: str) -> Dict[str, Any]:
        async with self._lock:
            channel = self.channels[channel_id]
            await self._stop_driver(channel_id)
            if channel["is_enabled"]:
                await self._start_driver(channel)
            return self.describe_channel(channel_id)

    async def reload_channels(self) -> None:
        """Re-read every channel from PostgreSQL and restart drivers."""
        if self.store is None:
            return
        rows = await asyncio.to_thread(self.store.list)
        async with self._lock:
            for channel_id in list(self.drivers):
                await self._stop_driver(channel_id)
            self.channels = {}
            for row in rows:
                channel = normalize_channel(row)
                self.channels[channel["id"]] = channel
                if channel["is_enabled"]:
                    await self._start_driver(channel)

    def describe_channel(self, channel_id: str) -> Dict[str, Any]:
        channel = copy.deepcopy(self.channels[channel_id])
        auth = channel.get("auth_config") or {}
        if auth.get("password"):
            auth["password"] = "********"
        driver = self.drivers.get(channel_id)
        channel["runtime"] = driver.get_status() if driver else {"state": "disabled" if not channel["is_enabled"] else "stopped",
                                                                  "clients": 0, "peers": []}
        if channel["config"].get("payload_format") == FORMAT_FMS_SLOTS:
            slots, invalid = self.registry.fms_slots(channel_id)
            channel["runtime"]["fms_slots"] = len(slots)
            channel["runtime"]["invalid_slot_ids"] = invalid
        return channel

    def list_channels(self) -> List[Dict[str, Any]]:
        return [self.describe_channel(cid) for cid in sorted(
            self.channels, key=lambda c: (-int(self.channels[c].get("priority") or 0), c))]

    # ── rules / slot registry ────────────────────────────────────────────
    def sync_rules(self, cam_id: str, rules: List[Dict[str, Any]]) -> None:
        snapshot = [{k: r.get(k) for k in ("id", "type", "rule_type", "name", "fms_slot_id",
                                            "comm_channel_id", "enable_fms_dispatch")} for r in rules or []]
        self._call(self._sync_rules_now, cam_id, snapshot)

    def _sync_rules_now(self, cam_id: str, rules: List[Dict[str, Any]]) -> None:
        self.registry.sync_rules(cam_id, rules)
        if self.started:
            self._push_fms_snapshots()

    def remove_camera(self, cam_id: str) -> None:
        self._call(self._remove_camera_now, cam_id)

    def _remove_camera_now(self, cam_id: str) -> None:
        self.registry.remove_camera(cam_id)
        if self.started:
            self._push_fms_snapshots()

    # ── AI hooks (any thread, non-blocking) ──────────────────────────────
    def on_slot_transition(self, cam_id: str, rule: Dict[str, Any], status: str, info: Dict[str, Any]) -> None:
        labels = info.get("occupant_labels") or []
        ids = info.get("occupant_ids") or []
        context = slot_event_context(
            slot_id=str(rule.get("fms_slot_id") or rule.get("id")), status=status, camera_id=cam_id,
            camera_name="", rule_id=str(rule.get("id")), rule_name=str(rule.get("name") or ""),
            occupant_label=str(labels[0]) if labels else ("unknown" if status == "CARFULL" else ""),
            occupant_id=ids[0] if ids else None, overlap_ratio=info.get("overlap_ratio") or 0.0,
            confidence=info.get("confidence"))
        context["_detected_at"] = time.perf_counter()
        self._call(self._handle_slot_transition, cam_id, str(rule.get("id")), status, context)

    def on_roi_alert(self, event: Dict[str, Any]) -> None:
        context = {
            "event": EVENT_ROI_ALERT, "slot_id": str(event.get("rule_id") or ""), "rule_id": str(event.get("rule_id") or ""),
            "rule_name": str(event.get("rule_name") or ""), "camera_id": str(event.get("cam_id") or ""),
            "occupant_id": event.get("global_id"), "occupant_label": str(event.get("occupant_label") or ""),
            "description": str(event.get("description") or ""), "status": str(event.get("rule_type") or "").upper(),
            "state": str(event.get("rule_type") or ""), "is_occupied": True, "overlap_ratio": 0.0, "confidence": None,
        }
        context.update(now_fields())
        self.dispatch_event(EVENT_ROI_ALERT, context)

    def dispatch_event(self, event_type: str, context: Dict[str, Any], channel_id: Optional[str] = None) -> None:
        """Public, thread-safe and non-blocking (plan §3.3)."""
        self._call(self._dispatch, event_type, dict(context), channel_id)

    def _handle_slot_transition(self, cam_id: str, rule_id: str, status: str, context: Dict[str, Any]) -> None:
        self.stats["transitions"] += 1
        context["camera_name"] = self.camera_name(cam_id)
        entry = self.registry.update(cam_id, rule_id, status, context)
        if entry is not None:
            context["slot_id"] = entry.effective_slot_id
            if not entry.enabled:
                return
        event = EVENT_SLOT_CARFULL if status == "CARFULL" else EVENT_SLOT_EMPTY
        self._dispatch(event, context, entry.channel_id if entry else None, entry)

    # ── routing (loop thread) ────────────────────────────────────────────
    def _dispatch(self, event_type: str, context: Dict[str, Any], channel_id: Optional[str] = None,
                  entry=None) -> None:
        self.stats["events"] += 1
        for cid, channel in self.channels.items():
            driver = self.drivers.get(cid)
            if driver is None or (channel_id and cid != channel_id):
                continue
            config = channel["config"]
            meta = {"event": event_type, "slot_id": context.get("slot_id"),
                    "detected_at": context.get("_detected_at")}
            if event_type in channel["trigger_events"]:
                if config.get("payload_format") == FORMAT_FMS_SLOTS:
                    if event_type in (EVENT_SLOT_CARFULL, EVENT_SLOT_EMPTY) and entry is not None:
                        if self._fms_ready_entry(cid, entry):
                            driver.enqueue(self.fms_message(cid, entry.slot_id), meta)
                            self.stats["dispatched"] += 1
                elif channel["protocol"] != "MODBUS_TCP":
                    payload = self._render(channel, channel["payload_template"], context, config.get("payload_type", "json"))
                    if payload is not None:
                        driver.enqueue(payload, meta)
                        self.stats["dispatched"] += 1
            for command_id in config.get("event_commands", {}).get(event_type, []):
                command = next((c for c in config.get("commands", []) if c["id"] == command_id), None)
                if command is None:
                    continue
                if command["kind"] == "send":
                    payload = self._render(channel, command["payload"], context, command.get("payload_type", "json"))
                else:
                    payload = json.dumps(command)
                if payload is not None:
                    driver.enqueue(payload, {**meta, "command": command_id})
                    self.stats["dispatched"] += 1

    def _fms_ready_entry(self, channel_id: str, entry) -> bool:
        slot_id = (entry.slot_id or "").strip()
        if not slot_id:
            return False
        if not is_fms_slot_id(slot_id):
            key = (channel_id, slot_id)
            if key not in self._warned_invalid:
                self._warned_invalid.add(key)
                self._record({"channel_id": channel_id, "event": "CONFIG", "ok": False,
                              "detail": f"Slot ID '{slot_id}' không phải số nguyên - FMS WCS (stoi) sẽ không đọc được, đã bỏ qua"})
            return False
        return True

    def _render(self, channel: Dict[str, Any], template: str, context: Dict[str, Any], payload_type: str) -> Optional[str]:
        try:
            return render_template(template, {k: v for k, v in context.items() if not k.startswith("_")}, payload_type)
        except TemplateError as exc:
            self.stats["render_errors"] += 1
            self._record({"channel_id": channel["id"], "event": context.get("event"), "ok": False,
                          "detail": f"Lỗi template: {exc}"})
            return None

    def fms_message(self, channel_id: str, changed_slot: Optional[str] = None,
                    override: Optional[Dict[str, str]] = None) -> str:
        """FMS WCS wire message.

        ``WsCameraClient::run`` iterates ``json["slots"]`` and stores the whole
        document as the camera's latest state in Redis (``CameraStateWCS``),
        so every message carries the complete snapshot.  The changed slot is
        repeated at top level for operators and plan-style consumers.
        """
        slots, _ = self.registry.fms_slots(channel_id)
        if override:
            slots = [s for s in slots if s["slot_id"] != override["slot_id"]] + [override]
            slots.sort(key=lambda s: int(s["slot_id"]))
        message: Dict[str, Any] = {}
        if override:
            message.update(override)
        elif changed_slot is not None:
            state = self.registry.merged_state(channel_id, changed_slot)
            if state is not None:
                message.update({"slot_id": changed_slot, "state": state})
        message["slots"] = slots
        return json.dumps(message, ensure_ascii=False)

    def _push_fms_snapshots(self) -> None:
        for cid, channel in self.channels.items():
            driver = self.drivers.get(cid)
            if driver and channel["config"].get("payload_format") == FORMAT_FMS_SLOTS and driver.peers():
                driver.enqueue(self.fms_message(cid), {"event": "SNAPSHOT", "kind": "resync"})
                self._last_heartbeat[cid] = time.monotonic()

    def _resync_payloads(self, driver: BaseDriver) -> List[str]:
        channel = self.channels.get(driver.id)
        if channel is None or not channel["config"].get("resync_on_connect", True):
            return []
        if channel["config"].get("payload_format") == FORMAT_FMS_SLOTS:
            self._record({"channel_id": driver.id, "event": "RESYNC", "ok": True,
                          "detail": "Client mới kết nối - gửi snapshot toàn bộ ô hàng"})
            return [self.fms_message(driver.id)]
        if channel["protocol"] == "MODBUS_TCP" or EVENT_SLOT_CARFULL not in channel["trigger_events"]:
            return []
        payloads = []
        for context in self.registry.occupied_contexts(driver.id):
            payload = self._render(channel, channel["payload_template"], context, channel["config"].get("payload_type", "json"))
            if payload is not None:
                payloads.append(payload)
        if payloads:
            self._record({"channel_id": driver.id, "event": "RESYNC", "ok": True,
                          "detail": f"Client mới kết nối - gửi lại {len(payloads)} ô đang có hàng"})
        return payloads

    async def _heartbeat_loop(self) -> None:
        while True:
            await asyncio.sleep(HEARTBEAT_TICK_SEC)
            now = time.monotonic()
            for cid, channel in list(self.channels.items()):
                interval = float(channel["config"].get("heartbeat_sec") or 0)
                driver = self.drivers.get(cid)
                if (interval <= 0 or driver is None or channel["config"].get("payload_format") != FORMAT_FMS_SLOTS
                        or not driver.peers()):
                    continue
                if now - self._last_heartbeat.get(cid, 0.0) >= interval:
                    self._last_heartbeat[cid] = now
                    driver.enqueue(self.fms_message(cid), {"event": "HEARTBEAT", "kind": "heartbeat"})

    # ── observability ────────────────────────────────────────────────────
    def _record(self, record: Dict[str, Any]) -> None:
        self._log_seq += 1
        record = {"seq": self._log_seq, "at": time.time(), **record}
        if isinstance(record.get("payload"), str) and len(record["payload"]) > 600:
            record["payload"] = record["payload"][:600] + "…"
        self.log.append(record)
        if not record.get("ok"):
            key = f"{record.get('channel_id')}:{record.get('event')}"
            if time.time() - self._last_warning.get(key, 0) > 30:
                self._last_warning[key] = time.time()
                logger.warning("[%s] %s", record.get("channel_id"), record.get("detail"))

    def _on_driver_result(self, driver: BaseDriver, meta: Dict[str, Any], result: SendResult) -> None:
        if meta.get("kind") in ("heartbeat", "resync") and result.ok:
            return  # keep the log readable; heartbeats are visible in the counters
        detected_at = meta.get("detected_at")
        record = {"channel_id": driver.id, "event": meta.get("event"), "slot_id": meta.get("slot_id"),
                  "command": meta.get("command"), "ok": result.ok, "detail": result.detail,
                  "latency_ms": result.latency_ms, "delivered": result.delivered}
        if detected_at:
            record["e2e_latency_ms"] = round((time.perf_counter() - detected_at) * 1000.0, 3)
        self._record(record)

    def _on_inbound(self, driver: BaseDriver, peer: str, text: str) -> None:
        self._record({"channel_id": driver.id, "event": "INBOUND", "ok": True, "detail": f"Nhận từ {peer}", "payload": text})

    def logs(self, after: int = 0, channel_id: Optional[str] = None, limit: int = 200) -> List[Dict[str, Any]]:
        items = [r for r in self.log if r["seq"] > after and (not channel_id or r.get("channel_id") == channel_id)]
        return items[-limit:]

    def fms_status(self) -> Dict[str, Any]:
        channels = []
        for cid, channel in self.channels.items():
            if channel["config"].get("payload_format") != FORMAT_FMS_SLOTS and channel["device_type"] != "FMS_WCS":
                continue
            driver = self.drivers.get(cid)
            peers = driver.peers() if driver else []
            channels.append({"id": cid, "name": channel["name"], "enabled": channel["is_enabled"],
                             "clients": len(peers), "peers": peers, "endpoint": driver.target if driver else ""})
        clients = sum(c["clients"] for c in channels)
        return {"configured": bool(channels), "connected": clients > 0, "clients": clients, "channels": channels}

    def status(self) -> Dict[str, Any]:
        slots = self.registry.entries()
        return {
            "started": self.started,
            "store_error": self.store_error,
            "app_port": self.app_port,
            "channels": len(self.channels),
            "enabled": sum(1 for c in self.channels.values() if c["is_enabled"]),
            "online": sum(1 for d in self.drivers.values() if d.state in ("online", "listening")),
            "fms": self.fms_status(),
            "slots": {"total": len(slots), "carfull": sum(1 for s in slots if s["status"] == "CARFULL"),
                      "empty": sum(1 for s in slots if s["status"] == "EMPTY"),
                      "unknown": sum(1 for s in slots if s["status"] == "UNKNOWN")},
            "stats": dict(self.stats),
        }

    # ── interactive tools (Send Test / command buttons) ──────────────────
    async def preview(self, template: str, payload_type: str = "json", overrides: Optional[Dict[str, Any]] = None,
                      payload_format: str = "template", channel_id: str = "") -> Dict[str, Any]:
        context = sample_context(overrides)
        if payload_format == FORMAT_FMS_SLOTS:
            override = self._test_override(context)
            return {"ok": True, "rendered": self.fms_message(channel_id or "__preview__", override=override), "context": context}
        try:
            return {"ok": True, "rendered": render_template(template, context, payload_type), "context": context}
        except TemplateError as exc:
            return {"ok": False, "error": str(exc), "context": context}

    @staticmethod
    def _test_override(context: Dict[str, Any]) -> Optional[Dict[str, str]]:
        slot_id = str(context.get("slot_id") or "").strip()
        if not is_fms_slot_id(slot_id):
            return None
        return {"slot_id": slot_id, "state": STATE_CARFULL if context.get("status") == "CARFULL" else STATE_EMPTY}

    async def test_dispatch(self, channel_id: Optional[str], overrides: Optional[Dict[str, Any]] = None,
                            draft: Optional[Dict[str, Any]] = None, command_id: Optional[str] = None) -> Dict[str, Any]:
        """Send one sample packet now and report the precise outcome (Gate 6)."""
        existing = self.channels.get(channel_id) if channel_id else None
        try:
            channel = normalize_channel({**(draft or {}), **({"id": channel_id} if channel_id else {})}, existing) \
                if draft else existing
        except ChannelValidationError as exc:
            return {"ok": False, "detail": "Cấu hình chưa hợp lệ: " + "; ".join(exc.errors)}
        if channel is None:
            return {"ok": False, "detail": f"Không tìm thấy kênh '{channel_id}'"}
        context = sample_context({**(overrides or {}), "event": (overrides or {}).get("event") or None})
        context.setdefault("event", EVENT_TEST)

        driver, temporary = self._driver_for_test(channel)
        if driver is None:
            return {"ok": False, "detail": f"Kênh server '{channel['name']}' chưa chạy - hãy bật và Lưu trước khi gửi thử",
                    "target": ""}
        try:
            if temporary:
                await driver.start()
            await driver.wait_ready(float(channel["config"].get("connect_timeout_sec") or 3.0))
            config = channel["config"]
            if channel["protocol"] == "MODBUS_TCP" or command_id:
                commands = config.get("commands") or []
                command = next((c for c in commands if c["id"] == command_id), commands[0] if commands else None)
                if command is None:
                    return {"ok": False, "detail": "Chưa cấu hình lệnh điều khiển nào để gửi thử", "target": driver.target}
                payload = None
                if command["kind"] == "send":
                    payload = render_template(command["payload"], context, command.get("payload_type", "json"))
                result = await driver.execute(command, payload)
                payload = payload or json.dumps(command, ensure_ascii=False)
            else:
                if config.get("payload_format") == FORMAT_FMS_SLOTS:
                    override = self._test_override(context)
                    if override is None:
                        return {"ok": False, "detail": f"Slot ID '{context.get('slot_id')}' phải là số nguyên (FMS dùng stoi)"}
                    payload = self.fms_message(channel["id"], override=override)
                else:
                    payload = render_template(channel["payload_template"], context, config.get("payload_type", "json"))
                result = await driver.send_now(payload, timeout=float(config.get("timeout_sec") or 3.0) + 2.0)
        except TemplateError as exc:
            return {"ok": False, "detail": f"Lỗi template: {exc}", "target": driver.target}
        finally:
            if temporary:
                await driver.stop()
        response = {"ok": result.ok, "detail": result.detail, "target": result.target or driver.target,
                    "delivered": result.delivered, "latency_ms": result.latency_ms, "payload": payload,
                    "response": result.response}
        self._record({"channel_id": channel["id"], "event": EVENT_TEST, "ok": result.ok, "detail": result.detail,
                      "latency_ms": result.latency_ms, "payload": payload})
        return response

    def _driver_for_test(self, channel: Dict[str, Any]):
        running = self.drivers.get(channel["id"])
        current = self.channels.get(channel["id"])
        same_endpoint = current is not None and all(
            current.get(k) == channel.get(k) for k in ("protocol", "mode", "host", "port", "endpoint_path"))
        if running is not None and same_endpoint:
            return running, False
        if channel["mode"] == "SERVER":
            return None, False
        hooks = DriverHooks(app_port=self.app_port)  # no resync noise from a throw-away driver
        return create_driver(channel, hooks), True

    async def execute_command(self, channel_id: str, command_id: str,
                              overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        channel = self.channels.get(channel_id)
        if channel is None:
            raise KeyError(channel_id)
        command = next((c for c in channel["config"].get("commands", []) if c["id"] == command_id), None)
        if command is None:
            raise KeyError(command_id)
        return await self.test_dispatch(channel_id, overrides, command_id=command_id)

    async def modbus_operation(self, channel_id: str, operation: Dict[str, Any]) -> Dict[str, Any]:
        channel = self.channels.get(channel_id)
        if channel is None:
            raise KeyError(channel_id)
        if channel["protocol"] != "MODBUS_TCP":
            return {"ok": False, "detail": "Kênh không phải Modbus TCP"}
        driver, temporary = self._driver_for_test(channel)
        try:
            if temporary:
                await driver.start()
            result = await driver.execute(operation)
        finally:
            if temporary:
                await driver.stop()
        self._record({"channel_id": channel_id, "event": "MANUAL", "ok": result.ok, "detail": result.detail,
                      "latency_ms": result.latency_ms})
        return {"ok": result.ok, "detail": result.detail, "target": result.target, "latency_ms": result.latency_ms,
                "response": result.response}

    # ── FastAPI WebSocket attachment (embedded server mode) ──────────────
    def embedded_server_for(self, path: str = "", channel_id: str = "") -> Optional[BaseDriver]:
        for cid, driver in self.drivers.items():
            channel = self.channels[cid]
            if not getattr(driver, "embedded", False):
                continue
            if channel_id and cid == channel_id:
                return driver
            if not channel_id and (channel["endpoint_path"] or "/").rstrip("/") == (path or "/").rstrip("/"):
                return driver
        return None

    async def attach_fastapi_ws(self, websocket, path: str, channel_id: str = "") -> None:
        driver = self.embedded_server_for(path, channel_id)
        if driver is None:
            await websocket.close(code=1008, reason="no enabled WebSocket server channel for this path")
            return
        await driver.attach_starlette(websocket, path)

    # ── helpers used by tests / API ──────────────────────────────────────
    def manual_slot_state(self, cam_id: str, rule_id: str, state: str) -> None:
        """Force a slot state (diagnostics); behaves exactly like an AI transition."""
        entry = self.registry.get(cam_id, rule_id)
        rule = {"id": rule_id, "name": entry.rule_name if entry else rule_id,
                "fms_slot_id": entry.slot_id if entry else None}
        self.on_slot_transition(cam_id, rule, status_from_state(state), {"occupant_labels": ["manual"]})


gateway_manager = CommGatewayManager()
