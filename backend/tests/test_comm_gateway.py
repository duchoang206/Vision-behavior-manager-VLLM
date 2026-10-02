"""Acceptance tests for the communication gateway (plan Gates 1-7).

The FMS side is emulated exactly like ``rtcserver_wcs`` ``WsCameraClient``:
iterate ``json["slots"]``, ``stoi(slot_id)``, ``state == "Car Full"``.
"""

import asyncio
import json
import os
import socket
import struct
import threading
import time
import unittest

import websockets
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient

from core.behavior_analytics import BehaviorAnalyticsEngine
from core.comm_gateway import CommGatewayManager, normalize_channel, render_template
from core.comm_gateway.channel_store import ChannelValidationError
from core.comm_gateway.slot_registry import SlotRegistry
from core.comm_gateway.template_renderer import TemplateError
from core.dashboard_auth import require_dashboard_user
from routers.comm_gateway import create_comm_gateway_router

GATE5_TEMPLATE = '{\n  "station": "ST_{slot_id}",\n  "occupied": {is_occupied},\n  "carrier": "{occupant_label}",\n  "cam": "{camera_name}"\n}'


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def fms_wcs_parse(message: str) -> dict:
    """Mirror of WsCameraClient::run/onMessage/hasGoods."""
    document = json.loads(message)
    return {int(slot["slot_id"]): slot.get("state") == "Car Full" for slot in document.get("slots", [])}


def slot_rule(rule_id: str, slot_id, name: str = "", **extra) -> dict:
    return {"id": rule_id, "type": "occupancy", "name": name or rule_id, "fms_slot_id": slot_id,
            "camera_points": [[0.1, 0.1], [0.5, 0.1], [0.5, 0.5], [0.1, 0.5]],
            "fms_points": [[100, 100], [101, 100], [101, 101], [100, 101]],
            "points": [[100, 100], [101, 100], [101, 101], [100, 101]],
            "coordinate_space": "hybrid", "target_objects": ["rack"], "threshold": 15, **extra}


async def wait_for(predicate, timeout=3.0, interval=0.01):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval)
    return False


class TemplateRendererTests(unittest.TestCase):
    def test_gate5_dynamic_template_native_types(self):
        context = {"slot_id": "5", "is_occupied": True, "occupant_label": "rack #400", "camera_name": "Cam 4"}
        rendered = json.loads(render_template(GATE5_TEMPLATE, context))
        self.assertEqual(rendered, {"station": "ST_5", "occupied": True, "carrier": "rack #400", "cam": "Cam 4"})
        self.assertIs(rendered["occupied"], True)

    def test_string_values_are_escaped(self):
        rendered = json.loads(render_template('{"cam": "{camera_name}", "n": {occupant_id}}',
                                              {"camera_name": 'Kho "A"\n01', "occupant_id": 400}))
        self.assertEqual(rendered, {"cam": 'Kho "A"\n01', "n": 400})

    def test_object_braces_are_not_placeholders(self):
        self.assertEqual(json.loads(render_template('{"a": {"b": "{slot_id}"}}', {"slot_id": "7"})), {"a": {"b": "7"}})

    def test_invalid_template_reports_error_without_crashing(self):
        with self.assertRaises(TemplateError) as ctx:
            render_template('{"slot_id": "{slot_id}", "state": "{state}"', {"slot_id": "5", "state": "Empty"})
        self.assertIn("JSON không hợp lệ", str(ctx.exception))

    def test_text_payload(self):
        self.assertEqual(render_template("SLOT={slot_id};OCC={is_occupied}", {"slot_id": "5", "is_occupied": False}, "text"),
                         "SLOT=5;OCC=false")

    def test_channel_validation_rejects_bad_template(self):
        with self.assertRaises(ChannelValidationError) as ctx:
            normalize_channel({"name": "WMS", "protocol": "HTTP_WEBHOOK", "host": "10.0.0.2", "port": 80,
                               "payload_template": '{"a": {slot_id}'})
        self.assertTrue(any("Payload template" in e for e in ctx.exception.errors))

    def test_client_mode_requires_host(self):
        with self.assertRaises(ChannelValidationError):
            normalize_channel({"name": "PLC", "protocol": "TCPIP", "mode": "CLIENT", "port": 502})


class SlotRegistryTests(unittest.TestCase):
    def test_fms_snapshot_merges_cameras_and_skips_non_numeric(self):
        registry = SlotRegistry()
        registry.sync_rules("cam_a", [slot_rule("r1", "5"), slot_rule("r2", "A1"), slot_rule("r3", "")])
        registry.sync_rules("cam_b", [slot_rule("r9", "5")])
        registry.update("cam_a", "r1", "EMPTY", {})
        registry.update("cam_b", "r9", "CARFULL", {})
        registry.update("cam_a", "r2", "CARFULL", {})
        registry.update("cam_a", "r3", "CARFULL", {})
        slots, invalid = registry.fms_slots("fms")
        self.assertEqual(slots, [{"slot_id": "5", "state": "Car Full"}])
        self.assertEqual(invalid, ["A1"])
        self.assertEqual(registry.get("cam_a", "r3").effective_slot_id, "r3")

    def test_unknown_state_not_reported(self):
        registry = SlotRegistry()
        registry.sync_rules("cam", [slot_rule("r1", "1")])
        self.assertEqual(registry.fms_slots("fms")[0], [])


class BehaviorHookTests(unittest.TestCase):
    """Gate 3 / Gate 4: 2 frames → CARFULL once, 5 empty frames → EMPTY, no repeats."""

    def setUp(self):
        self.engine = BehaviorAnalyticsEngine()
        self.calls = []
        self.engine.transition_listener = lambda cam, rule, status, info: self.calls.append((rule["fms_slot_id"], status, info))
        self.synced = []
        self.engine.rules_listener = lambda cam, rules: self.synced.append((cam, [r.get("fms_slot_id") for r in rules]))
        self.engine.set_rules("cam_4", [slot_rule("rule_5", "5", "Ô 5")])
        self.rack = {"id": 400, "class": "rack", "label": "rack #400", "x": 0.2, "y": 0.2, "w": 0.2, "h": 0.2,
                     "confidence": 0.94}

    def test_transitions_fire_once(self):
        self.assertEqual(self.synced, [("cam_4", ["5"])])
        self.engine.process_frame("cam_4", [self.rack])
        self.assertEqual(self.calls, [])
        self.engine.process_frame("cam_4", [self.rack])
        self.assertEqual([(c[0], c[1]) for c in self.calls], [("5", "CARFULL")])
        self.assertEqual(self.calls[0][2]["occupant_labels"], ["rack #400"])
        self.assertEqual(self.calls[0][2]["confidence"], 0.94)
        for _ in range(30):  # goods stay in the slot: no flooding
            self.engine.process_frame("cam_4", [self.rack])
        self.assertEqual(len(self.calls), 1)
        for _ in range(4):
            self.engine.process_frame("cam_4", [])
        self.assertEqual(len(self.calls), 1, "a person/robot passing for <5 frames must not release the slot")
        self.engine.process_frame("cam_4", [])
        self.assertEqual([(c[0], c[1]) for c in self.calls], [("5", "CARFULL"), ("5", "EMPTY")])
        for _ in range(20):
            self.engine.process_frame("cam_4", [])
        self.assertEqual(len(self.calls), 2)

    def test_initial_empty_is_confirmed_once(self):
        for _ in range(12):
            self.engine.process_frame("cam_4", [])
        self.assertEqual([(c[0], c[1]) for c in self.calls], [("5", "EMPTY")])

    def test_roi_state_carries_slot_id(self):
        _, _, rois = self.engine.process_frame("cam_4", [self.rack])
        self.assertEqual(rois[0]["fms_slot_id"], "5")


def fms_channel(port: int, **extra) -> dict:
    return {"id": "fms_wcs", "name": "FMS WCS", "device_type": "FMS_WCS", "protocol": "WEBSOCKET", "mode": "SERVER",
            "host": "127.0.0.1", "port": port, "endpoint_path": "/ws/wcs_camera",
            "config": {"payload_format": "fms_wcs_slots", "heartbeat_sec": 0}, **extra}


class GatewayIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.gw = CommGatewayManager(app_port=1)
        await self.gw.start()
        self.gw.sync_rules("cam_4", [slot_rule("r2", "2", "Ô 2"), slot_rule("r5", "5", "Ô 5"), slot_rule("r7", "7", "Ô 7")])

    async def asyncTearDown(self):
        await self.gw.stop()

    def transition(self, rule_id, slot_id, status, label="rack #400"):
        self.gw.on_slot_transition("cam_4", {"id": rule_id, "name": f"Ô {slot_id}", "fms_slot_id": slot_id}, status,
                                   {"occupant_labels": [label] if status == "CARFULL" else [],
                                    "occupant_ids": [400] if status == "CARFULL" else [], "overlap_ratio": 80.0})

    async def test_gate2_resync_and_gate3_gate4_transitions(self):
        port = free_port()
        await self.gw.save_channel(fms_channel(port))
        self.transition("r2", "2", "CARFULL")
        self.transition("r5", "5", "CARFULL")
        self.transition("r7", "7", "EMPTY")
        await asyncio.sleep(0.05)
        async with websockets.connect(f"ws://127.0.0.1:{port}/") as fms:  # FMS default config path "/"
            first = await asyncio.wait_for(fms.recv(), 2)
            self.assertEqual(fms_wcs_parse(first), {2: True, 5: True, 7: False})
            self.assertTrue(await wait_for(lambda: self.gw.fms_status()["clients"] == 1))
            self.assertEqual(self.gw.fms_status()["connected"], True)

            started = time.perf_counter()
            threading.Thread(target=self.transition, args=("r5", "5", "EMPTY")).start()  # from a "frame thread"
            message = await asyncio.wait_for(fms.recv(), 2)
            latency_ms = (time.perf_counter() - started) * 1000
            document = json.loads(message)
            self.assertEqual((document["slot_id"], document["state"]), ("5", "Empty"))
            self.assertEqual(fms_wcs_parse(message), {2: True, 5: False, 7: False})
            self.assertLess(latency_ms, 50.0)

            started = time.perf_counter()
            threading.Thread(target=self.transition, args=("r5", "5", "CARFULL")).start()
            message = await asyncio.wait_for(fms.recv(), 2)
            self.assertLess((time.perf_counter() - started) * 1000, 50.0)
            self.assertEqual({k: json.loads(message)[k] for k in ("slot_id", "state")}, {"slot_id": "5", "state": "Car Full"})
            with self.assertRaises(asyncio.TimeoutError):  # no per-frame flooding
                await asyncio.wait_for(fms.recv(), 0.3)
        self.assertTrue(await wait_for(lambda: self.gw.fms_status()["clients"] == 0))

    async def test_heartbeat_snapshot(self):
        port = free_port()
        await self.gw.save_channel(fms_channel(port, config={"payload_format": "fms_wcs_slots", "heartbeat_sec": 0.5}))
        self.transition("r2", "2", "CARFULL")
        async with websockets.connect(f"ws://127.0.0.1:{port}/ws/wcs_camera") as fms:
            await asyncio.wait_for(fms.recv(), 2)  # resync
            heartbeat = await asyncio.wait_for(fms.recv(), 2)
            self.assertEqual(json.loads(heartbeat), {"slots": [{"slot_id": "2", "state": "Car Full"}]})

    async def test_embedded_fastapi_route(self):
        import uvicorn

        port = free_port()
        self.gw.app_port = port
        app = FastAPI()

        @app.websocket("/ws/wcs_camera")
        async def endpoint(websocket: WebSocket):
            await self.gw.attach_fastapi_ws(websocket, "/ws/wcs_camera")

        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
        task = asyncio.create_task(server.serve())
        try:
            self.assertTrue(await wait_for(lambda: server.started, 5))
            await self.gw.save_channel(fms_channel(port))
            self.transition("r5", "5", "CARFULL")
            async with websockets.connect(f"ws://127.0.0.1:{port}/ws/wcs_camera") as fms:
                self.assertEqual(fms_wcs_parse(await asyncio.wait_for(fms.recv(), 2)), {5: True})
                result = await self.gw.test_dispatch("fms_wcs", {"slot_id": "2", "state": "Car Full"})
                self.assertTrue(result["ok"], result)
                self.assertIn("1/1", result["detail"])
                self.assertEqual(fms_wcs_parse(await asyncio.wait_for(fms.recv(), 2)), {2: True, 5: True})
        finally:
            server.should_exit = True
            await task

    async def test_gate5_template_channel_over_tcp_server_with_resync(self):
        port = free_port()
        self.gw.set_camera_name_resolver(lambda cam_id: "Cam 4")
        await self.gw.save_channel({"id": "wms", "name": "WMS", "protocol": "TCPIP", "mode": "SERVER", "host": "127.0.0.1",
                                    "port": port, "payload_template": GATE5_TEMPLATE})
        self.transition("r2", "2", "CARFULL", label="rack #12")
        await asyncio.sleep(0.05)
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        try:
            resync = json.loads(await asyncio.wait_for(reader.readline(), 2))
            self.assertEqual(resync["station"], "ST_2")
            self.transition("r5", "5", "CARFULL")
            line = json.loads(await asyncio.wait_for(reader.readline(), 2))
            self.assertEqual(line, {"station": "ST_5", "occupied": True, "carrier": "rack #400", "cam": "Cam 4"})
        finally:
            writer.close()

    async def test_gate6_send_test_reports_precise_errors(self):
        closed_port = free_port()
        refused = await self.gw.test_dispatch(None, {"slot_id": "5"}, draft={
            "id": "tmp_ws", "name": "FMS client", "protocol": "WEBSOCKET", "mode": "CLIENT", "host": "127.0.0.1",
            "port": closed_port, "config": {"connect_timeout_sec": 0.5}})
        self.assertFalse(refused["ok"])
        self.assertIn("Connection Refused", refused["detail"])

        # A server that accepts TCP but never answers HTTP → read timeout.
        silent = await asyncio.start_server(lambda r, w: asyncio.sleep(10), "127.0.0.1", 0)
        silent_port = silent.sockets[0].getsockname()[1]
        try:
            started = time.perf_counter()
            timeout = await self.gw.test_dispatch(None, {"slot_id": "5"}, draft={
                "id": "tmp_http", "name": "WMS", "protocol": "HTTP_WEBHOOK", "host": "127.0.0.1",
                "port": silent_port, "endpoint_path": "/slots", "config": {"timeout_sec": 1}})
            self.assertFalse(timeout["ok"])
            self.assertIn("Connection Timeout", timeout["detail"])
            self.assertLess(time.perf_counter() - started, 3.0)
        finally:
            silent.close()

        no_client = await self.gw.save_channel(fms_channel(free_port()))
        result = await self.gw.test_dispatch(no_client["id"], {"slot_id": "5"})
        self.assertFalse(result["ok"])
        self.assertIn("Không có client", result["detail"])

        bad = await self.gw.test_dispatch(None, {}, draft={"id": "x", "name": "x", "protocol": "TCPIP", "mode": "CLIENT",
                                                           "host": "127.0.0.1", "port": 9, "payload_template": "{bad"})
        self.assertFalse(bad["ok"])
        self.assertIn("template", bad["detail"].lower())

    async def test_gate6_webhook_success(self):
        received = []

        async def handle(reader, writer):
            data = await reader.read(65536)
            received.append(data)
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok")
            await writer.drain()
            writer.close()

        server = await asyncio.start_server(handle, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        try:
            result = await self.gw.test_dispatch(None, {"slot_id": "5", "state": "Car Full"}, draft={
                "id": "wms", "name": "WMS", "protocol": "HTTP_WEBHOOK", "host": "127.0.0.1", "port": port,
                "endpoint_path": "/api/slots"})
            self.assertTrue(result["ok"], result)
            self.assertIn("HTTP 200 OK", result["detail"])
            body = received[0].split(b"\r\n\r\n", 1)[1]
            self.assertEqual(json.loads(body), {"slot_id": "5", "state": "Car Full"})
        finally:
            server.close()

    async def test_gate7_frame_threads_never_block_on_network(self):
        silent = await asyncio.start_server(lambda r, w: asyncio.sleep(30), "127.0.0.1", 0)
        port = silent.sockets[0].getsockname()[1]
        try:
            await self.gw.save_channel({"id": "slow_wms", "name": "Slow WMS", "protocol": "HTTP_WEBHOOK",
                                        "host": "127.0.0.1", "port": port, "config": {"timeout_sec": 3}})
            durations = []
            frame_times = []

            def frame_loop():
                period = 1 / 25.0
                start = time.perf_counter()
                for frame in range(75):  # 3 s of video
                    tick = time.perf_counter()
                    status = "CARFULL" if frame % 2 else "EMPTY"
                    self.transition("r5", "5", status)
                    durations.append(time.perf_counter() - tick)
                    frame_times.append(time.perf_counter())
                    time.sleep(max(0.0, start + (frame + 1) * period - time.perf_counter()))

            thread = threading.Thread(target=frame_loop)
            thread.start()
            while thread.is_alive():
                await asyncio.sleep(0.01)
            fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
            self.assertLess(max(durations) * 1000, 5.0, "dispatch must be a non-blocking hand-off")
            self.assertAlmostEqual(fps, 25.0, delta=0.5)
            self.assertGreater(self.gw.drivers["slow_wms"].get_status()["queue"], 0)
        finally:
            silent.close()


class FakeModbusServer:
    def __init__(self):
        self.coils = [False] * 16
        self.registers = [0] * 16

    async def handle(self, reader, writer):
        try:
            while True:
                header = await reader.readexactly(7)
                tid, pid, length, unit = struct.unpack(">HHHB", header)
                pdu = await reader.readexactly(length - 1)
                fc = pdu[0]
                if fc == 5:
                    address, value = struct.unpack(">HH", pdu[1:5])
                    self.coils[address] = value == 0xFF00
                    response = pdu
                elif fc == 6:
                    address, value = struct.unpack(">HH", pdu[1:5])
                    self.registers[address] = value
                    response = pdu
                elif fc == 1:
                    address, count = struct.unpack(">HH", pdu[1:5])
                    bits = 0
                    for i in range(count):
                        bits |= int(self.coils[address + i]) << i
                    nbytes = (count + 7) // 8
                    response = bytes([1, nbytes]) + bits.to_bytes(nbytes, "little")
                elif fc == 3:
                    address, count = struct.unpack(">HH", pdu[1:5])
                    response = bytes([3, count * 2]) + struct.pack(f">{count}H", *self.registers[address:address + count])
                else:
                    response = bytes([fc | 0x80, 1])
                writer.write(struct.pack(">HHHB", tid, 0, len(response) + 1, unit) + response)
                await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionError):
            writer.close()


class PeripheralControlTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.gw = CommGatewayManager(app_port=1)
        await self.gw.start()
        self.gw.sync_rules("cam_4", [slot_rule("r5", "5")])
        self.plc = FakeModbusServer()
        self.server = await asyncio.start_server(self.plc.handle, "127.0.0.1", 0)
        self.port = self.server.sockets[0].getsockname()[1]

    async def asyncTearDown(self):
        await self.gw.stop()
        self.server.close()

    async def test_modbus_light_tower_follows_slot_events(self):
        await self.gw.save_channel({
            "id": "light_tower", "name": "Đèn tháp ô 5", "device_type": "LIGHT_TOWER", "protocol": "MODBUS_TCP",
            "host": "127.0.0.1", "port": self.port,
            "config": {"unit_id": 1, "commands": [
                {"id": "red_on", "name": "Đèn đỏ ON", "kind": "modbus_coil", "address": 3, "value": True},
                {"id": "red_off", "name": "Đèn đỏ OFF", "kind": "modbus_coil", "address": 3, "value": False},
                {"id": "set_count", "name": "Ghi thanh ghi", "kind": "modbus_register", "address": 2, "value": 1234}],
                "event_commands": {"SLOT_CARFULL": ["red_on"], "SLOT_EMPTY": ["red_off"]}}})
        self.gw.on_slot_transition("cam_4", {"id": "r5", "fms_slot_id": "5"}, "CARFULL", {})
        self.assertTrue(await wait_for(lambda: self.plc.coils[3] is True))
        self.gw.on_slot_transition("cam_4", {"id": "r5", "fms_slot_id": "5"}, "EMPTY", {})
        self.assertTrue(await wait_for(lambda: self.plc.coils[3] is False))

        manual = await self.gw.execute_command("light_tower", "set_count")
        self.assertTrue(manual["ok"], manual)
        self.assertEqual(self.plc.registers[2], 1234)
        read = await self.gw.modbus_operation("light_tower", {"kind": "modbus_read_registers", "address": 2, "count": 1})
        self.assertEqual(read["response"], [1234])

    async def test_modbus_exception_is_reported(self):
        await self.gw.save_channel({"id": "plc", "name": "PLC", "protocol": "MODBUS_TCP", "host": "127.0.0.1",
                                    "port": self.port})
        result = await self.gw.modbus_operation("plc", {"kind": "modbus_read_coils", "address": 40, "count": 1})
        self.assertFalse(result["ok"])

    async def test_tcp_client_command_to_plc(self):
        lines = []

        async def handle(reader, writer):
            lines.append((await reader.readline()).decode())

        server = await asyncio.start_server(handle, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        try:
            await self.gw.save_channel({"id": "plc_tcp", "name": "PLC TCP", "device_type": "PLC", "protocol": "TCPIP",
                                        "mode": "CLIENT", "host": "127.0.0.1", "port": port, "trigger_events": [],
                                        "config": {"payload_type": "text", "commands": [
                                            {"id": "open", "name": "Mở cửa", "kind": "send", "payload": "DOOR={slot_id};OPEN",
                                             "payload_type": "text"}]}})
            result = await self.gw.execute_command("plc_tcp", "open", {"slot_id": "5"})
            self.assertTrue(result["ok"], result)
            self.assertTrue(await wait_for(lambda: lines == ["DOOR=5;OPEN\n"]))
        finally:
            server.close()


class RouterTests(unittest.TestCase):
    def test_crud_and_validation(self):
        gateway = CommGatewayManager(app_port=1)
        app = FastAPI()
        app.include_router(create_comm_gateway_router(gateway))
        app.dependency_overrides[require_dashboard_user] = lambda: "tester"
        with TestClient(app) as client:
            meta = client.get("/api/comm/meta").json()
            self.assertIn("MODBUS_TCP", meta["protocols"])
            bad = client.post("/api/comm/channels", json={"name": "x", "protocol": "HTTP_WEBHOOK", "host": "1.2.3.4",
                                                          "port": 80, "payload_template": "{oops"})
            self.assertEqual(bad.status_code, 422)
            preview = client.post("/api/comm/preview", json={"payload_template": GATE5_TEMPLATE}).json()
            self.assertTrue(preview["ok"])
            self.assertIs(json.loads(preview["rendered"])["occupied"], True)
            broken = client.post("/api/comm/preview", json={"payload_template": '{"a": "{slot_id}"'}).json()
            self.assertFalse(broken["ok"])


@unittest.skipUnless(os.getenv("COMM_TEST_DB_URL"), "set COMM_TEST_DB_URL to a disposable PostgreSQL")
class PersistenceTests(unittest.IsolatedAsyncioTestCase):
    """Gate 1: configuration survives a backend restart."""

    async def test_channels_and_slot_ids_survive_restart(self):
        from core.comm_gateway import ChannelStore
        from core.database import DatabaseManager

        db = DatabaseManager(os.environ["COMM_TEST_DB_URL"])
        ChannelStore(db).delete("wms_cloud")  # idempotent re-runs
        first = CommGatewayManager(app_port=1)
        first.bind_store(ChannelStore(db))
        await first.start()
        self.assertIn("fms_wcs", first.channels)  # default FMS channel seeded on first boot
        await first.save_channel({"id": "wms_cloud", "name": "WMS Cloud", "protocol": "HTTP_WEBHOOK",
                                  "host": "10.1.2.3", "port": 8443, "endpoint_path": "/api/v1/slots",
                                  "payload_template": GATE5_TEMPLATE, "auth_config": {"bearer_token": "t0k"},
                                  "config": {"use_tls": True}})
        db.save_rule({"id": "persist_rule", "cam_id": "cam_persist", "type": "occupancy", "name": "Ô 5",
                      "points": [[1, 1], [2, 1], [2, 2]], "fms_slot_id": "5", "comm_channel_id": "fms_wcs",
                      "enable_fms_dispatch": True})
        await first.stop()

        second = CommGatewayManager(app_port=1)  # "restart"
        second.bind_store(ChannelStore(DatabaseManager(os.environ["COMM_TEST_DB_URL"])))
        await second.start()
        try:
            channel = second.channels["wms_cloud"]
            self.assertEqual(channel["payload_template"], GATE5_TEMPLATE.strip("\n"))
            self.assertEqual(channel["endpoint_path"], "/api/v1/slots")
            self.assertEqual(channel["auth_config"], {"bearer_token": "t0k"})
            self.assertTrue(channel["config"]["use_tls"])
            rules = db.get_rules_by_camera("cam_persist")
            self.assertEqual((rules[0]["fms_slot_id"], rules[0]["comm_channel_id"], rules[0]["enable_fms_dispatch"]),
                             ("5", "fms_wcs", True))
        finally:
            await second.stop()
            db.delete_rules_by_camera("cam_persist")


if __name__ == "__main__":
    unittest.main()
