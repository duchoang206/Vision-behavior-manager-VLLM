import asyncio
import sys
import json
import time
import unittest
from pathlib import Path

BACKEND_DIR = str(Path(__file__).resolve().parents[1])
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from core.metadata_broadcaster import LatestMetadataBroadcaster


class FakeSocket:
    def __init__(self, blocked=False, fail=False):
        self.messages = []
        self.closes = []
        self.gate = asyncio.Event()
        self.sending = asyncio.Event()
        self.fail = fail
        if not blocked:
            self.gate.set()

    async def send_text(self, message):
        payload = json.loads(message)
        if "streams" in payload:
            self.sending.set()
            await self.gate.wait()
            if self.fail:
                raise RuntimeError("connection failed")
        self.messages.append(payload)

    async def close(self, code, reason):
        self.closes.append((code, reason))

    def frames(self):
        return [message for message in self.messages if "streams" in message]


class MetadataBroadcasterTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.broadcaster = LatestMetadataBroadcaster(send_timeout=0.1, heartbeat_interval=0.02)
        self.broadcaster.start(asyncio.get_running_loop())

    async def asyncTearDown(self):
        await self.broadcaster.stop()

    def publish(self, camera, sequence):
        self.broadcaster.publish({"source": "identity_template", "timestamp": sequence,
                                  "streams": [{"cam_id": camera, "frame_id": sequence, "objects": []}]})

    async def wait_for(self, predicate):
        async def wait():
            while not predicate():
                await asyncio.sleep(0.001)
        await asyncio.wait_for(wait(), timeout=0.5)

    async def test_timeout_closes_client_without_blocking_healthy_client(self):
        slow, healthy = FakeSocket(blocked=True), FakeSocket()
        self.broadcaster.register(slow)
        self.broadcaster.register(healthy)
        self.publish("cam1", 1)
        await self.wait_for(lambda: len(healthy.frames()) == 1)
        with self.assertLogs(level="WARNING"):
            await self.wait_for(lambda: slow.closes)
        self.assertEqual(1013, slow.closes[0][0])
        self.assertNotIn(slow, self.broadcaster._clients)
        self.publish("cam1", 2)
        await self.wait_for(lambda: len(healthy.frames()) == 2)
        self.assertEqual(1, self.broadcaster.status()["send_timeouts"])

    async def test_pending_frames_keep_latest_per_camera(self):
        socket = FakeSocket(blocked=True)
        self.broadcaster.register(socket)
        self.publish("cam1", 1)
        await asyncio.wait_for(socket.sending.wait(), timeout=0.5)
        self.publish("cam1", 2)
        self.publish("cam2", 1)
        await asyncio.sleep(0)
        self.publish("cam1", 3)
        await asyncio.sleep(0)
        self.assertEqual(2, self.broadcaster.status()["queued_frames"])
        socket.gate.set()
        await self.wait_for(lambda: len(socket.frames()) == 3)
        frames = [(message["streams"][0]["cam_id"], message["timestamp"]) for message in socket.frames()]
        self.assertEqual([("cam1", 1), ("cam1", 3), ("cam2", 1)], frames)

    async def test_ingress_coalesces_burst_and_rejects_older_frame(self):
        socket = FakeSocket()
        self.broadcaster.register(socket)
        for sequence in range(100):
            self.publish("cam1", sequence)
        self.publish("cam1", 3)
        self.assertEqual(1, self.broadcaster.status()["pending_cameras"])
        await self.wait_for(lambda: socket.frames())
        self.assertEqual([99], [message["timestamp"] for message in socket.frames()])
        self.assertEqual(99, self.broadcaster.status()["replaced_frames"])
        self.assertEqual(1, self.broadcaster.status()["out_of_order_frames"])

    async def test_expired_frames_are_not_replayed(self):
        socket = FakeSocket()
        self.broadcaster.register(socket)
        self.broadcaster.max_age = 0.005
        self.publish("cam1", 1)
        time.sleep(0.01)
        await self.wait_for(lambda: self.broadcaster.status()["expired_frames"] == 1)
        self.assertEqual([], socket.frames())
        self.publish("cam1", 2)
        await self.wait_for(lambda: socket.frames())
        self.assertEqual(2, socket.frames()[0]["timestamp"])

    async def test_idle_connections_have_heartbeats_without_fake_frames(self):
        socket = FakeSocket()
        self.broadcaster.register(socket)
        await self.wait_for(lambda: len(socket.messages) >= 2)
        self.assertTrue(all(message["type"] == "metadata_heartbeat" for message in socket.messages))
        self.assertEqual([], socket.frames())

    async def test_unregister_stops_sender_and_reconnect_receives_fresh_frame(self):
        old = FakeSocket()
        self.broadcaster.register(old)
        self.publish("cam1", 1)
        await self.wait_for(lambda: old.frames())
        await self.broadcaster.unregister(old)
        fresh = FakeSocket()
        self.broadcaster.register(fresh)
        self.publish("cam1", 2)
        await self.wait_for(lambda: fresh.frames())
        self.assertEqual(1, len(old.frames()))
        self.assertEqual([2], [message["timestamp"] for message in fresh.frames()])
        self.assertEqual(1, self.broadcaster.status()["clients"])

    async def test_send_failure_closes_socket(self):
        socket = FakeSocket(fail=True)
        self.broadcaster.register(socket)
        with self.assertLogs(level="WARNING"):
            self.publish("cam1", 1)
            await self.wait_for(lambda: socket.closes)
        self.assertEqual(1013, socket.closes[0][0])
        self.assertEqual(1, self.broadcaster.status()["send_errors"])

    async def test_threaded_multi_camera_publish_preserves_stream_data(self):
        socket = FakeSocket()
        self.broadcaster.register(socket)
        payload = {"source": "deepstream", "timestamp": 1, "streams": [
            {"cam_id": "cam1", "objects": [], "rois": [{"id": "zone1"}]},
            {"cam_id": "cam2", "objects": [{"label": "Robot_2001", "mask": {"polygons": []}}]},
        ]}
        await asyncio.to_thread(self.broadcaster.publish, payload)
        await self.wait_for(lambda: len(socket.frames()) == 2)
        self.assertEqual(payload["streams"], [message["streams"][0] for message in socket.frames()])
        self.assertTrue(all(message["source"] == "deepstream" for message in socket.frames()))


if __name__ == "__main__":
    unittest.main()
