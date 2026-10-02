"""MQTT publisher driver (Mosquitto / VDA 5050 style brokers).

paho-mqtt runs its own network thread (``loop_start``); ``publish`` only
enqueues inside paho, so the asyncio loop never blocks on the broker.
"""

import asyncio
import logging
import threading
import uuid
from typing import Any, Dict

from .base_driver import BaseDriver, SendResult

logger = logging.getLogger("CommGateway")

try:
    import paho.mqtt.client as mqtt
except ImportError:  # pragma: no cover
    mqtt = None


class MqttDriver(BaseDriver):
    protocol = "MQTT"

    def __init__(self, channel: Dict[str, Any], hooks=None):
        super().__init__(channel, hooks)
        self._client = None
        self._connected = threading.Event()

    @property
    def topic(self) -> str:
        return (self.channel.get("endpoint_path") or "vision/slots").lstrip("/") or "vision/slots"

    @property
    def target(self) -> str:
        return f"mqtt://{self.channel.get('host')}:{self.channel.get('port')}/{self.topic}"

    async def _open(self) -> None:
        if mqtt is None:
            raise RuntimeError("Thư viện paho-mqtt chưa được cài đặt")
        client_id = f"vision-{self.id}-{uuid.uuid4().hex[:6]}"
        api_version = getattr(mqtt, "CallbackAPIVersion", None)
        client = (mqtt.Client(callback_api_version=api_version.VERSION2, client_id=client_id)
                  if api_version else mqtt.Client(client_id=client_id))
        auth = self.channel.get("auth_config") or {}
        if auth.get("username"):
            client.username_pw_set(auth.get("username"), auth.get("password") or None)
        if self.config.get("use_tls"):
            client.tls_set()
        client.on_connect = self._on_connect
        client.on_disconnect = self._on_disconnect
        client.reconnect_delay_set(min_delay=1, max_delay=30)
        self._client = client
        self.state = "connecting"
        client.connect_async(self.channel.get("host"), int(self.channel["port"]), keepalive=30)
        client.loop_start()

    def _on_connect(self, client, userdata, flags, reason_code, *args) -> None:
        if reason_code == 0:
            self._connected.set()
            self.state = "online"
            logger.info("[%s] MQTT connected to %s", self.id, self.target)
        else:
            self._record_error(f"Broker từ chối kết nối: {reason_code}")
            self.state = "error"

    def _on_disconnect(self, client, userdata, *args) -> None:
        self._connected.clear()
        if self._running:
            self.state = "connecting"

    async def _close(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            await asyncio.to_thread(self._shutdown, client)
        self._connected.clear()

    @staticmethod
    def _shutdown(client) -> None:
        try:
            client.disconnect()
        finally:
            client.loop_stop()

    async def wait_ready(self, timeout: float) -> bool:
        return await asyncio.to_thread(self._connected.wait, timeout)

    async def send(self, payload: str) -> SendResult:
        if self._client is None or not self._connected.is_set():
            detail = self.stats.get("last_error") or "chưa kết nối tới broker"
            return SendResult(False, f"Chưa kết nối {self.target} ({detail})", self.target)
        qos = int(self.config.get("mqtt_qos") or 0)
        info = self._client.publish(self.topic, payload, qos=qos, retain=bool(self.config.get("mqtt_retain")))
        if info.rc != 0:
            return SendResult(False, f"Publish thất bại (rc={info.rc}) tới {self.target}", self.target)
        if qos > 0:
            await asyncio.to_thread(info.wait_for_publish, 3.0)
            if not info.is_published():
                return SendResult(False, f"Broker chưa xác nhận (QoS {qos}) sau 3s: {self.target}", self.target)
        return SendResult(True, f"Đã publish lên {self.target} (QoS {qos})", self.target, 1)

    def peers(self):
        return [{"peer": self.target}] if self._connected.is_set() else []
