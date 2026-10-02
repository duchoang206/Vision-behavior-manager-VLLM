"""Driver factory: one driver instance per enabled channel."""

from typing import Any, Dict

from .base_driver import BaseDriver, DriverHooks, SendResult, describe_error
from .modbus_driver import ModbusTcpDriver
from .mqtt_driver import MqttDriver
from .tcp_driver import TcpClientDriver, TcpServerDriver
from .webhook_driver import WebhookDriver
from .websocket_driver import WebSocketClientDriver, WebSocketServerDriver


def create_driver(channel: Dict[str, Any], hooks: DriverHooks) -> BaseDriver:
    protocol = channel["protocol"]
    server = channel.get("mode") == "SERVER"
    if protocol == "WEBSOCKET":
        return WebSocketServerDriver(channel, hooks) if server else WebSocketClientDriver(channel, hooks)
    if protocol == "TCPIP":
        return TcpServerDriver(channel, hooks) if server else TcpClientDriver(channel, hooks)
    if protocol == "MQTT":
        return MqttDriver(channel, hooks)
    if protocol == "HTTP_WEBHOOK":
        return WebhookDriver(channel, hooks)
    if protocol == "MODBUS_TCP":
        return ModbusTcpDriver(channel, hooks)
    raise ValueError(f"Giao thức không hỗ trợ: {protocol}")


__all__ = ["BaseDriver", "DriverHooks", "SendResult", "create_driver", "describe_error",
           "WebSocketServerDriver", "WebSocketClientDriver", "TcpServerDriver", "TcpClientDriver",
           "MqttDriver", "WebhookDriver", "ModbusTcpDriver"]
