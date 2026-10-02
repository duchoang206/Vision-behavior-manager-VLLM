"""Persistence and validation of communication channels (``comm_channels``)."""

import copy
import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from .constants import (CLIENT_ONLY_PROTOCOLS, COMMAND_KINDS, DEFAULT_TEMPLATE, DEVICE_TYPES, EVENTS,
                        FORMAT_FMS_SLOTS, FORMAT_TEMPLATE, MODES, PAYLOAD_FORMATS, PAYLOAD_TYPES, PROTOCOLS)
from .template_renderer import validate_template

logger = logging.getLogger("CommGateway")

_SLUG = re.compile(r"^[a-z0-9][a-z0-9_\-]{0,63}$")
JSON_COLUMNS = ("auth_config", "trigger_events", "config")
COLUMNS = ("id", "name", "description", "device_type", "protocol", "mode", "host", "port", "endpoint_path",
           "auth_config", "payload_template", "trigger_events", "is_enabled", "priority", "config")

CONFIG_DEFAULTS: Dict[str, Any] = {
    "payload_format": FORMAT_TEMPLATE,
    "payload_type": "json",
    "heartbeat_sec": 2.0,
    "resync_on_connect": True,
    "strict_path": False,
    "line_terminator": "lf",
    "encoding": "utf-8",
    "hex_payload": False,
    "unit_id": 1,
    "connect_timeout_sec": 3.0,
    "timeout_sec": 3.0,
    "http_method": "POST",
    "mqtt_qos": 0,
    "mqtt_retain": False,
    "use_tls": False,
    "commands": [],
    "event_commands": {},
    "poll": {},
}


class ChannelValidationError(ValueError):
    def __init__(self, errors: List[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


def slugify(text: str) -> str:
    import unicodedata

    ascii_text = unicodedata.normalize("NFKD", str(text or "")).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "_", ascii_text.lower()).strip("_")
    return (slug or "channel")[:48]


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "on", "yes")
    return bool(value)


def _as_number(value: Any, default: float, errors: List[str], label: str, minimum: float = 0.0,
               maximum: Optional[float] = None) -> float:
    try:
        number = float(value if value not in (None, "") else default)
    except (TypeError, ValueError):
        errors.append(f"{label} phải là số")
        return default
    if number < minimum or (maximum is not None and number > maximum):
        errors.append(f"{label} phải trong khoảng {minimum:g}–{maximum:g}" if maximum is not None
                      else f"{label} phải ≥ {minimum:g}")
    return number


def _normalize_commands(raw: Any, payload_type: str, errors: List[str]) -> List[Dict[str, Any]]:
    commands = []
    seen = set()
    for index, item in enumerate(raw or []):
        if not isinstance(item, dict):
            errors.append(f"Lệnh #{index + 1} không hợp lệ")
            continue
        name = str(item.get("name") or "").strip() or f"Lệnh {index + 1}"
        command_id = str(item.get("id") or "").strip().lower() or slugify(name)
        if not _SLUG.match(command_id) or command_id in seen:
            errors.append(f"Mã lệnh '{command_id}' không hợp lệ hoặc bị trùng")
            continue
        seen.add(command_id)
        kind = str(item.get("kind") or "send")
        if kind not in COMMAND_KINDS:
            errors.append(f"Lệnh '{name}': loại '{kind}' không hỗ trợ")
            continue
        command = {"id": command_id, "name": name, "kind": kind}
        if kind == "send":
            payload = str(item.get("payload") or "")
            if not payload.strip():
                errors.append(f"Lệnh '{name}': payload rỗng")
            command["payload"] = payload
            command["payload_type"] = item.get("payload_type") if item.get("payload_type") in PAYLOAD_TYPES else payload_type
            if command["payload_type"] == "json" and payload.strip():
                error = validate_template(payload, "json")
                if error:
                    errors.append(f"Lệnh '{name}': {error}")
        else:
            address = int(_as_number(item.get("address"), 0, errors, f"Lệnh '{name}': địa chỉ", 0, 65535))
            command["address"] = address
            if kind == "modbus_coil":
                command["value"] = _as_bool(item.get("value"), True)
            elif kind == "modbus_register":
                command["value"] = int(_as_number(item.get("value"), 0, errors, f"Lệnh '{name}': giá trị", 0, 65535))
            else:
                command["count"] = int(_as_number(item.get("count"), 1, errors, f"Lệnh '{name}': số lượng", 1, 125))
        commands.append(command)
    return commands


def normalize_channel(data: Dict[str, Any], existing: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Merge ``data`` over ``existing``/defaults and validate.  Raises ChannelValidationError."""
    errors: List[str] = []
    base = copy.deepcopy(existing) if existing else {}
    merged = {**base, **{k: v for k, v in (data or {}).items() if k in COLUMNS}}
    config = {**CONFIG_DEFAULTS, **(base.get("config") or {}), **((data or {}).get("config") or {})}

    name = str(merged.get("name") or "").strip()
    if not name:
        errors.append("Tên kênh/thiết bị là bắt buộc")
    channel_id = str(merged.get("id") or "").strip().lower() or slugify(name)
    if not _SLUG.match(channel_id):
        errors.append("Mã kênh chỉ gồm a-z, 0-9, '_' hoặc '-' (tối đa 64 ký tự)")

    protocol = str(merged.get("protocol") or "WEBSOCKET").upper()
    if protocol not in PROTOCOLS:
        errors.append(f"Giao thức '{protocol}' không hỗ trợ")
    mode = str(merged.get("mode") or "SERVER").upper()
    if mode not in MODES:
        errors.append(f"Chế độ '{mode}' không hợp lệ")
    if protocol in CLIENT_ONLY_PROTOCOLS:
        mode = "CLIENT"

    device_type = str(merged.get("device_type") or "GENERIC").upper()
    if device_type not in DEVICE_TYPES:
        errors.append(f"Loại thiết bị '{device_type}' không hỗ trợ")

    host = str(merged.get("host") or "").strip()
    if mode == "SERVER":
        host = host or "0.0.0.0"
    elif not host or host in ("0.0.0.0", "::"):
        errors.append("Chế độ CLIENT cần địa chỉ IP/host đích")
    port = int(_as_number(merged.get("port"), 8000, errors, "Cổng", 1, 65535))

    path = str(merged.get("endpoint_path") or "").strip()
    if protocol == "WEBSOCKET":
        path = path or "/ws/wcs_camera"
        path = path if path.startswith("/") else "/" + path
    elif protocol == "MQTT":
        path = path.lstrip("/")
        if not path:
            errors.append("MQTT cần Topic (endpoint path)")
    elif protocol == "HTTP_WEBHOOK":
        path = path or "/"
        if not path.startswith(("/", "http://", "https://")):
            path = "/" + path

    payload_format = config.get("payload_format") if config.get("payload_format") in PAYLOAD_FORMATS else FORMAT_TEMPLATE
    if protocol == "MODBUS_TCP":
        payload_format = FORMAT_TEMPLATE
    payload_type = config.get("payload_type") if config.get("payload_type") in PAYLOAD_TYPES else "json"
    if payload_format == FORMAT_FMS_SLOTS:
        payload_type = "json"
    config["payload_format"] = payload_format
    config["payload_type"] = payload_type

    template = str(merged.get("payload_template") or "").strip("\n") or DEFAULT_TEMPLATE
    if payload_format == FORMAT_TEMPLATE and protocol != "MODBUS_TCP":
        error = validate_template(template, payload_type)
        if error:
            errors.append(f"Payload template: {error}")

    events = merged.get("trigger_events")
    if events is None:
        events = ["SLOT_CARFULL", "SLOT_EMPTY"]
    if isinstance(events, str):
        events = [events]
    events = [str(e).upper() for e in events]
    unknown = [e for e in events if e not in EVENTS]
    if unknown:
        errors.append(f"Sự kiện không hỗ trợ: {', '.join(unknown)}")

    config["heartbeat_sec"] = _as_number(config.get("heartbeat_sec"), 2.0, errors, "Chu kỳ heartbeat", 0, 3600)
    config["connect_timeout_sec"] = _as_number(config.get("connect_timeout_sec"), 3.0, errors, "Connect timeout", 0.2, 60)
    config["timeout_sec"] = _as_number(config.get("timeout_sec"), 3.0, errors, "Timeout", 0.2, 60)
    config["unit_id"] = int(_as_number(config.get("unit_id"), 1, errors, "Modbus Unit ID", 0, 255))
    config["mqtt_qos"] = int(_as_number(config.get("mqtt_qos"), 0, errors, "MQTT QoS", 0, 2))
    for flag in ("resync_on_connect", "strict_path", "hex_payload", "mqtt_retain", "use_tls"):
        config[flag] = _as_bool(config.get(flag), CONFIG_DEFAULTS[flag])
    if str(config.get("line_terminator")).lower() not in ("lf", "crlf", "cr", "none"):
        config["line_terminator"] = "lf"
    config["http_method"] = str(config.get("http_method") or "POST").upper()
    if config["http_method"] not in ("POST", "PUT", "PATCH"):
        errors.append("HTTP method phải là POST/PUT/PATCH")

    commands = _normalize_commands(config.get("commands"), payload_type, errors)
    config["commands"] = commands
    command_ids = {c["id"] for c in commands}
    event_commands = {}
    for event, ids in dict(config.get("event_commands") or {}).items():
        event = str(event).upper()
        if event not in EVENTS:
            errors.append(f"Ánh xạ lệnh: sự kiện '{event}' không hỗ trợ")
            continue
        ids = [ids] if isinstance(ids, str) else list(ids or [])
        missing = [i for i in ids if i not in command_ids]
        if missing:
            errors.append(f"Ánh xạ lệnh {event}: không tìm thấy lệnh {', '.join(missing)}")
        event_commands[event] = [i for i in ids if i in command_ids]
    config["event_commands"] = {k: v for k, v in event_commands.items() if v}
    poll = dict(config.get("poll") or {})
    if poll:
        poll = {"enabled": _as_bool(poll.get("enabled")), "kind": "registers" if poll.get("kind") == "registers" else "coils",
                "address": int(_as_number(poll.get("address"), 0, errors, "Poll address", 0, 65535)),
                "count": int(_as_number(poll.get("count"), 8, errors, "Poll count", 1, 125)),
                "interval_sec": _as_number(poll.get("interval_sec"), 1.0, errors, "Poll interval", 0.2, 3600)}
    config["poll"] = poll

    auth = merged.get("auth_config") or {}
    if not isinstance(auth, dict):
        errors.append("auth_config phải là object")
        auth = {}

    if errors:
        raise ChannelValidationError(errors)
    return {
        "id": channel_id,
        "name": name,
        "description": str(merged.get("description") or "").strip(),
        "device_type": device_type,
        "protocol": protocol,
        "mode": mode,
        "host": host,
        "port": port,
        "endpoint_path": path,
        "auth_config": auth,
        "payload_template": template,
        "trigger_events": events,
        "is_enabled": _as_bool(merged.get("is_enabled"), True),
        "priority": int(_as_number(merged.get("priority"), 1, [], "priority", 0)),
        "config": config,
        "created_at": base.get("created_at"),
        "updated_at": base.get("updated_at"),
    }


def _row_to_channel(row: Dict[str, Any]) -> Dict[str, Any]:
    channel = dict(row)
    for column in JSON_COLUMNS:
        value = channel.get(column)
        if isinstance(value, str):
            try:
                channel[column] = json.loads(value)
            except ValueError:
                channel[column] = {} if column != "trigger_events" else []
    for column in ("created_at", "updated_at"):
        if isinstance(channel.get(column), datetime):
            channel[column] = channel[column].isoformat()
    channel["config"] = {**CONFIG_DEFAULTS, **(channel.get("config") or {})}
    return channel


class ChannelStore:
    """CRUD over ``comm_channels`` using the shared ``DatabaseManager`` connection settings."""

    def __init__(self, db):
        self.db = db

    def ensure_schema(self) -> None:
        self.db.ensure_comm_schema()

    def _execute(self, sql: str, params: tuple = (), fetch: str = "") -> Any:
        import psycopg2.extras

        with self.db._lock:
            conn = self.db._get_connection()
            try:
                cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                cursor.execute(sql, params)
                result = cursor.fetchall() if fetch == "all" else cursor.fetchone() if fetch == "one" else cursor.rowcount
                conn.commit()
                return result
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def list(self) -> List[Dict[str, Any]]:
        rows = self._execute("SELECT * FROM comm_channels ORDER BY priority DESC, created_at ASC, id ASC", fetch="all")
        return [_row_to_channel(r) for r in rows or []]

    def get(self, channel_id: str) -> Optional[Dict[str, Any]]:
        row = self._execute("SELECT * FROM comm_channels WHERE id = %s", (channel_id,), fetch="one")
        return _row_to_channel(row) if row else None

    def upsert(self, channel: Dict[str, Any]) -> Dict[str, Any]:
        values = [json.dumps(channel[c], ensure_ascii=False) if c in JSON_COLUMNS else channel[c] for c in COLUMNS]
        placeholders = ", ".join(["%s"] * len(COLUMNS))
        updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in COLUMNS if c != "id")
        row = self._execute(
            f"INSERT INTO comm_channels ({', '.join(COLUMNS)}) VALUES ({placeholders}) "
            f"ON CONFLICT (id) DO UPDATE SET {updates}, updated_at = CURRENT_TIMESTAMP RETURNING *",
            tuple(values), fetch="one")
        return _row_to_channel(row)

    def delete(self, channel_id: str) -> bool:
        return bool(self._execute("DELETE FROM comm_channels WHERE id = %s", (channel_id,)))
