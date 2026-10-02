"""Shared vocabulary of the communication gateway (protocols, device types, events)."""

PROTOCOLS = ("WEBSOCKET", "TCPIP", "MQTT", "HTTP_WEBHOOK", "MODBUS_TCP")
MODES = ("SERVER", "CLIENT")
# Protocols that can only originate connections.
CLIENT_ONLY_PROTOCOLS = ("MQTT", "HTTP_WEBHOOK", "MODBUS_TCP")

# Device types mirror the FMS "Call Box" device list so operators recognise them.
DEVICE_TYPES = (
    "FMS_WCS",      # FMS WCS camera device (CAMERA_AI) - slot state feed
    "PLC",
    "ELEVATOR",
    "CALLER",
    "BOXSENSOR",
    "LIGHT_TOWER",  # Đèn tháp / còi báo
    "DOOR",
    "WMS",
    "GENERIC",
)

EVENT_SLOT_CARFULL = "SLOT_CARFULL"
EVENT_SLOT_EMPTY = "SLOT_EMPTY"
EVENT_ROI_ALERT = "ROI_ALERT"
EVENT_TEST = "TEST"
EVENTS = (EVENT_SLOT_CARFULL, EVENT_SLOT_EMPTY, EVENT_ROI_ALERT)

# Payload formats.  ``fms_wcs_slots`` is the wire format consumed by the FMS
# WCS ``WsCameraClient`` (rtcserver_wcs): it iterates ``json["slots"]``,
# parses ``slot_id`` with ``stoi`` and treats ``state == "Car Full"`` as goods.
FORMAT_TEMPLATE = "template"
FORMAT_FMS_SLOTS = "fms_wcs_slots"
PAYLOAD_FORMATS = (FORMAT_TEMPLATE, FORMAT_FMS_SLOTS)
PAYLOAD_TYPES = ("json", "text")

STATE_CARFULL = "Car Full"
STATE_EMPTY = "Empty"

COMMAND_KINDS = ("send", "modbus_coil", "modbus_register", "modbus_read_coils", "modbus_read_registers")

TEMPLATE_VARIABLES = (
    {"name": "slot_id", "type": "string", "description": "Mã ô hàng (Slot ID) gán trên ROI", "example": "5"},
    {"name": "state", "type": "string", "description": "Trạng thái chuẩn FMS WCS", "example": "Car Full"},
    {"name": "status", "type": "string", "description": "Trạng thái kỹ thuật backend", "example": "CARFULL"},
    {"name": "is_occupied", "type": "boolean", "description": "Cờ chiếm dụng (JSON boolean)", "example": True},
    {"name": "event", "type": "string", "description": "Loại sự kiện", "example": "SLOT_CARFULL"},
    {"name": "camera_id", "type": "string", "description": "Mã camera", "example": "cam_4"},
    {"name": "camera_name", "type": "string", "description": "Tên camera", "example": "Cam 4"},
    {"name": "rule_id", "type": "string", "description": "Mã ROI", "example": "rule_0412"},
    {"name": "rule_name", "type": "string", "description": "Tên ROI", "example": "Ô A1"},
    {"name": "occupant_label", "type": "string", "description": "Nhãn đối tượng chiếm giữ", "example": "rack #400"},
    {"name": "occupant_id", "type": "number", "description": "ID đối tượng chiếm giữ", "example": 400},
    {"name": "overlap_ratio", "type": "number", "description": "Tỷ lệ chồng lấn (%)", "example": 87.5},
    {"name": "confidence", "type": "number", "description": "Độ tin cậy nhận diện", "example": 0.94},
    {"name": "description", "type": "string", "description": "Mô tả cảnh báo (ROI_ALERT)", "example": "Xâm nhập vùng cấm"},
    {"name": "timestamp_iso", "type": "string", "description": "Thời gian ISO 8601 UTC", "example": "2026-10-02T14:30:00Z"},
    {"name": "timestamp_ms", "type": "number", "description": "Epoch mili-giây", "example": 1790932800000},
)
TEMPLATE_VARIABLE_NAMES = frozenset(v["name"] for v in TEMPLATE_VARIABLES)

DEFAULT_TEMPLATE = '{\n  "slot_id": "{slot_id}",\n  "state": "{state}"\n}'

TEMPLATE_PRESETS = (
    {"id": "fms_wcs", "name": "FMS WCS chuẩn", "payload_type": "json", "template": DEFAULT_TEMPLATE},
    {"id": "wms_slot", "name": "WMS vị trí lưu kho", "payload_type": "json", "template": (
        '{\n  "event": "SLOT_UPDATE",\n  "location_code": "{slot_id}",\n  "occupied": {is_occupied},\n'
        '  "carrier_tag": "{occupant_label}",\n  "source_camera": "{camera_name}",\n  "timestamp": "{timestamp_iso}"\n}')},
    {"id": "safety_alert", "name": "Cảnh báo an toàn", "payload_type": "json", "template": (
        '{\n  "alert_level": "WARNING",\n  "zone_id": "{slot_id}",\n  "intruder": "{occupant_label}",\n'
        '  "time_epoch": {timestamp_ms}\n}')},
    {"id": "plc_text", "name": "PLC dạng text", "payload_type": "text", "template": "SLOT={slot_id};STATE={status}"},
)

DEFAULT_FMS_CHANNEL_ID = "fms_wcs"
