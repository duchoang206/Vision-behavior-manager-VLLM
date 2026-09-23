import json
import math
import os
import re
from collections import deque

from pydantic import BaseModel, ConfigDict, Field, field_validator


class WorkflowNode(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    type: str = Field(max_length=40)
    config: dict = Field(default_factory=dict)
    x: float = Field(default=240, ge=0, le=4000)
    y: float = Field(default=80, ge=0, le=4000)

    @field_validator("config")
    @classmethod
    def bounded_config(cls, value):
        if len(json.dumps(value, allow_nan=False)) > 65536:
            raise ValueError("Cấu hình khối vượt quá 64 KB.")
        return value


class WorkflowEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: str = Field(max_length=64)
    target: str = Field(max_length=64)


class WorkflowDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=1000)
    nodes: list[WorkflowNode] = Field(min_length=1, max_length=64)
    edges: list[WorkflowEdge] = Field(default_factory=list, max_length=128)

    @field_validator("name")
    @classmethod
    def clean_name(cls, value):
        if not value.strip():
            raise ValueError("Tên pipeline không được để trống.")
        return value.strip()


BLOCKS = [
    ("source", "Camera source", "Nguồn metadata realtime của camera đã đăng ký.", "source"),
    ("detector", "Object Detector", "DeepStream detector đang chạy; có thể chọn TensorRT model đã build.", "processing"),
    ("roi", "ROI Filter", "Lọc điểm chân đối tượng nằm trong vùng đa giác.", "processing"),
    ("line", "Line Crossing", "Đếm lần đi qua đoạn thẳng, có hướng và chống rung.", "processing"),
    ("dwell", "Dwell Time", "Phát hiện đối tượng lưu lại liên tục quá thời gian đặt.", "processing"),
    ("matcher", "Object Matcher", "Lọc định danh đã đăng ký; tái sử dụng tracking/ReID hiện tại.", "processing"),
    ("classifier", "Classifier", "Lọc lớp metadata hiện có; không suy luận PPE hoặc loại hàng mới.", "processing"),
    ("counter", "Object Counter", "Điều kiện số lượng hiện tại, bao gồm không có hàng (0).", "processing"),
    ("proximity", "Safety Distance", "Cảnh báo người–robot gần nhau trên mặt phẳng đã calib, đơn vị mét.", "processing"),
    ("display", "Display Output", "Chọn hiển thị tracking trên Monitor hoặc chỉ xem kết quả trong pipeline.", "output"),
    ("event", "Event / MP4", "Ghi nhật ký Analytics và liên kết MP4 do FFmpeg ghi.", "output"),
    ("webhook", "Alert / Webhook", "Gửi sự kiện JSON tới connector FMS/WMS/ERP/thiết bị do quản trị cấu hình.", "output"),
    ("ppe", "PPE / Fall Detection", "Cần model bảo hộ/té ngã được tích hợp; hiện chưa khả dụng.", "unavailable"),
    ("anomaly", "Anomaly Detection", "Cần model và dữ liệu nghiệp vụ; hiện chưa khả dụng.", "unavailable"),
    ("attendance", "Attendance / 5S", "Cần dữ liệu nhân sự và quy tắc nghiệp vụ; hiện chưa khả dụng.", "unavailable"),
    ("email", "Email Alert", "Chưa có bộ gửi SMTP; dùng connector cảnh báo đã cấu hình.", "unavailable"),
]
BLOCK_MAP = {entry[0]: entry for entry in BLOCKS}
OUTPUTS = {entry[0] for entry in BLOCKS if entry[3] == "output"}
CONFIG_KEYS = {
    "source": {"camera_ids"}, "detector": {"classes", "confidence", "model_id"},
    "classifier": {"classes"}, "matcher": {"labels"}, "roi": {"points"},
    "line": {"points", "direction", "hysteresis"}, "dwell": {"seconds"},
    "counter": {"operator", "count"}, "proximity": {"distance_m"},
    "display": {"monitor"}, "event": {"severity", "message", "cooldown_seconds", "evidence"},
    "webhook": {"connector", "message", "cooldown_seconds", "require_fms"},
}


def connector_settings():
    try:
        settings = json.loads(os.getenv("WORKFLOW_CONNECTORS_JSON", "{}"))
        return {key: value for key, value in settings.items()
                if re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", key) and isinstance(value, dict)
                and isinstance(value.get("url"), str) and value["url"].startswith(("https://", "http://"))}
    except (ValueError, AttributeError):
        return {}


def block_catalog():
    connectors = connector_settings()
    return [{"type": kind, "title": title, "description": description, "group": group,
             "available": group != "unavailable" and (kind != "webhook" or bool(connectors))}
            for kind, title, description, group in BLOCKS]


def validate_definition(definition, cameras, labels=(), calibrated=(), recording_enabled=True, models=()):
    graph = definition.model_dump() if isinstance(definition, WorkflowDefinition) else WorkflowDefinition.model_validate(definition).model_dump()
    nodes = {node["id"]: node for node in graph["nodes"]}
    errors, warnings = [], []
    parents = {node_id: [] for node_id in nodes}
    children = {node_id: [] for node_id in nodes}
    if len(nodes) != len(graph["nodes"]):
        errors.append("ID khối bị trùng.")
    seen_edges = set()
    for edge in graph["edges"]:
        source, target = edge["source"], edge["target"]
        if source not in nodes or target not in nodes or source == target:
            errors.append("Kết nối chứa khối không tồn tại hoặc tự nối.")
        elif (source, target) in seen_edges:
            errors.append("Kết nối bị trùng.")
        else:
            parents[target].append(source)
            children[source].append(target)
            seen_edges.add((source, target))
    roots = [node for node in nodes.values() if node["type"] == "source"]
    if len(roots) != 1:
        errors.append("Pipeline cần đúng một khối Camera source.")
    camera_ids = roots[0]["config"].get("camera_ids", []) if roots else []
    if not isinstance(camera_ids, list) or not 1 <= len(camera_ids) <= 32 or any(not isinstance(item, str) for item in camera_ids):
        errors.append("Chọn từ 1 đến 32 camera nguồn.")
        camera_ids = []
    elif len(set(camera_ids)) != len(camera_ids) or any(camera not in cameras for camera in camera_ids):
        errors.append("Camera bị trùng hoặc không còn được đăng ký.")
    indegrees = {node_id: len(value) for node_id, value in parents.items()}
    pending = deque(node_id for node_id, degree in indegrees.items() if degree == 0)
    order = []
    while pending:
        node_id = pending.popleft()
        order.append(node_id)
        for child in children[node_id]:
            indegrees[child] -= 1
            if not indegrees[child]:
                pending.append(child)
    if len(order) != len(nodes):
        errors.append("Pipeline có vòng lặp; chỉ hỗ trợ graph có hướng không chu trình.")
    for node_id, node in nodes.items():
        kind, config = node["type"], node["config"]
        prefix = f"{node_id}: "
        if kind not in CONFIG_KEYS:
            errors.append(prefix + "khối chưa có runtime/model hỗ trợ.")
            continue
        if set(config) - CONFIG_KEYS[kind]:
            errors.append(prefix + "cấu hình có trường không được hỗ trợ.")
        if kind == "source" and parents[node_id]:
            errors.append(prefix + "nguồn không nhận đầu vào.")
        if kind != "source" and not parents[node_id]:
            errors.append(prefix + "chưa nối đầu vào.")
        if kind in OUTPUTS and children[node_id]:
            errors.append(prefix + "khối đầu ra phải nằm cuối nhánh.")
        if kind not in OUTPUTS and not children[node_id]:
            errors.append(prefix + "nhánh chưa có khối đầu ra.")
        if kind in {"detector", "classifier", "matcher"}:
            key = "labels" if kind == "matcher" else "classes"
            values = config.get(key, [])
            if not isinstance(values, list) or not 1 <= len(values) <= 100 or any(not isinstance(item, str) or not item.strip() or len(item) > 120 for item in values):
                errors.append(prefix + f"cần danh sách {key} hợp lệ.")
            elif kind == "matcher" and any(item.casefold() not in {label.casefold() for label in labels} for item in values):
                errors.append(prefix + "nhãn chưa được đăng ký.")
        if kind == "detector" and "model_id" in config and (not isinstance(config["model_id"], str)
                or (config["model_id"] and not re.fullmatch(r"[a-f0-9]{32}", config["model_id"]))):
            errors.append(prefix + "model_id không hợp lệ.")
        if kind == "detector" and isinstance(config.get("model_id"), str) and config["model_id"]:
            model = next((item for item in models if item.get("id") == config["model_id"]), None)
            if not model or model.get("state") != "ready":
                errors.append(prefix + "model TensorRT chưa sẵn sàng.")
            elif isinstance(config.get("classes"), list) and any(not isinstance(item, str) or item.casefold() not in {str(name).casefold() for name in model.get("labels", [])} for item in config["classes"]):
                errors.append(prefix + "class không có trong labels của model đã chọn.")
            if type(config.get("confidence", .25)) in (int, float) and config.get("confidence", .25) < .25:
                errors.append(prefix + "detector custom có ngưỡng tối thiểu 0.25.")
        if kind in {"roi", "line"}:
            points = config.get("points", [])
            valid = isinstance(points, list) and (len(points) == 2 if kind == "line" else 3 <= len(points) <= 128)
            valid = valid and all(isinstance(point, list) and len(point) == 2 and all(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1 for value in point) for point in points)
            if valid:
                from shapely.geometry import Polygon
                valid = math.dist(*points) > .005 if kind == "line" else Polygon(points).is_valid and Polygon(points).area > .00001
            if not valid:
                errors.append(prefix + "hình học không hợp lệ; dùng tọa độ chuẩn hóa 0..1.")
        numbers = {"detector": ("confidence", .25, 0, 1), "dwell": ("seconds", 5, .1, 86400),
                   "proximity": ("distance_m", 1.5, .05, 100), "line": ("hysteresis", .005, .0001, .1),
                   "counter": ("count", 1, 0, 10000), "event": ("cooldown_seconds", 10, 1, 86400),
                   "webhook": ("cooldown_seconds", 10, 1, 86400)}
        if kind in numbers:
            key, default, minimum, maximum = numbers[kind]
            value = config.get(key, default)
            if type(value) not in (int, float) or not math.isfinite(value) or not minimum <= value <= maximum or (key == "count" and int(value) != value):
                errors.append(prefix + f"{key} phải trong khoảng {minimum}..{maximum}.")
        for key, allowed, default in (("direction", {"both", "a_to_b", "b_to_a"}, "both"), ("operator", {"gte", "lte", "eq"}, "gte"), ("severity", {"info", "warning", "critical"}, "warning")):
            if key in config and (not isinstance(config[key], str) or config[key] not in allowed):
                errors.append(prefix + f"{key} không hợp lệ.")
        for key in ("evidence", "require_fms", "monitor"):
            if key in config and type(config[key]) is not bool:
                errors.append(prefix + f"{key} phải là boolean.")
        if "message" in config and (not isinstance(config["message"], str) or len(config["message"]) > 500):
            errors.append(prefix + "nội dung cảnh báo tối đa 500 ký tự.")
        if kind == "event" and config.get("evidence") and not recording_enabled:
            errors.append(prefix + "FFmpeg recorder chưa được bật.")
        if kind == "webhook" and (not isinstance(config.get("connector"), str) or config["connector"] not in connector_settings()):
            errors.append(prefix + "connector chưa được cấu hình ở backend.")
        if (kind == "proximity" or config.get("require_fms")) and any(camera not in calibrated for camera in camera_ids):
            errors.append(prefix + "tất cả camera cần được hiệu chuẩn trước.")
    if not any(node["type"] in OUTPUTS for node in nodes.values()):
        errors.append("Cần ít nhất một khối đầu ra.")
    from core.custom_detector import requested_models
    if len(requested_models(graph)) > 1:
        errors.append("Một pipeline chỉ dùng một custom detector dùng chung trên GPU.")
    if len(camera_ids) > 1 and any(node["type"] in {"roi", "line"} for node in nodes.values()):
        errors.append("ROI/đường kẻ thuộc riêng từng camera: dùng một camera cho pipeline này.")
    warnings.append("Model custom dùng parser YOLO raw v8/v11 của DeepStream; model không có metadata names phải nhập labels theo đúng class ID.")
    return {"valid": not errors, "errors": errors, "warnings": warnings, "order": order, "camera_ids": camera_ids}
