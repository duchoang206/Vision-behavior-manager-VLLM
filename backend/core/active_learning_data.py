import hashlib
import json
import math
import os
from pathlib import Path
import uuid


def atomic_write(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as output:
            output.write(content if isinstance(content, bytes) else content.encode("utf-8"))
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")


def checksum(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def validate_annotations(annotations, labels):
    from shapely.geometry import Polygon

    if not isinstance(annotations, list) or len(annotations) > 128:
        raise ValueError("Tối đa 128 đối tượng trong một frame.")
    result, identifiers, vertex_count = [], set(), 0
    for annotation in annotations:
        name = annotation.get("class_name")
        identifier = str(annotation.get("id", ""))
        if name not in labels or not identifier or len(identifier) > 128 or identifier in identifiers:
            raise ValueError("ID đối tượng phải duy nhất, lớp phải thuộc model.")
        identifiers.add(identifier)
        rings = annotation.get("polygons")
        if not isinstance(rings, list) or not 1 <= len(rings) <= 32:
            raise ValueError("Mỗi vật cần mask/polygon đã kiểm tra.")
        normalized = []
        for ring in rings:
            if not isinstance(ring, list) or not 3 <= len(ring) <= 1024:
                raise ValueError("Mỗi polygon cần 3–1024 đỉnh.")
            if any(not isinstance(point, (list, tuple)) or len(point) != 2
                   or any(type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1
                          for value in point) for point in ring):
                raise ValueError("Điểm polygon phải hữu hạn và nằm trong ảnh (0..1).")
            polygon = Polygon(ring)
            if not polygon.is_valid or polygon.area < 1e-7:
                raise ValueError("Polygon tự cắt hoặc không có diện tích; hãy sửa lại.")
            normalized.append([[round(float(value), 7) for value in point] for point in ring])
            vertex_count += len(ring)
        if vertex_count > 8192:
            raise ValueError("Một frame tối đa 8192 đỉnh polygon.")
        points = [point for ring in normalized for point in ring]
        left, right = min(point[0] for point in points), max(point[0] for point in points)
        top, bottom = min(point[1] for point in points), max(point[1] for point in points)
        label = str(annotation.get("label", name)).strip()
        if len(label) > 120:
            raise ValueError("Tên Label tối đa 120 ký tự.")
        result.append(dict(id=identifier, class_name=name, class_id=labels.index(name), label=label,
                           polygons=normalized, bbox=[left, top, right - left, bottom - top]))
    return result


def detection_lines(annotations):
    rows = []
    for annotation in annotations:
        left, top, width, height = annotation["bbox"]
        rows.append(f"{annotation['class_id']} {left + width / 2:.7f} {top + height / 2:.7f} {width:.7f} {height:.7f}")
    return "\n".join(rows) + ("\n" if rows else "")


def quality_gate(baseline, candidate, minimum_gain, max_class_drop):
    def valid(metrics):
        values = [metrics.get("map50_95"), metrics.get("map50"), *metrics.get("per_class", {}).values()]
        return bool(metrics.get("per_class")) and all(isinstance(value, (int, float)) and math.isfinite(value)
                                                      and 0 <= value <= 1 for value in values)
    if not valid(baseline) or not valid(candidate) or baseline["per_class"].keys() != candidate["per_class"].keys():
        return False, "Thiếu metric hữu hạn hoặc lớp validation không khớp."
    if candidate["map50_95"] <= baseline["map50_95"] + minimum_gain:
        return False, "mAP50–95 chưa tốt hơn baseline theo ngưỡng yêu cầu."
    if candidate["map50"] < baseline["map50"] - max_class_drop:
        return False, "mAP50 giảm quá ngưỡng."
    for name, score in baseline["per_class"].items():
        if candidate["per_class"][name] < score - max_class_drop:
            return False, f"Độ chính xác lớp {name} giảm quá ngưỡng."
    return True, "Đạt kiểm định trên cùng tập validation cố định."
