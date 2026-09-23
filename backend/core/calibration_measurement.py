import math
from typing import Any, Dict, List

import cv2
import numpy as np


def measure_calibrated_polyline(config: Dict[str, Any], points: List[List[float]], points_space="raw") -> Dict[str, Any]:
    """Project normalized camera points through one calibration snapshot and measure meters."""
    if len(points) < 2:
        raise ValueError("Cần ít nhất 2 điểm để đo khoảng cách.")
    if len(points) > 256:
        raise ValueError("Tối đa 256 điểm cho một đường đo.")

    source = np.asarray(points, dtype=np.float64)
    if source.ndim != 2 or source.shape[1:] != (2,) or not np.isfinite(source).all():
        raise ValueError("Tọa độ điểm đo không hợp lệ.")
    if (source < 0).any() or (source > 1).any():
        raise ValueError("Điểm đo phải nằm trong ảnh camera, trong khoảng 0..1.")
    if config.get("coordinate_space", "fms_floor_metric") != "fms_floor_metric":
        raise ValueError("Hiệu chuẩn này chưa dùng tọa độ mặt sàn theo mét.")

    matrix = np.asarray(config.get("matrix"), dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) < 3:
        raise ValueError("Camera chưa có ma trận hiệu chuẩn hợp lệ.")
    matrix = matrix / np.linalg.norm(matrix)
    if points_space not in {"raw", "rectified"}:
        raise ValueError("Miền tọa độ thước đo không hợp lệ.")
    if points_space == "rectified":
        if not config.get("deepcalib_points_rectified"):
            raise ValueError("Camera chưa lưu hiệu chuẩn DeepCalib.")
        from deep_calib.geometry import validate_visible_rectified_points
        validate_visible_rectified_points(source, config.get("intrinsic_profile") or {})
        source_for_projection = source
    elif config.get("deepcalib_points_rectified"):
        try:
            from deep_calib.geometry import undistort_normalized_points
            source_for_projection = undistort_normalized_points(source, config.get("intrinsic_profile") or {})
        except (ImportError, ValueError) as error:
            raise ValueError(f"Không thể rectified điểm đo bằng DeepCalib: {error}") from error
    else:
        source_for_projection = source
    homogeneous = np.column_stack((source_for_projection, np.ones(len(source_for_projection)))) @ matrix.T
    denominators = homogeneous[:, 2]
    if not np.isfinite(homogeneous).all() or (np.abs(denominators) < 1e-8).any():
        raise ValueError("Có điểm nằm ngoài miền chiếu của hiệu chuẩn.")
    reference_points = np.asarray(config.get("rectified_src_points") or config.get("src_points", []), dtype=np.float64)
    reference_depth = denominators[0]
    if reference_points.ndim == 2 and reference_points.shape[1:] == (2,) and len(reference_points):
        reference_depth = float(matrix[2] @ np.append(reference_points[0], 1.0))
    if (denominators * reference_depth <= 0).any():
        raise ValueError("Đường đo đi qua đường chân trời hoặc ra phía không hợp lệ của mặt sàn.")

    floor_points = homogeneous[:, :2] / denominators[:, None]
    if not np.isfinite(floor_points).all():
        raise ValueError("Không thể đổi điểm ảnh sang tọa độ mặt sàn.")

    coverage = config.get("rectified_coverage_polygon") if config.get("deepcalib_points_rectified") else config.get("coverage_polygon")
    if not coverage and reference_points.ndim == 2 and len(reference_points) >= 3:
        coverage = cv2.convexHull(reference_points.astype(np.float32)).reshape(-1, 2).tolist()
    if coverage:
        polygon = np.asarray(coverage, dtype=np.float32)
        inside = [bool(cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), True) >= -1e-7) for point in source_for_projection]
    else:
        inside = [None] * len(source)

    frame = config.get("fms_frame") or {}
    has_fms_frame = all(isinstance(frame.get(key), (int, float)) and math.isfinite(frame[key])
                        for key in ("origin_x", "origin_y", "layout_depth"))
    transformed_points = []
    for (camera_x, camera_y), (floor_x, floor_z), is_inside in zip(source, floor_points, inside):
        transformed_points.append({
            "camera": [round(float(camera_x), 6), round(float(camera_y), 6)],
            "floor": [round(float(floor_x), 6), round(float(floor_z), 6)],
            "fms": [round(float(floor_x + frame["origin_x"]), 6),
                    round(float(frame["origin_y"] + frame["layout_depth"] - floor_z), 6)] if has_fms_frame else None,
            "inside_calibrated_area": is_inside,
        })

    segments = []
    distances = []
    for index in range(1, len(floor_points)):
        distance = float(np.linalg.norm(floor_points[index] - floor_points[index - 1]))
        distances.append(distance)
        segments.append({"from": index, "to": index + 1, "distance_m": round(distance, 6)})
    total_distance = math.fsum(distances)
    direct_distance = float(np.linalg.norm(floor_points[-1] - floor_points[0]))
    warnings = []
    if any(value is False for value in inside):
        warnings.append("Một hoặc nhiều điểm nằm ngoài vùng đã được phủ bởi các điểm hiệu chuẩn; độ chính xác có thể giảm.")
    if any(value is None for value in inside):
        warnings.append("Hiệu chuẩn cũ không có dữ liệu vùng phủ để kiểm chứng các điểm đo.")

    return {
        "units": "m",
        "plane": "ground",
        "points_space": points_space,
        "points": transformed_points,
        "segments": segments,
        "total_distance_m": round(total_distance, 6),
        "direct_distance_m": round(direct_distance, 6),
        "warnings": warnings,
        "calibration": {
            "method": config.get("method", "homography"),
            "updated_at": config.get("calibration_updated_at") or config.get("calibrated_at"),
            "point_count": config.get("point_count") or len(config.get("src_points", [])),
            "reprojection_error_m": config.get("reprojection_error", config.get("residual_m")),
        },
    }
