import math

import cv2
import numpy as np


def project_calibration(config, points, source="floor", frame=None):
    if not config or config.get("coordinate_space", "fms_floor_metric") != "fms_floor_metric":
        raise ValueError("Camera chưa hiệu chuẩn theo mặt sàn FMS.")
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1:] != (2,) or not 1 <= len(values) <= 256 or not np.isfinite(values).all():
        raise ValueError("Cần từ 1 đến 256 cặp tọa độ hữu hạn.")
    matrix = np.asarray(config.get("matrix"), dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) != 3:
        raise ValueError("Ma trận hiệu chuẩn không hợp lệ.")
    matrix = matrix / np.linalg.norm(matrix)
    saved_frame = config.get("fms_frame") or frame or {}
    has_frame = all(type(saved_frame.get(key)) in (int, float) and math.isfinite(saved_frame[key])
                    for key in ("origin_x", "origin_y", "layout_depth"))
    if frame and config.get("fms_frame") and any(abs(saved_frame.get(key, float("inf")) - frame[key]) > 1e-6 for key in frame):
        raise ValueError("Hệ tọa độ FMS đã thay đổi; cần hiệu chuẩn lại camera.")
    rectified = bool(config.get("deepcalib_points_rectified"))
    profile = config.get("intrinsic_profile") or {}
    if source == "image":
        if (values < 0).any() or (values > 1).any():
            raise ValueError("Tọa độ ảnh phải thuộc 0..1.")
        image = values
        if rectified:
            from deep_calib.geometry import undistort_normalized_points
            image = undistort_normalized_points(values, profile)
        homogeneous = np.column_stack((image, np.ones(len(image)))) @ matrix.T
        if (np.abs(homogeneous[:, 2]) < 1e-8).any():
            raise ValueError("Điểm nằm trên đường chân trời.")
        floor = homogeneous[:, :2] / homogeneous[:, 2:]
    elif source in {"floor", "fms"}:
        floor = values.copy()
        if source == "fms":
            if not has_frame:
                raise ValueError("Thiếu hệ tọa độ FMS của hiệu chuẩn.")
            floor[:, 0] -= saved_frame["origin_x"]
            floor[:, 1] = saved_frame["origin_y"] + saved_frame["layout_depth"] - floor[:, 1]
        homogeneous = np.column_stack((floor, np.ones(len(floor)))) @ np.linalg.inv(matrix).T
        if (np.abs(homogeneous[:, 2]) < 1e-8).any():
            raise ValueError("Điểm nằm ngoài miền chiếu của camera.")
        image = homogeneous[:, :2] / homogeneous[:, 2:]
    else:
        raise ValueError("Miền tọa độ không hợp lệ.")
    reference = np.asarray(config.get("rectified_src_points") if rectified else config.get("src_points"), dtype=float)
    depth = np.column_stack((image, np.ones(len(image)))) @ matrix[2]
    reference_depth = matrix[2] @ np.append(reference[0], 1.0)
    if not np.isfinite(image).all() or not np.isfinite(floor).all() or (depth * reference_depth <= 0).any():
        raise ValueError("Điểm ở phía không hợp lệ của mặt sàn.")
    raw = image
    if rectified:
        from deep_calib.geometry import distort_normalized_points
        raw = distort_normalized_points(image, profile)
    if not np.isfinite(raw).all():
        raise ValueError("Điểm không có phép chiếu lên ảnh gốc.")
    coverage = cv2.convexHull(reference.astype(np.float32))
    result = []
    for image_point, raw_point, floor_point in zip(image, raw, floor):
        inside_image = bool((raw_point >= 0).all() and (raw_point <= 1).all() and
                            (image_point >= 0).all() and (image_point <= 1).all())
        result.append({"image": raw_point.tolist(), "rectified_image": image_point.tolist() if rectified else None,
                       "floor": floor_point.tolist(),
                       "fms": [float(floor_point[0] + saved_frame["origin_x"]),
                               float(saved_frame["origin_y"] + saved_frame["layout_depth"] - floor_point[1])] if has_frame else None,
                       "inside_image": inside_image,
                       "inside_calibrated_area": bool(cv2.pointPolygonTest(coverage, tuple(map(float, image_point)), True) >= -1e-7)})
    return {"save_id": config.get("save_id"), "points": result, "units": "m", "plane": "ground"}


def calibration_footprint(config, frame=None):
    rectified = bool(config.get("deepcalib_points_rectified"))
    source = np.asarray(config.get("rectified_src_points") if rectified else config.get("src_points"), dtype=np.float32)
    hull = cv2.convexHull(source).reshape(-1, 2)
    if rectified:
        from deep_calib.geometry import distort_normalized_points
        hull = distort_normalized_points(hull, config.get("intrinsic_profile") or {})
    projection = project_calibration(config, hull, "image", frame)
    floor = [point["floor"] for point in projection["points"]]
    center = np.mean(floor, axis=0)
    pose = None
    if all(type(config.get(key)) in (int, float) and math.isfinite(config[key]) for key in ("cam_x", "cam_y", "cam_z")):
        pose = [float(config["cam_x"]), float(config["cam_y"]), float(config["cam_z"])]
    return {"footprint": floor, "anchor": pose or [float(center[0]), .4, float(center[1])],
            "anchor_kind": "camera_pose" if pose else "coverage_center", "save_id": config.get("save_id"),
            "method": config.get("method", "homography"), "physical_pose_known": bool(pose),
            "yaw": float(config["yaw"]) if type(config.get("yaw")) in (int, float) and math.isfinite(config["yaw"]) else None}
