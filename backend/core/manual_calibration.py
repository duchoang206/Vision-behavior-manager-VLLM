"""Manual camera/FMS point pairs in the shared metric floor coordinate frame."""

import time

import cv2
import numpy as np


def prepare_manual_calibration(calibrator, source, destination, map_id, frame, metadata=None, length_constraints=None):
    config = calibrator.prepare_config(source, destination, metadata=metadata)
    for points in (config["src_points"], config["dst_points"]):
        if len(np.unique(np.round(points, 7), axis=0)) != len(points):
            raise ValueError("Có điểm trùng nhau. Mỗi điểm camera cần một điểm FMS riêng.")
    source_array = np.asarray(config["src_points"], dtype=np.float64)
    destination_array = np.asarray(config["dst_points"], dtype=np.float64)
    constraints = length_constraints or []
    if constraints:
        from scipy.optimize import least_squares
        endpoints, lengths = [], []
        for item in constraints:
            points = np.asarray(item.get("points"), dtype=float)
            distance = item.get("distance_m")
            if (points.shape != (2, 2) or not np.isfinite(points).all() or (points < 0).any() or (points > 1).any()
                    or np.linalg.norm(points[1] - points[0]) < 1e-5
                    or type(distance) not in (int, float) or not np.isfinite(distance) or not .001 <= distance <= 10000):
                raise ValueError("Đoạn chiều dài phải có hai điểm khác nhau và số mét dương hữu hạn.")
            hull = cv2.convexHull(source_array.astype(np.float32))
            if any(cv2.pointPolygonTest(hull, tuple(point), True) < -1e-6 for point in points):
                raise ValueError("Hai đầu đoạn đo phải nằm trong vùng các điểm neo FMS đã chấm.")
            endpoints.append(points)
            lengths.append(distance)
        endpoints = np.asarray(endpoints)
        lengths = np.asarray(lengths)
        initial = np.asarray(config["matrix"])
        if abs(initial[2, 2]) < 1e-10:
            raise ValueError("Vùng calib quá sát chân trời; hãy chọn điểm mặt sàn khác.")
        initial /= initial[2, 2]
        all_points = np.vstack([source_array, endpoints.reshape(-1, 2)])
        homogeneous = np.column_stack([all_points, np.ones(len(all_points))])

        def residual(parameters):
            transform = np.append(parameters, 1.).reshape(3, 3)
            projected = homogeneous @ transform.T
            depth = projected[:, 2]
            safe = np.where(np.abs(depth) > 1e-8, depth, np.where(depth >= 0, 1e-8, -1e-8))
            positions = projected[:, :2] / safe[:, None]
            segments = positions[len(source_array):].reshape(-1, 2, 2)
            measured = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)
            return np.concatenate([(positions[:len(source_array)] - destination_array).ravel() / .05,
                                   (measured - lengths) / .02])

        solution = least_squares(residual, initial.ravel()[:8], max_nfev=400, x_scale='jac')
        if not solution.success or not np.isfinite(solution.x).all():
            raise ValueError("Không hội tụ được hiệu chuẩn; hãy kiểm tra lại điểm và chiều dài.")
        transform = np.append(solution.x, 1.).reshape(3, 3)
        config = calibrator.prepare_config(source, destination, matrix=transform, metadata=metadata)
        homogeneous_result = homogeneous @ transform.T
        depths = homogeneous_result[:, 2]
        if np.min(np.abs(depths)) < 1e-8 or depths.min() * depths.max() <= 0:
            raise ValueError("Đoạn chiều dài đi qua miền chiếu không hợp lệ.")
        fitted = homogeneous_result[:, :2] / depths[:, None]
        fitted_segments = fitted[len(source_array):].reshape(-1, 2, 2)
        fitted_lengths = np.linalg.norm(fitted_segments[:, 1] - fitted_segments[:, 0], axis=1)
        if (np.linalg.norm(fitted[:len(source_array)] - destination_array, axis=1) > .25).any() or (np.abs(fitted_lengths - lengths) > np.maximum(.05, lengths * .05)).any():
            raise ValueError("Chiều dài và các điểm FMS mâu thuẫn (lệch neo >25 cm hoặc chiều dài >5%). Hiệu chuẩn cũ chưa thay đổi.")
        config.update(length_constraints=constraints, length_errors_m=(fitted_lengths - lengths).tolist(),
                      input_method="fms_anchors_and_metric_lengths")
    else:
        config.update(length_constraints=[], length_errors_m=[])
    projected = cv2.perspectiveTransform(source_array[None], np.asarray(config["matrix"]))[0]
    errors = np.linalg.norm(projected - destination_array, axis=1)
    config.update(method="manual_camera_fms_click", active_method="manual_camera_fms_click",
                  map_id=map_id, primary_map=True, active_calibration=True,
                  coordinate_space="fms_floor_metric", calibration_role="primary_camera_fms_map",
                  calibration_updated_at=time.time(), point_count=len(source_array),
                  fms_frame=frame, coverage_polygon=cv2.convexHull(source_array.astype(np.float32)).reshape(-1, 2).tolist(),
                  reprojection_error=float(np.sqrt(np.mean(errors ** 2))), point_errors_m=errors.tolist())
    return config
