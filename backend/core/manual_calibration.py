"""Manual camera/FMS point pairs in the shared metric floor coordinate frame."""

import time

import cv2
import numpy as np


def prepare_manual_calibration(calibrator, source, destination, map_id, frame, metadata=None):
    config = calibrator.prepare_config(source, destination, metadata=metadata)
    for points in (config["src_points"], config["dst_points"]):
        if len(np.unique(np.round(points, 7), axis=0)) != len(points):
            raise ValueError("Có điểm trùng nhau. Mỗi điểm camera cần một điểm FMS riêng.")
    source_array = np.asarray(config["src_points"], dtype=np.float64)
    destination_array = np.asarray(config["dst_points"], dtype=np.float64)
    projected = cv2.perspectiveTransform(source_array[None], np.asarray(config["matrix"]))[0]
    errors = np.linalg.norm(projected - destination_array, axis=1)
    config.update(method="manual_camera_fms_click", active_method="manual_camera_fms_click",
                  map_id=map_id, primary_map=True, active_calibration=True,
                  coordinate_space="fms_floor_metric", calibration_role="primary_camera_fms_map",
                  calibration_updated_at=time.time(), point_count=len(source_array),
                  fms_frame=frame, coverage_polygon=cv2.convexHull(source_array.astype(np.float32)).reshape(-1, 2).tolist(),
                  reprojection_error=float(np.sqrt(np.mean(errors ** 2))), point_errors_m=errors.tolist())
    return config
