import json
import copy
import threading
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple

class CameraCalibrator:
    """
    2D-to-Floor-Map Spatial Homography Calibration Toolkit.
    Maps pixel coordinates (normalized 0..1 or pixel space) from camera view
    to a common 2D floor plan coordinate space (0..1 or real-world meters).
    """
    def __init__(self):
        # cam_id -> 3x3 Homography Matrix (numpy ndarray)
        self.homographies: Dict[str, np.ndarray] = {}
        # cam_id -> Calibration points {"src_points": [[x,y]...], "dst_points": [[x,y]...]}
        self.configs: Dict[str, dict] = {}
        self.lock = threading.RLock()

    def set_calibration(self, cam_id: str, src_points: List[List[float]], dst_points: List[List[float]], cam_x: float = None, cam_y: float = None, cam_z: float = None, yaw: float = None) -> bool:
        """
        Compute and store the 3x3 Homography Matrix from at least 4 corresponding points.
        src_points: 4 points in camera normalized coords [[x0,y0], [x1,y1], [x2,y2], [x3,y3]]
        dst_points: 4 points in floor plan coords [[X0,Y0], [X1,Y1], [X2,Y2], [X3,Y3]]
        """
        try:
            config = self.prepare_config(src_points, dst_points)
            config.update(cam_x=cam_x, cam_y=cam_y, cam_z=cam_z, yaw=yaw)
            self.apply_config(cam_id, config)
            return True
        except (ValueError, TypeError, cv2.error, np.linalg.LinAlgError):
            return False

    @staticmethod
    def prepare_config(src_points, dst_points, matrix=None, metadata=None):
        source = np.asarray(src_points, dtype=np.float64)
        destination = np.asarray(dst_points, dtype=np.float64)
        if source.ndim != 2 or source.shape[1:] != (2,) or source.shape != destination.shape or len(source) < 4:
            raise ValueError("Cần ít nhất 4 cặp điểm ảnh–mặt sàn.")
        if not np.isfinite(source).all() or not np.isfinite(destination).all() or (source < 0).any() or (source > 1).any():
            raise ValueError("Điểm ảnh phải nằm trong 0..1; tọa độ phải hữu hạn.")
        for points in (source, destination):
            if np.linalg.matrix_rank(points - points.mean(axis=0), tol=1e-7) < 2:
                raise ValueError("Các điểm hiệu chuẩn không được thẳng hàng.")
        transform = cv2.findHomography(source, destination, 0)[0] if matrix is None else np.asarray(matrix, dtype=float)
        if transform is None or transform.shape != (3, 3) or not np.isfinite(transform).all() or np.linalg.matrix_rank(transform) < 3:
            raise ValueError("Ma trận hiệu chuẩn không hợp lệ.")
        transform = transform / np.linalg.norm(transform)
        denominators = np.column_stack((source, np.ones(len(source)))) @ transform[2]
        if np.min(np.abs(denominators)) < 1e-8 or np.min(denominators) * np.max(denominators) <= 0:
            raise ValueError("Phép chiếu có điểm vô cực trong vùng hiệu chuẩn.")
        corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        corner_depth = np.column_stack((corners, np.ones(4))) @ transform[2]
        fov = cv2.perspectiveTransform(corners[None], transform)[0].tolist() if np.min(np.abs(corner_depth)) > 1e-8 and np.min(corner_depth) * np.max(corner_depth) > 0 else []
        config = dict(metadata or {})
        config.update(src_points=source.tolist(), dst_points=destination.tolist(), matrix=transform.tolist(), fov_polygon=fov)
        return config

    def apply_config(self, cam_id, config):
        checked = self.prepare_config(config["src_points"], config["dst_points"], config.get("matrix"), config)
        with self.lock:
            self.homographies[cam_id] = np.asarray(checked["matrix"], dtype=float)
            self.configs[cam_id] = checked


    def restore_config(self, cam_id, config):
        try:
            self.apply_config(cam_id, config)
            return True
        except (ValueError, TypeError, cv2.error, np.linalg.LinAlgError):
            return False

    def _compute_homography_dlt(self, src: np.ndarray, dst: np.ndarray) -> Optional[np.ndarray]:
        """Compute 3x3 Homography matrix using SVD."""
        A = []
        for i in range(4):
            x, y = src[i][0], src[i][1]
            u, v = dst[i][0], dst[i][1]
            A.append([-x, -y, -1,  0,  0,  0, x * u, y * u, u])
            A.append([ 0,  0,  0, -x, -y, -1, x * v, y * v, v])
        A = np.array(A, dtype=np.float32)
        
        # SVD: A = U * S * Vh
        _, _, Vh = np.linalg.svd(A)
        H = Vh[-1].reshape((3, 3))
        
        # Normalize so that H[2, 2] == 1
        if abs(H[2, 2]) > 1e-7:
            H = H / H[2, 2]
        return H

    def camera_to_floor(self, cam_id: str, x: float, y: float) -> Tuple[float, float]:
        """
        Transform a 2D camera ground point (bottom-center of bounding box)
        to the Floor Plan coordinates (X_floor, Z_floor) in metric meters,
        strictly bounded inside the room's physical SLAM walls.
        """
        if cam_id == "86c5119c":  # Cam 1 (Showroom Floor)
            MIN_X, MAX_X = 6.5, 11.8
            MIN_Z, MAX_Z = 9.8, 13.0
        elif cam_id == "b1269e28":  # Cam 4 (Storage & Rack Floor)
            MIN_X, MAX_X = 9.8, 16.5
            MIN_Z, MAX_Z = 10.2, 16.0
        else:
            MIN_X, MAX_X = 6.0, 22.0
            MIN_Z, MAX_Z = 9.8, 16.0

        if cam_id not in self.homographies:
            # Default room perspective mapping based on camera installation
            if cam_id == "86c5119c":
                fx = 6.8 + x * 4.2
                fy = 10.0 + y * 2.5
            elif cam_id == "b1269e28":
                fx = 10.2 + x * 5.5
                fy = 10.5 + y * 5.2
            else:
                fx = 12.0 + (x - 0.5) * 6.0
                fy = 12.0 + (y - 0.5) * 4.0
            fx = min(max(MIN_X, fx), MAX_X)
            fy = min(max(MIN_Z, fy), MAX_Z)
            return (round(fx, 3), round(fy, 3))
            
        H = self.homographies[cam_id]
        pt = np.array([x, y, 1.0], dtype=np.float32)
        floor_pt = np.dot(H, pt)
        
        # Homogeneous normalization
        if abs(floor_pt[2]) > 1e-7:
            fx = float(floor_pt[0] / floor_pt[2])
            fy = float(floor_pt[1] / floor_pt[2])
        else:
            return (None, None)
            
        # A calibrated map can use any metric origin/extent; do not clamp it to
        # the legacy warehouse constants used only by the uncalibrated fallback.
        return (round(fx, 3), round(fy, 3))

    def project_ground_point(self, cam_id: str, x: float, y: float) -> dict:
        with self.lock:
            config = self.configs.get(cam_id)
            matrix = self.homographies.get(cam_id)
        invalid = {
                "x": None,
                "y": None,
                "z": None,
                "valid": False,
                "source": "uncalibrated",
                "confidence": 0.0,
        }
        if matrix is None or not np.isfinite([x, y]).all():
            return invalid
        projected = matrix @ np.array([x, y, 1.0])
        if not np.isfinite(projected).all() or abs(projected[2]) < 1e-8:
            return dict(invalid, source="invalid_projection")
        floor_x, floor_y = projected[:2] / projected[2]
        coverage = (config or {}).get("coverage_polygon")
        inside = not coverage or cv2.pointPolygonTest(np.asarray(coverage, dtype=np.float32), (float(x), float(y)), False) >= 0
        return {
            "x": round(float(floor_x), 4),
            "y": 0.0,
            "z": round(float(floor_y), 4),
            "valid": True,
            "source": (config or {}).get("method", "homography"),
            "confidence": 0.9 if inside else 0.35,
            "inside_calibrated_area": inside,
            "coordinate_space": "fms_floor_metric",
        }

    def get_config(self, cam_id: str) -> Optional[dict]:
        with self.lock:
            return copy.deepcopy(self.configs.get(cam_id))

    def load_from_db_records(self, records: List[dict]):
        for r in records:
            cam_id = r.get("cam_id") or r.get("id")
            calib = r.get("calibration_points") or {}
            if isinstance(calib, str):
                calib = json.loads(calib)
            src = r.get("src_points") or calib.get("src_points")
            dst = r.get("dst_points") or calib.get("dst_points")
            cam_x = r.get("cam_x")
            cam_y = r.get("cam_y")
            cam_z = r.get("cam_z")
            yaw = r.get("yaw")
            if cam_id and src and dst:
                matrix = r.get("homography_matrix") or calib.get("matrix")
                if isinstance(matrix, str):
                    matrix = json.loads(matrix)
                self.restore_config(cam_id, dict(calib, src_points=src, dst_points=dst,
                    matrix=matrix, cam_x=cam_x, cam_y=cam_y, cam_z=cam_z, yaw=yaw))

# Global singleton
camera_calibrator = CameraCalibrator()
