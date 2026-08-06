"""
Session 3: Homography Spatial Mapping Module.
Projects 2D image pixel ground contact points (u, v) into 3D metric warehouse world coordinates (X_world, Y_world).
"""

import numpy as np
from typing import Tuple, List, Optional, Union


class HomographyMapper:
    """
    Projective Geometry Homography Transformer.
    Converts pixel coordinates to real-world floor map coordinates in meters.
    """
    def __init__(self, H_matrix: Optional[Union[np.ndarray, List[List[float]]]] = None):
        if H_matrix is not None:
            self.H = np.array(H_matrix, dtype=np.float64)
            if self.H.shape != (3, 3):
                raise ValueError(f"Homography matrix must be 3x3, got {self.H.shape}")
            self.H_inv = np.linalg.pinv(self.H)
        else:
            # Default identity projective mapping
            self.H = np.eye(3, dtype=np.float64)
            self.H_inv = np.eye(3, dtype=np.float64)

    @classmethod
    def from_point_correspondences(
        cls,
        image_points: List[Tuple[float, float]],
        world_points: List[Tuple[float, float]]
    ) -> 'HomographyMapper':
        """
        Computes 3x3 Homography matrix from at least 4 point correspondences (cv2.findHomography).
        image_points: [(u1, v1), (u2, v2), (u3, v3), (u4, v4)]
        world_points: [(X1, Y1), (X2, Y2), (X3, Y3), (X4, Y4)] (in meters)
        """
        import cv2
        src_pts = np.array(image_points, dtype=np.float32).reshape(-1, 1, 2)
        dst_pts = np.array(world_points, dtype=np.float32).reshape(-1, 1, 2)
        
        H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None:
            raise RuntimeError("Failed to compute Homography matrix from given point correspondences")
        return cls(H)

    def pixel_to_world(self, u: float, v: float) -> Tuple[float, float]:
        """
        Converts single image pixel contact point (u, v) to metric world coordinates (X_world, Y_world).
        [x', y', w']^T = H * [u, v, 1]^T
        X_world = x' / w', Y_world = y' / w'
        """
        pt = np.array([u, v, 1.0], dtype=np.float64)
        projected = self.H @ pt
        w_norm = projected[2]
        
        if abs(w_norm) < 1e-7:
            # Point is at infinity / singularity
            return (0.0, 0.0)
            
        x_world = float(projected[0] / w_norm)
        y_world = float(projected[1] / w_norm)
        return (round(x_world, 3), round(y_world, 3))

    def world_to_pixel(self, x_world: float, y_world: float) -> Tuple[float, float]:
        """
        Converts metric world coordinate (X_world, Y_world) back to image pixel coordinate (u, v).
        """
        pt = np.array([x_world, y_world, 1.0], dtype=np.float64)
        projected = self.H_inv @ pt
        w_norm = projected[2]
        
        if abs(w_norm) < 1e-7:
            return (0.0, 0.0)
            
        u = float(projected[0] / w_norm)
        v = float(projected[1] / w_norm)
        return (round(u, 2), round(v, 2))

    def batch_pixel_to_world(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Batch transforms list of (u, v) points to [(X1, Y1), (X2, Y2), ...]"""
        if not points:
            return []
        pts_homo = np.ones((len(points), 3), dtype=np.float64)
        pts_homo[:, :2] = points
        
        projected = (self.H @ pts_homo.T).T
        w = projected[:, 2:3]
        w[np.abs(w) < 1e-7] = 1.0
        world_pts = projected[:, :2] / w
        return [(round(float(p[0]), 3), round(float(p[1]), 3)) for p in world_pts]


class HomographyEngine:
    """
    Multi-camera Homography transformation engine.
    Maintains HomographyMapper per camera.
    """
    def __init__(self):
        self.mappers: dict = {}

    def set_camera_homography(self, cam_id: str, H_matrix):
        self.mappers[cam_id] = HomographyMapper(H_matrix)

    def project_point(self, cam_id: str, norm_u: float, norm_v: float, width: float = 1920, height: float = 1080) -> Tuple[float, float]:
        if cam_id in self.mappers:
            u = norm_u * width if norm_u <= 1.0 else norm_u
            v = norm_v * height if norm_v <= 1.0 else norm_v
            return self.mappers[cam_id].pixel_to_world(u, v)
        
        # Fallback default scaling to warehouse floor (e.g. 26m x 18m)
        u_val = norm_u if norm_u <= 1.0 else norm_u / width
        v_val = norm_v if norm_v <= 1.0 else norm_v / height
        world_x = round(u_val * 26.0, 2)
        world_z = round(v_val * 18.0, 2)
        return (world_x, world_z)


homography_engine = HomographyEngine()

