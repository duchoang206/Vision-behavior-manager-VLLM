"""Tầng 3 – Rectify the 4-vertex floor ROI into a metric bird's-eye view.

The operator clicks the four corners of the floor cell on the camera image
(normalised ``[0, 1]`` coordinates, as stored in ``rules.points``) and enters
its real size: width along edge 1→2, length along edge 2→3.  The first click
becomes BEV ``(0, 0)`` and the view is metric at 1 px = 1 mm; the polygon is forced clockwise so a
counter-clockwise drawing cannot mirror the view (which would flip the sign of
every rotation).  Pixel *centres* are mapped, keeping the half-pixel bias out
of millimetre measurements.
"""

from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import DEFAULT_CELL_MM

MIN_POLYGON_AREA_PX = 400.0


class InvalidRoiError(ValueError):
    pass


def _as_points(points: Iterable[Sequence[float]]) -> np.ndarray:
    try:
        array = np.asarray([[float(p[0]), float(p[1])] for p in points], dtype=np.float64)
    except (TypeError, ValueError, IndexError) as exc:
        raise InvalidRoiError("ROI phải là danh sách điểm [x, y]") from exc
    if array.shape != (4, 2):
        raise InvalidRoiError("ROI kiểm định cần đúng 4 đỉnh")
    if not np.isfinite(array).all() or array.min() < -0.01 or array.max() > 1.01:
        raise InvalidRoiError("Tọa độ ROI phải nằm trong khung hình [0, 1]")
    return np.clip(array, 0.0, 1.0)


def order_clockwise(points: Iterable[Sequence[float]]) -> np.ndarray:
    """Keep the first vertex, make the traversal clockwise on screen (y down)."""
    array = _as_points(points)
    x, y = array[:, 0], array[:, 1]
    signed = float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    if signed < 0:
        array = array[[0, 3, 2, 1]]
    return array


def _check_convex(pixels: np.ndarray) -> None:
    crosses = []
    for index in range(4):
        a, b, c = pixels[index], pixels[(index + 1) % 4], pixels[(index + 2) % 4]
        crosses.append((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
    if not (all(value > 0 for value in crosses) or all(value < 0 for value in crosses)):
        raise InvalidRoiError("ROI phải là tứ giác lồi (đỉnh 1→2→3→4 theo chu vi ô sàn)")


CellSize = Tuple[float, float]


def compute_homography(points_norm: Iterable[Sequence[float]], frame_width: int, frame_height: int,
                       cell_mm: CellSize = DEFAULT_CELL_MM) -> np.ndarray:
    """Homography from camera pixels to the cell's BEV at 1 px = 1 mm (``W x H`` px)."""
    ordered = order_clockwise(points_norm)
    src = ordered * np.asarray([frame_width, frame_height], dtype=np.float64) - 0.5
    _check_convex(src)
    if abs(cv2.contourArea(src.astype(np.float32))) < MIN_POLYGON_AREA_PX:
        raise InvalidRoiError("ROI quá nhỏ trên ảnh camera")
    width, height = float(cell_mm[0]), float(cell_mm[1])
    dst = np.asarray([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float64) - 0.5
    return cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32))


def warp_to_metric_bev(frame: np.ndarray, polygon_norm: Iterable[Sequence[float]],
                       cell_mm: CellSize = DEFAULT_CELL_MM) -> np.ndarray:
    """Plan Phase 1 entry point: ROI → orthographic BEV, 1 px = 1 mm (1000 x 1000 for a 1 m cell)."""
    height, width = frame.shape[:2]
    matrix = compute_homography(polygon_norm, width, height, cell_mm)
    size = (int(round(cell_mm[0])), int(round(cell_mm[1])))
    return cv2.warpPerspective(frame, matrix, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


class MetricRectifier:
    """Caches the homography of one ROI (and its real cell size) for a frame size."""

    def __init__(self, points_norm: Iterable[Sequence[float]], cell_mm: CellSize = DEFAULT_CELL_MM):
        self.points = order_clockwise(points_norm)
        self.cell_mm: CellSize = (float(cell_mm[0]), float(cell_mm[1]))
        self._key: Optional[Tuple[int, int]] = None
        self._full: Optional[np.ndarray] = None

    def _matrix(self, frame_width: int, frame_height: int) -> np.ndarray:
        if self._key != (frame_width, frame_height):
            self._full = compute_homography(self.points, frame_width, frame_height, self.cell_mm)
            self._key = (frame_width, frame_height)
        return self._full

    def size(self, mm_per_px: float = 1.0) -> Tuple[int, int]:
        """``(width, height)`` in pixels of the cell view at ``mm_per_px``."""
        return (max(1, int(round(self.cell_mm[0] / mm_per_px))), max(1, int(round(self.cell_mm[1] / mm_per_px))))

    def matrix(self, frame_shape, mm_per_px: float = 1.0) -> np.ndarray:
        """Camera px → BEV px at ``mm_per_px`` (same metric frame)."""
        full = self._matrix(int(frame_shape[1]), int(frame_shape[0]))
        if mm_per_px == 1.0:
            return full
        scale = 1.0 / float(mm_per_px)
        # Scale about pixel centres: u' + 0.5 = (u + 0.5) * scale.
        rescale = np.asarray([[scale, 0, 0.5 * scale - 0.5], [0, scale, 0.5 * scale - 0.5], [0, 0, 1]])
        return rescale @ full

    def warp(self, frame: np.ndarray, mm_per_px: float = 1.0) -> np.ndarray:
        return cv2.warpPerspective(frame, self.matrix(frame.shape, mm_per_px), self.size(mm_per_px),
                                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    def warp_region(self, frame: np.ndarray, region_mm, mm_per_px: float = 2.0) -> np.ndarray:
        """Rectified sub-view of the cell covering ``region_mm`` = [x0, y0, x1, y1]."""
        x0, y0, x1, y1 = [float(v) for v in region_mm]
        width = max(1, int(np.ceil((x1 - x0) / mm_per_px)))
        height = max(1, int(np.ceil((y1 - y0) / mm_per_px)))
        scale = 1.0 / mm_per_px
        # out pixel centre u <-> mm = x0 + (u + 0.5) * mm_per_px <-> 1 mm BEV pixel centre mm - 0.5
        to_region = np.asarray([[scale, 0, (0.5 - x0) * scale - 0.5],
                                [0, scale, (0.5 - y0) * scale - 0.5], [0, 0, 1]], dtype=np.float64)
        return cv2.warpPerspective(frame, to_region @ self.matrix(frame.shape), (width, height),
                                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    def warp_crop(self, crop: np.ndarray, origin: Tuple[int, int], frame_shape,
                  mm_per_px: float = 1.0) -> np.ndarray:
        """Warp a sub-image of a ``frame_shape`` frame whose top-left pixel is ``origin``."""
        shift = np.asarray([[1, 0, origin[0]], [0, 1, origin[1]], [0, 0, 1]], dtype=np.float64)
        return cv2.warpPerspective(crop, self.matrix(frame_shape, mm_per_px) @ shift, self.size(mm_per_px),
                                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    def camera_px_of_bev_mm(self, frame_shape, points_mm: np.ndarray) -> np.ndarray:
        """``(N, 2)`` BEV millimetre points → camera pixel-centre coordinates."""
        inverse = np.linalg.inv(self._matrix(int(frame_shape[1]), int(frame_shape[0])))
        pts = np.asarray(points_mm, dtype=np.float64).reshape(-1, 1, 2) - 0.5
        return cv2.perspectiveTransform(pts, inverse).reshape(-1, 2)

    def camera_px_to_bev_mm(self, frame_shape, points_px: np.ndarray) -> np.ndarray:
        """``(N, 2)`` camera pixel-centre points → BEV millimetres (inverse of ``camera_px_of_bev_mm``)."""
        pts = np.asarray(points_px, dtype=np.float64).reshape(-1, 1, 2)
        if pts.size == 0:
            return np.zeros((0, 2), np.float64)
        return cv2.perspectiveTransform(pts, self._matrix(int(frame_shape[1]), int(frame_shape[0]))).reshape(-1, 2) + 0.5

    def camera_bounds(self, frame_shape, margin: int = 48) -> Tuple[int, int, int, int]:
        """Pixel box ``(x0, y0, x1, y1)`` around the ROI on the camera frame."""
        corners = self.points * np.asarray([frame_shape[1], frame_shape[0]], dtype=np.float64)
        x0, y0 = np.floor(corners.min(axis=0)).astype(int) - margin
        x1, y1 = np.ceil(corners.max(axis=0)).astype(int) + margin
        return (max(0, int(x0)), max(0, int(y0)), min(int(frame_shape[1]), int(x1)), min(int(frame_shape[0]), int(y1)))

    def bev_to_camera_norm(self, frame_shape, points_mm: Iterable[Sequence[float]]) -> List[List[float]]:
        """Map BEV millimetre points back to normalised camera coordinates (UI overlay)."""
        pts = np.asarray([[float(p[0]) - 0.5, float(p[1]) - 0.5] for p in points_mm], dtype=np.float64)
        if pts.size == 0:
            return []
        inverse = np.linalg.inv(self._matrix(int(frame_shape[1]), int(frame_shape[0])))
        mapped = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), inverse).reshape(-1, 2)
        mapped = (mapped + 0.5) / np.asarray([frame_shape[1], frame_shape[0]], dtype=np.float64)
        return [[round(float(x), 5), round(float(y), 5)] for x, y in mapped]
