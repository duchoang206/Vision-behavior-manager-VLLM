"""Synthetic camera scenes with millimetre ground truth for the inspection tests.

A 2 m x 2 m textured concrete floor (1 px = 1 mm) holds the 1 m x 1 m cell at
``(500..1500, 500..1500)``.  Articles, shadows, glare and debris are painted on
that floor plane, which is then projected into an oblique 1920x1080 camera
through a perspective homography, with background clutter and sensor noise.
The pipeline only ever sees the camera frame and the four clicked ROI
corners, exactly like production.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

FRAME_W, FRAME_H = 1920, 1080
FLOOR_MM = 2000
CELL_ORIGIN = 500
# Camera pixel positions of the cell corners (TL, TR, BR, BL): an oblique view.
CAMERA_QUAD = np.float32([[690, 250], [1290, 268], [1450, 860], [520, 835]])
ROI_POINTS = (CAMERA_QUAD / np.float32([FRAME_W, FRAME_H])).tolist()

FLOOR_BGR = (138.0, 141.0, 145.0)
CARTON_BGR = (70, 128, 182)
_SHIFT = 4
_SCALE = 1 << _SHIFT


def _world_to_camera(cell=(1000.0, 1000.0)) -> np.ndarray:
    """Canvas pixel centres -> camera pixel centres.

    Both ROI clicks and BEV millimetres are *edge* coordinates (a pixel ``i``
    spans ``[i, i + 1)``), the convention of the production rectifier.
    """
    w, h = float(cell[0]), float(cell[1])
    corners = np.float64([[CELL_ORIGIN, CELL_ORIGIN], [CELL_ORIGIN + w, CELL_ORIGIN],
                          [CELL_ORIGIN + w, CELL_ORIGIN + h], [CELL_ORIGIN, CELL_ORIGIN + h]])
    edge = cv2.getPerspectiveTransform(corners.astype(np.float32), CAMERA_QUAD)
    to_edge = np.float64([[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]])
    to_centre = np.float64([[1, 0, -0.5], [0, 1, -0.5], [0, 0, 1]])
    return to_centre @ edge @ to_edge


H_WORLD_TO_CAMERA = _world_to_camera()
_CAMERAS = {}


def world_to_camera(cell=(1000.0, 1000.0)) -> np.ndarray:
    """The same camera quad framing a cell of another real size (cell <= 1500 mm)."""
    key = (float(cell[0]), float(cell[1]))
    if key not in _CAMERAS:
        _CAMERAS[key] = _world_to_camera(key)
    return _CAMERAS[key]


def _floor_texture(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shape = (FLOOR_MM, FLOOR_MM)
    # Concrete: aggregate speckle (2-8 mm), trowel mottling (~25 mm), slab-scale stains.
    fine = cv2.GaussianBlur(rng.normal(0, 1, shape).astype(np.float32), (0, 0), 1.6) * 14.0
    aggregate = cv2.GaussianBlur(rng.normal(0, 1, shape).astype(np.float32), (0, 0), 3.5) * 28.0
    mottle = cv2.GaussianBlur(rng.normal(0, 1, shape).astype(np.float32), (0, 0), 12.0) * 60.0
    coarse = cv2.GaussianBlur(rng.normal(0, 1, shape).astype(np.float32), (0, 0), 60) * 260.0
    fine = fine + aggregate + mottle
    floor = np.empty((FLOOR_MM, FLOOR_MM, 3), np.float32)
    for channel, base in enumerate(FLOOR_BGR):
        floor[..., channel] = base + fine + coarse
    # Painted lane markings outside the cell and expansion joints (static structure).
    cv2.rectangle(floor, (300, 0), (340, FLOOR_MM), (40, 190, 215), -1)
    cv2.rectangle(floor, (1660, 0), (1700, FLOOR_MM), (40, 190, 215), -1)
    cv2.line(floor, (0, 250), (FLOOR_MM, 250), (95, 98, 100), 4)
    cv2.line(floor, (0, 1750), (FLOOR_MM, 1750), (95, 98, 100), 4)
    return floor


def _clutter(seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frame = np.full((FRAME_H, FRAME_W, 3), 60, np.float32)
    for _ in range(140):
        x, y = int(rng.integers(0, FRAME_W)), int(rng.integers(0, FRAME_H))
        w, h = int(rng.integers(20, 260)), int(rng.integers(20, 260))
        color = tuple(float(c) for c in rng.integers(20, 230, 3))
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)
    return frame


_FLOOR = _floor_texture()
_CLUTTER = _clutter()


def rect_corners(width: float, height: float, cx: float, cy: float, angle_deg: float) -> np.ndarray:
    """Corners of a rotated rectangle in BEV mm (positive angle = clockwise on screen)."""
    rad = math.radians(angle_deg)
    ux, uy = math.cos(rad), math.sin(rad)
    vx, vy = -math.sin(rad), math.cos(rad)
    hw, hh = width / 2.0, height / 2.0
    return np.float64([[cx - hw * ux - hh * vx, cy - hw * uy - hh * vy],
                       [cx + hw * ux - hh * vx, cy + hw * uy - hh * vy],
                       [cx + hw * ux + hh * vx, cy + hw * uy + hh * vy],
                       [cx - hw * ux + hh * vx, cy - hw * uy + hh * vy]])


def _canvas(polygon_bev) -> np.ndarray:
    """BEV edge-mm -> fixed-point canvas pixel-centre coordinates."""
    return (np.asarray(polygon_bev, np.float64) + CELL_ORIGIN - 0.5) * _SCALE


def _fill(canvas: np.ndarray, polygon_bev: np.ndarray, color) -> None:
    world = _canvas(polygon_bev)
    cv2.fillPoly(canvas, [np.round(world).astype(np.int32)], color, cv2.LINE_AA, _SHIFT)


def _outline(canvas: np.ndarray, polygon_bev: np.ndarray, color, thickness: int) -> None:
    world = _canvas(polygon_bev)
    cv2.polylines(canvas, [np.round(world).astype(np.int32)], True, color, thickness, cv2.LINE_AA, _SHIFT)


@dataclass
class Article:
    width: float = 800.0
    height: float = 600.0
    cx: float = 500.0
    cy: float = 500.0
    angle: float = 0.0
    color: Tuple[float, float, float] = CARTON_BGR
    edge_color: Optional[Tuple[float, float, float]] = (40, 70, 100)
    edge_mm: int = 4
    tape: bool = True
    damaged: bool = False
    slats: int = 0          # pallet top deck: number of dark gaps between boards

    @property
    def corners(self) -> np.ndarray:
        return rect_corners(self.width, self.height, self.cx, self.cy, self.angle)


@dataclass
class Scene:
    article: Optional[Article] = None
    debris: List[Tuple[str, Sequence[float]]] = field(default_factory=list)
    gain: float = 1.0
    noise_sigma: float = 2.0
    shadow: Optional[np.ndarray] = None       # polygon in BEV mm
    shadow_factor: float = 0.45
    glare: Optional[Tuple[Tuple[float, float], Tuple[float, float], float]] = None  # p1, p2, width (BEV mm)
    blur_sigma: float = 0.0
    seed: int = 11
    cell: Tuple[float, float] = (1000.0, 1000.0)
    # A coloured lamp pool (polygon in BEV mm) multiplying B, G, R: not a shadow, it shifts the chroma.
    light: Optional[np.ndarray] = None
    light_gain: Tuple[float, float, float] = (0.7, 1.0, 1.5)


def render(scene: Scene) -> np.ndarray:
    canvas = _FLOOR.copy()
    article = scene.article
    if article is not None:
        corners = article.corners
        _fill(canvas, corners, article.color)
        for index in range(article.slats):
            offset = (index + 1) / (article.slats + 1) - 0.5
            gap = rect_corners(article.width * 0.97, 22.0, article.cx - math.sin(math.radians(article.angle)) * offset * article.height,
                               article.cy + math.cos(math.radians(article.angle)) * offset * article.height, article.angle)
            _fill(canvas, gap, tuple(max(0.0, c - 70) for c in article.color))
        if article.tape:
            tape = rect_corners(article.width, 50.0, article.cx, article.cy, article.angle)
            _fill(canvas, tape, tuple(min(255.0, c + 35) for c in article.color))
        if article.damaged:
            hole = rect_corners(article.width * 0.28, article.height * 0.3, article.cx + 40, article.cy - 30,
                                article.angle + 20)
            _fill(canvas, hole, (25, 35, 45))
        if article.edge_color is not None and article.edge_mm > 0:
            # Physical edge shading lies on the article, never outside its footprint.
            inset = rect_corners(article.width - article.edge_mm, article.height - article.edge_mm,
                                 article.cx, article.cy, article.angle)
            _outline(canvas, inset, article.edge_color, article.edge_mm)
    for kind, params in scene.debris:
        if kind == "toolbox":
            _fill(canvas, rect_corners(*params), (170, 90, 30))          # blue steel box
            w, h, cx, cy, angle = params
            _outline(canvas, rect_corners(w - 4, h - 4, cx, cy, angle), (110, 50, 10), 4)
        elif kind == "nylon":
            cx, cy, rx, ry = params
            world = _canvas([[cx, cy]])[0]
            cv2.ellipse(canvas, (int(world[0]), int(world[1])), (int(rx * _SCALE), int(ry * _SCALE)), 15, 0, 360,
                        (22, 22, 24), -1, cv2.LINE_AA, _SHIFT)
    if scene.shadow is not None:
        mask = np.zeros(canvas.shape[:2], np.float32)
        cv2.fillPoly(mask, [np.round(_canvas(scene.shadow)).astype(np.int32)], 1.0, cv2.LINE_AA, _SHIFT)
        mask = cv2.GaussianBlur(mask, (0, 0), 8.0)
        canvas *= (1.0 - (1.0 - scene.shadow_factor) * mask)[..., None]
        # Skylight in shadow is slightly bluish.
        canvas[..., 0] += 2.0 * mask
    if scene.light is not None:
        mask = np.zeros(canvas.shape[:2], np.float32)
        cv2.fillPoly(mask, [np.round(_canvas(scene.light)).astype(np.int32)], 1.0, cv2.LINE_AA, _SHIFT)
        mask = cv2.GaussianBlur(mask, (0, 0), 8.0)
        for channel, gain in enumerate(scene.light_gain):
            canvas[..., channel] *= 1.0 + (gain - 1.0) * mask
    canvas *= scene.gain
    if scene.glare is not None:
        (x1, y1), (x2, y2), width = scene.glare
        mask = np.zeros(canvas.shape[:2], np.float32)
        p1, p2 = _canvas([[x1, y1], [x2, y2]])
        cv2.line(mask, tuple(int(v) for v in p1), tuple(int(v) for v in p2), 1.0, int(width), cv2.LINE_AA, _SHIFT)
        mask = cv2.GaussianBlur(mask, (0, 0), 3.0)
        canvas += (mask * 400.0)[..., None]

    frame = _CLUTTER * scene.gain
    # ~1.7 floor mm per camera pixel: integrate like a sensor pixel before sampling.
    canvas = cv2.GaussianBlur(canvas, (0, 0), 0.8)
    camera = world_to_camera(scene.cell)
    warped = cv2.warpPerspective(canvas, camera, (FRAME_W, FRAME_H), flags=cv2.INTER_LINEAR)
    coverage = cv2.warpPerspective(np.ones(canvas.shape[:2], np.float32), camera, (FRAME_W, FRAME_H),
                                   flags=cv2.INTER_LINEAR)
    frame = frame * (1.0 - coverage[..., None]) + warped * coverage[..., None]
    if scene.blur_sigma > 0:
        frame = cv2.GaussianBlur(frame, (0, 0), scene.blur_sigma)
    rng = np.random.default_rng(scene.seed)
    if scene.noise_sigma > 0:
        frame = frame + rng.normal(0, scene.noise_sigma, frame.shape).astype(np.float32)
    return np.clip(frame, 0, 255).astype(np.uint8)


def baseline_frames(count: int = 5, gain: float = 1.0, cell=(1000.0, 1000.0)) -> List[np.ndarray]:
    return [render(Scene(gain=gain, seed=1000 + index, cell=cell)) for index in range(count)]


# ── test doubles ─────────────────────────────────────────────────────────────
class OracleDetector:
    """Stands in for a trained YOLO: reports pre-set detections (BEV mm) that
    fall inside the crop it is shown, in that crop's pixel coordinates."""

    def __init__(self, detections=()):
        self.detections = list(detections)     # (class_name, confidence, [x0, y0, x1, y1] mm)
        self.calls = []

    def __call__(self, image, conf, region):
        self.calls.append((tuple(region), conf))
        height, width = image.shape[:2]
        sx = (region[2] - region[0]) / float(width)
        sy = (region[3] - region[1]) / float(height)
        out = []
        for name, score, (x0, y0, x1, y1) in self.detections:
            if score < conf:
                continue
            cx0, cy0 = max(x0, region[0]), max(y0, region[1])
            cx1, cy1 = min(x1, region[2]), min(y1, region[3])
            if cx1 <= cx0 or cy1 <= cy0:
                continue
            out.append((name, score, ((cx0 - region[0]) / sx, (cy0 - region[1]) / sy,
                                      (cx1 - region[0]) / sx, (cy1 - region[1]) / sy)))
        return out


def article_bbox(article: "Article", pad: float = 6.0):
    corners = article.corners
    return [float(corners[:, 0].min() - pad), float(corners[:, 1].min() - pad),
            float(corners[:, 0].max() + pad), float(corners[:, 1].max() + pad)]
