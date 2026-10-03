"""Tầng 4 – Metrology engine: adaptive background subtraction and OBB measurement.

Detection runs on the rectified cell (of any configured size) at the
4 mm/px pyramid level against the baseline of the empty cell.  The oriented
bounding box (OBB) found there is then refined to sub-millimetre accuracy by
sampling intensity profiles across each of its four sides directly on the
full-resolution camera frame (``refine_rectangle``).  Output is the article's
width/height, centre offset from the cell centre ``(W/2, H/2)``, rotation θ
and the clearance to the cell edge, all in BEV millimetres, checked against the
configured tolerances.

Safety notes
------------
* Every limit is applied with a *guard band* equal to the accuracy the system
  is accepted for (target.md §2: ±3 mm centre, ±0.5° angle).  A value that is
  truly over a limit can therefore never be measured as inside it, which is
  what the 0 % false-negative KPI requires.
* Shadow / illumination pixels are removed only when chroma *and* floor
  texture are verifiably unchanged; on a texture-less floor nothing is
  classified as shadow (an object is never explained away without evidence).
"""

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import (ERR_CENTER_OFFSET_EXCEEDED, ERR_DIMENSION_OUT_OF_SPEC, ERR_OUT_OF_BOUNDS_VIOLATION,
                     ERR_ROTATION_ANGLE_EXCEEDED, METROLOGY_PRIORITY, WORK_MM_PER_PX, InspectionConfig,
                     Tolerances)
from .homography_rectifier import MetricRectifier
from .illumination_normalizer import LabPlanes, NormalizedPair, normalize_pair, prepare_live

CV_EMPTY = "CV_EMPTY"
CV_PASS = "CV_PASS"
CV_NG = "CV_NG"

MIN_OBJECT_AREA_MM2 = 2500.0                 # 50 x 50 mm
MIN_OBJECT_AREA_PX = MIN_OBJECT_AREA_MM2 / (WORK_MM_PER_PX ** 2)
GLARE_UNRESOLVED_AREA_MM2 = 10000.0          # 1 % of the cell
# Guard bands = accepted measurement accuracy (target.md §2).
GUARD_OFFSET_MM = 3.0
GUARD_ANGLE_DEG = 0.5
GUARD_MARGIN_MM = 3.0
GUARD_DIMENSION_MM = 3.0

# Shadow / illumination filter
SHADOW_RATIO_RANGE = (0.20, 0.92)
HIGHLIGHT_RATIO_RANGE = (1.08, 2.50)
LIGHT_RATIO_RANGE = (0.20, 0.97, 1.03, 2.50)   # extended range for the penumbra fringe
SHADOW_LOCAL_NCC = 0.55
SHADOW_BLOB_NCC = 0.45
TEXTURE_MIN_STD = 0.8
NCC_WINDOW = 7
# Light changes R, G and B by the same factor, so normalised rgb chromaticity
# survives a shadow even on saturated paint (Lab a/b of a yellow lane line
# shrink with lightness and would look like a colour change).
SHADOW_CHROMATICITY_MAX = 12.0     # max |Δ(c/ΣRGB)| x 255
BAND_PASS_SIGMAS = (1.0, 3.0)
PENUMBRA_PX = 5
# Next to a confirmed shadow the band-pass sees the light step, not the floor:
# within FRINGE_PX of it, foreground must be at least FRINGE_OPEN_PX thick.
FRINGE_PX = 12
FRINGE_OPEN_PX = 9

# AI-guided re-scan (Kịch bản 1): mean live-baseline step across a new edge.
RESCAN_MIN_LINE_CONTRAST = 5.0

# Sub-pixel edge refinement
# (profiles per side, half window mm, step mm).  The coarse 4 mm/px OBB is
# normally within ±6 mm, so one tight pass settles it; when it does not (an
# attached shadow residue, a strongly blurred edge) the wide capture runs.
REFINE_FAST_PASS = (18, 12.0, 1.0)
REFINE_FAST_MAX_SHIFT_MM = 8.0
REFINE_PASSES = ((14, 24.0, 2.0), (24, 9.0, 1.0))
REFINE_MIN_FILL = 0.80
REFINE_MIN_SIDE_SUPPORT = 0.4
REFINE_MAX_SHIFT_MM = 22.0
REFINE_MAX_TURN_DEG = 3.0

_KERNEL3 = np.ones((3, 3), np.uint8)
_KERNEL5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
_KERNEL_FRINGE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (FRINGE_OPEN_PX, FRINGE_OPEN_PX))


def _high_pass(L: np.ndarray) -> np.ndarray:
    """Band-pass (difference of Gaussians): floor texture without sensor noise."""
    return cv2.GaussianBlur(L, (0, 0), BAND_PASS_SIGMAS[0]) - cv2.GaussianBlur(L, (0, 0), BAND_PASS_SIGMAS[1])


# ── baseline ─────────────────────────────────────────────────────────────────
@dataclass
class BaselineModel:
    """The empty cell: a median camera crop around the ROI and what derives from it.

    Every analysis image of the baseline is produced from ``camera_crop`` by
    the same warp as the live frame, so live and baseline are sampled
    identically at every resolution.
    """

    camera_crop: np.ndarray
    crop_origin: Tuple[int, int]
    frame_shape: Tuple[int, int]
    rectifier: MetricRectifier
    bgr: np.ndarray
    planes: LabPlanes
    high_pass: np.ndarray
    edges: np.ndarray
    timestamp: float = 0.0
    points: Optional[List[List[float]]] = None
    # One of several empty-cell captures of the station (one per lighting condition).
    baseline_id: str = ""
    label: str = ""
    captured_at: float = 0.0

    @classmethod
    def from_camera(cls, camera_crop: np.ndarray, crop_origin: Tuple[int, int], frame_shape: Sequence[int],
                    rectifier: MetricRectifier, timestamp: float = 0.0,
                    points: Optional[List[List[float]]] = None, baseline_id: str = "", label: str = "",
                    captured_at: float = 0.0) -> "BaselineModel":
        shape = (int(frame_shape[0]), int(frame_shape[1]))
        work = rectifier.warp_crop(camera_crop, crop_origin, shape, WORK_MM_PER_PX)
        planes = LabPlanes.from_bgr(work)
        edges = cv2.Canny(cv2.GaussianBlur(planes.gray, (3, 3), 0), 10, 30)
        edges = cv2.dilate(edges, _KERNEL3)
        return cls(camera_crop, (int(crop_origin[0]), int(crop_origin[1])), shape, rectifier, work, planes,
                   _high_pass(planes.L), edges, timestamp, points, baseline_id, label, captured_at or timestamp)

    @property
    def cell_mm(self) -> Tuple[float, float]:
        return self.rectifier.cell_mm

    def bev(self, mm_per_px: float = 1.0) -> np.ndarray:
        return self.rectifier.warp_crop(self.camera_crop, self.crop_origin, self.frame_shape, mm_per_px)


# ── geometry primitives ──────────────────────────────────────────────────────
@dataclass
class RectMM:
    """Oriented rectangle in BEV mm; ``angle`` is the direction of side ``w``."""

    cx: float
    cy: float
    w: float
    h: float
    angle: float

    @classmethod
    def from_corners(cls, corners: np.ndarray) -> "RectMM":
        corners = np.asarray(corners, dtype=np.float64)
        u = (corners[1] - corners[0] + corners[2] - corners[3]) / 2.0
        v = (corners[2] - corners[1] + corners[3] - corners[0]) / 2.0
        center = corners.mean(axis=0)
        return cls(float(center[0]), float(center[1]), float(np.hypot(*u)), float(np.hypot(*v)),
                   math.degrees(math.atan2(u[1], u[0])))

    def corners(self) -> np.ndarray:
        rad = math.radians(self.angle)
        u = np.asarray([math.cos(rad), math.sin(rad)]) * self.w / 2.0
        v = np.asarray([-math.sin(rad), math.cos(rad)]) * self.h / 2.0
        c = np.asarray([self.cx, self.cy])
        return np.asarray([c - u - v, c + u - v, c + u + v, c - u + v])


@dataclass
class Blob:
    contour: np.ndarray
    area_px: float
    bbox_px: Tuple[int, int, int, int]
    source: str = "background"
    shape: Tuple[int, int] = (250, 250)       # (rows, cols) of the analysis image

    @property
    def area_mm2(self) -> float:
        return self.area_px * WORK_MM_PER_PX ** 2

    @property
    def bbox_mm(self) -> List[float]:
        x, y, w, h = self.bbox_px
        return [x * WORK_MM_PER_PX, y * WORK_MM_PER_PX, (x + w) * WORK_MM_PER_PX, (y + h) * WORK_MM_PER_PX]

    def mask(self, shape=None) -> np.ndarray:
        mask = np.zeros(shape or self.shape, np.uint8)
        cv2.drawContours(mask, [self.contour], -1, 255, cv2.FILLED)
        return mask

    def touches_border(self) -> bool:
        points = self.contour.reshape(-1, 2)
        rows, cols = self.shape
        return bool(points.min() <= 0 or points[:, 0].max() >= cols - 1 or points[:, 1].max() >= rows - 1)

    def coarse_rect(self) -> RectMM:
        (cx, cy), (w, h), angle = cv2.minAreaRect(self.contour.reshape(-1, 2).astype(np.float32))
        # Contour vertices are pixel centres: the physical extent is one pixel more.
        corners = (cv2.boxPoints(((cx, cy), (w + 1.0, h + 1.0), angle)) + 0.5) * WORK_MM_PER_PX
        return RectMM.from_corners(corners)


@dataclass
class Measurement:
    width_mm: float
    height_mm: float
    center_x_mm: float
    center_y_mm: float
    offset_x_mm: float
    offset_y_mm: float
    offset_mm: float
    rotation_deg: float
    min_margin_mm: float
    touches_border: bool
    corners_mm: List[List[float]]
    area_mm2: float
    fill_ratio: float
    orientation_swapped: bool = False
    refined: bool = False

    def as_dict(self) -> dict:
        data = {key: (round(value, 2) if isinstance(value, float) else value) for key, value in self.__dict__.items()}
        data["corners_mm"] = [[round(x, 1), round(y, 1)] for x, y in self.corners_mm]
        return data


def _fold(angle: float) -> float:
    """Fold an undirected line angle into [-90, 90)."""
    return ((angle + 90.0) % 180.0) - 90.0


def measurement_from_rect(rect: RectMM, tolerances: Tolerances, touches_border: bool, area_mm2: float,
                          refined: bool = False) -> Measurement:
    corners = rect.corners()
    theta_w = _fold(rect.angle)
    if abs(theta_w) <= 45.0:
        width_x, height_y, theta = rect.w, rect.h, theta_w
    else:
        width_x, height_y, theta = rect.h, rect.w, _fold(rect.angle + 90.0)

    swapped = False
    err_a = max(abs(width_x - tolerances.width_mm) - tolerances.tolerance_w_mm,
                abs(height_y - tolerances.height_mm) - tolerances.tolerance_h_mm)
    err_b = max(abs(height_y - tolerances.width_mm) - tolerances.tolerance_w_mm,
                abs(width_x - tolerances.height_mm) - tolerances.tolerance_h_mm)
    if err_a > 0 and err_b <= 0 and abs(tolerances.width_mm - tolerances.height_mm) > 1e-6:
        # The article only fits when turned a quarter: it is lying across the forks.
        width_x, height_y = height_y, width_x
        theta = theta - 90.0 if theta > 0 else theta + 90.0
        swapped = True

    cell_w, cell_h = tolerances.cell_w_mm, tolerances.cell_h_mm
    offset_x, offset_y = rect.cx - cell_w / 2.0, rect.cy - cell_h / 2.0
    margin = float(np.min(np.concatenate([corners[:, 0], cell_w - corners[:, 0],
                                          corners[:, 1], cell_h - corners[:, 1]])))
    if touches_border:
        margin = min(margin, 0.0)
    return Measurement(
        width_mm=float(width_x), height_mm=float(height_y), center_x_mm=float(rect.cx), center_y_mm=float(rect.cy),
        offset_x_mm=float(offset_x), offset_y_mm=float(offset_y), offset_mm=float(math.hypot(offset_x, offset_y)),
        rotation_deg=float(theta), min_margin_mm=margin, touches_border=touches_border,
        corners_mm=[[float(x), float(y)] for x, y in corners], area_mm2=float(area_mm2),
        fill_ratio=min(1.0, area_mm2 / max(1.0, rect.w * rect.h)), orientation_swapped=swapped, refined=refined)


def evaluate_measurement(measurement: Measurement, tolerances: Tolerances) -> Tuple[List[str], Dict[str, dict]]:
    """Compare against tolerances (with guard bands). Errors come out in priority order."""
    checks = {
        "width": {"value": measurement.width_mm, "nominal": tolerances.width_mm,
                  "limit": tolerances.tolerance_w_mm,
                  "ok": abs(measurement.width_mm - tolerances.width_mm) <= tolerances.tolerance_w_mm - GUARD_DIMENSION_MM},
        "height": {"value": measurement.height_mm, "nominal": tolerances.height_mm,
                   "limit": tolerances.tolerance_h_mm,
                   "ok": abs(measurement.height_mm - tolerances.height_mm) <= tolerances.tolerance_h_mm - GUARD_DIMENSION_MM},
        "center_offset": {"value": measurement.offset_mm, "limit": tolerances.max_center_offset_mm,
                          "ok": measurement.offset_mm <= tolerances.max_center_offset_mm - GUARD_OFFSET_MM},
        "rotation": {"value": measurement.rotation_deg, "limit": tolerances.max_rotation_deg,
                     "ok": abs(measurement.rotation_deg) <= tolerances.max_rotation_deg - GUARD_ANGLE_DEG},
        "safe_margin": {"value": measurement.min_margin_mm, "limit": tolerances.safe_margin_mm,
                        "ok": (not measurement.touches_border)
                        and measurement.min_margin_mm >= tolerances.safe_margin_mm + GUARD_MARGIN_MM},
    }
    for check in checks.values():
        check["value"] = round(float(check["value"]), 2)
    failed = set()
    if not checks["safe_margin"]["ok"]:
        failed.add(ERR_OUT_OF_BOUNDS_VIOLATION)
    if not (checks["width"]["ok"] and checks["height"]["ok"]):
        failed.add(ERR_DIMENSION_OUT_OF_SPEC)
    if not checks["rotation"]["ok"]:
        failed.add(ERR_ROTATION_ANGLE_EXCEEDED)
    if not checks["center_offset"]["ok"]:
        failed.add(ERR_CENTER_OFFSET_EXCEEDED)
    return [code for code in METROLOGY_PRIORITY if code in failed], checks


# ── sub-pixel edge refinement ────────────────────────────────────────────────
class EdgeSampler:
    """Bilinear samples of the live frame and the baseline at BEV mm positions."""

    def __init__(self, frame: np.ndarray, rectifier: MetricRectifier, baseline: BaselineModel):
        self.frame = frame
        self.rectifier = rectifier
        self.baseline = baseline

    @staticmethod
    def _sample(image: np.ndarray, points_px: np.ndarray) -> np.ndarray:
        out = cv2.remap(image, points_px.reshape(1, -1, 2), None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return out.reshape(-1, image.shape[2] if image.ndim == 3 else 1).astype(np.float32)

    def pair(self, points_mm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        camera = self.rectifier.camera_px_of_bev_mm(self.frame.shape, points_mm).astype(np.float32)
        live = self._sample(self.frame, camera)
        base = self._sample(self.baseline.camera_crop, camera - np.asarray(self.baseline.crop_origin, np.float32))
        return live, base


def _fit_line(s: np.ndarray, t: np.ndarray) -> Tuple[float, float]:
    """Least-squares ``t = alpha + beta * s``."""
    s_mean, t_mean = float(s.mean()), float(t.mean())
    ds = s - s_mean
    var = float(np.dot(ds, ds))
    beta = float(np.dot(ds, t - t_mean)) / var if var > 1e-9 else 0.0
    return t_mean - beta * s_mean, beta


def refine_rectangle(rect: RectMM, sampler: EdgeSampler) -> Tuple[Optional[RectMM], dict]:
    """:func:`refine_once` with the fast pass, else wide capture then settle."""
    fast, info = refine_once(rect, sampler, REFINE_FAST_PASS[1], REFINE_FAST_PASS[2], REFINE_FAST_PASS[0])
    if fast is not None and info["sides_refined"] == 4:
        shift = max(abs(fast.w - rect.w), abs(fast.h - rect.h), math.hypot(fast.cx - rect.cx, fast.cy - rect.cy))
        turn = abs(_fold(fast.angle - rect.angle))
        if shift <= REFINE_FAST_MAX_SHIFT_MM and turn <= REFINE_MAX_TURN_DEG:
            info.update({"turn_deg": round(turn, 3), "shift_mm": round(shift, 2), "passes": 1})
            return fast, info
    current, info = rect, {}
    for profiles, half_window, step in REFINE_PASSES:
        refined, info = refine_once(current, sampler, half_window, step, profiles)
        if refined is None:
            return None, info
        current = refined
    turn = abs(_fold(current.angle - rect.angle))
    shift = max(abs(current.w - rect.w), abs(current.h - rect.h), math.hypot(current.cx - rect.cx, current.cy - rect.cy))
    info.update({"turn_deg": round(turn, 3), "shift_mm": round(shift, 2), "passes": 1 + len(REFINE_PASSES)})
    if turn > REFINE_MAX_TURN_DEG or shift > REFINE_MAX_SHIFT_MM:
        return None, info
    return current, info


def refine_once(rect: RectMM, sampler: EdgeSampler, half_window: float, step: float,
                profiles: int) -> Tuple[Optional[RectMM], dict]:
    """Locate each side of ``rect`` to sub-millimetre and rebuild the rectangle.

    Along every profile the live/baseline difference ``d(t)`` is computed with
    a gain fitted on the outer part of that same profile (so a cast shadow or
    lamp fall-off next to the article cancels out).  The edge is where ``d``
    falls fastest going outwards - the middle of a blurred step, or the outer
    flank of the dark edge line of a camouflaged carton.  A robust line is
    fitted per side; adjacent lines are intersected into the new corners.
    """
    corners = rect.corners()
    center = np.asarray([rect.cx, rect.cy])
    t = np.arange(-half_window, half_window + 0.5 * step, step, dtype=np.float32)
    fraction = np.linspace(-0.8, 0.8, profiles)
    mids, alongs, normals, positions = [], [], [], []
    for side in range(4):
        a, b = corners[side], corners[(side + 1) % 4]
        length = float(np.hypot(*(b - a)))
        along = (b - a) / max(length, 1e-6)
        normal = np.asarray([-along[1], along[0]])
        mid = (a + b) / 2.0
        if np.dot(normal, mid - center) < 0:
            normal = -normal
        mids.append(mid)
        alongs.append(along)
        normals.append(normal)
        positions.append(fraction * length / 2.0)
    mids, alongs, normals = np.asarray(mids), np.asarray(alongs), np.asarray(normals)
    s = np.asarray(positions)                                                     # (4, K)
    points = (mids[:, None, None, :] + s[:, :, None, None] * alongs[:, None, None, :]
              + t[None, None, :, None] * normals[:, None, None, :])              # (4, K, M, 2)
    live, base = sampler.pair(points.reshape(-1, 2))
    shape = points.shape[:3] + (live.shape[1],)
    live, base = live.reshape(shape), base.reshape(shape)

    outer = t > 0.4 * half_window
    gain = np.mean(live[:, :, outer, :], axis=2) / np.maximum(np.mean(base[:, :, outer, :], axis=2), 4.0)  # (4, K, C)
    d = np.max(np.abs(live / np.maximum(gain[:, :, None, :], 0.05) - base), axis=3)       # (4, K, M)
    ds = (d[..., :-2] + 2.0 * d[..., 1:-1] + d[..., 2:]) / 4.0                            # t[1:-1]
    g = ds[..., :-2] - ds[..., 2:]                                                         # t[2:-2]
    index = np.argmax(g, axis=-1)
    peak = np.take_along_axis(g, index[..., None], -1)[..., 0]
    outer_d = d[..., outer]
    noise = 1.2533 * np.mean(np.abs(outer_d - outer_d.mean(axis=-1, keepdims=True)), axis=-1)
    # ``g`` spans two samples: scale the noise floor with the step.
    valid = (peak >= np.maximum(6.0, 4.0 * noise)) & (index > 0) & (index < g.shape[-1] - 1)
    i0 = np.clip(index - 1, 0, g.shape[-1] - 1)
    i2 = np.clip(index + 1, 0, g.shape[-1] - 1)
    y0 = np.take_along_axis(g, i0[..., None], -1)[..., 0]
    y2 = np.take_along_axis(g, i2[..., None], -1)[..., 0]
    curvature = y0 - 2.0 * peak + y2
    delta = np.where(curvature < 0, 0.5 * (y0 - y2) / np.where(curvature < 0, curvature, -1.0), 0.0)
    edge_t = t[2:-2][index] + np.clip(delta, -0.5, 0.5) * step                             # (4, K)

    lines, support = [], []
    minimum = REFINE_MIN_SIDE_SUPPORT * profiles
    for side in range(4):
        mask = valid[side]
        alpha, beta, ok = 0.0, 0.0, False
        if mask.sum() >= minimum:
            ss, tt = s[side][mask], edge_t[side][mask]
            alpha, beta = _fit_line(ss, tt)
            for _ in range(2):
                residual = np.abs(tt - (alpha + beta * ss))
                keep = residual <= max(0.75, 3.0 * 1.4826 * float(np.partition(residual, len(residual) // 2)[len(residual) // 2]))
                if keep.sum() < minimum or keep.all():
                    break
                ss, tt = ss[keep], tt[keep]
                alpha, beta = _fit_line(ss, tt)
            ok = True
            support.append(int(len(ss)))
        else:
            support.append(int(mask.sum()))
        point = mids[side] + alpha * normals[side]
        direction = alongs[side] + beta * normals[side]
        lines.append((point, direction / np.hypot(*direction), ok))

    info = {"sides_refined": int(sum(1 for line in lines if line[2])), "support": support}
    if info["sides_refined"] < 3:
        return None, info
    new_corners = []
    for side in range(4):
        p1, d1, _ = lines[(side - 1) % 4]
        p2, d2, _ = lines[side]
        matrix = np.asarray([d1, -d2]).T
        if abs(np.linalg.det(matrix)) < 1e-6:
            return None, info
        k = np.linalg.solve(matrix, p2 - p1)
        new_corners.append(p1 + k[0] * d1)
    return RectMM.from_corners(np.asarray(new_corners)), info


# ── foreground ───────────────────────────────────────────────────────────────
@dataclass
class CVResult:
    verdict: str
    object_present: bool
    main: Optional[Blob] = None
    measurement: Optional[Measurement] = None
    errors: List[str] = field(default_factory=list)
    checks: Dict[str, dict] = field(default_factory=dict)
    extra_blobs: List[Blob] = field(default_factory=list)
    refinement: Dict[str, object] = field(default_factory=dict)
    foreground_ratio: float = 0.0
    shadow_ratio: float = 0.0
    glare_ratio: float = 0.0
    glare_unresolved: bool = False
    profile: str = "normal"
    luma_threshold: float = 0.0
    noise_luma: float = 0.0
    elapsed_ms: float = 0.0
    # Working data reused by the arbitration gate (never serialised).
    live: Optional[LabPlanes] = None
    pair: Optional[NormalizedPair] = None
    hp_live: Optional["_LazyBandPass"] = None
    mask: Optional[np.ndarray] = None

    def as_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "object_present": self.object_present,
            "measurement": self.measurement.as_dict() if self.measurement else None,
            "errors": list(self.errors),
            "checks": self.checks,
            "source": self.main.source if self.main else None,
            "bbox_mm": self.main.bbox_mm if self.main else None,
            "extra_objects": [{"bbox_mm": blob.bbox_mm, "area_mm2": round(blob.area_mm2, 1)} for blob in self.extra_blobs],
            "refinement": self.refinement,
            "foreground_ratio": round(self.foreground_ratio, 4),
            "shadow_ratio": round(self.shadow_ratio, 4),
            "glare_ratio": round(self.glare_ratio, 4),
            "glare_unresolved": self.glare_unresolved,
            "light_profile": self.profile,
            "luma_threshold": round(self.luma_threshold, 2),
            "noise_luma": round(self.noise_luma, 2),
            "elapsed_ms": round(self.elapsed_ms, 3),
        }


class _LazyBandPass:
    """Band-pass of the live luminance, computed only where and when needed."""

    def __init__(self, image: np.ndarray):
        self.image = image
        self._full: Optional[np.ndarray] = None

    def full(self) -> np.ndarray:
        if self._full is None:
            self._full = _high_pass(self.image)
        return self._full

    def window(self, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        if self._full is not None:
            return self._full[y0:y1, x0:x1]
        pad = int(3 * BAND_PASS_SIGMAS[1]) + 1
        Y0, X0 = max(0, y0 - pad), max(0, x0 - pad)
        Y1, X1 = min(self.image.shape[0], y1 + pad), min(self.image.shape[1], x1 + pad)
        return _high_pass(self.image[Y0:Y1, X0:X1])[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0]


def _local_ncc(hp_live: np.ndarray, hp_base: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    size = (NCC_WINDOW, NCC_WINDOW)
    cross = cv2.boxFilter(hp_live * hp_base, -1, size)
    energy_live = cv2.boxFilter(hp_live * hp_live, -1, size)
    energy_base = cv2.boxFilter(hp_base * hp_base, -1, size)
    ncc = cross / np.sqrt(energy_live * energy_base + 1e-6)
    return ncc, np.sqrt(energy_base)


def light_chroma_ok(pair: NormalizedPair, where: np.ndarray) -> np.ndarray:
    """Pixels of ``where`` whose colour change is explainable by light alone."""
    ok = where & (pair.diff_ab <= pair.chroma_threshold)
    rest = where & ~ok
    if rest.any() and pair.live_bgr is not None and pair.base_bgr is not None:
        live = pair.live_bgr[rest].astype(np.float32)
        base = pair.base_bgr[rest].astype(np.float32)
        live /= live.sum(axis=1, keepdims=True) + 6.0
        base /= base.sum(axis=1, keepdims=True) + 6.0
        ok[rest] = np.abs(live - base).max(axis=1) * 255.0 <= SHADOW_CHROMATICITY_MAX
    return ok


def illumination_mask(pair: NormalizedPair, hp_live, baseline: BaselineModel,
                      candidates: np.ndarray) -> np.ndarray:
    """Pixels whose change is explained by light only (shadow or highlight).

    The *core* needs every cue: unchanged chroma, a darkening/brightening
    ratio and the floor texture still there (local NCC of the band-pass;
    ``hp_live(y0, y1, x0, x1)`` returns the live band-pass of a window).  The
    core is then closed and grown by the penumbra width, absorbing only
    pixels that keep the floor's chroma and a light-only ratio: the soft
    fringe of a shadow has no texture match yet is no object.
    """
    ratio = pair.rel_ratio
    in_range = ((ratio >= SHADOW_RATIO_RANGE[0]) & (ratio <= SHADOW_RATIO_RANGE[1])) | \
               ((ratio >= HIGHLIGHT_RATIO_RANGE[0]) & (ratio <= HIGHLIGHT_RATIO_RANGE[1]))
    possible = light_chroma_ok(pair, candidates & in_range)
    result = np.zeros(candidates.shape, bool)
    if np.count_nonzero(possible) < MIN_OBJECT_AREA_PX / 4:
        return result
    # Edge pixels of a coloured article also meet the cues one by one; a
    # shadow is a region, so scattered candidates never reach the NCC stage.
    seeds = cv2.morphologyEx(possible.astype(np.uint8), cv2.MORPH_OPEN, _KERNEL3)
    if np.count_nonzero(seeds) < MIN_OBJECT_AREA_PX / 4:
        return result
    ys, xs = np.nonzero(seeds)
    pad = NCC_WINDOW + PENUMBRA_PX
    y0, y1 = max(0, ys.min() - pad), min(candidates.shape[0], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(candidates.shape[1], xs.max() + pad + 1)
    ncc, base_std = _local_ncc(hp_live(y0, y1, x0, x1), baseline.high_pass[y0:y1, x0:x1])
    textured = base_std >= TEXTURE_MIN_STD
    core = (possible[y0:y1, x0:x1] & textured & (ncc >= SHADOW_LOCAL_NCC)).astype(np.uint8)
    if not core.any():
        return result
    zone = cv2.dilate(cv2.morphologyEx(core, cv2.MORPH_CLOSE, _KERNEL5), _KERNEL3, iterations=PENUMBRA_PX) > 0
    sub_ratio = ratio[y0:y1, x0:x1]
    light_only = ((sub_ratio >= LIGHT_RATIO_RANGE[0]) & (sub_ratio <= LIGHT_RATIO_RANGE[1])) | \
                 ((sub_ratio >= LIGHT_RATIO_RANGE[2]) & (sub_ratio <= LIGHT_RATIO_RANGE[3]))
    grow = np.zeros(candidates.shape, bool)
    grow[y0:y1, x0:x1] = candidates[y0:y1, x0:x1] & zone & light_only
    result[y0:y1, x0:x1] = light_chroma_ok(pair, grow)[y0:y1, x0:x1]
    return result


def blob_illumination_evidence(blob_mask: np.ndarray, pair: NormalizedPair, hp_live: np.ndarray,
                               baseline: BaselineModel) -> dict:
    """Blob-level Lab/texture test (plan Kịch bản 2, step 2)."""
    region = blob_mask > 0
    count = int(region.sum())
    if count == 0:
        return {"is_shadow": False, "reason": "empty"}
    ratio = float(np.median(pair.rel_ratio[region]))
    light_only = ((SHADOW_RATIO_RANGE[0] <= ratio <= SHADOW_RATIO_RANGE[1])
                  or (HIGHLIGHT_RATIO_RANGE[0] <= ratio <= HIGHLIGHT_RATIO_RANGE[1]))
    if not light_only:
        return {"is_shadow": False, "reason": "luma_ratio", "median_luma_ratio": round(ratio, 3)}
    chroma_ok = float(light_chroma_ok(pair, region).sum()) / count
    if chroma_ok < 0.5:
        # A coloured article: light alone never changes the floor's chromaticity.
        return {"is_shadow": False, "reason": "chroma", "light_chroma_fraction": round(chroma_ok, 3)}
    live_hp = (hp_live.full() if isinstance(hp_live, _LazyBandPass) else hp_live)[region]
    base_hp = baseline.high_pass[region]
    base_std = float(np.std(base_hp))
    denominator = math.sqrt(float(np.sum(live_hp * live_hp)) * float(np.sum(base_hp * base_hp))) + 1e-6
    ncc = float(np.sum(live_hp * base_hp)) / denominator
    textured = base_std >= TEXTURE_MIN_STD
    is_shadow = bool(textured and ncc >= SHADOW_BLOB_NCC)
    return {"is_shadow": is_shadow, "texture_ncc": round(ncc, 3), "light_chroma_fraction": round(chroma_ok, 3),
            "median_luma_ratio": round(ratio, 3), "floor_texture_std": round(base_std, 2), "textured_floor": textured}


def fringe_explained(blob_mask: np.ndarray, pair: NormalizedPair, baseline: BaselineModel,
                     shadow_distance: np.ndarray) -> bool:
    """Is a residue next to a confirmed shadow just its soft fringe?

    Two models compete on the blob's pixels.  *Fringe*: the floor under a
    light field that depends only on the distance ``d`` to the shadow core,
    ``L = L_base · k(d) + plane`` (``k`` a piecewise-linear spline).
    *Surface*: something smooth lying there, ``L = s(d) + plane``.  A 1-D
    function of ``d`` can never reproduce the 2-D floor texture, so the fringe
    model only wins where that texture is really visible through the light
    change - it cannot explain away a flat object.
    """
    region = blob_mask > 0
    distance = shadow_distance[region]
    if region.sum() < 12 or np.median(distance) > FRINGE_PX:
        return False
    if light_chroma_ok(pair, region).sum() * 2 < region.sum():
        return False
    ratio = float(np.median(pair.rel_ratio[region]))
    if not (LIGHT_RATIO_RANGE[0] <= ratio <= LIGHT_RATIO_RANGE[1] or LIGHT_RATIO_RANGE[2] <= ratio <= LIGHT_RATIO_RANGE[3]):
        return False
    live = cv2.GaussianBlur(pair.live_L, (0, 0), 1.0)[region].astype(np.float64)
    base = cv2.GaussianBlur(baseline.planes.L, (0, 0), 1.0)[region].astype(np.float64)
    ys, xs = np.nonzero(region)
    xs = (xs - xs.mean()) / max(1.0, xs.std())
    ys = (ys - ys.mean()) / max(1.0, ys.std())
    knots = np.arange(0.0, float(distance.max()) + 2.0, 2.0)
    hats = np.clip(1.0 - np.abs(distance[:, None] - knots[None, :]) / 2.0, 0.0, None)
    plane = np.stack([xs, ys], axis=1)
    fringe_design = np.hstack([hats * base[:, None], plane])
    surface_design = np.hstack([hats, plane])

    def rss(design: np.ndarray) -> float:
        coeff, *_ = np.linalg.lstsq(design, live, rcond=None)
        return float(np.sum((live - design @ coeff) ** 2))

    return rss(fringe_design) <= 0.5 * rss(surface_design)


def _blob(contour: np.ndarray, source: str, shape: Tuple[int, int]) -> Blob:
    return Blob(contour, float(cv2.contourArea(contour)), tuple(int(v) for v in cv2.boundingRect(contour)), source,
                (int(shape[0]), int(shape[1])))


def _blobs_from_mask(mask: np.ndarray, source: str = "background") -> List[Blob]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    blobs = [_blob(contour, source, mask.shape[:2]) for contour in contours]
    blobs = [blob for blob in blobs if blob.area_px >= MIN_OBJECT_AREA_PX]
    blobs.sort(key=lambda blob: blob.area_px, reverse=True)
    return blobs


def _glare_bridged_blobs(mask: np.ndarray, glare: np.ndarray) -> List[Blob]:
    """Re-join article pieces separated by a saturated specular streak.

    Glare pixels carry no information and are never foreground; pieces of
    foreground connected *through* glare belong to one article, whose extent
    is the convex hull of the real (non-glare) pieces - so a streak running
    on across the floor cannot inflate the box (target.md Test 2.2).
    """
    count, labels = cv2.connectedComponents(((mask > 0) | (glare > 0)).astype(np.uint8), connectivity=8)
    blobs = []
    for label in range(1, count):
        member = (labels == label) & (mask > 0)
        points = cv2.findNonZero(member.astype(np.uint8))
        if points is None or len(points) < MIN_OBJECT_AREA_PX:
            continue
        crossed = bool(((labels == label) & (glare > 0)).any())
        hull = cv2.convexHull(points) if crossed else None
        if hull is not None:
            blob = _blob(hull, "glare_bridged", mask.shape[:2])
        else:
            contours, _ = cv2.findContours(member.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            blob = _blob(max(contours, key=cv2.contourArea), "background", mask.shape[:2])
        if blob.area_px >= MIN_OBJECT_AREA_PX:
            blobs.append(blob)
    blobs.sort(key=lambda blob: blob.area_px, reverse=True)
    return blobs


class GeometryCVInspector:
    def inspect(self, live_bgr: np.ndarray, baseline: BaselineModel, config: InspectionConfig,
                brightness: Optional[float] = None, frame: Optional[np.ndarray] = None,
                rectifier: Optional[MetricRectifier] = None) -> CVResult:
        """``live_bgr`` is the cell at the analysis level; ``frame`` enables edge refinement."""
        started = time.perf_counter()
        live, profile, brightness = prepare_live(live_bgr, baseline.planes, config.light_profile, brightness)
        pair = normalize_pair(live, baseline.planes, profile, brightness)
        raw = (np.abs(pair.diff_L) > pair.luma_threshold) | (pair.diff_ab > pair.chroma_threshold)
        glare = pair.glare > 0
        raw &= ~glare
        total = float(raw.size)
        result = CVResult(CV_EMPTY, False, profile=pair.profile, luma_threshold=pair.luma_threshold,
                          noise_luma=pair.noise_luma, live=live, pair=pair,
                          glare_ratio=float(glare.sum()) / total)
        band = _LazyBandPass(pair.live_L)
        fringe = None
        shadow_distance = None
        if raw.any() and config.enable_shadow_filter:
            shadow = illumination_mask(pair, band.window, baseline, raw)
            result.shadow_ratio = float(shadow.sum()) / total
            raw &= ~shadow
            if shadow.any():
                fringe = cv2.dilate(shadow.astype(np.uint8), _KERNEL3, iterations=FRINGE_PX) > 0

        mask = raw.astype(np.uint8) * 255
        if fringe is not None and mask.any():
            near = np.where(fringe, mask, 0).astype(np.uint8)
            mask = np.where(fringe, cv2.morphologyEx(near, cv2.MORPH_OPEN, _KERNEL_FRINGE), mask).astype(np.uint8)
        if mask.any():
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _KERNEL3)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _KERNEL5)
        result.foreground_ratio = float(np.count_nonzero(mask)) / total
        blobs = _glare_bridged_blobs(mask, pair.glare) if glare.any() and mask.any() else _blobs_from_mask(mask)

        if blobs and config.enable_shadow_filter:
            kept = []
            for blob in blobs:
                blob_mask = blob.mask()
                if blob_illumination_evidence(blob_mask, pair, band, baseline)["is_shadow"]:
                    continue
                if fringe is not None and np.count_nonzero(fringe & (blob_mask > 0)) * 2 >= np.count_nonzero(blob_mask):
                    if shadow_distance is None:
                        shadow_distance = cv2.distanceTransform((~shadow).astype(np.uint8), cv2.DIST_L2, 3)
                    if fringe_explained(blob_mask, pair, baseline, shadow_distance):
                        continue
                kept.append(blob)
            blobs = kept
        result.hp_live = band

        solid = np.zeros_like(mask)
        if blobs:
            cv2.drawContours(solid, [blob.contour for blob in blobs], -1, 255, cv2.FILLED)
        result.mask = solid

        if glare.any():
            near_objects = cv2.dilate(solid, _KERNEL5, iterations=2) > 0
            loose = float((glare & ~near_objects).sum()) * WORK_MM_PER_PX ** 2
            result.glare_unresolved = loose >= GLARE_UNRESOLVED_AREA_MM2

        if blobs:
            sampler = EdgeSampler(frame, rectifier, baseline) if frame is not None and rectifier is not None else None
            self._attach(result, blobs[0], blobs[1:], config, sampler)
        result.elapsed_ms = (time.perf_counter() - started) * 1000.0
        return result

    @staticmethod
    def _attach(result: CVResult, main: Blob, extra: Sequence[Blob], config: InspectionConfig,
                sampler: Optional[EdgeSampler]) -> None:
        tolerances = Tolerances.from_config(config)
        result.object_present = True
        result.main = main
        result.extra_blobs = list(extra)
        rect = main.coarse_rect()
        touches = main.touches_border()
        area = main.area_mm2
        refined = None
        result.refinement = {"applied": False}
        fill = area / max(1.0, rect.w * rect.h)
        if sampler is not None and fill >= REFINE_MIN_FILL:
            refined, info = refine_rectangle(rect, sampler)
            result.refinement = {"applied": refined is not None, **info}
            if refined is not None and info.get("sides_refined") == 4:
                # All four edges were located on the camera frame, beyond the
                # cell border if need be: the geometry, not the coarse mask
                # touching the border, decides the clearance.
                touches = False
        result.measurement = measurement_from_rect(refined or rect, tolerances, touches, area, refined is not None)
        result.errors, result.checks = evaluate_measurement(result.measurement, tolerances)
        result.verdict = CV_NG if result.errors else CV_PASS

    def adopt(self, result: CVResult, blob: Blob, config: InspectionConfig,
              sampler: Optional[EdgeSampler] = None) -> None:
        """Make ``blob`` the measured article of ``result``."""
        others = [b for b in result.extra_blobs if b is not blob]
        self._attach(result, blob, others, config, sampler)

    # ── Kịch bản 1: AI-guided local re-scan ──────────────────────────────────
    def rescan_with_hint(self, result: CVResult, baseline: BaselineModel, hint_bbox_mm: Sequence[float],
                         config: InspectionConfig) -> Optional[Blob]:
        """Find a camouflaged article inside the AI box from new physical edges.

        Background subtraction found nothing, so the global intensity change is
        below threshold.  Inside the AI box the live image is searched for
        edges that are *not* in the baseline (Canny on the gain-normalised
        luminance) and, independently, for a local Otsu contrast threshold.
        The candidate must be rectangular and agree with the AI box.
        """
        if result.pair is None:
            return None
        x0, y0, x1, y1 = [float(v) / WORK_MM_PER_PX for v in hint_bbox_mm]
        pad_x, pad_y = max(5.0, 0.06 * (x1 - x0)), max(5.0, 0.06 * (y1 - y0))
        rx0, ry0 = int(max(0, x0 - pad_x)), int(max(0, y0 - pad_y))
        rows, cols = result.pair.diff_L.shape[:2]
        rx1, ry1 = int(min(cols, x1 + pad_x)), int(min(rows, y1 + pad_y))
        if rx1 - rx0 < 6 or ry1 - ry0 < 6:
            return None
        hint = (x0, y0, x1, y1)
        offset = np.asarray([[rx0, ry0]], dtype=np.int32)
        candidates: List[Blob] = []

        live_u8 = np.clip(result.pair.live_L, 0, 255).astype(np.uint8)[ry0:ry1, rx0:rx1]
        live_edges = cv2.Canny(cv2.GaussianBlur(live_u8, (3, 3), 0), 10, 30)
        # A carton's mechanical edge is a long straight segment; floor texture is
        # not.  A segment counts only if the live-minus-baseline profile across
        # it, averaged along its length, shows a coherent change: texture and
        # static structure (painted lines, joints) average out to nothing.
        span = min(x1 - x0, y1 - y0)
        segments = cv2.HoughLinesP(live_edges, 1, np.pi / 360, threshold=25,
                                   minLineLength=max(20, int(0.3 * span)), maxLineGap=4)
        endpoints = []
        diff = result.pair.diff_L
        offsets = np.arange(-3, 4, dtype=np.float64)
        for x_a, y_a, x_b, y_b in (segments.reshape(-1, 4) if segments is not None else []):
            a = np.asarray([x_a + rx0, y_a + ry0], np.float64)
            b = np.asarray([x_b + rx0, y_b + ry0], np.float64)
            length = float(np.hypot(*(b - a)))
            normal = np.asarray([-(b - a)[1], (b - a)[0]]) / max(length, 1e-6)
            along = np.linspace(0.1, 0.9, 40)[:, None, None]
            points = a + along * (b - a) + offsets[None, :, None] * normal
            xs = np.clip(np.round(points[..., 0]).astype(int), 0, cols - 1)
            ys = np.clip(np.round(points[..., 1]).astype(int), 0, rows - 1)
            if np.max(np.abs(diff[ys, xs].mean(axis=0))) < RESCAN_MIN_LINE_CONTRAST:
                continue
            endpoints.append((a, b))
        corners = _rectangle_from_segments(endpoints, 0.3 * span)
        if corners is not None:
            blob = _blob(np.round(corners).astype(np.int32).reshape(-1, 1, 2), "edge_rescan", (rows, cols))
            if blob.area_px >= MIN_OBJECT_AREA_PX:
                candidates.append(blob)

        local_diff = np.clip(np.abs(result.pair.diff_L[ry0:ry1, rx0:rx1]), 0, 255).astype(np.uint8)
        otsu, local = cv2.threshold(local_diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if otsu >= 2.5 * result.pair.noise_luma:
            local = cv2.morphologyEx(local, cv2.MORPH_OPEN, _KERNEL3)
            local = cv2.morphologyEx(local, cv2.MORPH_CLOSE, _KERNEL5)
            for blob in _blobs_from_mask(local, "otsu_rescan")[:1]:
                candidates.append(_blob(blob.contour + offset, "otsu_rescan", (rows, cols)))

        best, best_score = None, 0.0
        for blob in candidates:
            rect_w, rect_h = cv2.minAreaRect(blob.contour.reshape(-1, 2).astype(np.float32))[1]
            rectangularity = blob.area_px / max(1.0, (rect_w + 1) * (rect_h + 1))
            x, y, w, h = blob.bbox_px
            iou = _iou((x, y, x + w, y + h), hint)
            if rectangularity >= 0.80 and iou >= 0.40 and iou + rectangularity > best_score:
                best, best_score = blob, iou + rectangularity
        return best


def _rectangle_from_segments(segments: Sequence[Tuple[np.ndarray, np.ndarray]],
                             min_separation: float) -> Optional[np.ndarray]:
    """Four sides from straight edge segments → the rectangle's corners.

    Segments are split into two perpendicular direction families; in each
    family the outermost lines (smallest / largest offset along the family's
    normal) are the article's sides - the outer flank of an edge line is its
    footprint.  Adjacent sides are intersected, so segment lengths (which can
    run on along collinear floor texture) do not matter.
    """
    if len(segments) < 4:
        return None
    angles = np.asarray([math.degrees(math.atan2(*(b - a)[::-1])) % 180.0 for a, b in segments])
    lengths = np.asarray([float(np.hypot(*(b - a))) for a, b in segments])
    reference = angles[int(np.argmax(lengths))]
    families = []
    for target in (reference, (reference + 90.0) % 180.0):
        delta = np.abs(((angles - target) + 90.0) % 180.0 - 90.0)
        families.append([index for index in np.nonzero(delta < 12.0)[0]])
    lines = []
    for family, base_angle in zip(families, (reference, reference + 90.0)):
        if len(family) < 2:
            return None
        direction = np.asarray([math.cos(math.radians(base_angle)), math.sin(math.radians(base_angle))])
        normal = np.asarray([-direction[1], direction[0]])
        offsets = np.asarray([float(np.dot((segments[i][0] + segments[i][1]) / 2.0, normal)) for i in family])
        low, high = offsets.min(), offsets.max()
        if high - low < min_separation:
            return None
        for extreme in (low, high):
            members = [i for i, value in zip(family, offsets) if abs(value - extreme) <= 3.0]
            points = np.vstack([np.vstack(segments[i]) for i in members])
            mean = points.mean(axis=0)
            _, _, vt = np.linalg.svd(points - mean)
            lines.append((mean, vt[0]))
    # lines: A-low, A-high, B-low, B-high → corners A-low∩B-low, B-low∩A-high, A-high∩B-high, B-high∩A-low
    order = [(0, 2), (2, 1), (1, 3), (3, 0)]
    corners = []
    for i, j in order:
        (p1, d1), (p2, d2) = lines[i], lines[j]
        matrix = np.asarray([d1, -d2]).T
        if abs(np.linalg.det(matrix)) < 1e-6:
            return None
        k = np.linalg.solve(matrix, p2 - p1)
        corners.append(p1 + k[0] * d1)
    return np.asarray(corners)


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def iou_mm(a: Sequence[float], b: Sequence[float]) -> float:
    return _iou(a, b)
