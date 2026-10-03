"""Tầng 2 – Illumination normalisation between the live cell and its baseline.

Lighting in a warehouse changes globally (day/night shift, lamps switched
off) and smoothly (fall-off across the cell).  Normalising each image on its
own would also flatten a large uniform carton, so the model is relative: the
live luminance is divided by a smooth *gain field* ``G = L_live / L_baseline``
that is estimated only from pixels which still look like the background: the
dominant mode of the ratio histogram on the cell's border ring (an article
standing in the cell can cover at most part of it), then a coarse grid of
inlier means, Gaussian-smoothed.  This is the Gaussian background division of
the plan made object-safe.

Light profiles (target.md Bộ 2):

* **A – low** (brightness < 70): the live cell is brightened to the baseline
  level in BGR *before* the Lab conversion (dark colours lose their a/b
  chroma, which is often the only cue separating a brown carton from grey
  concrete), then bilateral-filtered; thresholds scale with the measured noise
  so the darker image keeps its contours.
* **B – normal**.
* **C – high** (brightness > 170): saturated specular pixels are masked with a
  lower clip level so a glare streak cannot grow the object outline.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

LOW_LIGHT_MAX = 70.0
HIGH_LIGHT_MIN = 170.0
GRID_CELLS = 8
INLIER_LOG_RATIO = 0.12
MAX_LOCAL_GAIN_DEVIATION = 0.10
SATURATION_L = 250
SATURATION_L_HIGH_PROFILE = 240
BASE_LUMA_THRESHOLD = 18.0
BASE_CHROMA_THRESHOLD = 10.0
NOISE_SIGMAS = 4.0


def select_light_profile(brightness: float, requested: str = "adaptive") -> str:
    if requested in ("low", "normal", "high"):
        return requested
    if brightness < LOW_LIGHT_MAX:
        return "low"
    if brightness > HIGH_LIGHT_MIN:
        return "high"
    return "normal"


@dataclass
class LabPlanes:
    """Lab planes of one rectified image (OpenCV 8-bit Lab: L∈[0,255], a/b offset 128)."""

    L: np.ndarray        # float32
    a: np.ndarray        # float32
    b: np.ndarray        # float32
    bgr: np.ndarray      # uint8 source

    @classmethod
    def from_bgr(cls, bgr: np.ndarray) -> "LabPlanes":
        L, a, b = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab))
        return cls(L.astype(np.float32), a.astype(np.float32), b.astype(np.float32), bgr)

    @property
    def gray(self) -> np.ndarray:
        cached = self.__dict__.get("_gray")
        if cached is None:
            cached = self.__dict__["_gray"] = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2GRAY)
        return cached


@dataclass
class NormalizedPair:
    profile: str
    brightness: float
    global_gain: float
    gain: np.ndarray        # float32 smooth gain field (live / baseline)
    live_L: np.ndarray      # float32 live luminance after gain division (and de-noising)
    diff_L: np.ndarray      # float32 live_L - baseline L
    diff_ab: np.ndarray     # float32 max(|Δa|, |Δb|)
    rel_ratio: np.ndarray   # float32 (raw live / baseline) / gain, 1.0 == unchanged
    glare: np.ndarray       # uint8 0/255 saturated specular pixels
    noise_luma: float
    noise_chroma: float
    luma_threshold: float
    chroma_threshold: float
    live_bgr: Optional[np.ndarray] = None
    base_bgr: Optional[np.ndarray] = None


def _robust_mode(values: np.ndarray, low: float = -2.5, high: float = 2.5, bins: int = 250) -> float:
    width = (high - low) / bins
    index = ((values - low) / width).astype(np.int32)
    hist = np.bincount(index[(index >= 0) & (index < bins)], minlength=bins).astype(np.float32)
    hist = np.convolve(hist, np.asarray([1.0, 2.0, 1.0], dtype=np.float32), mode="same")
    return float(low + (int(np.argmax(hist)) + 0.5) * width)


def _ring(values: np.ndarray, width: int) -> np.ndarray:
    return np.concatenate([values[:width].ravel(), values[-width:].ravel(),
                           values[width:-width, :width].ravel(), values[width:-width, -width:].ravel()])


def _gain_field(log_ratio_sub: np.ndarray, mode: float, shape) -> np.ndarray:
    """Smooth multiplicative gain (live / baseline) at full analysis size."""
    cells = GRID_CELLS
    block_h, block_w = max(1, log_ratio_sub.shape[0] // cells), max(1, log_ratio_sub.shape[1] // cells)
    sub = log_ratio_sub[:block_h * cells, :block_w * cells]
    inlier = (np.abs(sub - mode) < INLIER_LOG_RATIO).astype(np.float32)
    sums = (sub * inlier).reshape(cells, block_h, cells, block_w).sum(axis=(1, 3))
    counts = inlier.reshape(cells, block_h, cells, block_w).sum(axis=(1, 3))
    grid = np.where(counts >= 0.4 * block_h * block_w, sums / np.maximum(counts, 1.0), mode).astype(np.float32)
    grid = np.clip(grid, mode - MAX_LOCAL_GAIN_DEVIATION, mode + MAX_LOCAL_GAIN_DEVIATION)
    grid = np.exp(cv2.GaussianBlur(grid, (3, 3), 0.8, borderType=cv2.BORDER_REPLICATE)).astype(np.float32)
    return cv2.resize(grid, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


def _ring_gain(live_gray: np.ndarray, base_gray: np.ndarray) -> float:
    ratio = np.log((live_gray[::2, ::2].astype(np.float32) + 4.0) / (base_gray[::2, ::2].astype(np.float32) + 4.0))
    return float(np.exp(_robust_mode(_ring(ratio, max(2, ratio.shape[0] // 25)))))


def prepare_live(live_bgr: np.ndarray, baseline: LabPlanes, requested_profile: str = "adaptive",
                 brightness: Optional[float] = None) -> Tuple[LabPlanes, str, float]:
    """Select the light profile and return the live Lab planes ready for comparison."""
    if brightness is None:
        brightness = float(cv2.mean(cv2.cvtColor(live_bgr, cv2.COLOR_BGR2GRAY))[0])
    profile = select_light_profile(brightness, requested_profile)
    if profile != "low":
        return LabPlanes.from_bgr(live_bgr), profile, brightness
    gain = _ring_gain(cv2.cvtColor(live_bgr, cv2.COLOR_BGR2GRAY), baseline.gray)
    brightened = LabPlanes.from_bgr(cv2.convertScaleAbs(live_bgr, alpha=1.0 / max(gain, 0.05)))
    # Edge-preserving on lightness (contours), plain smoothing on the noisier chroma.
    brightened.L = cv2.bilateralFilter(brightened.L, 5, 12.0, 3.0)
    brightened.a = cv2.GaussianBlur(brightened.a, (5, 5), 1.2)
    brightened.b = cv2.GaussianBlur(brightened.b, (5, 5), 1.2)
    return brightened, profile, brightness


def normalize_pair(live: LabPlanes, baseline: LabPlanes, profile: str = "normal",
                   brightness: float = 0.0) -> NormalizedPair:
    """Express the live cell in the baseline's illumination."""
    live_L = live.L
    raw_ratio = (live_L + 4.0) / (baseline.L + 4.0)
    sub = np.log(raw_ratio[::2, ::2])
    ring = max(2, sub.shape[0] // 25)
    mode = _robust_mode(_ring(sub, ring))
    gain = _gain_field(sub, mode, live_L.shape)

    inverse_gain = 1.0 / gain
    normalized = live_L * inverse_gain
    diff_L = normalized - baseline.L
    diff_ab = cv2.max(cv2.absdiff(live.a, baseline.a), cv2.absdiff(live.b, baseline.b))
    rel_ratio = raw_ratio * inverse_gain

    # Noise from the border ring only: an article whose lightness matches the
    # floor passes the ratio test and would otherwise inflate the chroma noise.
    inlier = _ring(np.abs(sub - mode) < INLIER_LOG_RATIO, ring)
    sample_luma = _ring(diff_L[::2, ::2], ring)[inlier]
    sample_chroma = _ring(diff_ab[::2, ::2], ring)[inlier]
    noise_luma = max(1.0, 1.4826 * float(np.median(np.abs(sample_luma))) if sample_luma.size else 1.0)
    noise_chroma = max(1.0, 1.4826 * float(np.median(sample_chroma)) if sample_chroma.size else 1.0)

    clip = SATURATION_L_HIGH_PROFILE if profile == "high" else SATURATION_L
    glare = ((live.L >= clip) & (baseline.L < clip - 15)).astype(np.uint8) * 255
    if glare.any():
        glare = cv2.dilate(glare, np.ones((3, 3), np.uint8))

    return NormalizedPair(
        profile=profile, brightness=brightness, global_gain=float(np.exp(mode)), gain=gain,
        live_L=normalized, diff_L=diff_L, diff_ab=diff_ab, rel_ratio=rel_ratio, glare=glare,
        noise_luma=noise_luma, noise_chroma=noise_chroma,
        luma_threshold=max(BASE_LUMA_THRESHOLD, NOISE_SIGMAS * noise_luma),
        chroma_threshold=max(BASE_CHROMA_THRESHOLD, NOISE_SIGMAS * noise_chroma),
        live_bgr=live.bgr, base_bgr=baseline.bgr,
    )


def enhance_for_ai(bgr: np.ndarray, profile: str) -> np.ndarray:
    """Local Lab brightening for the AI branch in low light (CLAHE on L)."""
    if profile != "low":
        return bgr
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    L, a, b = cv2.split(lab)
    L = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(L)
    return cv2.cvtColor(cv2.merge((L, a, b)), cv2.COLOR_Lab2BGR)
