"""Tầng 1 – Image Quality Assessment (IQA).

Blocks blurred, shaken or badly exposed frames before CV or AI can guess on
them (target.md Test 3.4).  Sharpness is the variance of the Laplacian of the
whole camera frame: blur and vibration affect every pixel, and a plain empty
floor cell alone would look "blurry" even on a perfect lens.  The frame is
first decimated to a fixed analysis width so the threshold does not depend on
the camera resolution, and the Laplacian is expressed at a reference mean
grey of 128: dimming the lights scales every gradient, and a dark but sharp
night frame must stay measurable (target.md Test 2.1).  Brightness is
measured on the rectified cell itself.
"""

from dataclasses import asdict, dataclass
from typing import Optional

import cv2
import numpy as np

from .config import ERR_BLURRY_FRAME, ERR_LIGHTING_OUT_OF_RANGE

MIN_LAPLACIAN_VAR = 80.0
MIN_BRIGHTNESS = 25.0
MAX_BRIGHTNESS = 235.0
ANALYSIS_WIDTH = 640
REFERENCE_GREY = 128.0


@dataclass
class ImageQualityReport:
    ok: bool
    laplacian_var: float
    brightness: float
    error_code: Optional[str] = None
    message: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def _gray(image: np.ndarray) -> np.ndarray:
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def laplacian_variance(frame: np.ndarray) -> float:
    """Brightness-normalised Laplacian variance at ``ANALYSIS_WIDTH``."""
    height, width = frame.shape[:2]
    if width > ANALYSIS_WIDTH:
        size = (ANALYSIS_WIDTH, max(1, int(round(height * ANALYSIS_WIDTH / float(width)))))
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_NEAREST)
    gray = _gray(frame)
    mean = max(1.0, float(cv2.mean(gray)[0]))
    _, std = cv2.meanStdDev(cv2.Laplacian(gray, cv2.CV_16S))
    return float(std[0, 0]) ** 2 * (REFERENCE_GREY / mean) ** 2


def mean_brightness(image: np.ndarray) -> float:
    return float(cv2.mean(_gray(image))[0])


def check_image_quality(frame: np.ndarray, region: Optional[np.ndarray] = None,
                        min_laplacian: float = MIN_LAPLACIAN_VAR,
                        min_brightness: float = MIN_BRIGHTNESS,
                        max_brightness: float = MAX_BRIGHTNESS) -> ImageQualityReport:
    """Return whether ``frame`` is usable.

    ``region`` is the rectified ROI; when given its brightness is checked,
    otherwise the whole frame is used.
    """
    if frame is None or frame.size == 0:
        return ImageQualityReport(False, 0.0, 0.0, ERR_BLURRY_FRAME, "Không có khung hình")
    sharpness = laplacian_variance(frame)
    brightness = mean_brightness(region if region is not None and region.size else frame)
    if sharpness < min_laplacian:
        return ImageQualityReport(False, sharpness, brightness, ERR_BLURRY_FRAME,
                                  f"Khung hình mờ/rung: Laplacian {sharpness:.1f} < {min_laplacian:.0f}")
    if brightness < min_brightness:
        return ImageQualityReport(False, sharpness, brightness, ERR_LIGHTING_OUT_OF_RANGE,
                                  f"Quá tối: độ sáng {brightness:.1f} < {min_brightness:.0f}")
    if brightness > max_brightness:
        return ImageQualityReport(False, sharpness, brightness, ERR_LIGHTING_OUT_OF_RANGE,
                                  f"Quá chói: độ sáng {brightness:.1f} > {max_brightness:.0f}")
    return ImageQualityReport(True, sharpness, brightness)
