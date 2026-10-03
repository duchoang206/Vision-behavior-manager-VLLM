"""Inspection configuration, presets and the shared result vocabulary.

``InspectionConfig`` mirrors the ``InspectionConfig`` TypeScript type of the
plan (§5.1) field for field so the Building view and the backend agree on one
schema.  Everything here is pure data: no OpenCV, no I/O.
"""

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator

# The rectified bird's-eye view covers the floor cell at 1 px = 1 mm; it is the
# metric reference frame (and the stored baseline image).  The cell size is
# configured per station (``roi_width_mm`` x ``roi_height_mm``); 1 m x 1 m is
# only the default.
BEV_SIZE_MM = 1000
DEFAULT_CELL_MM = (1000.0, 1000.0)
# Detection runs on a 4 mm/px pyramid level of the same view; the article's
# edges are then refined to sub-millimetre on the full-resolution camera frame
# (geometry_cv_inspector.refine_rectangle), which keeps a frame under 3 ms.
WORK_MM_PER_PX = 4
WORK_SIZE_PX = BEV_SIZE_MM // WORK_MM_PER_PX   # analysis size of the default 1 m cell

Mode = Literal["CV_ONLY", "AI_ONLY", "HYBRID"]
Preset = Literal["carton_800x600", "pallet_1200x1000", "pallet_1100x1100", "custom"]
LightProfile = Literal["adaptive", "low", "normal", "high"]

MODES = ("CV_ONLY", "AI_ONLY", "HYBRID")

PRESETS: Dict[str, Dict[str, Any]] = {
    "carton_800x600": {"label": "Thùng Carton (800x600)", "noun": "Thùng carton", "width_mm": 800, "height_mm": 600},
    "pallet_1200x1000": {"label": "Pallet EUR (1200x1000)", "noun": "Pallet", "width_mm": 1200, "height_mm": 1000},
    "pallet_1100x1100": {"label": "Pallet vuông (1100x1100)", "noun": "Pallet", "width_mm": 1100, "height_mm": 1100},
    "custom": {"label": "Tùy chỉnh", "noun": "Kiện hàng"},
}

# ── Final states ──────────────────────────────────────────────────────────────
STATUS_OK = "OK"
STATUS_NG = "NG"
STATUS_UNCERTAIN = "UNCERTAIN"
# AI_ONLY never certifies alignment (plan §2): a recognised object is reported
# as DETECTED, never as OK.
STATUS_DETECTED = "DETECTED"

SLOT_EMPTY = "EMPTY"
SLOT_OCCUPIED = "OCCUPIED"
SLOT_UNKNOWN = "UNKNOWN"

# Does the cell hold goods?  The station's headline answer (occupancy.py),
# reported next to the metrology verdict above.
OCC_CARFULL = "CARFULL"
OCC_EMPTY = "EMPTY"
OCC_UNKNOWN = "UNKNOWN"

# ── Result / error codes (target.md) ─────────────────────────────────────────
OK_PASS = "OK_PASS"
OK_EMPTY = "OK_EMPTY"
AI_DETECTED = "AI_DETECTED"
ERR_CENTER_OFFSET_EXCEEDED = "ERR_CENTER_OFFSET_EXCEEDED"
ERR_ROTATION_ANGLE_EXCEEDED = "ERR_ROTATION_ANGLE_EXCEEDED"
ERR_DIMENSION_OUT_OF_SPEC = "ERR_DIMENSION_OUT_OF_SPEC"
ERR_OUT_OF_BOUNDS_VIOLATION = "ERR_OUT_OF_BOUNDS_VIOLATION"
ERR_UNKNOWN_OBSTACLE_DETECTED = "ERR_UNKNOWN_OBSTACLE_DETECTED"
ERR_DEFECT_DAMAGED_CARGO = "ERR_DEFECT_DAMAGED_CARGO"
ERR_BLURRY_FRAME = "ERR_BLURRY_FRAME"
ERR_LIGHTING_OUT_OF_RANGE = "ERR_LIGHTING_OUT_OF_RANGE"
ERR_GLARE_SATURATION = "ERR_GLARE_SATURATION"
ERR_NO_BASELINE = "ERR_NO_BASELINE"
ERR_BASELINE_ROI_MISMATCH = "ERR_BASELINE_ROI_MISMATCH"
ERR_NO_FRAME = "ERR_NO_FRAME"
ERR_INVALID_ROI = "ERR_INVALID_ROI"
ERR_AI_UNAVAILABLE = "ERR_AI_UNAVAILABLE"
ERR_UNSTABLE_STATE = "ERR_UNSTABLE_STATE"
ERR_OPERATOR_REJECTED = "ERR_OPERATOR_REJECTED"
CV_MISSED_OBJECT = "CV_MISSED_OBJECT"
AI_MISSED_OBJECT = "AI_MISSED_OBJECT"

# Metrology errors in reporting priority: a box protruding over the cell edge
# is a collision hazard first, then a wrong article, then its placement.
METROLOGY_PRIORITY = (
    ERR_OUT_OF_BOUNDS_VIOLATION,
    ERR_DIMENSION_OUT_OF_SPEC,
    ERR_ROTATION_ANGLE_EXCEEDED,
    ERR_CENTER_OFFSET_EXCEEDED,
)


class InspectionConfig(BaseModel):
    """Per-rule inspection settings (plan §5.1)."""

    mode: Mode = "HYBRID"
    # Real size of the floor cell outlined by the four ROI vertices:
    # width along edge 1→2, length along edge 2→3.
    roi_width_mm: float = Field(1000.0, ge=200, le=10000, allow_inf_nan=False)
    roi_height_mm: float = Field(1000.0, ge=200, le=10000, allow_inf_nan=False)
    target_preset: Preset = "carton_800x600"
    width_mm: float = Field(800.0, gt=0, le=5000, allow_inf_nan=False)
    height_mm: float = Field(600.0, gt=0, le=5000, allow_inf_nan=False)
    tolerance_w_mm: float = Field(30.0, ge=0, le=1000, allow_inf_nan=False)
    tolerance_h_mm: float = Field(30.0, ge=0, le=1000, allow_inf_nan=False)
    max_center_offset_mm: float = Field(50.0, ge=0, le=1000, allow_inf_nan=False)
    max_rotation_deg: float = Field(5.0, ge=0, le=90, allow_inf_nan=False)
    safe_margin_mm: float = Field(30.0, ge=0, le=500, allow_inf_nan=False)
    light_profile: LightProfile = "adaptive"
    enable_ai_assisted_cv: bool = True
    enable_shadow_filter: bool = True
    temporal_window_frames: int = Field(3, ge=1, le=30)
    has_background_baseline: Optional[bool] = None
    # Monitor view display of the station (no effect on any decision).
    monitor_overlay: bool = True        # outline the cell on the camera tile
    monitor_label: bool = True          # occupancy label next to the outline
    monitor_object: bool = True         # outline what CV / AI found in the cell

    @field_validator("mode", mode="before")
    @classmethod
    def _upper_mode(cls, value):
        return str(value).upper() if isinstance(value, str) else value

    @model_validator(mode="after")
    def _apply_preset(self):
        preset = PRESETS.get(self.target_preset) or {}
        if "width_mm" in preset and "width_mm" not in self.model_fields_set:
            self.width_mm = float(preset["width_mm"])
        if "height_mm" in preset and "height_mm" not in self.model_fields_set:
            self.height_mm = float(preset["height_mm"])
        return self

    @property
    def article_noun(self) -> str:
        return (PRESETS.get(self.target_preset) or PRESETS["custom"])["noun"]

    @property
    def cell_mm(self) -> Tuple[float, float]:
        return float(self.roi_width_mm), float(self.roi_height_mm)


DISPLAY_FIELDS = frozenset({"monitor_overlay", "monitor_label", "monitor_object", "has_background_baseline"})


def parse_config(raw: Optional[Dict[str, Any]]) -> InspectionConfig:
    return InspectionConfig.model_validate(raw or {})


def decision_settings(config: InspectionConfig) -> Dict[str, Any]:
    """The fields that change a verdict (display-only fields left out)."""
    return config.model_dump(exclude=set(DISPLAY_FIELDS))


@dataclass(frozen=True)
class Tolerances:
    """Flat view of the numeric limits used by the metrology engine."""

    width_mm: float
    height_mm: float
    tolerance_w_mm: float
    tolerance_h_mm: float
    max_center_offset_mm: float
    max_rotation_deg: float
    safe_margin_mm: float
    cell_w_mm: float = DEFAULT_CELL_MM[0]
    cell_h_mm: float = DEFAULT_CELL_MM[1]

    @classmethod
    def from_config(cls, config: InspectionConfig) -> "Tolerances":
        return cls(config.width_mm, config.height_mm, config.tolerance_w_mm, config.tolerance_h_mm,
                   config.max_center_offset_mm, config.max_rotation_deg, config.safe_margin_mm,
                   float(config.roi_width_mm), float(config.roi_height_mm))
