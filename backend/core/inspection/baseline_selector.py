"""Pick, per frame, the empty-cell baseline whose lighting matches the live cell.

The operator captures the empty cell under each lighting condition the
station meets (morning sun, lamps only, night shift ...).  Each baseline is
compared with the live cell through the detector's own relative normalisation
(gain field from the border ring) and the one leaving the fewest changed
pixels at the *fixed* base thresholds wins.  Fixed thresholds matter: the
detector's thresholds grow with the residual noise on the ring, so a badly
matching baseline would look quiet by hiding everything - the article too.

An article changes the cell against every baseline alike (none of them holds
it), so the choice follows the lighting, never the article.
"""

from typing import List, Optional, Sequence, Tuple

import numpy as np

from .config import InspectionConfig
from .geometry_cv_inspector import BaselineModel
from .illumination_normalizer import (BASE_CHROMA_THRESHOLD, BASE_LUMA_THRESHOLD, LabPlanes, normalize_pair,
                                      prepare_live)

# The ranking only needs the lighting pattern: half the analysis resolution
# (8 mm per pixel) costs a quarter and ranks alike.
FIT_STEP = 2


def _half(planes: LabPlanes) -> LabPlanes:
    return LabPlanes(*(np.ascontiguousarray(a[::FIT_STEP, ::FIT_STEP]) for a in (planes.L, planes.a, planes.b, planes.bgr)))


def _fit_planes(baseline: BaselineModel) -> LabPlanes:
    cached = baseline.__dict__.get("_fit_planes")
    if cached is None:
        cached = baseline.__dict__["_fit_planes"] = _half(baseline.planes)
    return cached


def baseline_fit(work_bgr: np.ndarray, baseline: BaselineModel, config: InspectionConfig,
                 brightness: Optional[float] = None) -> float:
    """Share of the cell that still differs from ``baseline`` after light normalisation."""
    base = _fit_planes(baseline)
    live, profile, brightness = prepare_live(np.ascontiguousarray(work_bgr[::FIT_STEP, ::FIT_STEP]), base,
                                             config.light_profile, brightness)
    pair = normalize_pair(live, base, profile, brightness)
    changed = (np.abs(pair.diff_L) > BASE_LUMA_THRESHOLD) | (pair.diff_ab > BASE_CHROMA_THRESHOLD)
    changed &= pair.glare == 0
    return float(np.count_nonzero(changed)) / float(changed.size)


def _describe(baseline: BaselineModel, score: Optional[float]) -> dict:
    return {"id": baseline.baseline_id, "label": baseline.label, "captured_at": baseline.captured_at or None,
            "score": None if score is None else round(score, 4)}


def select_baseline(work_bgr: np.ndarray, baselines: Sequence[BaselineModel], config: InspectionConfig,
                    brightness: Optional[float] = None) -> Tuple[BaselineModel, List[dict]]:
    """Best-matching baseline and the ranking (best first) of all candidates."""
    if len(baselines) == 1:
        return baselines[0], [_describe(baselines[0], None)]
    scored = sorted(((baseline_fit(work_bgr, b, config, brightness), index) for index, b in enumerate(baselines)))
    ranking = [_describe(baselines[index], score) for score, index in scored]
    return baselines[scored[0][1]], ranking
