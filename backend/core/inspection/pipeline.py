"""One frame through the seven inspection layers (plan §6).

``InspectionPipeline.run`` is shared by the Building view's "Test thử" button
and the realtime runtime, so what the operator validates is exactly what
drives the AGV.
"""

import base64
import threading
import time
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from .ai_defect_inspector import AIDefectInspector, AIResult
from .arbitration_gate import ArbitrationContext, ArbitrationGate, Verdict, describe
from .baseline_selector import select_baseline
from .config import (DEFAULT_CELL_MM, ERR_BASELINE_ROI_MISMATCH, ERR_INVALID_ROI, ERR_NO_BASELINE, ERR_NO_FRAME,
                     OCC_CARFULL, OCC_EMPTY, OCC_UNKNOWN, SLOT_UNKNOWN, STATUS_DETECTED, STATUS_NG, STATUS_OK, STATUS_UNCERTAIN, WORK_MM_PER_PX,
                     InspectionConfig)
from .geometry_cv_inspector import (GUARD_ANGLE_DEG, GUARD_DIMENSION_MM, GUARD_MARGIN_MM, GUARD_OFFSET_MM,
                                    BaselineModel, CVResult, EdgeSampler, GeometryCVInspector,
                                    blob_illumination_evidence)
from .homography_rectifier import InvalidRoiError, MetricRectifier, order_clockwise
from .illumination_normalizer import enhance_for_ai
from .image_quality import check_image_quality
from .live_detections import LiveDetections, cell_detections, merge_live
from .occupancy import decide_occupancy, unknown_decision

AI_MM_PER_PX = 2.0
AI_REGION_PAD_MM = 60.0
OVERLAY_MM_PER_PX = 2
STATUS_COLORS = {STATUS_OK: (60, 200, 60), STATUS_NG: (40, 40, 235), STATUS_UNCERTAIN: (0, 200, 255),
                 STATUS_DETECTED: (230, 200, 40)}
OCCUPANCY_COLORS = {OCC_CARFULL: (94, 63, 244), OCC_EMPTY: (129, 185, 16), OCC_UNKNOWN: (11, 158, 245)}


def _padded(bbox: Sequence[float], pad: float, cell: Tuple[float, float]) -> List[float]:
    return [max(0.0, bbox[0] - pad), max(0.0, bbox[1] - pad),
            min(float(cell[0]), bbox[2] + pad), min(float(cell[1]), bbox[3] + pad)]


def baseline_matches(baseline: BaselineModel, points: Sequence[Sequence[float]], frame_shape,
                     cell_mm: Tuple[float, float] = DEFAULT_CELL_MM) -> bool:
    if baseline.points is None:
        return True
    if tuple(baseline.frame_shape[:2]) != tuple(int(v) for v in frame_shape[:2]):
        return False
    if any(abs(a - b) > 0.5 for a, b in zip(baseline.cell_mm, cell_mm)):
        return False
    try:
        return bool(np.allclose(order_clockwise(baseline.points), order_clockwise(points), atol=2e-3))
    except InvalidRoiError:
        return False


def effective_limits(config: InspectionConfig) -> dict:
    """The limits actually applied, guard bands included (reported for transparency)."""
    return {
        "width_mm": [config.width_mm - config.tolerance_w_mm + GUARD_DIMENSION_MM,
                     config.width_mm + config.tolerance_w_mm - GUARD_DIMENSION_MM],
        "height_mm": [config.height_mm - config.tolerance_h_mm + GUARD_DIMENSION_MM,
                      config.height_mm + config.tolerance_h_mm - GUARD_DIMENSION_MM],
        "max_center_offset_mm": config.max_center_offset_mm - GUARD_OFFSET_MM,
        "max_rotation_deg": config.max_rotation_deg - GUARD_ANGLE_DEG,
        "min_safe_margin_mm": config.safe_margin_mm + GUARD_MARGIN_MM,
        "guard_bands": {"dimension_mm": GUARD_DIMENSION_MM, "offset_mm": GUARD_OFFSET_MM,
                        "angle_deg": GUARD_ANGLE_DEG, "margin_mm": GUARD_MARGIN_MM},
    }


class InspectionPipeline:
    def __init__(self, ai: Optional[AIDefectInspector] = None):
        self.ai = ai if ai is not None else AIDefectInspector(None)
        self.cv = GeometryCVInspector()
        self.gate = ArbitrationGate()
        self._rectifiers: Dict[Tuple[float, ...], MetricRectifier] = {}
        self._lock = threading.Lock()

    def rectifier(self, points: Sequence[Sequence[float]],
                  cell_mm: Tuple[float, float] = DEFAULT_CELL_MM) -> MetricRectifier:
        key = tuple(round(float(v), 6) for point in points for v in point[:2]) + tuple(float(v) for v in cell_mm)
        with self._lock:
            rect = self._rectifiers.get(key)
            if rect is None:
                rect = MetricRectifier(points, cell_mm)
                if len(self._rectifiers) > 64:
                    self._rectifiers.clear()
                self._rectifiers[key] = rect
            return rect

    # ── result helpers ───────────────────────────────────────────────────────
    @staticmethod
    def _result(verdict: Verdict, config: InspectionConfig, timing: dict, **extra) -> dict:
        report = {
            "status": verdict.status, "slot_state": verdict.slot_state, "code": verdict.code,
            "message": verdict.message, "errors": verdict.errors, "mode": verdict.mode,
            "ai_class": verdict.ai_class, "ai_confidence": verdict.ai_confidence,
            "arbitration": {"dispute": verdict.dispute, "resolution": verdict.resolution, "steps": verdict.steps},
            "limits": effective_limits(config), "timing_ms": {k: round(v, 3) for k, v in timing.items()},
            "timestamp": time.time(), "measurement": None, "checks": {}, "iqa": None, "cv": None, "ai": None,
            "overlay": None, "occupancy": None, "baseline_choice": None,
        }
        report.update(extra)
        return report

    def _early(self, code: str, message: str, config: InspectionConfig, timing: dict, **extra) -> dict:
        verdict = Verdict(STATUS_UNCERTAIN, SLOT_UNKNOWN, code, message, config.mode, [code])
        extra.setdefault("occupancy", unknown_decision(message))
        return self._result(verdict, config, timing, **extra)

    # ── main entry ───────────────────────────────────────────────────────────
    def run(self, frame: Optional[np.ndarray], points: Sequence[Sequence[float]], config: InspectionConfig,
            baseline: Union[None, BaselineModel, Sequence[BaselineModel]], overlay: bool = False,
            targets: Optional[Sequence[str]] = None, live: Optional[LiveDetections] = None) -> dict:
        """``baseline``: one empty-cell baseline or all of the station's (the best match is used);
        ``targets``: the classes that count as goods; ``live``: the camera model's detections."""
        started = time.perf_counter()
        timing: Dict[str, float] = {}
        if frame is None or getattr(frame, "size", 0) == 0:
            return self._early(ERR_NO_FRAME, "Không lấy được khung hình từ camera", config, timing)
        cell = config.cell_mm
        try:
            rect = self.rectifier(points, cell)
            rect.matrix(frame.shape)
        except InvalidRoiError as exc:
            return self._early(ERR_INVALID_ROI, str(exc), config, timing)

        mark = time.perf_counter()
        work = rect.warp(frame, WORK_MM_PER_PX)
        timing["warp"] = (time.perf_counter() - mark) * 1000.0
        mark = time.perf_counter()
        iqa = check_image_quality(frame, work)
        timing["iqa"] = (time.perf_counter() - mark) * 1000.0
        if not iqa.ok:
            # Tầng 1 stops here: neither CV nor AI may guess on this frame.
            timing["total"] = (time.perf_counter() - started) * 1000.0
            return self._early(iqa.error_code, iqa.message, config, timing, iqa=iqa.as_dict())

        needs_cv = config.mode in ("CV_ONLY", "HYBRID")
        baselines = [b for b in (baseline if isinstance(baseline, (list, tuple)) else [baseline]) if b is not None]
        choice = None
        if needs_cv and not baselines:
            timing["total"] = (time.perf_counter() - started) * 1000.0
            return self._early(ERR_NO_BASELINE, "Chưa có ảnh nền ô trống – chụp ảnh nền khi ô trống hoàn toàn",
                               config, timing, iqa=iqa.as_dict())
        if needs_cv:
            usable = [b for b in baselines if baseline_matches(b, points, frame.shape, cell)]
            if not usable:
                timing["total"] = (time.perf_counter() - started) * 1000.0
                return self._early(ERR_BASELINE_ROI_MISMATCH,
                                   "ROI, kích thước ô hoặc độ phân giải camera đã đổi – cần chụp lại ảnh nền",
                                   config, timing, iqa=iqa.as_dict())
            mark = time.perf_counter()
            baseline, ranking = select_baseline(work, usable, config, iqa.brightness)
            timing["baseline_select"] = (time.perf_counter() - mark) * 1000.0
            choice = {**ranking[0], "count": len(baselines), "usable": len(usable), "candidates": ranking}

        cv_result: Optional[CVResult] = None
        if needs_cv:
            mark = time.perf_counter()
            cv_result = self.cv.inspect(work, baseline, config, iqa.brightness, frame, rect)
            timing["cv"] = (time.perf_counter() - mark) * 1000.0

        ai_result: Optional[AIResult] = None
        live_found = []
        profile = cv_result.profile if cv_result else "normal"
        if config.mode in ("AI_ONLY", "HYBRID"):
            mark = time.perf_counter()
            live_found = cell_detections(live, rect, frame.shape, targets)
            region = (_padded(cv_result.main.bbox_mm, AI_REGION_PAD_MM, cell) if cv_result and cv_result.object_present
                      else [0.0, 0.0, float(cell[0]), float(cell[1])])
            ai_result = merge_live(self._ai(frame, rect, region, profile, low=False), live, live_found)
            timing["ai"] = (time.perf_counter() - mark) * 1000.0

        def rescan(bbox):
            return self.cv.rescan_with_hint(cv_result, baseline, bbox, config)

        def adopt(blob):
            self.cv.adopt(cv_result, blob, config, EdgeSampler(frame, rect, baseline))

        second_results: List[AIResult] = []

        def second_pass(region):
            result = merge_live(self._ai(frame, rect, region, profile, low=True), live, live_found)
            second_results.append(result)
            return result

        def shadow(blob):
            return blob_illumination_evidence(blob.mask(), cv_result.pair, cv_result.hp_live, baseline)

        mark = time.perf_counter()
        verdict = self.gate.decide(config.mode, config, cv_result, ai_result,
                                   ArbitrationContext(rescan, adopt, second_pass, shadow))
        occupancy = decide_occupancy(config.mode, cv_result, ai_result, verdict, second_results)
        timing["arbitration"] = (time.perf_counter() - mark) * 1000.0
        timing["total"] = (time.perf_counter() - started) * 1000.0

        measurement = cv_result.measurement if cv_result else None
        report = self._result(
            verdict, config, timing, iqa=iqa.as_dict(),
            cv=cv_result.as_dict() if cv_result else None, ai=ai_result.as_dict() if ai_result else None,
            measurement=measurement.as_dict() if measurement else None,
            checks=cv_result.checks if cv_result else {}, occupancy=occupancy, baseline_choice=choice)
        report["overlay"] = self._overlay(frame, rect, report, cv_result, [ai_result, *second_results], overlay)
        return report

    def _ai(self, frame: np.ndarray, rect: MetricRectifier, region: Sequence[float], profile: str,
            low: bool) -> AIResult:
        if not self.ai.available:
            return self.ai.inspect(np.zeros((1, 1, 3), np.uint8), region, low)
        image = enhance_for_ai(rect.warp_region(frame, region, AI_MM_PER_PX), profile)
        return self.ai.inspect(image, region, low_sensitivity=low)

    # ── overlay ──────────────────────────────────────────────────────────────
    @staticmethod
    def _ai_detections(ai_results: Sequence[Optional[AIResult]]) -> list:
        seen, found = set(), []
        for result in ai_results:
            for det in (result.detections if result is not None and result.available else []):
                key = (det.class_name, det.source, tuple(round(v) for v in det.bbox_mm))
                if key not in seen:
                    seen.add(key)
                    found.append(det)
        return found

    def _overlay(self, frame: np.ndarray, rect: MetricRectifier, report: dict, cv_result: Optional[CVResult],
                 ai_results: Sequence[Optional[AIResult]], with_image: bool) -> dict:
        measurement = report.get("measurement")
        cell_center = rect.bev_to_camera_norm(frame.shape, [[rect.cell_mm[0] / 2.0, rect.cell_mm[1] / 2.0]])[0]
        data = {"cell_center": cell_center, "obb": None, "center": None, "ai_boxes": [], "objects": []}
        if measurement:
            data["obb"] = rect.bev_to_camera_norm(frame.shape, measurement["corners_mm"])
            data["center"] = rect.bev_to_camera_norm(frame.shape, [[measurement["center_x_mm"],
                                                                     measurement["center_y_mm"]]])[0]
        detections = self._ai_detections(ai_results)
        for det in detections:
            if det.camera_box:
                x0, y0, x1, y1 = det.camera_box
                polygon = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
            else:
                x0, y0, x1, y1 = det.bbox_mm
                polygon = rect.bev_to_camera_norm(frame.shape, [[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
            data["ai_boxes"].append({"label": f"{det.class_name} {det.confidence:.0%}", "role": det.role,
                                     "source": det.source, "polygon": polygon})
        occupancy = report.get("occupancy") or {}
        if cv_result is not None and cv_result.object_present and occupancy.get("cv_object"):
            for index, blob in enumerate([cv_result.main, *cv_result.extra_blobs]):
                if index == 0 and data["obb"]:
                    data["objects"].append(data["obb"])
                    continue
                x0, y0, x1, y1 = blob.bbox_mm
                data["objects"].append(rect.bev_to_camera_norm(frame.shape, [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]))
        if with_image:
            data["bev_image"] = self._bev_image(frame, rect, report, detections)
        return data

    @staticmethod
    def _bev_image(frame: np.ndarray, rect: MetricRectifier, report: dict, detections: Sequence) -> str:
        width, height = rect.size(OVERLAY_MM_PER_PX)
        image = rect.warp(frame, OVERLAY_MM_PER_PX)
        for mm in range(100, int(rect.cell_mm[0]), 100):
            cv2.line(image, (mm // OVERLAY_MM_PER_PX, 0), (mm // OVERLAY_MM_PER_PX, height), (90, 90, 90), 1)
        for mm in range(100, int(rect.cell_mm[1]), 100):
            cv2.line(image, (0, mm // OVERLAY_MM_PER_PX), (width, mm // OVERLAY_MM_PER_PX), (90, 90, 90), 1)
        color = STATUS_COLORS.get(report["status"], (255, 255, 255))
        center = (width // 2, height // 2)
        cv2.drawMarker(image, center, (255, 255, 255), cv2.MARKER_CROSS, 18, 1)
        for det in detections:
            x0, y0, x1, y1 = [int(v / OVERLAY_MM_PER_PX) for v in det.bbox_mm]
            cv2.rectangle(image, (x0, y0), (x1, y1), (230, 160, 40), 1)
            cv2.putText(image, f"{det.class_name} {det.confidence:.0%}", (x0 + 3, max(12, y0 + 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 160, 40), 1, cv2.LINE_AA)
        m = report.get("measurement")
        if m:
            corners = (np.asarray(m["corners_mm"], np.float64) / OVERLAY_MM_PER_PX * 16).astype(np.int32)
            cv2.polylines(image, [corners], True, color, 2, cv2.LINE_AA, 4)
            obj = (int(m["center_x_mm"] / OVERLAY_MM_PER_PX), int(m["center_y_mm"] / OVERLAY_MM_PER_PX))
            cv2.arrowedLine(image, center, obj, color, 2, cv2.LINE_AA, tipLength=0.25)
            cv2.circle(image, obj, 4, color, -1, cv2.LINE_AA)
            label = f"{m['width_mm']:.0f}x{m['height_mm']:.0f}mm  d={m['offset_mm']:.0f}mm  {m['rotation_deg']:+.1f}deg"
            cv2.putText(image, label, (8, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        occupancy = (report.get("occupancy") or {}).get("state")
        occupancy_color = OCCUPANCY_COLORS.get(occupancy, (255, 255, 255))
        cv2.rectangle(image, (0, 0), (width - 1, height - 1), occupancy_color, 3)
        cv2.putText(image, str(occupancy or "-"), (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, occupancy_color, 2, cv2.LINE_AA)
        cv2.putText(image, f"{report['status']}  {report['code']}", (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color, 1, cv2.LINE_AA)
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 82])
        return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii") if ok else ""


__all__ = ["InspectionPipeline", "baseline_matches", "effective_limits", "describe"]
