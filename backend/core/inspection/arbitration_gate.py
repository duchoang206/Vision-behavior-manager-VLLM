"""Tầng 6 – Arbitration gate: one verdict from the CV and AI branches.

Rules that are never traded away (target.md §1):

1. A mechanical defect measured by CV (offset, rotation, size, clearance) is
   always ``NG``.  The AI can add reasons to reject, never a reason to accept.
2. Something CV sees but nobody can name is an obstacle (``NG``) unless it is
   proven to be light only (shadow filter).
3. Whatever cannot be resolved is ``UNCERTAIN`` with the reason spelled out.

The two classic disputes of the plan (§3):

* **Kịch bản 1 – CV_MISSED_OBJECT**: CV sees an empty cell, AI sees an
  article.  The AI box guides a local edge/Otsu re-scan; if the mechanical
  outline is found it is measured and judged normally.
* **Kịch bản 2 – AI_MISSED_OBJECT**: CV sees something, AI recognises
  nothing.  Safety lock first, then the shadow test, then a low-confidence
  AI pass; still unknown means ``ERR_UNKNOWN_OBSTACLE_DETECTED``.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

from .ai_defect_inspector import ROLE_DEFECT, ROLE_OTHER, ROLE_TARGET, AIDetection, AIResult
from .config import (AI_DETECTED, AI_MISSED_OBJECT, CV_MISSED_OBJECT, ERR_AI_UNAVAILABLE, ERR_CENTER_OFFSET_EXCEEDED,
                     ERR_DEFECT_DAMAGED_CARGO, ERR_DIMENSION_OUT_OF_SPEC, ERR_GLARE_SATURATION,
                     ERR_OUT_OF_BOUNDS_VIOLATION, ERR_ROTATION_ANGLE_EXCEEDED, ERR_UNKNOWN_OBSTACLE_DETECTED, OK_EMPTY,
                     OK_PASS, SLOT_EMPTY, SLOT_OCCUPIED, SLOT_UNKNOWN, STATUS_DETECTED, STATUS_NG, STATUS_OK,
                     STATUS_UNCERTAIN, InspectionConfig)
from .geometry_cv_inspector import Blob, CVResult, Measurement, iou_mm

# Final error code when several apply: collision hazards first, then the
# measured mechanical faults, then the AI's visual judgement.
VERDICT_PRIORITY = (
    ERR_OUT_OF_BOUNDS_VIOLATION,
    ERR_UNKNOWN_OBSTACLE_DETECTED,
    ERR_DIMENSION_OUT_OF_SPEC,
    ERR_ROTATION_ANGLE_EXCEEDED,
    ERR_CENTER_OFFSET_EXCEEDED,
    ERR_DEFECT_DAMAGED_CARGO,
)
MATCH_IOU = 0.30


@dataclass
class Verdict:
    status: str
    slot_state: str
    code: str
    message: str
    mode: str
    errors: List[str] = field(default_factory=list)
    dispute: Optional[str] = None
    resolution: Optional[str] = None
    steps: List[str] = field(default_factory=list)
    ai_class: Optional[str] = None
    ai_confidence: Optional[float] = None

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class ArbitrationContext:
    """Callbacks into the pipeline for the dispute resolutions."""

    rescan: Callable[[Sequence[float]], Optional[Blob]]
    adopt: Callable[[Blob], None]
    ai_second_pass: Callable[[Sequence[float]], AIResult]
    shadow_evidence: Callable[[Blob], dict]


def _matches(detections: Sequence[AIDetection], bbox_mm: Sequence[float]) -> List[AIDetection]:
    found = []
    for det in detections:
        cx, cy = (det.bbox_mm[0] + det.bbox_mm[2]) / 2.0, (det.bbox_mm[1] + det.bbox_mm[3]) / 2.0
        inside = bbox_mm[0] <= cx <= bbox_mm[2] and bbox_mm[1] <= cy <= bbox_mm[3]
        if inside or iou_mm(det.bbox_mm, bbox_mm) >= MATCH_IOU:
            found.append(det)
    return found


def _best(detections: Sequence[AIDetection], role: Optional[str] = None) -> Optional[AIDetection]:
    pool = [d for d in detections if role is None or d.role == role]
    return max(pool, key=lambda d: d.confidence) if pool else None


def _primary(codes: Sequence[str]) -> str:
    ordered = [code for code in VERDICT_PRIORITY if code in codes]
    return ordered[0] if ordered else codes[0]


def describe(code: str, config: InspectionConfig, measurement: Optional[Measurement] = None,
             detection: Optional[AIDetection] = None, blob: Optional[Blob] = None) -> str:
    noun = config.article_noun
    m = measurement
    if code == OK_EMPTY:
        return "Ô trống – không có vật thể"
    if code == OK_PASS and m:
        text = (f"Đạt chuẩn: {m.width_mm:.0f}x{m.height_mm:.0f} mm, lệch tâm {m.offset_mm:.0f} mm, "
                f"góc {m.rotation_deg:+.1f}°")
        return text + (f" | AI: {detection.class_name} {detection.confidence:.0%}" if detection else "")
    if code == AI_DETECTED and detection:
        return (f"AI nhận diện {detection.class_name} ({detection.confidence:.0%}) – chế độ AI_ONLY "
                "không đo cơ khí, không cấp quyền bốc dỡ tự động")
    if code == ERR_CENTER_OFFSET_EXCEEDED and m:
        return (f"{noun} lệch tâm {m.offset_mm:.1f} mm (ΔX={m.offset_x_mm:+.1f}, ΔY={m.offset_y_mm:+.1f}) "
                f"vượt ngưỡng cho phép {config.max_center_offset_mm:.0f} mm")
    if code == ERR_ROTATION_ANGLE_EXCEEDED and m:
        return f"{noun} lệch góc {abs(m.rotation_deg):.1f}° vượt ngưỡng cho phép {config.max_rotation_deg:.1f}°"
    if code == ERR_DIMENSION_OUT_OF_SPEC and m:
        return (f"Kích thước {m.width_mm:.0f}x{m.height_mm:.0f} mm ngoài dung sai "
                f"{config.width_mm:.0f}±{config.tolerance_w_mm:.0f} x {config.height_mm:.0f}±{config.tolerance_h_mm:.0f} mm")
    if code == ERR_OUT_OF_BOUNDS_VIOLATION and m:
        return f"{noun} cách mép ô {max(m.min_margin_mm, 0.0):.0f} mm < khoảng an toàn {config.safe_margin_mm:.0f} mm"
    if code == ERR_UNKNOWN_OBSTACLE_DETECTED:
        if detection is not None and detection.role == ROLE_OTHER:
            return f"Vật cản trong ô: AI nhận diện {detection.class_name} ({detection.confidence:.0%}) – không phải kiện hàng"
        size = ""
        if blob is not None:
            box = blob.bbox_mm
            size = f" {box[2] - box[0]:.0f}x{box[3] - box[1]:.0f} mm"
        return f"CẢNH BÁO DỊ VẬT/CHƯỚNG NGẠI VẬT LẠ{size} – AI không nhận diện được, cấm AGV tiến vào ô"
    if code == ERR_DEFECT_DAMAGED_CARGO and detection:
        return f"AI phát hiện hàng hư hỏng/móp rách: {detection.class_name} ({detection.confidence:.0%})"
    if code == ERR_GLARE_SATURATION:
        return "Vùng cháy sáng lớn che khuất ô – không thể xác nhận ô trống"
    if code == ERR_AI_UNAVAILABLE:
        return "Nhánh AI không khả dụng – không thể đối chất, giữ UNCERTAIN an toàn"
    if code == CV_MISSED_OBJECT:
        return "AI thấy vật nhưng CV không tìm lại được viền cơ khí – giữ UNCERTAIN"
    return code


class ArbitrationGate:
    def decide(self, mode: str, config: InspectionConfig, cv: Optional[CVResult], ai: Optional[AIResult],
               context: Optional[ArbitrationContext] = None) -> Verdict:
        if mode == "CV_ONLY":
            return self._cv_only(config, cv)
        if mode == "AI_ONLY":
            return self._ai_only(config, ai)
        return self._hybrid(config, cv, ai, context)

    # ── CV_ONLY ──────────────────────────────────────────────────────────────
    def _cv_only(self, config: InspectionConfig, cv: CVResult) -> Verdict:
        if not cv.object_present:
            if cv.glare_unresolved:
                return Verdict(STATUS_UNCERTAIN, SLOT_UNKNOWN, ERR_GLARE_SATURATION,
                               describe(ERR_GLARE_SATURATION, config), "CV_ONLY", [ERR_GLARE_SATURATION])
            return Verdict(STATUS_OK, SLOT_EMPTY, OK_EMPTY, describe(OK_EMPTY, config), "CV_ONLY")
        errors = list(cv.errors)
        if cv.extra_blobs:
            errors.append(ERR_UNKNOWN_OBSTACLE_DETECTED)
        if errors:
            code = _primary(errors)
            blob = cv.extra_blobs[0] if code == ERR_UNKNOWN_OBSTACLE_DETECTED and cv.extra_blobs else cv.main
            return Verdict(STATUS_NG, SLOT_OCCUPIED, code, describe(code, config, cv.measurement, blob=blob),
                           "CV_ONLY", errors)
        return Verdict(STATUS_OK, SLOT_OCCUPIED, OK_PASS, describe(OK_PASS, config, cv.measurement), "CV_ONLY")

    # ── AI_ONLY ──────────────────────────────────────────────────────────────
    def _ai_only(self, config: InspectionConfig, ai: AIResult) -> Verdict:
        if ai is None or not ai.available:
            return Verdict(STATUS_UNCERTAIN, SLOT_UNKNOWN, ERR_AI_UNAVAILABLE, describe(ERR_AI_UNAVAILABLE, config),
                           "AI_ONLY", [ERR_AI_UNAVAILABLE])
        defect, other, target = ai.best(ROLE_DEFECT), ai.best(ROLE_OTHER), ai.best(ROLE_TARGET)
        if defect is not None:
            return self._with_ai(Verdict(STATUS_NG, SLOT_OCCUPIED, ERR_DEFECT_DAMAGED_CARGO,
                                         describe(ERR_DEFECT_DAMAGED_CARGO, config, detection=defect), "AI_ONLY",
                                         [ERR_DEFECT_DAMAGED_CARGO]), defect)
        if other is not None:
            return self._with_ai(Verdict(STATUS_NG, SLOT_OCCUPIED, ERR_UNKNOWN_OBSTACLE_DETECTED,
                                         describe(ERR_UNKNOWN_OBSTACLE_DETECTED, config, detection=other), "AI_ONLY",
                                         [ERR_UNKNOWN_OBSTACLE_DETECTED]), other)
        if target is not None:
            return self._with_ai(Verdict(STATUS_DETECTED, SLOT_OCCUPIED, AI_DETECTED,
                                         describe(AI_DETECTED, config, detection=target), "AI_ONLY"), target)
        return Verdict(STATUS_OK, SLOT_EMPTY, OK_EMPTY, describe(OK_EMPTY, config), "AI_ONLY")

    @staticmethod
    def _with_ai(verdict: Verdict, detection: Optional[AIDetection]) -> Verdict:
        if detection is not None:
            verdict.ai_class, verdict.ai_confidence = detection.class_name, round(detection.confidence, 4)
        return verdict

    # ── HYBRID ───────────────────────────────────────────────────────────────
    def _hybrid(self, config: InspectionConfig, cv: CVResult, ai: AIResult,
                context: ArbitrationContext) -> Verdict:
        steps: List[str] = []
        if ai is None or not ai.available:
            steps.append("Nhánh AI không khả dụng" + (f": {ai.error}" if ai is not None and ai.error else ""))
            errors = list(cv.errors) + ([ERR_UNKNOWN_OBSTACLE_DETECTED] if cv.extra_blobs else [])
            if errors:
                steps.append("CV đã đủ căn cứ từ chối (lỗi cơ khí đo được)")
                code = _primary(errors)
                return Verdict(STATUS_NG, SLOT_OCCUPIED, code, describe(code, config, cv.measurement, blob=cv.main),
                               "HYBRID", errors, steps=steps)
            return Verdict(STATUS_UNCERTAIN, SLOT_UNKNOWN, ERR_AI_UNAVAILABLE, describe(ERR_AI_UNAVAILABLE, config),
                           "HYBRID", [ERR_AI_UNAVAILABLE], steps=steps)

        if not cv.object_present:
            if not ai.detections:
                if cv.glare_unresolved:
                    return Verdict(STATUS_UNCERTAIN, SLOT_UNKNOWN, ERR_GLARE_SATURATION,
                                   describe(ERR_GLARE_SATURATION, config), "HYBRID", [ERR_GLARE_SATURATION], steps=steps)
                steps.append("CV và AI cùng xác nhận ô trống")
                return Verdict(STATUS_OK, SLOT_EMPTY, OK_EMPTY, describe(OK_EMPTY, config), "HYBRID", steps=steps)
            return self._dispute_cv_missed(config, cv, ai, context, steps)

        matched = _matches(ai.detections, cv.main.bbox_mm)
        unmatched = [d for d in ai.detections if d not in matched]
        if not matched:
            box = cv.main.bbox_mm
            steps.append(f"Mâu thuẫn 2: CV thấy vật {box[2] - box[0]:.0f}x{box[3] - box[1]:.0f} mm, AI không nhận diện "
                         "→ khóa an toàn UNCERTAIN (AI_MISSED_OBJECT), cấm AGV tiến vào ô")
            if config.enable_shadow_filter:
                evidence = context.shadow_evidence(cv.main)
                steps.append("Bộ lọc bóng râm Lab/vân sàn: " + ("chỉ là bóng đổ" if evidence.get("is_shadow")
                                                                else "vật thể 3D thật") + f" {evidence}")
                if evidence.get("is_shadow") and not cv.extra_blobs and not unmatched:
                    return Verdict(STATUS_OK, SLOT_EMPTY, OK_EMPTY, "Bóng râm – ô thực tế đang trống", "HYBRID",
                                   dispute=AI_MISSED_OBJECT, resolution="SHADOW_FILTER", steps=steps)
            else:
                steps.append("Bộ lọc bóng râm đang tắt – không loại trừ bóng đổ")
            second = context.ai_second_pass(_padded(cv.main.bbox_mm, 60.0, config.cell_mm))
            matched = _matches(second.detections, cv.main.bbox_mm) if second.available else []
            steps.append(f"AI quét cấp 2 (ngưỡng ≥ {second.confidence_threshold:.2f}): "
                         + (", ".join(f"{d.class_name} {d.confidence:.0%}" for d in matched) or "không thấy class nào"))
            if not matched:
                errors = [ERR_UNKNOWN_OBSTACLE_DETECTED] + list(cv.errors)
                return Verdict(STATUS_NG, SLOT_OCCUPIED, ERR_UNKNOWN_OBSTACLE_DETECTED,
                               describe(ERR_UNKNOWN_OBSTACLE_DETECTED, config, blob=cv.main), "HYBRID", errors,
                               dispute=AI_MISSED_OBJECT, resolution="FOD_LOCK", steps=steps)
            verdict = self._judge_occupied(config, cv, matched, unmatched, context, steps)
            verdict.dispute, verdict.resolution = AI_MISSED_OBJECT, "AI_LOW_CONFIDENCE_CONFIRM"
            return verdict
        return self._judge_occupied(config, cv, matched, unmatched, context, steps)

    def _dispute_cv_missed(self, config: InspectionConfig, cv: CVResult, ai: AIResult,
                           context: ArbitrationContext, steps: List[str]) -> Verdict:
        det = _best([d for d in ai.detections if d.role in (ROLE_TARGET, ROLE_DEFECT)]) or _best(ai.detections)
        steps.append(f"Mâu thuẫn 1: CV không thấy vật (trừ nền ΔI < ngưỡng), AI thấy {det.class_name} "
                     f"({det.confidence:.0%}) → UNCERTAIN (CV_MISSED_OBJECT)")
        if not config.enable_ai_assisted_cv:
            steps.append("AI chỉ điểm cho CV đang tắt")
            return self._with_ai(Verdict(STATUS_UNCERTAIN, SLOT_UNKNOWN, CV_MISSED_OBJECT,
                                         describe(CV_MISSED_OBJECT, config), "HYBRID", [CV_MISSED_OBJECT],
                                         dispute=CV_MISSED_OBJECT, steps=steps), det)
        steps.append("AI gửi BBox [" + ", ".join(f"{v:.0f}" for v in det.bbox_mm)
                     + "] mm → CV quét gờ cạnh Canny/Otsu cục bộ")
        blob = context.rescan(det.bbox_mm)
        if blob is None:
            steps.append("Không tìm lại được viền cơ khí trong BBox")
            return self._with_ai(Verdict(STATUS_UNCERTAIN, SLOT_UNKNOWN, CV_MISSED_OBJECT,
                                         describe(CV_MISSED_OBJECT, config), "HYBRID", [CV_MISSED_OBJECT],
                                         dispute=CV_MISSED_OBJECT, steps=steps), det)
        context.adopt(blob)
        m = cv.measurement
        steps.append(f"CV tìm lại viền ({blob.source}): {m.width_mm:.1f}x{m.height_mm:.1f} mm, θ={m.rotation_deg:+.2f}°")
        unmatched = [d for d in ai.detections if d is not det and d not in _matches(ai.detections, cv.main.bbox_mm)]
        verdict = self._judge_occupied(config, cv, [det], unmatched, context, steps)
        verdict.dispute, verdict.resolution = CV_MISSED_OBJECT, "AI_GUIDED_RESCAN"
        return verdict

    def _judge_occupied(self, config: InspectionConfig, cv: CVResult, matched: Sequence[AIDetection],
                        unmatched: Sequence[AIDetection], context: ArbitrationContext, steps: List[str]) -> Verdict:
        errors = list(cv.errors)
        if errors:
            steps.append("CV đo lỗi cơ khí → NG bất kể AI (không bao giờ để AI cho qua)")
        defect = _best(matched, ROLE_DEFECT)
        other = _best(matched, ROLE_OTHER)
        target = _best(matched, ROLE_TARGET)
        obstacle_blob = None
        if other is not None:
            errors.append(ERR_UNKNOWN_OBSTACLE_DETECTED)
        if defect is not None:
            errors.append(ERR_DEFECT_DAMAGED_CARGO)
            steps.append(f"AI phát hiện dị tật ngoại quan: {defect.class_name} {defect.confidence:.0%}")
        for blob in cv.extra_blobs:
            evidence = context.shadow_evidence(blob) if config.enable_shadow_filter else {"is_shadow": False}
            if not evidence.get("is_shadow"):
                errors.append(ERR_UNKNOWN_OBSTACLE_DETECTED)
                obstacle_blob = obstacle_blob or blob
                steps.append(f"Vật thể thứ hai ngoài kiện hàng tại {[round(v) for v in blob.bbox_mm]} mm → dị vật")
        uncertain = False
        for det in unmatched:
            found = context.rescan(det.bbox_mm) if config.enable_ai_assisted_cv else None
            if found is not None:
                errors.append(ERR_UNKNOWN_OBSTACLE_DETECTED)
                obstacle_blob = obstacle_blob or found
                steps.append(f"AI thấy thêm {det.class_name} ngoài kiện hàng, CV xác nhận viền → dị vật")
            else:
                uncertain = True
                steps.append(f"AI thấy thêm {det.class_name} ({det.confidence:.0%}) mà CV không xác nhận được")
        detection = defect or other or target
        if errors:
            errors = list(dict.fromkeys(errors))
            code = _primary(errors)
            blob = obstacle_blob if code == ERR_UNKNOWN_OBSTACLE_DETECTED else cv.main
            det = defect if code == ERR_DEFECT_DAMAGED_CARGO else (other if code == ERR_UNKNOWN_OBSTACLE_DETECTED else target)
            return self._with_ai(Verdict(STATUS_NG, SLOT_OCCUPIED, code,
                                         describe(code, config, cv.measurement, det, blob), "HYBRID", errors,
                                         steps=steps), detection)
        if uncertain:
            return self._with_ai(Verdict(STATUS_UNCERTAIN, SLOT_UNKNOWN, CV_MISSED_OBJECT,
                                         describe(CV_MISSED_OBJECT, config), "HYBRID", [CV_MISSED_OBJECT],
                                         dispute=CV_MISSED_OBJECT, steps=steps), detection)
        steps.append("CV PASS + AI xác nhận kiện hàng nguyên vẹn → OK")
        return self._with_ai(Verdict(STATUS_OK, SLOT_OCCUPIED, OK_PASS,
                                     describe(OK_PASS, config, cv.measurement, target), "HYBRID", steps=steps), target)


def _padded(bbox_mm: Sequence[float], pad: float, cell_mm: Sequence[float]) -> List[float]:
    return [max(0.0, bbox_mm[0] - pad), max(0.0, bbox_mm[1] - pad),
            min(float(cell_mm[0]), bbox_mm[2] + pad), min(float(cell_mm[1]), bbox_mm[3] + pad)]
