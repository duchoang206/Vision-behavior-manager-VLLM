"""Inspection core: Phase 1-3 of the plan and target.md test groups 1-3.

Every test renders an oblique 1920x1080 camera frame from a floor scene with
millimetre ground truth (inspection_scenes) and runs the production pipeline
on it.  The AI branch uses ``OracleDetector`` in place of a trained YOLO: it
reports what a model would report (class, confidence, box) so that the
arbitration logic - not a model's accuracy - is what is under test.
"""

import math

import numpy as np
import pytest

from core.inspection.ai_defect_inspector import AIDefectInspector, CallableBackend
from core.inspection.baseline_store import BaselineStore
from core.inspection.config import InspectionConfig, WORK_SIZE_PX
from core.inspection.homography_rectifier import (InvalidRoiError, MetricRectifier, compute_homography,
                                                  warp_to_metric_bev)
from core.inspection.image_quality import check_image_quality
from core.inspection.pipeline import InspectionPipeline
from core.inspection.temporal_inspection_manager import EVENT_CONFIRMED, EVENT_INTERLOCK, TemporalInspectionManager
from inspection_scenes import (FRAME_H, FRAME_W, H_WORLD_TO_CAMERA, ROI_POINTS, Article, OracleDetector, Scene,
                               article_bbox, baseline_frames, rect_corners, render, world_to_camera)


@pytest.fixture(scope="module")
def baseline(tmp_path_factory):
    store = BaselineStore(tmp_path_factory.mktemp("baselines"))
    store.save("cam_test", "rule_test", baseline_frames(5), ROI_POINTS)
    return store.load("cam_test", "rule_test")


def run(scene, baseline, mode="CV_ONLY", detections=None, ai=True, **config):
    oracle = OracleDetector(detections or [])
    inspector = AIDefectInspector(CallableBackend(oracle, "oracle")) if ai else AIDefectInspector(None)
    report = InspectionPipeline(inspector).run(render(scene), ROI_POINTS, InspectionConfig(mode=mode, **config),
                                               baseline)
    report["_oracle_calls"] = len(oracle.calls)
    return report


def carton(article, confidence=0.98, name="carton"):
    return [(name, confidence, article_bbox(article))]


# ── Phase 1: rectification and image quality ────────────────────────────────
def test_rectifier_maps_floor_millimetres_exactly():
    matrix = compute_homography(ROI_POINTS, FRAME_W, FRAME_H)
    rng = np.random.default_rng(3)
    bev_mm = rng.uniform(0, 1000, (200, 2))
    # world mm (edge coords) -> canvas pixel centre -> camera pixel centre -> BEV pixel centre -> mm
    world_centre = np.hstack([bev_mm + 500 - 0.5, np.ones((200, 1))])
    camera = (H_WORLD_TO_CAMERA @ world_centre.T).T
    camera = camera[:, :2] / camera[:, 2:]
    back = (matrix @ np.hstack([camera, np.ones((200, 1))]).T).T
    back = back[:, :2] / back[:, 2:] + 0.5
    assert np.abs(back - bev_mm).max() < 0.05


def test_warp_produces_upright_1000px_bev():
    frame = render(Scene(article=Article(angle=0.0)))
    bev = warp_to_metric_bev(frame, ROI_POINTS)
    assert bev.shape == (1000, 1000, 3)
    # The oblique carton becomes an upright rectangle: its edges are columns/rows.
    gray = bev.mean(axis=2)
    column_profile = gray[300:700].mean(axis=0)
    left_edge = int(np.argmax(np.abs(np.diff(column_profile[60:140])))) + 60
    right_edge = int(np.argmax(np.abs(np.diff(column_profile[860:940])))) + 860
    assert abs(left_edge - 100) <= 2 and abs(right_edge - 899) <= 2
    assert check_image_quality(frame, bev).ok


def test_counter_clockwise_roi_gives_the_same_view():
    frame = render(Scene(article=Article(angle=3.0)))
    reversed_points = [ROI_POINTS[0], ROI_POINTS[3], ROI_POINTS[2], ROI_POINTS[1]]
    a = MetricRectifier(ROI_POINTS).warp(frame, 4.0).astype(int)
    b = MetricRectifier(reversed_points).warp(frame, 4.0).astype(int)
    assert np.abs(a - b).max() == 0


@pytest.mark.parametrize("points", [
    [[0.1, 0.1], [0.5, 0.1], [0.5, 0.5]],                           # three vertices
    [[0.1, 0.1], [0.5, 0.5], [0.5, 0.1], [0.1, 0.5]],               # self-intersecting
    [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [1.4, 0.9]],               # outside the frame
])
def test_invalid_roi_is_rejected(points):
    with pytest.raises(InvalidRoiError):
        compute_homography(points, FRAME_W, FRAME_H)


def test_image_quality_gate():
    sharp = render(Scene(article=Article()))
    assert check_image_quality(sharp).ok
    blurred = check_image_quality(render(Scene(article=Article(), blur_sigma=2.5)))
    assert not blurred.ok and blurred.error_code == "ERR_BLURRY_FRAME" and blurred.laplacian_var < 80
    dark = check_image_quality(render(Scene(gain=0.12, noise_sigma=1.0)))
    assert not dark.ok and dark.error_code == "ERR_LIGHTING_OUT_OF_RANGE"
    # A dim but sharp night frame is not "blurry" (Test 2.1 must still measure).
    night = check_image_quality(render(Scene(article=Article(), gain=0.33, noise_sigma=3.0)))
    assert night.ok and night.brightness < 70


# ── Phase 2 / Bộ 1: metrology ───────────────────────────────────────────────
@pytest.mark.parametrize("angle", [0.0, 3.0, -3.0, 10.0])
def test_phase2_measurement_accuracy_at_angles(baseline, angle):
    report = run(Scene(article=Article(angle=angle)), baseline)
    m = report["measurement"]
    assert abs(m["width_mm"] - 800) <= 5 and abs(m["height_mm"] - 600) <= 5
    assert abs(m["rotation_deg"] - angle) <= 0.5
    assert abs(m["center_x_mm"] - 500) <= 3 and abs(m["center_y_mm"] - 500) <= 3


def test_tc_1_1_golden_sample_ok(baseline):
    article = Article()
    for mode in ("CV_ONLY", "HYBRID"):
        report = run(Scene(article=article), baseline, mode, carton(article))
        m = report["measurement"]
        assert report["status"] == "OK" and report["code"] == "OK_PASS", report["message"]
        assert 795 <= m["width_mm"] <= 805 and 595 <= m["height_mm"] <= 605
        assert m["offset_mm"] <= 5 and abs(m["rotation_deg"]) <= 0.5


def test_tc_1_2_center_offset_exceeded(baseline):
    report = run(Scene(article=Article(cx=565)), baseline)
    assert report["status"] == "NG" and report["code"] == "ERR_CENTER_OFFSET_EXCEEDED"
    assert abs(report["measurement"]["offset_x_mm"] - 65) <= 3


def test_center_offset_accuracy_30mm(baseline):
    report = run(Scene(article=Article(cx=530)), baseline)
    assert abs(report["measurement"]["offset_x_mm"] - 30) <= 3 and abs(report["measurement"]["offset_y_mm"]) <= 3
    assert report["status"] == "OK"


def test_tc_1_3_rotation_exceeded(baseline):
    report = run(Scene(article=Article(angle=7.5)), baseline)
    assert 7.0 <= report["measurement"]["rotation_deg"] <= 8.0
    assert report["status"] == "NG" and report["code"] == "ERR_ROTATION_ANGLE_EXCEEDED"


def test_tc_1_4_dimension_out_of_spec(baseline):
    report = run(Scene(article=Article(width=900)), baseline)
    assert abs(report["measurement"]["width_mm"] - 900) <= 5
    assert report["status"] == "NG" and report["code"] == "ERR_DIMENSION_OUT_OF_SPEC"


def test_tc_1_5_boundary_safe_margin(baseline):
    report = run(Scene(article=Article(cx=410)), baseline)          # left edge 10 mm from the cell border
    assert abs(report["measurement"]["min_margin_mm"] - 10) <= 3
    assert report["status"] == "NG" and report["code"] == "ERR_OUT_OF_BOUNDS_VIOLATION"


# ── Bộ 2: lighting ──────────────────────────────────────────────────────────
def test_tc_2_1_low_light(baseline):
    report = run(Scene(article=Article(), gain=0.33, noise_sigma=3.0), baseline)
    m = report["measurement"]
    assert report["cv"]["light_profile"] == "low" and report["iqa"]["brightness"] < 70
    assert abs(m["width_mm"] - 800) <= 5 and abs(m["height_mm"] - 600) <= 5
    assert report["status"] == "OK"


def test_tc_2_2_glare_does_not_inflate_the_box(baseline):
    report = run(Scene(article=Article(), gain=1.25, glare=((150, -50), (700, 1050), 60)), baseline)
    m = report["measurement"]
    assert report["cv"]["light_profile"] == "high" and report["iqa"]["brightness"] > 170
    assert abs(m["width_mm"] - 800) <= 5 and abs(m["height_mm"] - 600) <= 5
    assert report["status"] == "OK"


def test_tc_2_3_dark_shadow_on_empty_cell(baseline):
    shadow = np.float64([[-50, -50], [1050, -50], [1050, 450], [-50, 560]])      # ~50 % of the cell
    for mode in ("CV_ONLY", "HYBRID"):
        report = run(Scene(shadow=shadow, shadow_factor=0.35), baseline, mode)
        assert report["status"] == "OK" and report["code"] == "OK_EMPTY", report["message"]
        assert report["cv"]["shadow_ratio"] > 0.3


def test_shadow_rejection_rate_random_shadows(baseline):
    rng = np.random.default_rng(2024)
    pipeline = InspectionPipeline()
    config = InspectionConfig(mode="CV_ONLY")
    false_full = 0
    trials = 60
    for index in range(trials):
        poly = rect_corners(rng.uniform(400, 1000), rng.uniform(300, 900), *rng.uniform(150, 850, 2),
                            rng.uniform(0, 180))
        frame = render(Scene(shadow=poly, shadow_factor=rng.uniform(0.3, 0.75), seed=9000 + index))
        if pipeline.run(frame, ROI_POINTS, config, baseline)["slot_state"] != "EMPTY":
            false_full += 1
    assert (trials - false_full) / trials >= 0.99


def test_shadow_next_to_carton_is_not_measured(baseline):
    shadow = np.float64([[-50, -50], [1050, -50], [1050, 450], [-50, 560]])
    report = run(Scene(article=Article(), shadow=shadow), baseline)
    m = report["measurement"]
    assert abs(m["width_mm"] - 800) <= 5 and abs(m["height_mm"] - 600) <= 5


# ── Bộ 3: arbitration (HYBRID) ──────────────────────────────────────────────
def test_tc_3_1_camouflage_ai_guided_rescan(baseline):
    article = Article(color=(141, 144, 148), edge_color=(110, 112, 115), tape=False)
    report = run(Scene(article=article), baseline, "HYBRID", carton(article, 0.91))
    assert report["arbitration"]["dispute"] == "CV_MISSED_OBJECT"
    assert report["arbitration"]["resolution"] == "AI_GUIDED_RESCAN"
    assert any("UNCERTAIN (CV_MISSED_OBJECT)" in step for step in report["arbitration"]["steps"])
    m = report["measurement"]
    assert abs(m["width_mm"] - 800) <= 5 and abs(m["height_mm"] - 600) <= 5
    assert report["status"] == "OK"
    rotated = Article(color=(141, 144, 148), edge_color=(110, 112, 115), tape=False, angle=8.0)
    report = run(Scene(article=rotated), baseline, "HYBRID", carton(rotated, 0.91))
    assert report["status"] == "NG" and report["code"] == "ERR_ROTATION_ANGLE_EXCEEDED"


@pytest.mark.parametrize("debris", [("toolbox", (400, 300, 500, 500, 0)), ("nylon", (500, 500, 200, 140))])
def test_tc_3_2_unknown_obstacle(baseline, debris):
    report = run(Scene(debris=[debris]), baseline, "HYBRID", [])
    assert report["status"] == "NG" and report["code"] == "ERR_UNKNOWN_OBSTACLE_DETECTED"
    assert report["arbitration"]["dispute"] == "AI_MISSED_OBJECT"
    assert report["arbitration"]["resolution"] == "FOD_LOCK"
    steps = " ".join(report["arbitration"]["steps"])
    assert "khóa an toàn" in steps and "vật thể 3D thật" in steps and "0.20" in steps
    assert report["_oracle_calls"] == 2                     # normal pass + low-sensitivity pass


def test_tc_3_3_damaged_cargo(baseline):
    article = Article(damaged=True)
    report = run(Scene(article=article), baseline, "HYBRID", carton(article, 0.88, "carton_damaged"))
    assert report["cv"]["verdict"] == "CV_PASS"
    assert report["status"] == "NG" and report["code"] == "ERR_DEFECT_DAMAGED_CARGO"


def test_tc_3_4_blurry_frame_stops_at_iqa(baseline):
    article = Article()
    report = run(Scene(article=article, blur_sigma=3.0), baseline, "HYBRID", carton(article))
    assert report["status"] == "UNCERTAIN" and report["code"] == "ERR_BLURRY_FRAME"
    assert report["cv"] is None and report["ai"] is None and report["_oracle_calls"] == 0


def test_low_confidence_second_pass_confirms_article(baseline):
    article = Article()
    report = run(Scene(article=article), baseline, "HYBRID", carton(article, 0.31))
    assert report["arbitration"]["dispute"] == "AI_MISSED_OBJECT"
    assert report["arbitration"]["resolution"] == "AI_LOW_CONFIDENCE_CONFIRM"
    assert report["status"] == "OK"


def test_hybrid_without_model_never_certifies(baseline):
    good = run(Scene(article=Article()), baseline, "HYBRID", ai=False)
    assert good["status"] == "UNCERTAIN" and good["code"] == "ERR_AI_UNAVAILABLE"
    bad = run(Scene(article=Article(angle=9)), baseline, "HYBRID", ai=False)
    assert bad["status"] == "NG" and bad["code"] == "ERR_ROTATION_ANGLE_EXCEEDED"


def test_ai_only_reports_detected_not_ok(baseline):
    article = Article(angle=9)                                 # misaligned - AI_ONLY cannot know
    report = run(Scene(article=article), baseline, "AI_ONLY", carton(article))
    assert report["status"] == "DETECTED" and report["code"] == "AI_DETECTED"


def test_no_baseline_is_uncertain():
    report = InspectionPipeline().run(render(Scene()), ROI_POINTS, InspectionConfig(mode="CV_ONLY"), None)
    assert report["status"] == "UNCERTAIN" and report["code"] == "ERR_NO_BASELINE"


def test_baseline_for_another_roi_is_refused(baseline):
    moved = [[x + 0.01, y] for x, y in ROI_POINTS]
    report = InspectionPipeline().run(render(Scene()), moved, InspectionConfig(mode="CV_ONLY"), baseline)
    assert report["code"] == "ERR_BASELINE_ROI_MISMATCH"


# ── 0 % false negatives (target.md §2) ──────────────────────────────────────
def test_zero_false_negative_sweep(baseline):
    """Every article past a limit is NG, even when the AI is certain it is a good carton."""
    rng = np.random.default_rng(77)
    misses = []
    for index in range(40):
        if index % 2:
            angle, distance, direction = rng.uniform(5.05, 15.0) * rng.choice([-1, 1]), rng.uniform(0, 30), rng.uniform(0, 360)
        else:
            angle, distance, direction = rng.uniform(-4.0, 4.0), rng.uniform(50.05, 110), rng.uniform(0, 360)
        cx = 500 + distance * math.cos(math.radians(direction))
        cy = 500 + distance * math.sin(math.radians(direction))
        article = Article(cx=cx, cy=cy, angle=angle)
        for mode in ("CV_ONLY", "HYBRID"):
            report = run(Scene(article=article, seed=300 + index), baseline, mode, carton(article, 0.99))
            if report["status"] != "NG":
                misses.append((mode, round(angle, 2), round(distance, 1), report["code"]))
    assert not misses


# ── Tầng 7: temporal FSM ────────────────────────────────────────────────────
def _frame(status, slot="OCCUPIED", code="X"):
    return {"status": status, "slot_state": slot, "code": code, "message": code}


def test_temporal_confirms_after_window_and_interlocks_immediately():
    fsm = TemporalInspectionManager(window_frames=3)
    events = [fsm.update(_frame("OK"), now=t * 0.2).event for t in range(3)]
    assert events == [None, None, EVENT_CONFIRMED] and fsm.snapshot()["agv_permission"]
    assert fsm.update(_frame("NG", code="ERR_ROTATION_ANGLE_EXCEEDED"), now=0.8).event == EVENT_INTERLOCK
    assert not fsm.snapshot()["agv_permission"]
    fsm.update(_frame("NG"), now=1.0)
    assert fsm.update(_frame("NG"), now=1.2).event == EVENT_CONFIRMED
    assert fsm.snapshot()["confirmed_status"] == "NG"


def test_temporal_single_frame_glitch_does_not_flip_state():
    fsm = TemporalInspectionManager(window_frames=3)
    for t in range(3):
        fsm.update(_frame("OK", "EMPTY"), now=t * 0.2)
    fsm.update(_frame("OK", "OCCUPIED"), now=0.6)           # one frame of a passing person
    for t in range(4, 7):
        fsm.update(_frame("OK", "EMPTY"), now=t * 0.2)
    snap = fsm.snapshot()
    assert snap["confirmed"]["slot_state"] == "EMPTY" and snap["agv_permission"]


def test_temporal_uncertain_needs_1_5_seconds_and_operator_can_resolve():
    fsm = TemporalInspectionManager(window_frames=3, uncertain_hold_sec=1.5)
    for t in range(3):
        fsm.update(_frame("OK", "EMPTY"), now=t * 0.2)
    first = fsm.update(_frame("UNCERTAIN", "UNKNOWN"), now=1.0)
    assert first.event == EVENT_INTERLOCK
    times = [1.2, 1.4, 1.8, 2.2, 2.6]
    events = [fsm.update(_frame("UNCERTAIN", "UNKNOWN"), now=t).event for t in times]
    assert events.index(EVENT_CONFIRMED) == 4                  # at t = 2.6 s, i.e. after > 1.5 s
    decided = fsm.apply_operator_decision(False, "supervisor", now=3.0)
    assert decided.state["confirmed_status"] == "NG"
    assert fsm.update(_frame("UNCERTAIN", "UNKNOWN"), now=3.2).event is None   # decision holds


def test_temporal_flip_flop_becomes_uncertain():
    fsm = TemporalInspectionManager(window_frames=3, uncertain_hold_sec=1.5)
    status = ["OK", "NG"]
    last = None
    for index in range(12):
        update = fsm.update(_frame(status[index % 2]), now=index * 0.2)
        last = update if update.event else last
    assert last is not None and last.state["confirmed"]["code"] == "ERR_UNSTABLE_STATE"


# ── Cells of any real size (the 1 m x 1 m cell is only the default) ────────
PALLET_CELL = (1400.0, 1200.0)
WOOD = (82, 140, 186)


@pytest.fixture(scope="module")
def pallet_baseline(tmp_path_factory):
    store = BaselineStore(tmp_path_factory.mktemp("pallet_baselines"))
    meta = store.save("cam_test", "rule_pallet", baseline_frames(5, cell=PALLET_CELL), ROI_POINTS, PALLET_CELL)
    assert meta["size_px"] == [1400, 1200] and meta["cell_mm"] == [1400.0, 1200.0]
    return store.load("cam_test", "rule_pallet")


def pallet(**overrides):
    values = dict(width=1200.0, height=1000.0, cx=700.0, cy=600.0, color=WOOD, edge_color=(45, 80, 110),
                  tape=False, slats=4)
    values.update(overrides)
    return Article(**values)


def run_pallet(article, baseline, mode="CV_ONLY", detections=None, scene_extra=None, **config):
    oracle = OracleDetector(detections or [])
    scene = Scene(article=article, cell=PALLET_CELL, **(scene_extra or {}))
    cfg = InspectionConfig(mode=mode, target_preset="pallet_1200x1000", roi_width_mm=PALLET_CELL[0],
                           roi_height_mm=PALLET_CELL[1], **config)
    return InspectionPipeline(AIDefectInspector(CallableBackend(oracle, "oracle"))).run(render(scene), ROI_POINTS, cfg, baseline)


def test_rectifier_is_metric_for_a_rectangular_cell():
    matrix = compute_homography(ROI_POINTS, FRAME_W, FRAME_H, PALLET_CELL)
    rng = np.random.default_rng(5)
    bev_mm = rng.uniform(0, 1, (200, 2)) * np.asarray(PALLET_CELL)
    camera = (world_to_camera(PALLET_CELL) @ np.hstack([bev_mm + 500 - 0.5, np.ones((200, 1))]).T).T
    camera = camera[:, :2] / camera[:, 2:]
    back = (matrix @ np.hstack([camera, np.ones((200, 1))]).T).T
    back = back[:, :2] / back[:, 2:] + 0.5
    assert np.abs(back - bev_mm).max() < 0.05
    assert warp_to_metric_bev(render(Scene(cell=PALLET_CELL)), ROI_POINTS, PALLET_CELL).shape == (1200, 1400, 3)


def test_pallet_on_1400x1200_cell_ok(pallet_baseline):
    article = pallet(angle=1.0, cx=708, cy=594)
    for mode in ("CV_ONLY", "HYBRID"):
        report = run_pallet(article, pallet_baseline, mode, [("pallet", 0.95, article_bbox(article))])
        m = report["measurement"]
        assert report["status"] == "OK", report["message"]
        assert abs(m["width_mm"] - 1200) <= 5 and abs(m["height_mm"] - 1000) <= 5
        assert abs(m["offset_x_mm"] - 8) <= 3 and abs(m["offset_y_mm"] + 6) <= 3 and abs(m["rotation_deg"] - 1.0) <= 0.5


@pytest.mark.parametrize("article, code", [
    (pallet(cx=765), "ERR_CENTER_OFFSET_EXCEEDED"),
    (pallet(angle=6.0), "ERR_ROTATION_ANGLE_EXCEEDED"),
    (pallet(width=1100, height=1100), "ERR_DIMENSION_OUT_OF_SPEC"),
    (pallet(cx=612), "ERR_OUT_OF_BOUNDS_VIOLATION"),           # 12 mm from the left edge of the cell
])
def test_pallet_limits_on_rectangular_cell(pallet_baseline, article, code):
    report = run_pallet(article, pallet_baseline)
    assert report["status"] == "NG" and report["code"] == code, (report["code"], report["measurement"])


def test_pallet_cell_dark_shadow_stays_empty(pallet_baseline):
    shadow = np.float64([[-50, -50], [1450, -50], [1450, 520], [-50, 680]])
    report = InspectionPipeline().run(render(Scene(shadow=shadow, shadow_factor=0.35, cell=PALLET_CELL)), ROI_POINTS,
                                      InspectionConfig(mode="CV_ONLY", roi_width_mm=1400, roi_height_mm=1200),
                                      pallet_baseline)
    assert report["code"] == "OK_EMPTY", report["message"]


def test_changing_cell_size_requires_new_baseline(pallet_baseline):
    report = InspectionPipeline().run(render(Scene(cell=PALLET_CELL)), ROI_POINTS,
                                      InspectionConfig(mode="CV_ONLY"), pallet_baseline)
    assert report["status"] == "UNCERTAIN" and report["code"] == "ERR_BASELINE_ROI_MISMATCH"
