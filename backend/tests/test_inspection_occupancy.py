"""Inspection stations answering "does the cell hold goods?" (CARFULL / EMPTY / UNKNOWN).

* CV_ONLY: image processing against the empty-cell baselines decides alone.
* HYBRID: image processing finds the object, the camera's deployed model names
  it - a selected class → CARFULL, anything else → UNKNOWN.
* AI_ONLY: a selected class standing on the cell → CARFULL.
* Several empty-cell baselines, one per lighting condition; each frame is
  compared with the best-matching one.

The AI layer here is the camera model's live detections (``LiveDetections``),
built from the article's true camera footprint, i.e. what DeepStream reports.
"""

import json
import shutil
import subprocess
import time

import numpy as np
import pytest

from core.behavior_analytics import BehaviorAnalyticsEngine
from core.inspection.baseline_selector import select_baseline
from core.inspection.baseline_store import MAX_BASELINES, BaselineLimitError, BaselineStore
from core.inspection.config import InspectionConfig, WORK_MM_PER_PX
from core.inspection.homography_rectifier import MetricRectifier
from core.inspection.live_detections import LiveDetections, cell_detections, target_matches
from core.inspection.occupancy import OccupancyDebouncer
from core.inspection.pipeline import InspectionPipeline
from core.inspection.recorded_frames import RecordedFrames, RecordingNotFound
from core.inspection.runtime import InspectionRuntime, fms_payload
from inspection_scenes import (CELL_ORIGIN, FRAME_H, FRAME_W, H_WORLD_TO_CAMERA, ROI_POINTS, Article, Scene,
                               baseline_frames, rect_corners, render)

LAMP = rect_corners(520, 420, 300, 330, 12)          # a warm lamp pool on part of the cell (BEV mm)
CARTON = Article(cx=520, cy=560)
TOOLBOX = [("toolbox", (220, 160, 760, 760, 20))]


def camera_box(article: Article, raise_px: float = 0.0, shift_y_px: float = 0.0) -> dict:
    """Normalised camera box of an article's footprint, as the deployed detector reports it."""
    world = np.hstack([article.corners + CELL_ORIGIN - 0.5, np.ones((4, 1))])
    camera = (H_WORLD_TO_CAMERA @ world.T).T
    camera = camera[:, :2] / camera[:, 2:] + 0.5
    (x0, y0), (x1, y1) = camera.min(axis=0), camera.max(axis=0)
    y0, y1 = y0 - raise_px + shift_y_px, y1 + shift_y_px
    return {"x": x0 / FRAME_W, "y": y0 / FRAME_H, "w": (x1 - x0) / FRAME_W, "h": (y1 - y0) / FRAME_H}


def live(*objects, age=0.2):
    return LiveDetections([dict(obj) for obj in objects], age)


def detection(name, article=CARTON, confidence=0.91, **box):
    return {"class": name, "label": "", "confidence": confidence, **camera_box(article, **box)}


@pytest.fixture(scope="module")
def day_store(tmp_path_factory):
    store = BaselineStore(tmp_path_factory.mktemp("occupancy"))
    store.save("cam", "day", baseline_frames(5), ROI_POINTS, label="Ban ngày")
    return store


def run(scene, baselines, mode, targets=("rack",), detections=None):
    return InspectionPipeline().run(render(scene), ROI_POINTS, InspectionConfig(mode=mode), baselines,
                                    targets=list(targets), live=detections)


# ── the AI layer: same classes and matching as the occupancy rule ────────────
@pytest.mark.parametrize("class_name, label, targets", [
    ("Rack", "", ["rack"]), ("pallet_wood", "", ["rack"]), ("rack", "", ["pallet"]), ("shelf", "", ["shelf"]),
    ("Robot_1", "", ["robot"]), ("agv", "", ["robot"]), ("robot", "", ["amr"]), ("person", "", ["worker"]),
    ("worker", "", ["person"]), ("carton", "", ["rack"]), ("person", "", ["rack", "robot"]),
    ("box", "Rack_07", ["rack"]), ("thing", "", []), ("thing", "", ["  "]), ("forklift", "", ["fork"]),
])
def test_target_matching_is_the_occupancy_rule_matching(class_name, label, targets):
    rule = BehaviorAnalyticsEngine()._target_matches({"class": class_name, "label": label}, targets)
    assert target_matches(class_name, label, targets) == rule


def test_only_detections_standing_on_the_cell_count():
    rect = MetricRectifier(ROI_POINTS)
    shape = (FRAME_H, FRAME_W)
    on_cell = detection("rack", raise_px=260)                       # tall rack, feet in the cell
    in_front = detection("rack", Article(cx=500, cy=500, width=500, height=300), raise_px=500, shift_y_px=330)
    found = cell_detections(live(on_cell, in_front), rect, shape, ["rack"])
    assert len(found) == 1 and found[0].role == "target" and found[0].source == "live"
    x0, y0, x1, y1 = found[0].bbox_mm
    assert 0 <= x0 < CARTON.cx < x1 <= 1000 and 0 <= y0 < CARTON.cy < y1 <= 1000
    # The tracker's ground point wins over the box bottom.
    moved = dict(on_cell, ground_point=[0.05, 0.95])
    assert cell_detections(live(moved), rect, shape, ["rack"]) == []
    # Stale metadata: the camera's model is not running.
    assert cell_detections(live(on_cell, age=9.0), rect, shape, ["rack"]) == []
    person = cell_detections(live(detection("person")), rect, shape, ["rack"])
    assert [d.role for d in person] == ["other"]


# ── CV_ONLY ──────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("scene, expected", [
    (Scene(seed=21), "EMPTY"),
    (Scene(article=CARTON, seed=22), "CARFULL"),
    (Scene(debris=TOOLBOX, seed=23), "CARFULL"),
    (Scene(shadow=rect_corners(600, 380, 420, 520, 25), shadow_factor=0.4, seed=24), "EMPTY"),
])
def test_cv_only_object_means_carfull(day_store, scene, expected):
    report = run(scene, day_store.load_all("cam", "day"), "CV_ONLY")
    assert report["occupancy"]["state"] == expected, report["occupancy"]
    assert report["occupancy"]["source"] == "CV"


# ── HYBRID ───────────────────────────────────────────────────────────────────
def test_hybrid_cv_finds_and_the_ai_names_the_goods(day_store):
    report = run(Scene(article=CARTON, seed=31), day_store.load_all("cam", "day"), "HYBRID",
                 detections=live(detection("Rack", raise_px=200)))
    occupancy = report["occupancy"]
    assert occupancy["state"] == "CARFULL" and occupancy["label"] == "Rack" and occupancy["source"] == "CV+AI"
    assert occupancy["cv_object"] and occupancy["confidence"] == pytest.approx(0.91)
    assert any(box["source"] == "live" for box in report["overlay"]["ai_boxes"])


@pytest.mark.parametrize("detections, why", [
    (live(), "AI không nhận ra"),                                   # model running, sees nothing
    (live(detection("person")), "không thuộc đối tượng áp dụng"),   # knows it, not a selected class
    (LiveDetections([], None), "AI không khả dụng"),                # no model on this camera
])
def test_hybrid_object_the_ai_cannot_name_is_unknown(day_store, detections, why):
    report = run(Scene(article=CARTON, seed=32), day_store.load_all("cam", "day"), "HYBRID", detections=detections)
    assert report["occupancy"]["state"] == "UNKNOWN", report["occupancy"]
    assert why in report["occupancy"]["reason"]
    assert report["occupancy"]["cv_object"]


def test_hybrid_empty_cell_and_camouflage(day_store):
    baselines = day_store.load_all("cam", "day")
    assert run(Scene(seed=33), baselines, "HYBRID", detections=live())["occupancy"]["state"] == "EMPTY"
    # A person passing by the cell is ignored when image processing sees nothing on it.
    passer = run(Scene(seed=34), baselines, "HYBRID", detections=live(detection("person")))
    assert passer["occupancy"]["state"] == "EMPTY"
    # Goods the AI recognises are never reported empty, even where CV cannot separate them.
    hidden = run(Scene(seed=35), baselines, "HYBRID", detections=live(detection("rack")))
    assert hidden["occupancy"]["state"] == "CARFULL" and hidden["occupancy"]["source"] == "AI"


# ── AI_ONLY ──────────────────────────────────────────────────────────────────
def test_ai_only_needs_a_selected_class_on_the_cell():
    scene = Scene(article=CARTON, seed=41)
    assert run(scene, None, "AI_ONLY", detections=live(detection("rack")))["occupancy"]["state"] == "CARFULL"
    assert run(scene, None, "AI_ONLY", detections=live(detection("person")))["occupancy"]["state"] == "EMPTY"
    assert run(scene, None, "AI_ONLY", detections=live())["occupancy"]["state"] == "EMPTY"
    assert run(scene, None, "AI_ONLY", detections=LiveDetections())["occupancy"]["state"] == "UNKNOWN"
    every_class = run(scene, None, "AI_ONLY", targets=(), detections=live(detection("person")))
    assert every_class["occupancy"]["state"] == "CARFULL"


def test_no_answer_is_unknown_with_the_reason(day_store):
    assert run(Scene(article=CARTON, seed=42), [], "CV_ONLY")["occupancy"]["state"] == "UNKNOWN"
    blurred = run(Scene(article=CARTON, blur_sigma=6.0, seed=43), day_store.load_all("cam", "day"), "CV_ONLY")
    assert blurred["occupancy"]["state"] == "UNKNOWN" and blurred["occupancy"]["reason"] == blurred["message"]


# ── several baselines, one per lighting condition ────────────────────────────
@pytest.fixture(scope="module")
def lamp_store(tmp_path_factory):
    store = BaselineStore(tmp_path_factory.mktemp("lamp"))
    store.save("cam", "r", baseline_frames(5), ROI_POINTS, label="Ban ngày")
    store.save("cam", "r", [render(Scene(light=LAMP, seed=900 + i)) for i in range(5)], ROI_POINTS, label="Đèn vàng")
    return store


def test_one_baseline_mistakes_lamp_light_for_goods(day_store):
    report = run(Scene(light=LAMP, seed=51), day_store.load_all("cam", "day"), "CV_ONLY")
    assert report["occupancy"]["state"] == "CARFULL"          # why a second baseline is needed


@pytest.mark.parametrize("scene, state, chosen", [
    (Scene(light=LAMP, seed=52), "EMPTY", "Đèn vàng"),
    (Scene(seed=53), "EMPTY", "Ban ngày"),
    (Scene(light=LAMP, article=CARTON, seed=54), "CARFULL", "Đèn vàng"),
    (Scene(article=CARTON, seed=55), "CARFULL", "Ban ngày"),
    (Scene(light=LAMP, debris=TOOLBOX, seed=56), "CARFULL", "Đèn vàng"),
])
def test_best_matching_baseline_per_frame(lamp_store, scene, state, chosen):
    report = run(scene, lamp_store.load_all("cam", "r"), "CV_ONLY")
    assert report["occupancy"]["state"] == state
    assert report["baseline_choice"]["label"] == chosen
    assert report["baseline_choice"]["count"] == 2 and len(report["baseline_choice"]["candidates"]) == 2


def test_baseline_ranking_is_cheap(lamp_store):
    baselines = lamp_store.load_all("cam", "r")
    rect = MetricRectifier(ROI_POINTS)
    work = rect.warp(render(Scene(light=LAMP, seed=57)), WORK_MM_PER_PX)
    select_baseline(work, baselines, InspectionConfig())
    started = time.perf_counter()
    for _ in range(20):
        best, ranking = select_baseline(work, baselines, InspectionConfig())
    per_baseline_ms = (time.perf_counter() - started) * 1000.0 / 20 / len(baselines)
    assert best.label == "Đèn vàng" and ranking[0]["score"] < ranking[1]["score"]
    assert per_baseline_ms < 2.5


# ── baseline store ───────────────────────────────────────────────────────────
def test_store_keeps_several_baselines(tmp_path):
    store = BaselineStore(tmp_path)
    first = store.save("cam", "r", baseline_frames(3), ROI_POINTS, label="Sáng")
    second = store.save("cam", "r", baseline_frames(3, gain=0.6), ROI_POINTS, label="Tối", source="recording",
                        captured_at=first["captured_at"] - 3600, extra={"recording": {"file": "x.mp4"}})
    entries = store.entries("cam", "r")
    assert [e["label"] for e in entries] == ["Tối", "Sáng"]                # by capture moment
    assert second["count"] == 2 and entries[0]["source"] == "recording" and entries[0]["recording"]["file"] == "x.mp4"
    assert store.info("cam", "r")["id"] == second["id"] and store.info("cam", "r")["count"] == 2
    assert {m.label for m in store.load_all("cam", "r")} == {"Sáng", "Tối"}
    assert store.bev_jpeg("cam", "r", first["id"])[:2] == b"\xff\xd8"
    assert store.find("cam", ROI_POINTS) == "r"
    assert store.delete("cam", "r", first["id"]) == 1
    assert [e["id"] for e in store.entries("cam", "r")] == [second["id"]]
    assert store.delete("cam", "r") == 1 and store.entries("cam", "r") == [] and store.info("cam", "r") is None


def test_new_roi_replaces_the_old_baselines(tmp_path):
    store = BaselineStore(tmp_path)
    store.save("cam", "r", baseline_frames(2), ROI_POINTS)
    store.save("cam", "r", baseline_frames(2, gain=0.8), ROI_POINTS)
    moved = [[x + 0.01, y] for x, y in ROI_POINTS]
    result = store.save("cam", "r", baseline_frames(2), moved)
    assert result["replaced"] == 2 and result["count"] == 1 and len(store.entries("cam", "r")) == 1


def test_baseline_limit(tmp_path, monkeypatch):
    monkeypatch.setattr("core.inspection.baseline_store.MAX_BASELINES", 2)
    store = BaselineStore(tmp_path)
    frames = baseline_frames(1)
    store.save("cam", "r", frames, ROI_POINTS)
    store.save("cam", "r", frames, ROI_POINTS)
    with pytest.raises(BaselineLimitError):
        store.save("cam", "r", frames, ROI_POINTS)
    assert MAX_BASELINES == 12


def test_format_1_baseline_is_migrated(tmp_path):
    old = BaselineStore(tmp_path / "staging")
    meta = old.save("cam", "r", baseline_frames(2), ROI_POINTS)
    source = tmp_path / "staging" / "cam" / "r"
    legacy = tmp_path / "live" / "cam"
    legacy.mkdir(parents=True)
    shutil.copy(source / f"{meta['id']}.png", legacy / "r.png")
    shutil.copy(source / f"{meta['id']}_camera.png", legacy / "r_camera.png")
    v1 = {key: meta[key] for key in ("cam_id", "rule_id", "timestamp", "points", "frame_shape", "crop_origin",
                                     "frames_used", "brightness", "laplacian_var", "size_px", "cell_mm")}
    (legacy / "r.json").write_text(json.dumps({**v1, "format_version": 1}), encoding="utf-8")
    store = BaselineStore(tmp_path / "live")
    entries = store.entries("cam", "r")
    assert len(entries) == 1 and entries[0]["format_version"] == 2 and entries[0]["source"] == "live"
    assert not (legacy / "r.json").exists() and (legacy / "r" / f"{entries[0]['id']}_camera.png").exists()
    assert store.load("cam", "r") is not None and store.list_rules("cam") == ["r"]


# ── temporal confirmation ────────────────────────────────────────────────────
def state(name, label=None):
    return {"state": name, "label": label}


def feed(debouncer, names):
    return [debouncer.update(state(name, "rack" if name == "CARFULL" else None), float(i)) for i, name in enumerate(names)]


def test_occupancy_needs_consecutive_frames():
    debouncer = OccupancyDebouncer(3)
    assert feed(debouncer, ["EMPTY", "EMPTY"]) == [False, False] and debouncer.confirmed is None
    assert debouncer.update(state("EMPTY"), 2.0) and debouncer.confirmed["state"] == "EMPTY"
    # One frame of a passer-by does not change the state; three do, the majority naming it.
    assert feed(debouncer, ["CARFULL", "EMPTY", "CARFULL", "UNKNOWN"]) == [False, False, False, False]
    assert debouncer.update(state("CARFULL", "rack"), 9.0) and debouncer.confirmed["state"] == "CARFULL"


def test_ai_score_hovering_at_threshold_neither_flaps_nor_empties():
    debouncer = OccupancyDebouncer(3)
    feed(debouncer, ["CARFULL"] * 3)
    assert not any(feed(debouncer, ["UNKNOWN", "CARFULL", "UNKNOWN", "UNKNOWN", "CARFULL", "UNKNOWN"]))
    assert debouncer.confirmed["state"] == "CARFULL"
    assert any(feed(debouncer, ["UNKNOWN"] * 3)) and debouncer.confirmed["state"] == "UNKNOWN"
    assert not any(feed(debouncer, ["EMPTY", "EMPTY"])) and debouncer.confirmed["state"] == "UNKNOWN"


# ── runtime: confirmed occupancy on the dashboard and in the FMS message ─────
class FakeFrames:
    pinned: set = set()

    def status(self, cam_id):
        return {"running": True}

    def stop_all(self):
        pass


def test_runtime_confirms_and_reports_occupancy(tmp_path):
    store = BaselineStore(tmp_path)
    store.save("cam", "insp", baseline_frames(5), ROI_POINTS, label="Ban ngày")
    events, published = [], []
    current = {"objects": live()}
    runtime = InspectionRuntime(store, InspectionPipeline(), FakeFrames(), publisher=lambda p: published.append(p) or True,
                                event_sink=events.append, detections=lambda cam: current["objects"])
    runtime.sync_rules("cam", [{"id": "insp", "type": "inspection", "name": "Ô 7", "points": ROI_POINTS,
                                "fms_slot_id": "7", "target_objects": ["rack"],
                                "inspection_config": {"mode": "HYBRID", "monitor_object": False}}])
    station = runtime.station("cam", "insp")
    for index in range(3):
        runtime.process(station, render(Scene(seed=60 + index)), 100.0 + index)
    assert station.occupancy.confirmed["state"] == "EMPTY"
    current["objects"] = live(detection("rack", raise_px=200))
    for index in range(3):
        runtime.process(station, render(Scene(article=CARTON, seed=70 + index)), 110.0 + index)
    described = runtime.describe(station)
    assert described["occupancy"]["state"] == "CARFULL" and described["occupancy"]["label"] == "rack"
    assert described["targets"] == ["rack"] and described["baseline_count"] == 1
    assert described["display"] == {"overlay": True, "label": True, "object": False}
    assert described["frame_size"] == [FRAME_W, FRAME_H] and described["overlay"]["ai_boxes"]
    changes = [e["occupancy"]["state"] for e in events if e["type"] == "inspection_occupancy"]
    assert changes == ["EMPTY", "CARFULL"]
    assert published[-1]["occupancy"] == "CARFULL" and published[-1]["occupancy_class"] == "rack"
    # Display options alone do not reset the station.
    runtime.sync_rules("cam", [{"id": "insp", "type": "inspection", "name": "Ô 7", "points": ROI_POINTS,
                                "fms_slot_id": "7", "target_objects": ["rack"],
                                "inspection_config": {"mode": "HYBRID", "monitor_label": False}}])
    assert runtime.station("cam", "insp") is station and station.config.monitor_label is False
    # The selected classes are part of the decision: changing them starts over.
    runtime.sync_rules("cam", [{"id": "insp", "type": "inspection", "name": "Ô 7", "points": ROI_POINTS,
                                "fms_slot_id": "7", "target_objects": ["robot"],
                                "inspection_config": {"mode": "HYBRID"}}])
    assert runtime.station("cam", "insp") is not station


def test_fms_payload_state_mapping_is_unchanged():
    confirmed = {"confirmed": {"status": "OK", "slot_state": "EMPTY", "code": "OK_EMPTY", "message": "Ô trống"}}
    plain = fms_payload({"id": "r", "fms_slot_id": "7"}, "cam", "CONFIRMED", confirmed, 1.0)
    with_occupancy = fms_payload({"id": "r", "fms_slot_id": "7"}, "cam", "CONFIRMED", confirmed, 1.0,
                                 {"state": "EMPTY", "label": None})
    assert "occupancy" not in plain and plain["state"] == with_occupancy["state"] == "Empty"
    assert with_occupancy["occupancy"] == "EMPTY"


# ── empty-cell frames from the recorded video ────────────────────────────────
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_frames_at_a_recorded_moment(tmp_path):
    folder = tmp_path / "cam1"
    folder.mkdir()
    start = float(int(time.time() - 600.0))          # the segment name has whole seconds
    name = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(start)) + "_0123456789ab.mp4"
    width, height, fps, seconds = 320, 180, 10, 12
    # Each second of the clip has its own grey level: the frames' level proves the seek.
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-g", str(fps * 2),
               "-pix_fmt", "yuv420p", "-movflags", "frag_keyframe+empty_moov+default_base_moof", str(folder / name)]
    frames = b"".join(np.full((height, width, 3), 20 + 15 * (i // fps), np.uint8).tobytes() for i in range(fps * seconds))
    subprocess.run(command, input=frames, check=True, timeout=60)
    source = RecordedFrames(lambda: tmp_path)
    got, info = source.frames_at("cam1", start + 7.3, 5)
    assert len(got) == 5 and got[0].shape == (height, width, 3)
    assert abs(float(got[0].mean()) - (20 + 15 * 7)) < 6
    assert abs(info["at"] - (start + 7.3)) < 0.25
    first, last = source.coverage("cam1")
    assert first == start and last >= start
    for moment in (start - 100.0, time.time() + 60.0, start + 3600.0):
        with pytest.raises(RecordingNotFound):
            source.frames_at("cam1", moment)
    with pytest.raises(RecordingNotFound):
        source.frames_at("../cam1", start + 7.3)
