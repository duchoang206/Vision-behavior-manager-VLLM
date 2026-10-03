"""Realtime inspection: camera frames → pipeline → temporal FSM → FMS / dashboard.

* ``FrameSource`` keeps one decoding thread per camera that holds the latest
  frame (MediaMTX relay first, the camera's own URL as fallback), so the
  Building view's "Test" answers from a live frame without reopening RTSP.
* ``InspectionRuntime`` receives the ``inspection`` rules from the behaviour
  engine, evaluates every station at ``INSPECTION_RUNTIME_FPS`` and, on each
  confirmed state change or safety interlock, publishes the FMS message
  (``fms_payload``) and a dashboard event.  It never blocks a video thread:
  publishing callbacks only hand data over.
"""

import base64
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .baseline_store import BaselineStore
from .config import (OCC_UNKNOWN, SLOT_EMPTY, STATUS_DETECTED, STATUS_NG, STATUS_OK, STATUS_UNCERTAIN,
                     InspectionConfig, decision_settings, parse_config)
from .live_detections import LiveDetections
from .occupancy import OCCUPANCY_LABEL, OccupancyDebouncer
from .pipeline import InspectionPipeline
from .temporal_inspection_manager import EVENT_CONFIRMED, EVENT_INTERLOCK, TemporalInspectionManager

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
logger = logging.getLogger("Inspection.Runtime")

FMS_STATE_FULL = "Car Full"
FMS_STATE_EMPTY = "Empty"
FMS_STATE_BLOCKED = "Slot Blocked"
FMS_STATE_HOLD = "Hold"
INTERLOCK_CODE = "SAFETY_INTERLOCK"


# ── frames ───────────────────────────────────────────────────────────────────
class CameraFrameReader:
    def __init__(self, cam_id: str, urls: Sequence[str]):
        self.cam_id = cam_id
        self.urls = [u for u in urls if u]
        self.frame: Optional[np.ndarray] = None
        self.frame_at = 0.0
        self.seq = 0
        self.last_used = time.monotonic()
        self.error = ""
        self._running = True
        self._cond = threading.Condition()
        self._thread = threading.Thread(target=self._run, name=f"inspection-frames-{cam_id}", daemon=True)
        self._thread.start()

    def _open(self, url: str):
        try:
            return cv2.VideoCapture(url, cv2.CAP_FFMPEG, [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
                                                         cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000])
        except Exception:
            return cv2.VideoCapture(url)

    def _run(self) -> None:
        index = 0
        while self._running:
            if not self.urls:
                self.error = "Camera không có URL"
                time.sleep(1.0)
                continue
            url = self.urls[index % len(self.urls)]
            capture = self._open(url)
            if not capture.isOpened():
                self.error = f"Không mở được luồng {index % len(self.urls) + 1}/{len(self.urls)}"
                capture.release()
                index += 1
                time.sleep(0.5)
                continue
            self.error = ""
            failures = 0
            while self._running:
                ok, frame = capture.read()
                if not ok or frame is None:
                    failures += 1
                    if failures >= 3:
                        break
                    continue
                failures = 0
                with self._cond:
                    self.frame, self.frame_at, self.seq = frame, time.time(), self.seq + 1
                    self._cond.notify_all()
            capture.release()
            index += 1

    def latest(self, max_age: float, timeout: float) -> Optional[Tuple[np.ndarray, float, int]]:
        self.last_used = time.monotonic()
        deadline = time.monotonic() + timeout
        with self._cond:
            while self.frame is None or time.time() - self.frame_at > max_age:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cond.wait(remaining)
            return self.frame, self.frame_at, self.seq

    def next_after(self, seq: int, timeout: float) -> Optional[Tuple[np.ndarray, float, int]]:
        self.last_used = time.monotonic()
        deadline = time.monotonic() + timeout
        with self._cond:
            while self.frame is None or self.seq <= seq:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cond.wait(remaining)
            return self.frame, self.frame_at, self.seq

    def stop(self) -> None:
        self._running = False


class FrameSource:
    def __init__(self, url_resolver: Callable[[str], List[str]], idle_sec: float = 120.0):
        self.url_resolver = url_resolver
        self.idle_sec = idle_sec
        self.pinned: set = set()
        self._readers: Dict[str, CameraFrameReader] = {}
        self._lock = threading.Lock()

    def reader(self, cam_id: str) -> CameraFrameReader:
        with self._lock:
            self._reap()
            reader = self._readers.get(cam_id)
            if reader is None:
                reader = CameraFrameReader(cam_id, self.url_resolver(cam_id))
                self._readers[cam_id] = reader
            return reader

    def _reap(self) -> None:
        now = time.monotonic()
        for cam_id in [c for c, r in self._readers.items()
                       if c not in self.pinned and now - r.last_used > self.idle_sec]:
            self._readers.pop(cam_id).stop()

    def latest(self, cam_id: str, max_age: float = 2.0, timeout: float = 6.0):
        return self.reader(cam_id).latest(max_age, timeout)

    def frames(self, cam_id: str, count: int = 5, timeout: float = 8.0) -> List[np.ndarray]:
        reader = self.reader(cam_id)
        first = reader.latest(2.0, timeout)
        if first is None:
            return []
        frames, seq = [first[0].copy()], first[2]
        deadline = time.monotonic() + timeout
        while len(frames) < count and time.monotonic() < deadline:
            packet = reader.next_after(seq, max(0.0, deadline - time.monotonic()))
            if packet is None:
                break
            frames.append(packet[0].copy())
            seq = packet[2]
            time.sleep(0.12)              # spread the median over ~0.5 s of sensor noise
        return frames

    def status(self, cam_id: str) -> dict:
        reader = self._readers.get(cam_id)
        if reader is None:
            return {"running": False}
        return {"running": True, "seq": reader.seq, "age_sec": round(time.time() - reader.frame_at, 3)
                if reader.frame_at else None, "error": reader.error}

    def stop_all(self) -> None:
        with self._lock:
            for reader in self._readers.values():
                reader.stop()
            self._readers.clear()


# ── FMS message ──────────────────────────────────────────────────────────────
def _r(value, digits: int = 1):
    return None if value is None else round(float(value), digits)


def fms_payload(rule: dict, cam_id: str, event: str, state: dict, now: Optional[float] = None,
                occupancy: Optional[dict] = None) -> dict:
    """WCS message for a confirmed inspection state (target.md Tests 5.1-5.3).

    ``occupancy`` (the confirmed CARFULL / EMPTY / UNKNOWN of occupancy.py) is
    added as ``occupancy`` / ``occupancy_class``; ``state`` is unchanged."""
    confirmed = state.get("confirmed") or {}
    slot_id = str(rule.get("fms_slot_id") or rule.get("id"))
    base = {"slot_id": slot_id, "camera_id": cam_id, "rule_id": str(rule.get("id")),
            "timestamp": round(now if now is not None else time.time(), 3)}
    if occupancy:
        base.update(occupancy=occupancy.get("state"), occupancy_class=occupancy.get("label"))
    if event == EVENT_INTERLOCK:
        return {**base, "state": FMS_STATE_HOLD, "action": "AGV_STOP_AND_WAIT",
                "inspection": {"status": STATUS_UNCERTAIN, "error_code": INTERLOCK_CODE,
                               "message": "Khóa an toàn: trạng thái ô vừa thay đổi – AGV dừng chờ xác nhận"}}
    status = confirmed.get("status")
    m = confirmed.get("measurement") or {}
    code = confirmed.get("code")
    if status == STATUS_OK and confirmed.get("slot_state") == SLOT_EMPTY:
        return {**base, "state": FMS_STATE_EMPTY, "quality": "OK",
                "inspection": {"status": STATUS_OK, "error_code": code, "message": confirmed.get("message")}}
    if status == STATUS_OK:
        inspection = {"status": STATUS_OK, "width_mm": _r(m.get("width_mm")), "height_mm": _r(m.get("height_mm")),
                      "rotation_deg": _r(m.get("rotation_deg")), "center_offset_mm": _r(m.get("offset_mm")),
                      "code": code}
        if confirmed.get("operator_decision"):
            inspection["operator_decision"] = confirmed["operator_decision"]
        return {**base, "state": FMS_STATE_FULL, "quality": "OK", "inspection": inspection}
    if status == STATUS_NG:
        return {**base, "state": FMS_STATE_BLOCKED, "quality": "NG", "reason": "Defect / Misaligned",
                "action": "AGV_CANCEL_APPROACH",
                "inspection": {"status": STATUS_NG, "error_code": code, "message": confirmed.get("message"),
                               "errors": confirmed.get("errors") or [code]}}
    if status == STATUS_DETECTED:
        return {**base, "state": FMS_STATE_FULL, "quality": "UNVERIFIED",
                "inspection": {"status": STATUS_DETECTED, "error_code": code, "class": confirmed.get("ai_class"),
                               "confidence": confirmed.get("ai_confidence"), "message": confirmed.get("message")}}
    return {**base, "state": FMS_STATE_HOLD, "action": "AGV_WAIT_CONFIRMATION",
            "inspection": {"status": STATUS_UNCERTAIN, "error_code": code, "message": confirmed.get("message")}}


# ── runtime ──────────────────────────────────────────────────────────────────
def _targets(rule: dict) -> List[str]:
    raw = rule.get("target_objects")
    return [str(t).strip() for t in raw if str(t).strip()] if isinstance(raw, (list, tuple)) else []


@dataclass
class Station:
    cam_id: str
    rule: dict
    config: InspectionConfig
    fsm: TemporalInspectionManager
    occupancy: OccupancyDebouncer = field(default_factory=OccupancyDebouncer)
    last_report: Optional[dict] = None
    last_payload: Optional[dict] = None
    snapshot: Optional[str] = None
    frame_size: Optional[List[int]] = None
    processed: int = 0
    timings: Deque[float] = field(default_factory=lambda: deque(maxlen=50))

    @property
    def rule_id(self) -> str:
        return str(self.rule.get("id"))

    @property
    def targets(self) -> List[str]:
        return _targets(self.rule)


def _summary(report: dict) -> dict:
    keys = ("status", "slot_state", "code", "message", "errors", "measurement", "ai_class", "ai_confidence",
            "timestamp", "occupancy")
    data = {key: report.get(key) for key in keys}
    data["arbitration"] = report.get("arbitration")
    return data


def _overlay(report: Optional[dict]) -> Optional[dict]:
    overlay = (report or {}).get("overlay")
    return {k: v for k, v in overlay.items() if k != "bev_image"} if overlay else None


class InspectionRuntime:
    def __init__(self, store: BaselineStore, pipeline: InspectionPipeline, frames: FrameSource,
                 publisher: Optional[Callable[[dict], bool]] = None,
                 event_sink: Optional[Callable[[dict], None]] = None, fps: Optional[float] = None,
                 uncertain_hold_sec: float = 1.5,
                 detections: Optional[Callable[[str], LiveDetections]] = None):
        self.store = store
        self.pipeline = pipeline
        self.frames = frames
        self.publisher = publisher
        self.event_sink = event_sink
        self.detections = detections
        self.fps = float(fps or os.getenv("INSPECTION_RUNTIME_FPS", "5"))
        self.uncertain_hold_sec = uncertain_hold_sec
        self.stations: Dict[Tuple[str, str], Station] = {}
        self.published: Deque[dict] = deque(maxlen=200)
        self._lock = threading.RLock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._wake = threading.Event()

    def live(self, cam_id: str) -> Optional[LiveDetections]:
        """The camera model's latest detections (the AI layer shared with the occupancy rule)."""
        if self.detections is None:
            return None
        try:
            return self.detections(cam_id)
        except Exception:
            logger.exception("Live detections unavailable for camera %s", cam_id)
            return None

    # ── rules ────────────────────────────────────────────────────────────────
    def sync_rules(self, cam_id: str, rules: Iterable[dict]) -> None:
        """Behaviour-engine listener: (re)load the inspection stations of a camera."""
        wanted = {}
        for rule in rules or []:
            if str(rule.get("type") or rule.get("rule_type") or "").lower() != "inspection":
                continue
            points = rule.get("camera_points") or rule.get("points") or []
            if len(points) != 4:
                continue
            try:
                config = parse_config(rule.get("inspection_config"))
            except Exception:
                logger.warning("Inspection rule %s has an invalid configuration", rule.get("id"))
                continue
            wanted[(cam_id, str(rule.get("id")))] = (dict(rule, points=points), config)
        with self._lock:
            for key in [k for k in self.stations if k[0] == cam_id and k not in wanted]:
                del self.stations[key]
            for key, (rule, config) in wanted.items():
                current = self.stations.get(key)
                # Display-only settings (Monitor overlay) never reset the station's confirmed state.
                if current is None or current.rule.get("points") != rule.get("points") \
                        or decision_settings(current.config) != decision_settings(config) \
                        or current.rule.get("fms_slot_id") != rule.get("fms_slot_id") \
                        or _targets(current.rule) != _targets(rule):
                    self.stations[key] = Station(cam_id, rule, config,
                                                 TemporalInspectionManager(config.temporal_window_frames,
                                                                           self.uncertain_hold_sec),
                                                 OccupancyDebouncer(config.temporal_window_frames))
                else:
                    current.rule = rule
                    current.config = config
            self.frames.pinned = {k[0] for k in self.stations}
        self._wake.set()

    def remove_camera(self, cam_id: str) -> None:
        self.sync_rules(cam_id, [])

    # ── lifecycle ────────────────────────────────────────────────────────────
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="inspection-runtime", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        self.frames.stop_all()

    def _loop(self) -> None:
        period = 1.0 / max(0.5, self.fps)
        last_seq: Dict[str, int] = {}
        while self._running:
            started = time.monotonic()
            with self._lock:
                stations = list(self.stations.values())
            by_camera: Dict[str, List[Station]] = {}
            for station in stations:
                by_camera.setdefault(station.cam_id, []).append(station)
            for cam_id, group in by_camera.items():
                try:
                    packet = self.frames.latest(cam_id, max_age=2.0, timeout=0.0)
                    if packet is None:
                        for station in group:
                            self.process(station, None, time.time())
                        continue
                    if packet[2] == last_seq.get(cam_id):
                        continue
                    last_seq[cam_id] = packet[2]
                    for station in group:
                        self.process(station, packet[0], packet[1])
                except Exception:
                    logger.exception("Inspection runtime failed on camera %s", cam_id)
            self._wake.wait(max(0.0, period - (time.monotonic() - started)))
            self._wake.clear()

    # ── evaluation ───────────────────────────────────────────────────────────
    def process(self, station: Station, frame: Optional[np.ndarray], now: float) -> dict:
        baselines = self.store.load_all(station.cam_id, station.rule_id)
        report = self.pipeline.run(frame, station.rule["points"], station.config, baselines,
                                   targets=station.targets, live=self.live(station.cam_id))
        station.last_report = report
        station.processed += 1
        if frame is not None:
            station.frame_size = [int(frame.shape[1]), int(frame.shape[0])]
        station.timings.append(float(report.get("timing_ms", {}).get("total") or 0.0))
        if report.get("occupancy") and station.occupancy.update(report["occupancy"], now):
            self._occupancy_event(station)
        update = station.fsm.update(_summary(report), now)
        if update.event:
            if update.event == EVENT_CONFIRMED and update.state["confirmed_status"] == STATUS_UNCERTAIN \
                    and frame is not None:
                station.snapshot = self._snapshot(frame, station)
            elif update.event == EVENT_CONFIRMED:
                station.snapshot = None
            self._publish(station, update.event, update.state, now)
        return report

    def _snapshot(self, frame: np.ndarray, station: Station) -> Optional[str]:
        try:
            rect = self.pipeline.rectifier(station.rule["points"], station.config.cell_mm)
            image = rect.warp(frame, max(rect.cell_mm) / 320.0)
            ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
            return "data:image/jpeg;base64," + base64.b64encode(data.tobytes()).decode("ascii") if ok else None
        except Exception:
            return None

    def _occupancy_event(self, station: Station) -> None:
        """Dashboard event when the confirmed occupancy of the cell changes."""
        if self.event_sink is None:
            return
        occupancy = station.occupancy.confirmed or {}
        state = occupancy.get("state")
        what = f" ({occupancy['label']} {occupancy['confidence']:.0%})" \
            if occupancy.get("label") and occupancy.get("confidence") is not None else ""
        try:
            self.event_sink({
                "type": "inspection_occupancy", "rule_type": "inspection", "cam_id": station.cam_id,
                "rule_id": station.rule_id, "severity": "warning" if state == OCC_UNKNOWN else "info",
                "description": f"[{station.rule.get('name') or station.rule_id}] "
                               f"{OCCUPANCY_LABEL.get(state, state)}{what} – {occupancy.get('reason') or ''}",
                "occupancy": occupancy, "fms_slot_id": station.rule.get("fms_slot_id"),
            })
        except Exception as exc:
            logger.warning("Inspection event sink failed: %s", exc)

    def _publish(self, station: Station, event: str, state: dict, now: float) -> None:
        payload = fms_payload(station.rule, station.cam_id, event, state, now, station.occupancy.confirmed)
        station.last_payload = payload
        delivered = None
        if self.publisher is not None and station.rule.get("enable_fms_dispatch", True) is not False:
            try:
                delivered = bool(self.publisher(payload))
            except Exception as exc:
                logger.warning("Inspection FMS publish failed: %s", exc)
                delivered = False
        self.published.append({"at": now, "event": event, "payload": payload, "delivered": delivered})
        if self.event_sink is not None:
            status = payload["inspection"]["status"]
            message = payload["inspection"].get("message") or payload["inspection"].get("code") or status
            try:
                self.event_sink({
                    "type": "inspection_state", "rule_type": "inspection", "cam_id": station.cam_id,
                    "rule_id": station.rule_id,
                    "severity": {"NG": "critical", "UNCERTAIN": "warning"}.get(status, "info"),
                    "description": f"[{station.rule.get('name') or station.rule_id}] {payload['state']} – {message}",
                    "inspection": payload, "fsm_event": event,
                })
            except Exception as exc:
                logger.warning("Inspection event sink failed: %s", exc)

    # ── operator / API ───────────────────────────────────────────────────────
    def station(self, cam_id: str, rule_id: str) -> Optional[Station]:
        with self._lock:
            return self.stations.get((cam_id, str(rule_id)))

    def apply_decision(self, cam_id: str, rule_id: str, approve: bool, operator: str = "") -> dict:
        station = self.station(cam_id, rule_id)
        if station is None:
            raise KeyError(rule_id)
        now = time.time()
        update = station.fsm.apply_operator_decision(approve, operator, now)
        station.snapshot = None
        self._publish(station, update.event, update.state, now)
        return self.describe(station)

    def describe(self, station: Station) -> dict:
        timings = list(station.timings)
        info = self.store.info(station.cam_id, station.rule_id)
        report = station.last_report or {}
        occupancy = station.occupancy.snapshot()
        return {
            "cam_id": station.cam_id, "rule_id": station.rule_id, "name": station.rule.get("name"),
            "fms_slot_id": station.rule.get("fms_slot_id"), "mode": station.config.mode,
            "points": station.rule.get("points"), "state": station.fsm.snapshot(),
            "occupancy": occupancy["confirmed"], "occupancy_since": occupancy["since"],
            "latest_occupancy": occupancy["latest"],
            "latest": _summary(station.last_report) if station.last_report else None,
            "overlay": _overlay(station.last_report), "baseline_choice": report.get("baseline_choice"),
            "display": {"overlay": station.config.monitor_overlay, "label": station.config.monitor_label,
                        "object": station.config.monitor_object},
            "targets": station.targets, "frame_size": station.frame_size,
            "last_payload": station.last_payload, "snapshot": station.snapshot,
            "baseline": info is not None, "baseline_count": info.get("count", 0) if info else 0,
            "processed": station.processed,
            "latency_ms": {"mean": round(float(np.mean(timings)), 3) if timings else None,
                           "max": round(float(np.max(timings)), 3) if timings else None},
            "frames": self.frames.status(station.cam_id),
        }

    def states(self, cam_id: Optional[str] = None) -> List[dict]:
        with self._lock:
            stations = [s for s in self.stations.values() if cam_id is None or s.cam_id == cam_id]
        return [self.describe(station) for station in stations]
