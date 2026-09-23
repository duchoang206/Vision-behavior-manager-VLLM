import copy
import logging
import math
import queue
import threading
import time
import uuid

import requests

from core.workflow_definition import connector_settings
from core.custom_detector import requested_models


logger = logging.getLogger(__name__)


def object_key(obj):
    identity = obj.get("id")
    if identity is None:
        identity = obj.get("local_id")
    return str(identity) if isinstance(identity, (int, str)) else ""


def ground_point(obj):
    return (float(obj.get("x", 0)) + float(obj.get("w", 0)) / 2,
            float(obj.get("y", 0)) + float(obj.get("h", 0)))


def inside_polygon(point, points):
    inside = False
    previous = points[-1]
    for current in points:
        if (current[1] > point[1]) != (previous[1] > point[1]):
            boundary = (previous[0] - current[0]) * (point[1] - current[1]) / (previous[1] - current[1]) + current[0]
            if point[0] < boundary:
                inside = not inside
        previous = current
    return inside


def fresh_objects(objects, timestamp):
    result = []
    for obj in objects:
        if not isinstance(obj, dict) or obj.get("tracking_state") in {"predicted", "lost", "expired"}:
            continue
        observed = obj.get("observed_at", timestamp)
        try:
            observed = float(observed)
            if observed > 1e11:
                observed /= 1000
            if not math.isfinite(observed) or not -.25 <= timestamp - observed <= 1:
                continue
            if not all(math.isfinite(float(obj.get(key, 0))) for key in ("x", "y", "w", "h", "confidence")):
                continue
        except (TypeError, ValueError):
            continue
        result.append(obj)
    return result


class GraphEvaluator:
    def __init__(self, definition, order):
        self.definition = definition
        self.nodes = {node["id"]: node for node in definition["nodes"]}
        self.order = order
        self.parents = {node_id: [] for node_id in self.nodes}
        for edge in definition["edges"]:
            self.parents[edge["target"]].append(edge["source"])
        self.memory = {}
        self.cooldowns = {}
        self.totals = {}

    def evaluate(self, camera_id, objects, timestamp):
        values, summaries, actions = {}, {}, []
        source_objects = fresh_objects(objects, timestamp)
        for node_id in self.order:
            node = self.nodes[node_id]
            kind, config = node["type"], node["config"]
            empty_match = False
            if kind == "source":
                selected, signal = source_objects, True
                complete = len(source_objects) == len(objects)
            else:
                upstream = [values[parent] for parent in self.parents[node_id]]
                merged = {object_key(obj) or (obj.get("label"), obj.get("x"), obj.get("y"), obj.get("w"), obj.get("h")): obj
                          for selected, signal, empty, complete in upstream if signal for obj in selected}
                selected = list(merged.values())
                signal = any(signal for selected, signal, empty, complete in upstream)
                empty_match = any(signal and empty for selected, signal, empty, complete in upstream)
                complete = all(complete for selected, signal, empty, complete in upstream)
            state = self.memory.setdefault((node_id, camera_id), {})
            if kind in {"detector", "classifier", "matcher"}:
                allowed = {value.casefold() for value in config.get("labels" if kind == "matcher" else "classes", [])}
                selected = [obj for obj in selected if (
                    str(obj.get("label", "")).casefold() in allowed if kind == "matcher" else
                    bool({str(obj.get("class", "")).casefold(), str(obj.get("category", "")).casefold()} & allowed))
                    and float(obj.get("confidence", 1)) >= (config.get("confidence", .25) if kind == "detector" else 0)
                    and (kind != "detector" or not config.get("model_id") or obj.get("model_id") == config["model_id"])]
                empty_match = False
            elif kind == "roi":
                selected = [obj for obj in selected if inside_polygon(ground_point(obj), config["points"])]
                empty_match = False
            elif kind == "line":
                start, end = config["points"]
                direction_x, direction_y = end[0] - start[0], end[1] - start[1]
                length = math.hypot(direction_x, direction_y)
                crossings = []
                for obj in selected:
                    key, point = object_key(obj), ground_point(obj)
                    if not key:
                        continue
                    distance = (direction_x * (point[1] - start[1]) - direction_y * (point[0] - start[0])) / length
                    if abs(distance) < config.get("hysteresis", .005):
                        continue
                    previous = state.get(key)
                    if previous and timestamp - previous["at"] <= 1 and previous["distance"] * distance < 0:
                        fraction = previous["distance"] / (previous["distance"] - distance)
                        crossing = [previous["point"][axis] + fraction * (point[axis] - previous["point"][axis]) for axis in (0, 1)]
                        along = ((crossing[0] - start[0]) * direction_x + (crossing[1] - start[1]) * direction_y) / length ** 2
                        direction = "a_to_b" if previous["distance"] < 0 else "b_to_a"
                        if 0 <= along <= 1 and config.get("direction", "both") in {"both", direction}:
                            crossings.append(obj)
                    state[key] = {"distance": distance, "point": point, "at": timestamp}
                selected = crossings
                total_key = (node_id, camera_id)
                self.totals[total_key] = self.totals.get(total_key, 0) + len(crossings)
                signal = signal and bool(selected)
            elif kind == "dwell":
                dwelling, present = [], set()
                for obj in selected:
                    key = object_key(obj)
                    if not key:
                        continue
                    present.add(key)
                    previous = state.get(key)
                    started = previous["started"] if previous and timestamp - previous["at"] <= 1 else timestamp
                    state[key] = {"at": timestamp, "started": started}
                    if timestamp - started >= config.get("seconds", 5):
                        dwelling.append(obj)
                for key in set(state) - present:
                    state.pop(key, None)
                selected = dwelling
                signal = signal and bool(selected)
            elif kind == "counter":
                count, threshold = len(selected), config.get("count", 1)
                operator = config.get("operator", "gte")
                signal = signal and {"gte": count >= threshold, "lte": count <= threshold, "eq": count == threshold}[operator]
                signal = signal and (complete or (operator == "gte" and count > 0))
                empty_match = signal
            elif kind == "proximity":
                valid = [obj for obj in selected if obj.get("spatial_valid") and obj.get("inside_calibrated_area")
                         and all(type(obj.get(key)) in (int, float) and math.isfinite(obj[key]) for key in ("floor_x", "floor_y"))]
                people = [obj for obj in valid if str(obj.get("category", obj.get("class"))).lower() in {"person", "human", "worker"}]
                robots = [obj for obj in valid if str(obj.get("category", obj.get("class"))).lower() in {"robot", "delivery-robot", "agv", "amr"}]
                close = {}
                for person in people:
                    for robot in robots:
                        if math.hypot(person["floor_x"] - robot["floor_x"], person["floor_y"] - robot["floor_y"]) <= config.get("distance_m", 1.5):
                            close[object_key(person)], close[object_key(robot)] = person, robot
                selected = list(close.values())
                signal = signal and bool(selected)
            elif kind in {"event", "webhook"}:
                signal = signal and (bool(selected) or empty_match)
                if config.get("require_fms"):
                    selected = [obj for obj in selected if obj.get("spatial_valid") and obj.get("inside_calibrated_area") and obj.get("fms_world_position")]
                    signal = signal and bool(selected)
                cooldown_key = (node_id, camera_id)
                last = self.cooldowns.get(cooldown_key, -float("inf"))
                if signal and timestamp - last >= config.get("cooldown_seconds", 10):
                    actions.append({"node": node, "objects": selected, "count": len(selected), "timestamp": timestamp})
                    self.cooldowns[cooldown_key] = timestamp
            for key in [key for key, value in state.items() if timestamp - value["at"] > 2]:
                state.pop(key, None)
            values[node_id] = (selected, signal, empty_match, complete)
            summaries[node_id] = {"type": kind, "count": len(selected), "signal": signal and (bool(selected) or empty_match),
                                  "total_crossings": self.totals.get((node_id, camera_id)),
                                  "objects": [{key: obj.get(key) for key in ("id", "label", "class", "category", "confidence", "fms_world_position")} for obj in selected[:25]]}
        return summaries, actions


class WorkflowRuntime:
    fields = ("id", "local_id", "label", "class", "category", "confidence", "x", "y", "w", "h",
              "tracking_state", "observed_at", "floor_x", "floor_y", "spatial_valid", "inside_calibrated_area", "fms_world_position", "model_id")

    def __init__(self, store, validator, broadcast=None):
        self.store, self.validator, self.broadcast = store, validator, broadcast
        self.lock = threading.Lock()
        self.pending = {}
        self.last_timestamp = {}
        self.camera_ids = frozenset()
        self.evaluators = {}
        self.snapshots = {}
        self.events = queue.Queue(maxsize=256)
        self.wake = threading.Event()
        self.reload = threading.Event()
        self.stopping = threading.Event()
        self.threads = []
        self.last_error = None
        self.replaced_frames = 0
        self.dropped_events = 0

    def start(self):
        if any(thread.is_alive() for thread in self.threads):
            return
        self.store.initialize()
        self.threads = []
        self.stopping.clear()
        for name, target in (("workflow-metadata", self._run), ("workflow-actions", self._actions), ("workflow-connectors", self._deliveries)):
            thread = threading.Thread(name=name, target=target, daemon=True)
            thread.start()
            self.threads.append(thread)

    def stop(self):
        self.stopping.set()
        self.wake.set()
        for thread in self.threads:
            thread.join(timeout=6)

    def refresh(self):
        self.reload.set()
        self.wake.set()

    def publish(self, payload):
        if not self.camera_ids or self.stopping.is_set():
            return
        now = time.time()
        try:
            timestamp = float(payload.get("timestamp", now))
            if timestamp > 1e11:
                timestamp /= 1000
            if not math.isfinite(timestamp) or not -.25 <= now - timestamp <= 1:
                return
        except (TypeError, ValueError):
            return
        streams = payload.get("streams")
        if not isinstance(streams, list):
            return
        for stream in streams:
            if not isinstance(stream, dict) or not isinstance(stream.get("objects"), list):
                continue
            camera_id = str(stream.get("cam_id", ""))
            if camera_id not in self.camera_ids:
                continue
            objects = [{key: obj[key] for key in self.fields if key in obj} for obj in stream.get("objects", [])[:256] if isinstance(obj, dict)]
            model_id = stream.get("model_id") if payload.get("source") == "custom_deepstream" else None
            mailbox_key = (camera_id, model_id) if model_id else camera_id
            with self.lock:
                if timestamp <= self.last_timestamp.get(mailbox_key, 0):
                    continue
                self.last_timestamp[mailbox_key] = timestamp
                if mailbox_key in self.pending:
                    self.replaced_frames += 1
                self.pending[mailbox_key] = (timestamp, objects, model_id)
        self.wake.set()

    def status(self, deployment_id=None):
        with self.lock:
            snapshots = dict(self.snapshots)
        if deployment_id:
            result = copy.deepcopy(snapshots.get(deployment_id, {"state": "not_running", "cameras": {}}))
            fresh = [time.time() - camera.get("last_frame_at", 0) <= 3 for camera in result["cameras"].values()]
            if result["state"] == "running" and not any(fresh):
                result["state"] = "waiting_metadata"
            return result
        return {"healthy": bool(self.threads) and all(thread.is_alive() for thread in self.threads) and not self.last_error,
                "error": self.last_error, "replaced_frames": self.replaced_frames, "dropped_events": self.dropped_events,
                "queued_actions": self.events.qsize()}

    def _refresh(self):
        active = self.store.active()
        evaluators, camera_ids, new_snapshots = {}, set(), {}
        for deployment in active:
            validation = self.validator(deployment["definition"])
            deployment_id = deployment["id"]
            if not validation["valid"]:
                new_snapshots[deployment_id] = {"state": "blocked", "error": "; ".join(validation["errors"]), "cameras": {}}
                continue
            previous, evaluator = self.evaluators.get(deployment_id, (None, None))
            if previous and previous["updated_at"] != deployment["updated_at"]:
                evaluator = None
            evaluators[deployment_id] = (deployment, evaluator or GraphEvaluator(deployment["definition"], validation["order"]))
            camera_ids.update(validation["camera_ids"])
            new_snapshots[deployment_id] = self.snapshots.get(deployment_id, {"state": "waiting_metadata", "cameras": {}})
        self.evaluators = evaluators
        self.camera_ids = frozenset(camera_ids)
        with self.lock:
            self.snapshots = new_snapshots
            self.last_timestamp = {key: timestamp for key, timestamp in self.last_timestamp.items()
                                   if (key[0] if isinstance(key, tuple) else key) in camera_ids}

    def _run(self):
        next_refresh = 0
        while not self.stopping.is_set():
            try:
                if time.monotonic() >= next_refresh or self.reload.is_set():
                    self.reload.clear()
                    self._refresh()
                    next_refresh = time.monotonic() + 2
                    self.last_error = None
                self.wake.wait(.1)
                self.wake.clear()
                with self.lock:
                    pending, self.pending = self.pending, {}
                for mailbox_key, (timestamp, objects, model_id) in pending.items():
                    camera_id = mailbox_key[0] if isinstance(mailbox_key, tuple) else mailbox_key
                    if time.time() - timestamp > 1:
                        continue
                    for deployment_id, (deployment, evaluator) in self.evaluators.items():
                        source = next(node for node in deployment["definition"]["nodes"] if node["type"] == "source")
                        if camera_id not in source["config"]["camera_ids"]:
                            continue
                        if timestamp < deployment["updated_at"].timestamp():
                            continue
                        required_models = requested_models(deployment["definition"])
                        if required_models and model_id not in required_models:
                            continue
                        summaries, actions = evaluator.evaluate(camera_id, objects, timestamp)
                        with self.lock:
                            snapshot = self.snapshots[deployment_id]
                            previous = snapshot["cameras"].get(camera_id, {"frames": 0, "events": 0})
                            camera = dict(previous, last_frame_at=timestamp, nodes=summaries, frames=previous["frames"] + 1)
                            self.snapshots[deployment_id] = dict(snapshot, state="running", cameras={**snapshot["cameras"], camera_id: camera})
                        for action in actions:
                            first = action["objects"][0] if action["objects"] else {}
                            node, config = action["node"], action["node"]["config"]
                            event = {"event_id": uuid.uuid4().hex, "deployment_id": deployment_id,
                                     "pipeline_id": deployment["pipeline_id"], "revision": deployment["revision"],
                                     "deployment_generation": deployment["updated_at"].isoformat(),
                                     "cam_id": camera_id, "node_id": node["id"], "rule_type": "workflow",
                                     "timestamp": timestamp, "severity": config.get("severity", "info"),
                                     "description": config.get("message") or deployment["definition"]["name"],
                                     "count": action["count"], "objects": summaries[node["id"]]["objects"],
                                     "bbox": [first.get(key, 0) for key in ("x", "y", "w", "h")],
                                     "fms_position": first.get("fms_world_position"), "connector": config.get("connector"),
                                     "evidence_requested": bool(config.get("evidence"))}
                            try:
                                self.events.put_nowait(event)
                                with self.lock:
                                    snapshot = self.snapshots[deployment_id]
                                    camera = dict(camera, events=camera["events"] + 1)
                                    self.snapshots[deployment_id] = dict(snapshot, cameras={**snapshot["cameras"], camera_id: camera})
                            except queue.Full:
                                self.dropped_events += 1
            except Exception as error:
                self.last_error = type(error).__name__ + ": " + str(error)[:300]
                logger.exception("Workflow metadata runtime failed")
                self.stopping.wait(2)

    def _actions(self):
        while not self.stopping.is_set():
            try:
                event = self.events.get(timeout=.5)
            except queue.Empty:
                continue
            try:
                if self.store.write_event(event) and self.broadcast:
                    self.broadcast(event)
            except Exception:
                self.dropped_events += 1
                logger.exception("Workflow event persistence failed")
            finally:
                self.events.task_done()

    def _deliveries(self):
        session = requests.Session()
        session.trust_env = False
        while not self.stopping.is_set():
            try:
                if not connector_settings():
                    self.stopping.wait(2)
                    continue
                delivery = self.store.claim_delivery()
                if not delivery:
                    self.stopping.wait(1)
                    continue
                connector = connector_settings().get(delivery["connector"])
                error = None
                try:
                    if not connector:
                        raise ValueError("Connector đã bị gỡ cấu hình.")
                    headers = {"Content-Type": "application/json", "Idempotency-Key": delivery["id"]}
                    if connector.get("token"):
                        headers["Authorization"] = "Bearer " + connector["token"]
                    with session.post(connector["url"], json=delivery["payload"], headers=headers, timeout=(2, 3), allow_redirects=False, stream=True) as response:
                        if not 200 <= response.status_code < 300:
                            error = f"HTTP {response.status_code}"
                except Exception as failure:
                    error = type(failure).__name__
                self.store.finish_delivery(delivery, error)
            except Exception:
                logger.exception("Workflow connector worker failed")
                self.stopping.wait(3)
        session.close()
