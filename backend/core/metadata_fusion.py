import threading
import time

from core.identity_utils import identity_global_id


def is_person(obj):
    return str(obj.get("class", "")).lower() in {"person", "human", "worker"}


def unique_registered(objects):
    registered = {}
    for obj in objects:
        category = str(obj.get("category") or obj.get("class") or "").lower()
        label = str(obj.get("label") or "").strip()
        if category not in {"robot", "rack"} or not label:
            continue
        canonical = dict(obj, **{"label": label, "class": category, "category": category})
        canonical["id"] = identity_global_id(label, category)
        key = label.casefold()
        priority = (obj.get("tracking_state") != "predicted", float(obj.get("confidence", 0)))
        if key not in registered or priority > registered[key][0]:
            registered[key] = (priority, canonical)
    return [item[1] for item in registered.values()]


class MetadataFusion:
    def __init__(self, ttl=1.0):
        self.ttl = ttl
        self.snapshots = {}
        self.lock = threading.RLock()

    def update(self, cam_id, source, objects, now=None):
        now = time.monotonic() if now is None else now
        with self.lock:
            sources = self.snapshots.setdefault(cam_id, {})
            if source == "identity_template":
                sources[source] = (now, unique_registered(objects))
            elif source in {"deepstream", "cpu_fallback"}:
                sources[source] = (now, [dict(obj, label=None, category="person") for obj in objects if is_person(obj)])
            for name in list(sources):
                if now - sources[name][0] > self.ttl:
                    del sources[name]
            people = sources.get("deepstream", sources.get("cpu_fallback", (now, [])))[1]
            registered = sources.get("identity_template", (now, []))[1]
            return [dict(obj) for obj in people + registered]

    def remove_camera(self, cam_id):
        with self.lock:
            self.snapshots.pop(cam_id, None)
