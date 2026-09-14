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

def label_registered_people(people, identities):
    labeled = [dict(person) for person in people]
    candidates = []
    for person_index, person in enumerate(people):
        for identity in identities:
            left = max(person.get("x", 0), identity.get("x", 0))
            top = max(person.get("y", 0), identity.get("y", 0))
            right = min(person.get("x", 0) + person.get("w", 0), identity.get("x", 0) + identity.get("w", 0))
            bottom = min(person.get("y", 0) + person.get("h", 0), identity.get("y", 0) + identity.get("h", 0))
            intersection = max(0, right - left) * max(0, bottom - top)
            union = person.get("w", 0) * person.get("h", 0) + identity.get("w", 0) * identity.get("h", 0) - intersection
            overlap = intersection / union if union > 0 else 0
            if overlap >= 0.3:
                candidates.append((overlap, person_index, identity["label"]))
    assigned_people = set()
    assigned_labels = set()
    for overlap, person_index, label in sorted(candidates, reverse=True):
        if person_index in assigned_people or label in assigned_labels:
            continue
        labeled[person_index]["label"] = label
        labeled[person_index]["identity_registered"] = True
        labeled[person_index]["id"] = identity_global_id(label, "person")
        assigned_people.add(person_index)
        assigned_labels.add(label)
    return labeled


class MetadataFusion:
    def __init__(self, ttl=1.0, template_ttl=None):
        self.ttl = ttl
        self.template_ttl = template_ttl if template_ttl is not None else ttl
        self.snapshots = {}
        self.lock = threading.RLock()

    def update(self, cam_id, source, objects, now=None):
        now = time.monotonic() if now is None else now
        with self.lock:
            sources = self.snapshots.setdefault(cam_id, {})
            if source == "identity_template":
                sources[source] = (now, unique_registered(objects))
                sources["identity_people"] = (now, [dict(obj) for obj in objects
                    if is_person(obj) and obj.get("label") and obj.get("tracking_state") == "tracked"])
            elif source in {"deepstream", "cpu_fallback"}:
                sources[source] = (now, [dict(obj, label=None, category="person") for obj in objects if is_person(obj)])
            for name in list(sources):
                source_ttl = self.template_ttl if name == "identity_template" else self.ttl
                if now - sources[name][0] > source_ttl:
                    del sources[name]
            people = sources.get("deepstream", sources.get("cpu_fallback", (now, [])))[1]
            people = label_registered_people(people, sources.get("identity_people", (now, []))[1])
            registered = sources.get("identity_template", (now, []))[1]
            return [dict(obj) for obj in people + registered]

    def remove_camera(self, cam_id):
        with self.lock:
            self.snapshots.pop(cam_id, None)
