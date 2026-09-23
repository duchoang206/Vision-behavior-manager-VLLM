import math
import re
import threading


def model_identity(label, category):
    if category == "robot" and re.fullmatch(r"(?:robot|amr|agv)[_ -]?\d+", label, re.IGNORECASE):
        return label
    return None


def track_key(obj):
    return obj.get("model_id"), obj["id"], obj.get("class_id"), obj.get("class")


def bbox(obj):
    return [float(obj[name]) for name in ("x", "y", "w", "h")]

def box_iou(first, second):
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[0] + first[2], second[0] + second[2]), min(first[1] + first[3], second[1] + second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    return intersection / max(1e-9, first[2] * first[3] + second[2] * second[3] - intersection)


def compatible_boxes(previous, current):
    if not all(math.isfinite(value) for value in previous + current) or min(previous[2:] + current[2:]) <= 0:
        return False
    scale_x, scale_y = current[2] / previous[2], current[3] / previous[3]
    distance = math.hypot(current[0] + current[2] / 2 - previous[0] - previous[2] / 2,
                          current[1] + current[3] / 2 - previous[1] - previous[3] / 2)
    return .5 <= scale_x <= 2 and .5 <= scale_y <= 2 and distance <= math.hypot(*previous[2:]) * .8


class TrackMaskCache:
    def __init__(self, max_age_ms=700):
        self.max_age_ms = max_age_ms
        self.entries = {}
        self.latest = {}
        self.lock = threading.Lock()

    def store(self, camera_id, obj, mask, frame_id, captured_at, identity=None):
        key = camera_id, track_key(obj)
        with self.lock:
            latest = self.latest.get(camera_id)
            if latest and latest["frame_id"] >= frame_id:
                current = latest["objects"].get(track_key(obj))
                if not obj.get("label_prompt_id") and (current is None or not compatible_boxes(bbox(obj), current)):
                    return
            previous = self.entries.get(key)
            if previous and frame_id <= previous["frame_id"]:
                return
            self.entries[key] = {"box": bbox(obj), "mask": mask, "frame_id": frame_id,
                                 "captured_at": captured_at, "identity": identity,
                                 "label_prompt": bool(obj.get('label_prompt_id'))}

    def reject(self, camera_id, obj):
        with self.lock:
            self.entries.pop((camera_id, track_key(obj)), None)

    def has_newer(self, camera_id, obj, captured_at):
        with self.lock:
            entry = self.entries.get((camera_id, track_key(obj)))
            return bool(entry and entry['captured_at'] > captured_at)

    def verified_labels(self, camera_id, objects, now, policies):
        labels = set()
        with self.lock:
            for obj in objects:
                entry = self.entries.get((camera_id, track_key(obj)))
                policy = policies.get(obj.get("category"), {})
                if obj.get("label_prompt_id") or not entry or policy.get("error"):
                    continue
                identity = entry.get("identity") or {}
                if not identity.get("accepted") or identity.get("revision") != policy.get("revision"):
                    continue
                if identity.get("match_source") not in {"user_gallery", "user_reference"} and identity.get("version") != policy.get("version"):
                    continue
                if 0 <= now - entry["captured_at"] <= self.max_age_ms and compatible_boxes(entry["box"], bbox(obj)):
                    labels.add(identity["label"].casefold())
        return labels

    def attach(self, camera_id, objects, frame_id, now):
        with self.lock:
            self.latest[camera_id] = {"frame_id": frame_id, "objects": {track_key(obj): bbox(obj) for obj in objects}}
            active = {track_key(obj) for obj in objects}
            for key, entry in list(self.entries.items()):
                if now - entry["captured_at"] > self.max_age_ms or (key[0] == camera_id and key[1] not in active and not entry['label_prompt']):
                    del self.entries[key]
            for obj in objects:
                entry = self.entries.get((camera_id, track_key(obj)))
                obj["mask"] = None
                obj["mask_stale"] = True
                obj["identity_verification"] = None
                if not entry or entry["frame_id"] > frame_id or now < entry["captured_at"]:
                    continue
                current = bbox(obj)
                old = entry["box"]
                if not compatible_boxes(old, current):
                    continue
                polygons = [[[min(1., max(0., current[0] + (point[0] - old[0]) * current[2] / old[2])),
                              min(1., max(0., current[1] + (point[1] - old[1]) * current[3] / old[3]))]
                             for point in ring] for ring in entry["mask"]["polygons"]]
                label_prompt = bool(obj.get("label_prompt_id"))
                obj["mask"] = dict(entry["mask"], polygons=polygons, source="sam2_live" if label_prompt else "sam2_nvdcf",
                                   observed_at=entry["captured_at"], frame_id=entry["frame_id"],
                                   attached_at=obj.get("positioned_at", entry["captured_at"]) if label_prompt else now)
                obj["mask_stale"] = False
                obj["identity_verification"] = entry.get("identity")


def boundary_polygons(edges, width, height):
    outgoing = {}
    for left, top, right, bottom in edges:
        outgoing.setdefault((left, top), []).append((right, bottom))
    rings = []
    while outgoing:
        start = next(iter(outgoing))
        point = start
        ring = []
        while point in outgoing:
            ring.append(point)
            next_point = outgoing[point].pop()
            if not outgoing[point]:
                del outgoing[point]
            point = next_point
            if point == start:
                break
        if point != start or len(ring) < 4:
            continue
        simplified = []
        for index, current in enumerate(ring):
            before, after = ring[index - 1], ring[(index + 1) % len(ring)]
            if (current[0] - before[0]) * (after[1] - current[1]) != (current[1] - before[1]) * (after[0] - current[0]):
                simplified.append(current)
        area = abs(sum(current[0] * following[1] - following[0] * current[1]
                       for current, following in zip(ring, ring[1:] + ring[:1]))) / 2
        if area < 4 or len(simplified) < 3:
            continue
        stride = max(1, math.ceil(len(simplified) / 160))
        rings.append((area, [[round(point[0] / width, 6), round(point[1] / height, 6)] for point in simplified[::stride]]))
    return [ring for _, ring in sorted(rings, reverse=True)[:16] if len(ring) >= 3]
