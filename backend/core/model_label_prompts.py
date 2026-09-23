import hashlib
import math

from core.model_track_masks import bbox, box_iou, compatible_boxes, track_key

def latest_prompts(samples):
    cameras = {}
    for sample in samples:
        box = sample.get("bbox")
        camera_id = sample.get("camera_id")
        if sample.get("negative") or not camera_id or not sample.get("id"):
            continue
        if not isinstance(box, list) or len(box) != 4:
            continue
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in box):
            continue
        if min(box[:2]) < 0 or min(box[2:]) <= 0 or box[0] + box[2] > 1.000001 or box[1] + box[3] > 1.000001:
            continue
        mask = sample.get("mask")
        polygons = mask.get("polygons", []) if isinstance(mask, dict) else []
        if not polygons or not all(isinstance(ring, list) and len(ring) >= 3 and all(
                isinstance(point, list) and len(point) == 2 and all(isinstance(value, (int, float))
                and math.isfinite(value) and 0 <= value <= 1 for value in point) for point in ring) for ring in polygons):
            continue
        cameras.setdefault(camera_id, {})[sample["label"].casefold()] = sample
    return cameras


class LabelPromptTracks:
    def __init__(self, max_age_ms=350, retry_ms=1000, memory_age_ms=900):
        self.max_age_ms = max_age_ms
        self.memory_age_ms = max(max_age_ms, memory_age_ms)
        self.retry_ms = retry_ms
        self.entries = {}
        self.attempts = {}

    def needs_frame(self, camera_id, samples, now, blocked_labels=()):
        blocked = {str(label).casefold() for label in blocked_labels}
        return any(sample["label"].casefold() not in blocked and (
            (camera_id, sample["id"]) in self.entries
            or now - self.attempts.get((camera_id, sample["id"]), -math.inf) >= self.retry_ms
        ) for sample in samples)

    def candidates(self, camera_id, samples, frame_info, now, blocked_labels=()):
        blocked = {str(label).casefold() for label in blocked_labels}
        output = []
        active = {sample["id"] for sample in samples}
        for sample in samples:
            key = camera_id, sample['id']
            if key in self.entries:
                continue
            previous_key = next((old_key for old_key, entry in self.entries.items()
                                 if old_key[0] == camera_id and old_key[1] not in active
                                 and entry['object']['label_prompt_expected'].casefold() == sample['label'].casefold()
                                 and entry.get('generation') == frame_info.get('generation')), None)
            if previous_key:
                entry = self.entries.pop(previous_key)
                entry['object'] = dict(entry['object'], label_prompt_id=sample['id'])
                self.entries[key] = entry
        for key in self.entries.keys() | self.attempts.keys():
            if key[0] == camera_id and key[1] not in active:
                self.entries.pop(key, None)
                self.attempts.pop(key, None)
        for sample in samples:
            if sample["label"].casefold() in blocked:
                continue
            key = camera_id, sample["id"]
            entry = self.entries.get(key)
            if entry and now < entry['observed_at']:
                continue
            if entry and (now - entry["observed_at"] > self.memory_age_ms
                          or entry.get("generation") != frame_info.get("generation")):
                self.entries.pop(key, None)
                entry = None
            if entry is None and now - self.attempts.get(key, -math.inf) < self.retry_ms:
                continue
            if entry is None:
                token = f"{frame_info['model_id']}:{frame_info['generation']}:{camera_id}:label:{sample['label'].casefold()}"
                identifier = 2**40 + int.from_bytes(hashlib.blake2b(token.encode(), digest_size=6).digest(), "big")
                box = sample["bbox"]
                target = dict(id=identifier, local_id=f"label:{sample['label'].casefold()}", model_id=frame_info["model_id"],
                              class_id=-1, category=sample["category"], label=sample["label"],
                              confidence=1.0, tracking_state="candidate", observed_at=now,
                              label_prompt_id=sample["id"], label_prompt_expected=sample["label"],
                              label_prompt_generation=frame_info.get("generation"),
                              frame_width=1280, frame_height=720, keypoints=[],
                              **{"class": sample["class_name"], "x": box[0], "y": box[1], "w": box[2], "h": box[3]})
                self.attempts[key] = now
            else:
                target = dict(entry["object"])
            output.append(target)
        return output

    def observe(self, camera_id, target, mask, identity, captured_at, frame_id, detections=()):
        if not target.get("label_prompt_id"):
            return
        key = camera_id, target["label_prompt_id"]
        previous = self.entries.get(key, {})
        if captured_at < previous.get('observed_at', -math.inf):
            return
        if identity and (not identity.get("accepted") or identity.get("label") != target["label_prompt_expected"]):
            self.entries.pop(key, None)
            return
        if not mask or not identity:
            return
        for old_key, old_entry in list(self.entries.items()):
            if old_key != key and old_key[0] == camera_id and old_entry['object']['label_prompt_expected'] == target['label_prompt_expected']:
                if old_entry['observed_at'] > captured_at:
                    return
                self.entries.pop(old_key)
        self.entries[key] = dict(object=dict(target, observed_at=captured_at, frame_id=frame_id, tracking_state="tracked"),
                                 observed_at=captured_at, generation=target.get("label_prompt_generation"))
        binding = previous.get('binding')
        candidates = [(box_iou(bbox(target), bbox(detection)), detection) for detection in detections
                      if detection.get('category') == target.get('category') and not detection.get('label_prompt_id')
                      and (not binding or track_key(detection) == binding[0])
                      and 0 <= captured_at - detection.get('detected_at', -math.inf) <= 300]
        overlap, detection = max(candidates, key=lambda item: item[0], default=(0, None))
        if overlap >= .5:
            self.entries[key]['binding'] = (track_key(detection), bbox(detection))
        elif binding:
            self.entries[key]['binding'] = (binding[0], None)

    def live(self, camera_id, samples, now, blocked_labels=(), detections=()):
        blocked = {str(label).casefold() for label in blocked_labels}
        active = {sample['label'].casefold() for sample in samples}
        positions = {track_key(detection): detection for detection in detections}
        output = []
        for (camera, identifier), entry in self.entries.items():
            label = entry['object'].get('label', '').casefold()
            if camera != camera_id or label not in active or label in blocked:
                continue
            age = now - entry['observed_at']
            if age < 0 or age > 700:
                continue
            target = dict(entry['object'])
            binding = entry.get('binding')
            detection = positions.get(binding[0]) if binding else None
            if detection and binding[1] is not None and 0 <= now - detection.get('detected_at', -math.inf) <= 300 and compatible_boxes(binding[1], bbox(detection)):
                source, current = binding[1], bbox(detection)
                target.update(x=current[0] + (target['x'] - source[0]) * current[2] / source[2],
                              y=current[1] + (target['y'] - source[1]) * current[3] / source[3],
                              w=target['w'] * current[2] / source[2], h=target['h'] * current[3] / source[3],
                              observed_at=now, positioned_at=now, tracking_state='predicted')
                output.append(target)
            elif age <= self.max_age_ms:
                output.append(target)
        return output
