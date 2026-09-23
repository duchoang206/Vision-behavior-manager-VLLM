import math

from core.model_track_masks import bbox, compatible_boxes, track_key


class ModelTrackGate:
    def __init__(self, acquire_confidence=.50, retain_confidence=.30):
        self.acquire_confidence = acquire_confidence
        self.retain_confidence = retain_confidence
        self.entries = {}

    def filter(self, camera_id, objects, now, verification_categories=()):
        output = []
        reasons = {}
        for obj in objects:
            key = camera_id, track_key(obj)
            previous = self.entries.get(key)
            box = bbox(obj)
            score = float(obj.get("confidence", 0))
            detected_at = float(obj.get("detected_at", 0))
            valid = all(math.isfinite(value) for value in box + [score, detected_at]) and min(box[2:]) > 0
            continuous = previous and now - previous["seen"] <= 300 and compatible_boxes(previous["box"], box)
            confirmed = bool(continuous and previous["confirmed"])
            threshold = self.retain_confidence if confirmed else self.acquire_confidence
            can_verify = obj.get("category") in verification_categories and score >= .25
            reason = None
            if not valid or min(box[2] * 1280, box[3] * 720) < 8:
                reason = "invalid_or_tiny_box"
            elif now - detected_at > 300 or detected_at > now + 50:
                reason = "detector_stale"
            elif score < threshold and not can_verify:
                reason = "low_confidence"
            else:
                hits = ((previous["hits"] if continuous else 0) + 1) if score >= threshold else 0
                candidate_hits = (previous.get("candidate_hits", 0) if continuous else 0) + 1
                confirmed = hits >= 2 or confirmed
                self.entries[key] = dict(box=box, hits=hits, candidate_hits=candidate_hits, confirmed=confirmed, seen=now)
                if confirmed and score >= threshold:
                    output.append(obj)
                elif can_verify and candidate_hits >= 2:
                    output.append(dict(obj, requires_label_verification=True))
                else:
                    reason = "confirming_track"
            if reason:
                reasons[reason] = reasons.get(reason, 0) + 1
        self.entries = {key: entry for key, entry in self.entries.items() if now - entry["seen"] <= 700}
        return output, reasons
