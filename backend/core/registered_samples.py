"""SSD archive for manual registrations, separate from the bounded live gallery."""

import hashlib
import json
import uuid
from pathlib import Path


def sample_fingerprint(frame_image, mask):
    return hashlib.sha256((frame_image + json.dumps(mask["polygons"], separators=(",", ":"))).encode()).hexdigest()


class RegisteredSampleStore:
    def __init__(self, root):
        self.root = Path(root)

    def append(self, label, cam_id, sample):
        group = hashlib.sha256(f"{cam_id}\0{label}".encode()).hexdigest()[:24]
        relative = Path(group) / f"{uuid.uuid4().hex}.json"
        destination = self.root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(sample, separators=(",", ":")), encoding="utf-8")
        temporary.replace(destination)
        return dict(sample_ref=str(relative), bbox=sample.get("bbox"), created_at=sample.get("created_at"),
                    has_mask=bool(sample.get("mask")), cam_id=cam_id)

    def read(self, reference):
        if not reference.get("sample_ref"):
            return reference
        path = (self.root / reference["sample_ref"]).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError("Invalid registration archive path")
        return json.loads(path.read_text(encoding="utf-8"))
