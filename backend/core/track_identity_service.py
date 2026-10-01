"""Background ReID sampling for TrackIdentityVoter.

main.py hands over each camera's latest robot objects (cheap, non-blocking). A worker thread grabs the newest
RTSP frame, crops each tracked robot, embeds the crop and feeds the per-track vote. Robots whose label was
position-verified by FMS also teach the voter new appearance prototypes.
"""

import logging
import os
import threading
import time

from core.reid_encoder import encode_crop
from core.track_identity_vote import TrackIdentityVoter

STORE_PATH = os.getenv("TRACK_REID_STORE", "/app/data/track_identity/prototypes.json")


class TrackIdentityService:
    def __init__(self, voter=None, reader_factory=None, encoder=encode_crop, sample_interval=.4, max_crops_per_cycle=6, stable_interval=1.5):
        self.voter = voter or TrackIdentityVoter(store_path=STORE_PATH)
        self.reader_factory = reader_factory or self._default_reader
        self.encoder = encoder
        self.sample_interval, self.max_crops_per_cycle = sample_interval, max_crops_per_cycle
        self.stable_interval = float(os.getenv("TRACK_REID_STABLE_INTERVAL", stable_interval))  # CPU: ~65 ms per crop
        self.pending, self.readers, self.last_sample, self.last_save = {}, {}, {}, time.time()
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.enabled = os.getenv("TRACK_REID_VOTE", "1") != "0"
        self.thread = None

    @staticmethod
    def _default_reader(cam_id):
        from core.person_tracker import ZeroLatencyRTSPReader
        return ZeroLatencyRTSPReader(f"rtsp://127.0.0.1:8554/{cam_id}", cam_id)

    # ---- called from the metadata callback (must stay cheap) --------------------------------------------
    def apply(self, cam_id, objects):
        """Override robot labels the vote disagrees with; keeps the detector label in model_label."""
        if not self.enabled:
            return
        for obj in objects:
            if str(obj.get("category", "")).lower() != "robot":
                continue
            corrected = self.voter.decision((cam_id, obj.get("id")))
            if corrected and corrected != obj.get("label"):
                obj["model_label"] = obj.get("label")
                obj["label"] = corrected
                obj["reid_corrected"] = True

    def submit(self, cam_id, objects):
        if not self.enabled:
            return
        robots = [dict(obj) for obj in objects if str(obj.get("category", "")).lower() == "robot"]
        if not robots:
            return
        with self.lock:
            self.pending[cam_id] = robots
            if self.thread is None:
                self.thread = threading.Thread(target=self._run, name="track-reid-vote", daemon=True)
                self.thread.start()
        self.wake.set()

    # ---- worker ------------------------------------------------------------------------------------------
    def _run(self):
        while True:
            self.wake.wait(1.0)
            self.wake.clear()
            with self.lock:
                batch, self.pending = self.pending, {}
            for cam_id, robots in batch.items():
                try:
                    self._sample(cam_id, robots)
                except Exception:
                    logging.exception("Track ReID sampling failed for %s", cam_id)
            if time.time() - self.last_save > 60:
                self.last_save = time.time()
                try:
                    self.voter.save()
                except OSError:
                    logging.exception("Track ReID prototype save failed")

    def _sample(self, cam_id, robots):
        reader = self.readers.get(cam_id)
        if reader is None:
            reader = self.readers[cam_id] = self.reader_factory(cam_id)
        frame = reader.get_latest_frame()
        if frame is None:
            return
        height, width = frame.shape[:2]
        crops = 0
        for obj in robots:
            key = (cam_id, obj.get("id"))
            now = time.time()
            interval = self.stable_interval if self.voter.settled(key) else self.sample_interval
            if now - self.last_sample.get(key, 0) < interval or crops >= self.max_crops_per_cycle:
                continue
            if obj.get("tracking_state") != "tracked" or float(obj.get("confidence", 0) or 0) < .5:
                continue
            x, y, w, h = (float(obj.get(name, 0) or 0) for name in ("x", "y", "w", "h"))
            x0, y0 = int(max(0, (x - w * .03) * width)), int(max(0, (y - h * .03) * height))
            x1, y1 = int(min(width, (x + w * 1.03) * width)), int(min(height, (y + h * 1.03) * height))
            if x1 - x0 < 48 or y1 - y0 < 48:
                continue
            vector = self.encoder(frame[y0:y1, x0:x1])
            self.last_sample[key] = now
            crops += 1
            if vector is None:
                continue
            label = obj.get("label")
            if (obj.get("spatial_identity") or {}).get("accepted") is True and label:
                self.voter.add_prototype(label, vector)  # FMS confirmed this label is at this position
            self.voter.observe(key, obj.get("model_label") or label, vector, now)
        if len(self.last_sample) > 512:
            cutoff = time.time() - 60
            self.last_sample = {key: seen for key, seen in self.last_sample.items() if seen > cutoff}
