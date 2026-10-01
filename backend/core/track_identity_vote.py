"""Per-track appearance voting that corrects robot labels the detector gets wrong.

The detector classifies each robot from a single frame, so a robot can flicker between labels. Here every
sampled crop of a track votes for the closest registered robot appearance; a label is only overridden once a
clear majority of the recent votes agrees (default 15 votes, 70% majority).
"""

import json
import math
import os
import threading
import time
from collections import Counter, deque

import numpy as np


class TrackIdentityVoter:
    def __init__(self, window=15, min_votes=8, majority=.7, min_similarity=.50, min_margin=.08,
                 max_prototypes=60, duplicate_similarity=.97, track_ttl=30.0, store_path=None):
        self.window, self.min_votes, self.majority = window, min_votes, majority
        self.min_similarity, self.min_margin = min_similarity, min_margin
        self.max_prototypes, self.duplicate_similarity, self.track_ttl = max_prototypes, duplicate_similarity, track_ttl
        self.store_path = store_path
        self.prototypes = {}
        self.tracks = {}
        self.lock = threading.Lock()
        if store_path:
            self.load()

    # ---- prototypes -------------------------------------------------------------------------------------
    @staticmethod
    def _unit(vector):
        array = np.asarray(vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(array))
        return array / norm if norm > 1e-6 and math.isfinite(norm) else None

    def add_prototype(self, label, vector):
        """Returns True when the vector was kept (skips near-duplicates; evicts the oldest at capacity)."""
        unit = self._unit(vector)
        if unit is None or not label:
            return False
        with self.lock:
            bucket = self.prototypes.setdefault(label, [])
            if bucket and max(float(unit @ item) for item in bucket) >= self.duplicate_similarity:
                return False
            bucket.append(unit)
            if len(bucket) > self.max_prototypes:
                del bucket[0]
        return True

    def labels(self):
        with self.lock:
            return sorted(label for label, items in self.prototypes.items() if items)

    def classify(self, vector):
        """-> (label, similarity, margin) or None when fewer than two labels are registered."""
        unit = self._unit(vector)
        if unit is None:
            return None
        with self.lock:
            scores = {}
            for label, items in self.prototypes.items():
                if items:
                    similarities = sorted((float(unit @ item) for item in items), reverse=True)[:3]
                    scores[label] = sum(similarities) / len(similarities)
        if len(scores) < 2:
            return None
        ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
        return ranked[0][0], ranked[0][1], ranked[0][1] - ranked[1][1]

    # ---- voting -----------------------------------------------------------------------------------------
    def observe(self, key, model_label, vector, now=None):
        now = time.time() if now is None else now
        with self.lock:
            known = bool(self.prototypes.get(model_label))
        if not known:
            return None  # the detector's label is not a registered robot: nothing to refute, never relabel
        vote = self.classify(vector)
        if vote is None:
            return self.decision(key)
        with self.lock:
            state = self.tracks.setdefault(key, dict(votes=deque(maxlen=self.window), label=None, seen=now))
            state["seen"] = now
            state["votes"].append(vote)
            winner = self._winner(state["votes"])
            if winner is not None:
                state["label"] = winner if winner != model_label else None
            self._expire(now)
            return state["label"]

    def _winner(self, votes):
        if len(votes) < self.min_votes:
            return None
        label, count = Counter(vote[0] for vote in votes).most_common(1)[0]
        if count / len(votes) < self.majority:
            return None
        mine = [vote for vote in votes if vote[0] == label]
        if sum(vote[1] for vote in mine) / len(mine) < self.min_similarity:
            return None
        if sum(vote[2] for vote in mine) / len(mine) < self.min_margin:
            return None
        return label

    def status(self):
        with self.lock:
            return dict(
                prototypes={label: len(items) for label, items in self.prototypes.items()},
                tracks=[dict(camera=key[0], track=key[1], votes=len(state["votes"]), corrected_to=state["label"],
                             tally=dict(Counter(vote[0] for vote in state["votes"])))
                        for key, state in self.tracks.items()])

    def settled(self, key, agreement=.9):
        """True once a full window of votes agrees (>= agreement): the track needs far fewer samples after that."""
        with self.lock:
            state = self.tracks.get(key)
            if not state or len(state["votes"]) < self.window:
                return False
            return Counter(vote[0] for vote in state["votes"]).most_common(1)[0][1] / len(state["votes"]) >= agreement

    def decision(self, key):
        with self.lock:
            state = self.tracks.get(key)
            return state["label"] if state else None

    def _expire(self, now):
        if len(self.tracks) > 64:
            self.tracks = {key: state for key, state in self.tracks.items() if now - state["seen"] <= self.track_ttl}

    # ---- persistence ------------------------------------------------------------------------------------
    def save(self):
        if not self.store_path:
            return
        with self.lock:
            payload = {label: [item.tolist() for item in items] for label, items in self.prototypes.items()}
        os.makedirs(os.path.dirname(self.store_path), exist_ok=True)
        temporary = self.store_path + ".tmp"
        with open(temporary, "w") as handle:
            json.dump(payload, handle)
        os.replace(temporary, self.store_path)

    def load(self):
        try:
            with open(self.store_path) as handle:
                payload = json.load(handle)
        except (OSError, ValueError):
            return
        with self.lock:
            for label, items in payload.items():
                self.prototypes[label] = [unit for unit in (self._unit(item) for item in items) if unit is not None]
