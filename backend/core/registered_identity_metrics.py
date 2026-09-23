"""Category-independent triplet metrics; incompatible encoders never share a gallery."""

import hashlib
import os
import threading
from pathlib import Path


class RegisteredIdentityMetrics:
    def __init__(self, directory=None, device="cuda:0", auto_train=True):
        self.directory = Path(directory or Path(__file__).resolve().parents[1] / "data/identity_metrics")
        self.device = device
        self.auto_train = auto_train
        self.metrics = {}
        self.lock = threading.RLock()
        self.last_error = None

    def _metric(self, embedding_type):
        if embedding_type not in {"reid_512", "clip_512"}:
            return None
        with self.lock:
            if embedding_type not in self.metrics:
                from core.identity_metric import IdentityMetric
                model_path = Path(os.getenv("REID_ONNX_FILE", "/app/models/reid_512.onnx")) if embedding_type == "reid_512" else None
                signature = embedding_type
                if model_path and model_path.is_file():
                    with model_path.open("rb") as model_file:
                        signature += ":" + hashlib.file_digest(model_file, "sha256").hexdigest()[:16]
                self.metrics[embedding_type] = IdentityMetric(
                    self.directory / embedding_type, device=self.device, auto_train=self.auto_train,
                    model_signature=signature, descriptor_name=f"{embedding_type}_triplet_v1",
                    minimum_similarity=float(os.getenv("REGISTERED_REID_MIN_SIMILARITY", "0.78")),
                )
            return self.metrics[embedding_type]

    def learn(self, label, cam_id, vector, embedding_type, fingerprint=None):
        metric = self._metric(embedding_type)
        if metric is None:
            return None
        if fingerprint is None:
            import numpy as np
            fingerprint = hashlib.sha256(np.asarray(vector, dtype=np.float32).tobytes()).hexdigest()
        return metric.learn(label, cam_id or "unknown", fingerprint, vector)

    def verify(self, label, vector, embedding_type="reid_512"):
        metric = self.metrics.get(embedding_type)
        if metric is None:
            return None
        try:
            return metric.verify(label, vector)
        except (ValueError, TypeError, RuntimeError) as error:
            self.last_error = str(error)
            return dict(accepted=False, reason="identity_metric_error")

    def scores(self, vector, labels, embedding_type="reid_512"):
        return {label: result for label in set(labels)
                if (result := self.verify(label, vector, embedding_type)) is not None}

    def restore(self, registry):
        with registry._write_lock:
            targets = [(target.get("label", key.split("::")[-1]), target.get("cam_id") or target.get("last_cam"),
                        target.get("embedding_type"), list(target.get("crops", [])),
                        list(target.get("vectors", []))) for key, target in registry.targets.items()]
        for label, cam_id, embedding_type, samples, vectors in targets:
            if embedding_type not in {"reid_512", "clip_512"}:
                continue
            restored = False
            for reference in samples:
                try:
                    sample = registry.sample_store.read(reference)
                    if sample.get("embedding_type") == embedding_type and sample.get("vector") is not None:
                        self.learn(label, cam_id, sample["vector"], embedding_type, reference.get("sample_ref"))
                        restored = True
                except (OSError, ValueError, TypeError) as error:
                    self.last_error = str(error)
            if not restored:
                for vector in vectors:
                    self.learn(label, cam_id, vector, embedding_type)
        return self.status()

    def forget(self, label, cam_id=None):
        with self.lock:
            metrics = list(self.metrics.values())
        for metric in metrics:
            if label in metric.gallery:
                metric.forget(label, cam_id)

    def status(self):
        with self.lock:
            metrics = list(self.metrics.items())
        return {"encoders": {name: metric.status() for name, metric in metrics}, "last_error": self.last_error}


registered_identity_metrics = RegisteredIdentityMetrics()
