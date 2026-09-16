"""Trusted multi-view descriptors and a bounded, asynchronously learned identity metric."""

import hashlib
import json
import os
import random
import threading
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as functional


def masked_descriptor(features, bitmap, image_shape, input_shape):
    feature_height, feature_width = features.shape[-2:]
    image_height, image_width = image_shape[:2]
    input_height, input_width = input_shape[-2:]
    scale = min(input_height / image_height, input_width / image_width)
    resized_height, resized_width = round(image_height * scale), round(image_width * scale)
    weights = torch.as_tensor(bitmap, device=features.device, dtype=torch.float32)[None, None]
    weights = functional.interpolate(weights, size=(resized_height, resized_width), mode="nearest")
    weights = functional.pad(weights, (0, input_width - resized_width, 0, input_height - resized_height))
    weights = functional.interpolate(weights, size=(feature_height, feature_width), mode="area")
    total = weights.sum().clamp_min(1e-6)
    values = features.float()
    mean = (values * weights).sum(dim=(2, 3)) / total
    deviation = (((values - mean[..., None, None]).square() * weights).sum(dim=(2, 3)) / total).clamp_min(1e-8).sqrt()
    return functional.normalize(torch.cat((functional.normalize(mean, dim=1), functional.normalize(deviation, dim=1)), dim=1), dim=1)[0]


def triplet_loss(embeddings, labels, margin=0.25, negative_weight=2.0):
    embeddings = functional.normalize(embeddings, dim=1)
    similarities = embeddings @ embeddings.T
    same = labels[:, None].eq(labels[None, :])
    positive = same & ~torch.eye(len(labels), device=labels.device, dtype=torch.bool)
    negative = ~same
    valid = positive.any(dim=1) & negative.any(dim=1)
    positive_distance = (1 - similarities).masked_fill(~positive, -1).max(dim=1).values
    negative_distance = (1 - similarities).masked_fill(~negative, 3).min(dim=1).values
    penalties = functional.relu(positive_distance - negative_distance + margin)
    penalties = penalties + negative_weight * functional.relu(0.35 - negative_distance)
    return (penalties * valid).sum() / valid.sum().clamp_min(1)




class IdentityMetric:
    def __init__(self, directory, device="cuda:0", auto_train=True, model_signature=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device)
        self.lock = threading.RLock()
        self.records = {}
        self.gallery = {}
        self.projected_gallery = {}
        self.projection = None
        self.revision = 0
        self.pending = False
        self.training = False
        self.last_error = None
        self.last_loss = None
        self.auto_train = auto_train
        self.last_fit_at = 0.0
        self.model_signature = model_signature
        self.training_stream = torch.cuda.Stream(device=self.device, priority=0) if self.device.type == "cuda" else None
        for path in self.directory.glob("samples/*/*.json"):
            try:
                record = json.loads(path.read_text())
                if record.get("model_signature") != self.model_signature:
                    continue
                self.records[record["key"]] = {key: record[key] for key in ("key", "label", "cam_id")}
                self.records[record["key"]]["path"] = path
                self._add_gallery(record["label"], record["vector"])
            except (OSError, ValueError, KeyError):
                continue
        checkpoint = self.directory / "projection.pt"
        if checkpoint.is_file():
            try:
                state = torch.load(checkpoint, map_location=self.device, weights_only=True)
                if state.get("descriptor") == "sam2_mask_mean_std_v1" and state.get("model_signature") == self.model_signature:
                    self.projection = state["weight"].float()
                    self.revision = int(state["revision"])
                    self.last_loss = state.get("loss")
            except Exception as error:
                self.last_error = str(error)
        self.pending = bool(self.records)

    @torch.inference_mode()
    def _project_gallery(self):
        self.projected_gallery = {
            label: functional.normalize(functional.linear(vectors, self.projection), dim=1)
            for label, vectors in self.gallery.items()
        } if self.projection is not None else dict(self.gallery)

    def _add_gallery(self, label, vector):
        feature = functional.normalize(torch.as_tensor(vector, device=self.device, dtype=torch.float32).reshape(1, 512), dim=1)
        gallery = self.gallery.get(label)
        if gallery is None:
            self.gallery[label] = feature
            self.projected_gallery.clear()
            return
        if (gallery @ feature.T).max().item() > 0.9995:
            return
        gallery = torch.cat((gallery, feature))
        if len(gallery) > 64:
            similarities = gallery @ gallery.T
            similarities.fill_diagonal_(-1)
            remove_index = int(similarities[:-1].max(dim=1).values.argmax())
            gallery = torch.cat((gallery[:remove_index], gallery[remove_index + 1:]))
        self.gallery[label] = gallery
        self.projected_gallery.clear()

    def learn(self, label, cam_id, fingerprint, vector):
        feature = torch.as_tensor(vector, dtype=torch.float32).reshape(-1)
        if len(feature) != 512 or not torch.isfinite(feature).all() or feature.norm() < 1e-6:
            raise ValueError("Đặc trưng SAM không hợp lệ.")
        key = hashlib.sha256(f"{cam_id}\0{label}\0{fingerprint}".encode()).hexdigest()
        with self.lock:
            if key not in self.records:
                folder = self.directory / "samples" / hashlib.sha256(label.encode()).hexdigest()[:24]
                folder.mkdir(parents=True, exist_ok=True)
                path = folder / f"{key}.json"
                record = dict(key=key, label=label, cam_id=cam_id, vector=feature.tolist(), model_signature=self.model_signature)
                temporary = path.with_suffix(".tmp")
                temporary.write_text(json.dumps(record, separators=(",", ":")))
                temporary.replace(path)
                self.records[key] = dict(key=key, label=label, cam_id=cam_id, path=path)
                self._add_gallery(label, feature)
                self.pending = True
            if self.auto_train and self.pending and not self.training:
                self.training = True
                threading.Thread(target=self._training_loop, name="identity-triplet", daemon=True).start()
        return self.status()

    def forget(self, label, cam_id=None):
        with self.lock:
            for key, record in list(self.records.items()):
                if record["label"] == label and (cam_id is None or record["cam_id"] == cam_id):
                    record["path"].unlink(missing_ok=True)
                    del self.records[key]
            self.gallery.pop(label, None)
            for record in self.records.values():
                if record["label"] == label:
                    self._add_gallery(label, json.loads(record["path"].read_text())["vector"])
            self.projection = None
            self.projected_gallery.clear()
            (self.directory / "projection.pt").unlink(missing_ok=True)
            self.pending = True

    def _training_loop(self):
        try:
            context = torch.cuda.stream(self.training_stream) if self.training_stream is not None else nullcontext()
            with context:
                while True:
                    time.sleep(2)
                    with self.lock:
                        if not self.pending:
                            return
                        self.pending = False
                    self.fit()
        except Exception as error:
            self.last_error = str(error)
        finally:
            with self.lock:
                self.training = False
                if self.pending and self.last_error is None:
                    self.training = True
                    threading.Thread(target=self._training_loop, name="identity-triplet", daemon=True).start()

    def fit(self, steps=80):
        with self.lock:
            grouped = {}
            record_keys = set(self.records)
            for record in self.records.values():
                grouped.setdefault(record["label"], []).append(record["path"])
            if len(grouped) < 2 or not any(len(paths) >= 2 for paths in grouped.values()):
                return False
            initial = self.projection.clone() if self.projection is not None else torch.eye(512, device=self.device)
        samples, labels = [], []
        for label_index, paths in enumerate(grouped.values()):
            for path in paths[-8:] + random.sample(paths[:-8], min(max(0, len(paths) - 8), 8)):
                samples.append(json.loads(path.read_text())["vector"])
                labels.append(label_index)
        with torch.inference_mode(False), torch.enable_grad():
            features = functional.normalize(torch.tensor(samples, device=self.device, dtype=torch.float32), dim=1)
            identities = torch.tensor(labels, device=self.device)
            weight = torch.nn.Parameter(initial.detach().clone())
            optimizer = torch.optim.AdamW([weight], lr=0.001, weight_decay=0.001)
            baseline = float(triplet_loss(functional.linear(features, weight), identities).detach())
            for _step in range(steps):
                selected_labels = torch.randperm(len(grouped), device=self.device)[:8]
                indices = torch.cat([torch.where(identities == label)[0] for label in selected_labels])
                projected = functional.linear(features[indices], weight)
                loss = triplet_loss(projected, identities[indices]) + 0.02 * (weight - initial).square().mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if self.training_stream is not None:
                    self.training_stream.synchronize()
                time.sleep(0.005)
            final_loss = float(triplet_loss(functional.linear(features, weight), identities).detach())
            if not torch.isfinite(weight).all() or final_loss > baseline + 1e-5:
                return False
            with self.lock:
                if not record_keys.issubset(self.records):
                    return False
                checkpoint = dict(descriptor="sam2_mask_mean_std_v1", weight=weight.detach(), revision=self.revision + 1, loss=final_loss,
                                  model_signature=self.model_signature)
                temporary = self.directory / "projection.tmp"
                torch.save(checkpoint, temporary)
                temporary.replace(self.directory / "projection.pt")
                self.projection = weight.detach()
                self.projected_gallery.clear()
                self.revision += 1
                self.last_loss = final_loss
                self.last_fit_at = time.time()
                self.last_error = None
        return True

    @torch.inference_mode()
    def verify(self, label, descriptor):
        with self.lock:
            gallery = dict(self.gallery)
            projection = self.projection
            if not self.projected_gallery:
                self._project_gallery()
            projected_gallery = dict(self.projected_gallery)
        if label not in gallery:
            return dict(accepted=False, reason="identity_unavailable")
        query = functional.normalize(descriptor.to(self.device).float().reshape(1, 512), dim=1)
        raw_score = float((gallery[label] @ query.T).max())
        if projection is not None:
            query = functional.normalize(functional.linear(query, projection), dim=1)
        scores = {}
        for candidate, projected in projected_gallery.items():
            scores[candidate] = float((projected @ query.T).max())
        score = scores[label]
        rival = max((value for candidate, value in scores.items() if candidate != label), default=-1)
        threshold = float(os.getenv("REGISTERED_IDENTITY_MIN_SIMILARITY", "0.86"))
        margin = float(os.getenv("REGISTERED_IDENTITY_RIVAL_MARGIN", "0.035"))
        accepted = raw_score >= threshold and score >= 0.75 and score - rival >= margin
        reason = None if accepted else ("identity_ambiguous" if score - rival < margin else "identity_mismatch")
        return dict(accepted=accepted, reason=reason, score=round(score, 4), raw_score=round(raw_score, 4), rival_score=round(rival, 4), revision=self.revision)

    def status(self):
        with self.lock:
            counts = {}
            for record in self.records.values():
                counts[record["label"]] = counts.get(record["label"], 0) + 1
            ready = len(counts) >= 2 and max(counts.values(), default=0) >= 2
            return dict(descriptor="sam2_mask_mean_std_v1", device=str(self.device), samples=len(self.records), labels=counts,
                        model_signature=self.model_signature,
                        revision=self.revision, training=self.training, ready=ready, last_loss=self.last_loss,
                        last_error=self.last_error, last_fit_at=self.last_fit_at,
                        state="training" if self.training else "trained" if self.projection is not None else "waiting_for_positive_and_negative_samples")
