import json
import logging
import threading
import time
from pathlib import Path

import torch
import torch.nn.functional as functional

from core.identity_metric import triplet_loss


class ModelLabelMetric:
    def __init__(self, directory, signature, device="cuda:0", start=True):
        self.directory = Path(directory)
        self.signature = signature
        self.device = torch.device(device)
        self.lock = threading.RLock()
        self.stopping = threading.Event()
        self.reload_event = threading.Event()
        self.loaded = threading.Condition(self.lock)
        self.revision = -1
        self.continuity_revision = -1
        self.require_labels = False
        self.samples = []
        self.vectors = None
        self.projected = None
        self.weight = None
        self.training = False
        self.loss = None
        self.error = None
        self.thread = None
        self.declared_categories = set()
        self.incompatible_samples = 0
        self.references = []
        self.group_indices = None
        self.version = 0
        self.prompts = {}
        if start:
            self.start()

    def start(self):
        if self.thread is None:
            self.thread = threading.Thread(target=self._watch, name="model-label-triplet", daemon=True)
            self.thread.start()

    def _watch(self):
        stamp = None
        stream = torch.cuda.Stream(device=self.device, priority=0) if self.device.type == "cuda" else None
        while not self.stopping.is_set():
            self.reload_event.wait(.5)
            self.reload_event.clear()
            if self.stopping.is_set():
                break
            try:
                path = self.directory / "gallery.json"
                modified = path.stat().st_mtime_ns if path.exists() else None
                if modified == stamp:
                    continue
                if modified is None:
                    with self.lock:
                        self.samples = []
                        self.vectors = None
                        self.projected = None
                        self.weight = None
                        self.revision = 0
                        self.require_labels = True
                        self.declared_categories = set()
                        self.incompatible_samples = 0
                        self.loss = None
                        self.error = "Gallery bị xóa ngoài chương trình; đã khóa xác thực Label."
                        self.version += 1
                        self.prompts = {}
                        self.loaded.notify_all()
                    stamp = None
                    continue
                document = json.loads(path.read_text())
                if stream:
                    with torch.cuda.stream(stream):
                        self.load(document)
                        self.fit()
                else:
                    self.load(document)
                    self.fit()
                stamp = modified
                self.error = None
            except Exception as error:
                self.error = str(error)
                logging.exception("Model label gallery")

    def reload(self, revision, timeout=3):
        deadline = time.monotonic() + timeout
        self.reload_event.set()
        with self.loaded:
            while self.revision < revision and not self.stopping.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Gallery đã lưu nhưng worker chưa xác nhận revision mới.")
                self.loaded.wait(remaining)
            if self.revision < revision or self.error:
                raise RuntimeError(self.error or "Worker Label đã dừng.")
            return dict(mode="applied", revision=self.revision, samples=len(self.samples),
                        prompt_count=sum(len(rows) for rows in self.prompts.values()))

    def load(self, document):
        samples = [sample for sample in document.get("samples", []) if sample["signature"] == self.signature]
        vectors = functional.normalize(torch.tensor([sample["vector"] for sample in samples], device=self.device, dtype=torch.float32), dim=1) if samples else None
        if vectors is not None and (vectors.shape[1] != 512 or not bool(torch.isfinite(vectors).all())):
            raise ValueError("Gallery có vector không hợp lệ.")
        names = list(dict.fromkeys(sample["label"] for sample in samples))
        references = [next(sample for sample in samples if sample["label"] == name) for name in names]
        group_indices = torch.tensor([names.index(sample["label"]) for sample in samples], device=self.device, dtype=torch.long)
        from core.model_label_prompts import latest_prompts
        prompts = latest_prompts(samples)
        if self.device.type == "cuda":
            torch.cuda.current_stream(self.device).synchronize()
        with self.lock:
            old_ids = {sample.get('id') for sample in self.samples}
            new_ids = {sample.get('id') for sample in samples}
            old_negatives = {sample.get('id') for sample in self.samples if sample['negative']}
            new_negatives = {sample.get('id') for sample in samples if sample['negative']}
            old_labels = {(sample['label'], sample['category']) for sample in self.samples if not sample['negative']}
            new_labels = {(sample['label'], sample['category']) for sample in samples if not sample['negative']}
            if self.revision < 0 or not old_ids <= new_ids or old_negatives != new_negatives or old_labels != new_labels:
                self.continuity_revision = int(document.get('revision', 0))
            self.samples, self.vectors, self.projected = samples, vectors, vectors
            self.weight = None
            self.revision = int(document.get("revision", 0))
            self.require_labels = bool(document.get("require_labels"))
            self.declared_categories = {sample["category"] for sample in document.get("samples", [])}
            self.incompatible_samples = len(document.get("samples", [])) - len(samples)
            self.loss = None
            self.error = None
            self.references, self.group_indices = references, group_indices
            self.version += 1
            self.prompts = prompts
            self.loaded.notify_all()

    def prompt_samples(self, camera_id):
        with self.lock:
            return [] if self.error or self.incompatible_samples else list(self.prompts.get(camera_id, {}).values())

    def saved_sample(self, camera_id, identifier):
        with self.lock:
            return next((sample for sample in self.samples if sample.get('id') == identifier
                         and sample.get('camera_id') == camera_id and not sample['negative']), None)

    def reference_samples(self, label, camera_id, limit=3):
        with self.lock:
            if self.error or self.incompatible_samples:
                return []
            candidates = [sample for sample in self.samples if sample.get('id') and sample.get('mask')
                          and not sample['negative'] and sample['label'].casefold() == label.casefold()]
        local = [sample for sample in candidates if sample.get('camera_id') == camera_id]
        candidates = local or candidates
        if not candidates:
            return []
        selected = [candidates.pop()]
        distances = {sample['id']: 1. for sample in candidates}
        while candidates and len(selected) < max(1, limit):
            vector = selected[-1]['vector']
            for sample in candidates:
                similarity = sum(first * second for first, second in zip(sample['vector'], vector))
                distances[sample['id']] = min(distances[sample['id']], 1. - similarity)
            winner = max(candidates, key=lambda sample: distances[sample['id']])
            selected.append(winner)
            candidates.remove(winner)
        return selected

    def reference_ids(self):
        with self.lock:
            return {sample['id'] for sample in self.samples if sample.get('id') and not sample['negative']}

    def fit(self, steps=60):
        with self.lock:
            samples, gallery, revision = self.samples, self.vectors, self.revision
        names = sorted({sample["label"] for sample in samples if not sample["negative"]})
        positive_groups = [[index for index, sample in enumerate(samples) if sample["label"] == name and not sample["negative"]]
                           for name in names]
        groups = positive_groups + [[index] for index, sample in enumerate(samples) if sample["negative"]]
        anchors = [index for index, group in enumerate(positive_groups) if len(group) >= 2]
        if len(groups) < 2 or not anchors:
            return False
        self.training = True
        try:
            group_ids = {position: group_id for group_id, group in enumerate(groups) for position in group}
            baseline_groups = [anchors[0]] + [index for index in range(len(groups)) if index != anchors[0]][:63]
            selected = [position for group_id in baseline_groups for position in groups[group_id][:8]]
            features = gallery[selected].clone()
            identifiers = torch.tensor([group_ids[index] for index in selected], device=self.device)
            with torch.inference_mode(False), torch.enable_grad():
                initial = torch.eye(512, device=self.device)
                weight = torch.nn.Parameter(initial.clone())
                optimizer = torch.optim.AdamW([weight], lr=.001, weight_decay=.001)
                baseline = float(triplet_loss(features, identifiers))
                for step in range(steps):
                    if self.stopping.is_set():
                        return False
                    document = self.directory / "gallery.json"
                    if step % 5 == 0 and document.exists() and int(json.loads(document.read_text()).get("revision", 0)) != revision:
                        return False
                    anchor = anchors[step % len(anchors)]
                    other_groups = [(step * 63 + offset) % len(groups) for offset in range(min(64, len(groups)))]
                    active_groups = [groups[anchor]] + [groups[index] for index in other_groups if index != anchor][:63]
                    batch_indices = [group[(step * 8 + offset) % len(group)] for group in active_groups for offset in range(min(8, len(group)))]
                    batch = gallery[batch_indices]
                    batch_labels = torch.tensor([group_ids[index] for index in batch_indices], device=self.device)
                    loss = triplet_loss(functional.linear(batch, weight), batch_labels) + .02 * (weight - initial).square().mean()
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    if self.device.type == "cuda" and step % 4 == 3:
                        torch.cuda.current_stream(self.device).synchronize()
                    time.sleep(.005)
                final_loss = float(triplet_loss(functional.linear(features, weight), identifiers).detach())
                if not bool(torch.isfinite(weight).all()) or final_loss > baseline + 1e-5:
                    return False
                trained = weight.detach()
                projected = functional.normalize(functional.linear(gallery, trained), dim=1)
            if self.device.type == "cuda":
                torch.cuda.current_stream(self.device).synchronize()
            self.directory.mkdir(parents=True, exist_ok=True)
            temporary = self.directory / "triplet.tmp"
            torch.save(dict(weight=trained, signature=self.signature, revision=revision, loss=final_loss), temporary)
            with self.lock:
                if revision != self.revision:
                    return False
                temporary.replace(self.directory / "triplet.pt")
                self.weight, self.projected, self.loss = trained, projected, final_loss
                self.version += 1
            return True
        finally:
            self.training = False

    def policy(self, category):
        with self.lock:
            present = any(sample["category"] == category and not sample["negative"] for sample in self.samples)
            return dict(revision=self.revision, version=self.version, continuity_revision=self.continuity_revision,
                        required=self.require_labels or category in self.declared_categories,
                        available=present, error=self.error or ("Mẫu khác phiên bản SAM2; đăng ký lại." if self.incompatible_samples else None))

    @torch.inference_mode()
    def verify(self, descriptor, category, expected_label=None):
        with self.lock:
            samples, gallery, projected, weight, revision = self.samples, self.vectors, self.projected, self.weight, self.revision
            references, group_indices, version = self.references, self.group_indices, self.version
            error = self.error or self.incompatible_samples
        if error:
            return dict(accepted=False, reason="gallery_unavailable", revision=revision, version=version)
        if not samples:
            return dict(accepted=False, reason="no_label_samples", revision=revision)
        query = functional.normalize(torch.as_tensor(descriptor, device=self.device, dtype=torch.float32).reshape(1, 512), dim=1)
        if not bool(torch.isfinite(query).all()) or float(query.norm()) < 1e-6:
            return dict(accepted=False, reason="invalid_descriptor", revision=revision)
        raw = (gallery @ query.T).flatten()
        embedded = functional.normalize(functional.linear(query, weight), dim=1) if weight is not None else query
        similarities = (projected @ embedded.T).flatten()
        grouped = torch.full((len(references),), -1., device=self.device)
        grouped.scatter_reduce_(0, group_indices, similarities, reduce="amax", include_self=True)
        raw_grouped = torch.full_like(grouped, -1.)
        raw_grouped.scatter_reduce_(0, group_indices, raw, reduce="amax", include_self=True)
        raw_scores, raw_indices = raw_grouped.topk(min(2, len(references)))
        raw_winner = int(raw_indices[0])
        nearest_score = float(raw_scores[0])
        nearest_rival = float(raw_scores[1]) if len(references) > 1 else -1.
        nearest = references[raw_winner]
        if expected_label:
            expected_index = next((index for index, sample in enumerate(references)
                                   if sample['label'].casefold() == expected_label.casefold()
                                   and not sample['negative'] and sample['category'] == category), None)
            if expected_index is None:
                return dict(accepted=False, reason='reference_removed', revision=revision, version=version)
            expected_score = float(raw_grouped[expected_index])
            rival_score = float(raw_grouped[[index for index in range(len(references)) if index != expected_index]].max()) if len(references) > 1 else -1.
            reason = None
            if nearest['negative'] and nearest_score >= .84 and nearest_score >= expected_score - .02:
                reason = 'negative_sample'
            elif raw_winner != expected_index and nearest_score >= .84 and nearest_score - expected_score >= .06:
                reason = 'reference_identity_mismatch'
            elif expected_score < .78:
                reason = 'appearance_mismatch'
            reference = references[expected_index]
            return dict(accepted=reason is None, reason=reason, label=reference['label'] if reason is None else None,
                        candidate_label=reference['label'], expected_label=reference['label'],
                        class_name=reference['class_name'], score=round(expected_score, 4),
                        raw_score=round(expected_score, 4), rival_score=round(rival_score, 4),
                        projected_score=round(float(grouped[expected_index]), 4),
                        match_source='user_reference', revision=revision, version=version)
        if nearest_score >= .84 and (nearest["negative"] or nearest_score - nearest_rival >= .02):
            reason = "negative_sample" if nearest["negative"] else "category_conflict" if nearest["category"] != category else None
            return dict(accepted=reason is None, reason=reason, label=nearest["label"] if reason is None else None,
                        candidate_label=nearest["label"], class_name=nearest["class_name"],
                        score=round(nearest_score, 4), raw_score=round(nearest_score, 4),
                        projected_score=round(float(grouped[raw_winner]), 4), rival_score=round(nearest_rival, 4),
                        raw_rival_score=round(nearest_rival, 4), match_source="user_gallery",
                        revision=revision, version=version)
        scores, indices = grouped.topk(min(2, len(references)))
        winning_indices = indices.tolist()
        score, raw_score = torch.stack((scores[0], raw_grouped[indices[0]])).tolist()
        reference = references[winning_indices[0]]
        rival = float(scores[1]) if len(winning_indices) > 1 else -1
        reason = None
        if reference["negative"]:
            reason = "negative_sample"
        elif reference["category"] != category:
            reason = "category_conflict"
        elif score < .82 or raw_score < .78:
            reason = "appearance_mismatch"
        elif score - rival < .05:
            reason = "ambiguous_identity"
        return dict(accepted=reason is None, reason=reason, label=reference["label"] if reason is None else None,
                    candidate_label=reference["label"],
                    class_name=reference["class_name"], score=round(score, 4), raw_score=round(raw_score, 4),
                    projected_score=round(score, 4), raw_rival_score=round(nearest_rival, 4),
                    rival_score=round(rival, 4), match_source="triplet", revision=revision, version=version)

    def status(self):
        with self.lock:
            return dict(revision=self.revision, samples=len(self.samples), require_labels=self.require_labels,
                        training=self.training, last_loss=self.loss, error=self.error,
                        incompatible_samples=self.incompatible_samples,
                        state="training" if self.training else "trained" if self.weight is not None else "waiting_for_positive_and_negative_samples",
                        descriptor="sam2_mask_mean_std", detector_weights_trained=False,
                        prompt_count=sum(len(rows) for rows in self.prompts.values()),
                        active_labels=[sample['label'] for sample in self.references if not sample['negative']],
                        reference_min_similarity=.78,
                        matching_policy="user_gallery_first", direct_min_similarity=.84, direct_min_margin=.02)

    def stop(self):
        self.stopping.set()
        self.reload_event.set()
        with self.loaded:
            self.loaded.notify_all()
        if self.thread:
            self.thread.join(timeout=3)
