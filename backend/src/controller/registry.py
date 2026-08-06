"""
Session 3: Target Identity Registry & Re-ID Cosine Similarity Matcher.
Manages persistent signatures of AMR robots (e.g. Robot_9001, Robot_2001, Robot_6868).
Performs high-speed Cosine Similarity matching for cross-camera re-identification.
"""

import os
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from core.identity_utils import robot_number_from_label
from core.registered_target_mask import validate_mask


class TargetRegistry:
    """
    Persistent AMR Robot Identity Registry.
    Stores 512-dimensional Re-ID signatures and matches incoming tracks in real-time.
    """
    def __init__(self, persistence_file: Optional[str] = None):
        if persistence_file:
            self.persist_path = Path(persistence_file)
        else:
            base_dir = Path(__file__).resolve().parent.parent.parent
            self.persist_path = base_dir / "data" / "registered_targets.json"
            
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Structure: { label: { "label": str, "vector": np.ndarray, "created_at": float, "last_seen": float, "samples_count": int } }
        self.targets: Dict[str, Dict[str, Any]] = {}
        self.local_track_labels: Dict[Tuple[str, int], str] = {}
        self.global_track_labels: Dict[int, str] = {}
        self.track_last_seen: Dict[Tuple[str, int], float] = {}
        self.global_track_last_seen: Dict[int, float] = {}
        self.load_from_disk()

    def _normalize_bbox(self, bbox: Optional[List[float]]) -> Optional[List[float]]:
        if not bbox or len(bbox) != 4:
            return None
        try:
            x, y, w, h = [float(v) for v in bbox]
        except (TypeError, ValueError):
            return None

        x = min(1.0, max(0.0, x))
        y = min(1.0, max(0.0, y))
        w = min(1.0 - x, max(0.0, w))
        h = min(1.0 - y, max(0.0, h))
        if w <= 0.0 or h <= 0.0:
            return None
        return [round(x, 4), round(y, 4), round(w, 4), round(h, 4)]

    def _bbox_iou(self, a: List[float], b: List[float]) -> float:
        ax1, ay1, aw, ah = a
        bx1, by1, bw, bh = b
        ax2, ay2 = ax1 + aw, ay1 + ah
        bx2, by2 = bx1 + bw, by1 + bh

        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        union = aw * ah + bw * bh - inter
        if union <= 1e-6:
            return 0.0
        return inter / union

    def _center_distance(self, a: List[float], b: List[float]) -> float:
        acx, acy = a[0] + a[2] / 2.0, a[1] + a[3] / 2.0
        bcx, bcy = b[0] + b[2] / 2.0, b[1] + b[3] / 2.0
        return float(np.hypot(acx - bcx, acy - bcy))

    def _cleanup_expired_assignments(self, max_age: float = 6.0):
        now = time.time()
        expired_local = [k for k, t in self.track_last_seen.items() if now - t > max_age]
        for k in expired_local:
            self.track_last_seen.pop(k, None)
            self.local_track_labels.pop(k, None)

        expired_global = [k for k, t in self.global_track_last_seen.items() if now - t > max_age]
        for k in expired_global:
            self.global_track_last_seen.pop(k, None)
            self.global_track_labels.pop(k, None)

    def _remember_assignment(
        self,
        label: str,
        cam_id: Optional[str],
        global_id: Optional[int],
        local_id: Optional[int],
        live_bbox: Optional[List[float]] = None
    ):
        now = time.time()
        if cam_id and local_id is not None:
            # If this label had a different active local track on this cam, clear it
            for (c, l), lbl in list(self.local_track_labels.items()):
                if c == cam_id and lbl == label and l != int(local_id):
                    self.local_track_labels.pop((c, l), None)
                    self.track_last_seen.pop((c, l), None)

            self.local_track_labels[(cam_id, int(local_id))] = label
            self.track_last_seen[(cam_id, int(local_id))] = now

        if global_id is not None:
            for g, lbl in list(self.global_track_labels.items()):
                if lbl == label and g != int(global_id):
                    self.global_track_labels.pop(g, None)
                    self.global_track_last_seen.pop(g, None)

            self.global_track_labels[int(global_id)] = label
            self.global_track_last_seen[int(global_id)] = now

        target_key = self._target_key(label, cam_id)
        target = self.targets.get(target_key) or self.targets.get(label)
        if not target:
            for k, d in self.targets.items():
                if d.get("label") == label or k.split("::")[-1] == label:
                    target = d
                    break
        if target:
            target["last_seen"] = now
            target["has_acquired"] = True
            if cam_id:
                target["last_cam"] = cam_id
                target["live_cam"] = cam_id
            if live_bbox:
                norm_box = self._normalize_bbox(live_bbox)
                if norm_box:
                    target["live_bbox"] = norm_box
                    target["live_last_seen"] = now

    def bind_label_to_track(
        self,
        label: str,
        cam_id: str,
        global_id: Optional[int],
        local_id: Optional[int],
        live_bbox: Optional[List[float]] = None
    ) -> bool:
        """Locks a registered user label to the current live tracker IDs."""
        target_key = self._target_key(label, cam_id)
        if target_key not in self.targets and label not in self.targets:
            matching = [k for k, d in self.targets.items() if d.get("label") == label or k.split("::")[-1] == label]
            if not matching:
                return False
        self._remember_assignment(label, cam_id, global_id, local_id, live_bbox=live_bbox)
        return True

    def get_assignments_for_label(self, label: str) -> Dict[str, Any]:
        local_ids = [
            {"cam_id": cam_id, "local_id": local_id}
            for (cam_id, local_id), assigned_label in self.local_track_labels.items()
            if assigned_label == label
        ]
        global_ids = [
            global_id
            for global_id, assigned_label in self.global_track_labels.items()
            if assigned_label == label
        ]
        return {"local": local_ids, "global": global_ids}

    def get_crop_image(self, label: str, cam_id: Optional[str] = None) -> Optional[str]:
        target = None
        if cam_id:
            target = self.targets.get(self._target_key(label, cam_id))
        if not target:
            target = self.targets.get(label)
        if not target:
            for k, d in self.targets.items():
                if d.get("label") == label or k.split("::")[-1] == label:
                    if not cam_id or d.get("cam_id") == cam_id or d.get("last_cam") == cam_id or cam_id in (d.get("samples_by_cam") or {}):
                        target = d
                        break
        if not target:
            return None
        if cam_id:
            sample = (target.get("samples_by_cam") or {}).get(cam_id)
            if sample and sample.get("crop_image"):
                return sample.get("crop_image")
        return target.get("crop_image")

    def get_camera_sample(self, label: str, cam_id: str) -> Optional[Dict[str, Any]]:
        target = self.targets.get(self._target_key(label, cam_id)) or self.targets.get(label)
        if not target:
            for k, d in self.targets.items():
                if d.get("label") == label or k.split("::")[-1] == label:
                    if d.get("cam_id") == cam_id or d.get("last_cam") == cam_id or cam_id in (d.get("samples_by_cam") or {}):
                        target = d
                        break
        if not target:
            return None
        sample = (target.get("samples_by_cam") or {}).get(cam_id)
        if sample:
            return sample
        if (target.get("cam_id") == cam_id or target.get("last_cam") == cam_id) and target.get("bbox"):
            return {
                "cam_id": cam_id,
                "bbox": target.get("bbox"),
                "crop_image": target.get("crop_image"),
                "category": target.get("category", "object"),
            }
        return None

    def get_template_samples(self, label: str, cam_id: str) -> List[Dict[str, Any]]:
        """Return every trusted crop collected for one label on one camera."""
        target = self.targets.get(self._target_key(label, cam_id)) or self.targets.get(label)
        if not target:
            target = next((
                data for key, data in self.targets.items()
                if (data.get("label") == label or key.split("::")[-1] == label)
                and (
                    data.get("cam_id") == cam_id
                    or data.get("last_cam") == cam_id
                    or cam_id in (data.get("samples_by_cam") or {})
                )
            ), None)
        if not target:
            return []

        samples = []
        for crop in target.get("crops") or []:
            if crop.get("crop_image") and self._normalize_bbox(crop.get("bbox")):
                samples.append({
                    "cam_id": cam_id,
                    "bbox": self._normalize_bbox(crop["bbox"]),
                    "crop_image": crop["crop_image"],
                    "mask": crop.get("mask"),
                    "frame_image": crop.get("frame_image"),
                    "category": target.get("category", "object"),
                    "created_at": crop.get("created_at", 0),
                })

        latest = (target.get("samples_by_cam") or {}).get(cam_id)
        if latest and latest.get("crop_image") and self._normalize_bbox(latest.get("bbox")):
            latest_sample = {
                "cam_id": cam_id,
                "bbox": self._normalize_bbox(latest["bbox"]),
                "crop_image": latest["crop_image"],
                "mask": latest.get("mask"),
                "frame_image": latest.get("frame_image"),
                "category": latest.get("category") or target.get("category", "object"),
                "created_at": latest.get("updated_at", 0),
            }
            if not any(s["crop_image"] == latest_sample["crop_image"] for s in samples):
                samples.append(latest_sample)

        if not samples:
            fallback = self.get_camera_sample(label, cam_id)
            if fallback and fallback.get("crop_image") and fallback.get("bbox"):
                samples.append(fallback)
        return samples[-16:]

    def _target_key(self, label: str, cam_id: Optional[str] = None) -> str:
        clean_label = (label or "").strip()
        if cam_id:
            return f"{cam_id}::{clean_label}"
        return clean_label

    def _find_target_data(self, label: str, cam_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if cam_id:
            target = self.targets.get(self._target_key(label, cam_id))
            if target:
                return target
        target = self.targets.get(label)
        if target:
            return target
        return next((
            data for key, data in self.targets.items()
            if (data.get("label") == label or key.split("::")[-1] == label)
            and (
                not cam_id
                or data.get("cam_id") == cam_id
                or data.get("last_cam") == cam_id
                or cam_id in (data.get("samples_by_cam") or {})
            )
        ), None)

    def register_target(
        self,
        label: str,
        vector_512: List[float],
        cam_id: Optional[str] = None,
        world_pos: Optional[Tuple[float, float]] = None,
        category: Optional[str] = None,
        bbox: Optional[List[float]] = None,
        crop_image: Optional[str] = None,
        embedding_type: str = "reid_512",
        rack_3d_size: Optional[List[float]] = None,
        mask: Optional[dict] = None,
        frame_image: Optional[str] = None,
    ) -> bool:
        """
        Registers or enriches a target’s multi-angle 512-dim embedding signatures.
        Accepts both 'reid_512' (NvDCF Re-ID) and 'clip_512' (CLIP ViT-B/32).
        Each crop adds a new viewpoint angle into the target’s feature gallery (up to 16 angles).
        """
        mask = validate_mask(mask) if mask is not None else None
        if mask is not None and (category not in {"robot", "rack"} or not frame_image):
            raise ValueError("Mask chỉ dành cho robot/kệ và cần frame đăng ký.")
        embedding_type = str(embedding_type or "reid_512").strip().lower()
        # Accept reid_512 (NvDCF) and clip_512 (CLIP ViT-B/32 for Robot/Rack)
        if embedding_type not in ("reid_512", "clip_512"):
            print(f"❌ Unsupported embedding type '{embedding_type}' (Expected: reid_512 or clip_512)")
            return False

        if not vector_512 or len(vector_512) != 512:
            print(f"❌ Invalid vector length {len(vector_512) if vector_512 else 0} (Expected: 512)")
            return False
            
        vec = np.array(vector_512, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 1e-6:
            vec = vec / norm
        else:
            return False
            
        now = time.time()
        normalized_bbox = self._normalize_bbox(bbox)
        target_key = self._target_key(label, cam_id)
        fms_robot_id = robot_number_from_label(label) if (category or "").lower() == "robot" or label.lower().startswith("robot") else None
        
        sample_payload = None
        if cam_id and (normalized_bbox or crop_image):
            sample_payload = {
                "cam_id": cam_id,
                "bbox": normalized_bbox,
                "crop_image": crop_image,
                "category": category or "object",
                "updated_at": now,
            }

        crop_entry = {
            "bbox": normalized_bbox,
            "crop_image": crop_image,
            "created_at": now
        } if (normalized_bbox or crop_image) else None
        if mask is not None:
            if sample_payload is not None:
                sample_payload.update(mask=mask, frame_image=frame_image)
            if crop_entry is not None:
                crop_entry.update(mask=mask, frame_image=frame_image)

        if target_key in self.targets:
            # Update and enrich existing target on this camera with new angle view
            target_data = self.targets[target_key]
            # Reset gallery if embedding type changes (reid ↔ clip)
            if target_data.get("embedding_type", "legacy_hsv") != embedding_type:
                existing_vectors = []
                target_data["embedding_type"] = embedding_type
                target_data["migration_note"] = f"gallery_replaced_by_{embedding_type}"
            else:
                existing_vectors = target_data.setdefault("vectors", [target_data["vector"]])
            existing_crops = target_data.setdefault("crops", [])

            # Check similarity against existing angle vectors
            sims = [float(np.dot(vec, v / (np.linalg.norm(v) + 1e-6))) for v in existing_vectors]
            max_existing_sim = max(sims) if sims else -1.0
            
            # If distinct viewpoint or gallery not full, append as new angle
            if max_existing_sim < 0.985 or len(existing_vectors) < 3:
                existing_vectors.append(vec)
                if len(existing_vectors) > 16:
                    existing_vectors.pop(0)
            else:
                # Update closest matching angle vector with slight EMA
                best_idx = int(np.argmax(sims))
                blended = 0.6 * existing_vectors[best_idx] + 0.4 * vec
                existing_vectors[best_idx] = blended / (np.linalg.norm(blended) + 1e-6)

            if crop_entry:
                existing_crops.append(crop_entry)
                if len(existing_crops) > 16:
                    existing_crops.pop(0)

            # Recompute centroid vector across all angle samples
            target_data["vectors"] = existing_vectors
            centroid = np.mean(existing_vectors, axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-6)
            target_data["vector"] = centroid
            target_data["last_seen"] = now
            target_data["samples_count"] = len(existing_vectors)
            target_data["embedding_type"] = embedding_type
            target_data["cam_id"] = cam_id or target_data.get("cam_id", "unknown")
            target_data["last_cam"] = cam_id or target_data.get("last_cam", "unknown")
            if world_pos:
                target_data["last_pos"] = world_pos
            if category:
                target_data["category"] = category
            if fms_robot_id is not None:
                target_data["fms_robot_id"] = fms_robot_id
            if normalized_bbox:
                target_data["bbox"] = normalized_bbox
            if crop_image:
                target_data["crop_image"] = crop_image
            if sample_payload:
                samples_by_cam = target_data.setdefault("samples_by_cam", {})
                samples_by_cam[cam_id] = sample_payload
            print(f"🔄 Enriched multi-angle feature gallery for '{label}' on camera {cam_id} (Total angles: {len(existing_vectors)})")
        else:
            # Register new camera-scoped target with 1st angle
            self.targets[target_key] = {
                "key": target_key,
                "label": label,
                "cam_id": cam_id or "unknown",
                "last_cam": cam_id or "unknown",
                "vector": vec,
                "vectors": [vec],
                "crops": [crop_entry] if crop_entry else [],
                "created_at": now,
                "last_seen": now,
                "last_pos": world_pos or (0.0, 0.0),
                "samples_count": 1,
                "category": category or "object",
                "embedding_type": embedding_type,
                "fms_robot_id": fms_robot_id,
                "bbox": normalized_bbox,
                "crop_image": crop_image,
                "samples_by_cam": {cam_id: sample_payload} if sample_payload else {}
            }
            print(f"✅ Registered new target: '{label}' for camera {cam_id} (1st angle view, type={embedding_type})")

        # Lưu rack_3d_size nếu có
        if rack_3d_size and category and category.lower() == "rack":
            target_data = self.targets.get(target_key)
            if target_data:
                target_data["rack_3d_size"] = rack_3d_size[:3]  # [width_m, depth_m, height_m]
            
        self.save_to_disk()
        return True

    def match_reid(
        self,
        vector_input: List[float],
        threshold: float = 0.65
    ) -> Tuple[Optional[str], float]:
        """
        Computes Cosine Similarity against all registered target multi-angle signatures.
        Returns (best_matching_label, similarity_score) if score >= threshold, else (None, best_score).
        """
        if not self.targets or not vector_input or len(vector_input) != 512:
            return (None, 0.0)
            
        q_vec = np.array(vector_input, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm < 1e-6:
            return (None, 0.0)
        q_vec = q_vec / q_norm
        
        best_label = None
        best_score = -1.0
        
        for label, data in self.targets.items():
            if data.get("embedding_type", "legacy_hsv") != "reid_512":
                continue
            gallery = data.get("vectors") or [data["vector"]]
            for t_vec in gallery:
                t_norm = float(np.linalg.norm(t_vec))
                if t_norm > 1e-6:
                    sim = float(np.dot(q_vec, t_vec / t_norm))
                    if sim > best_score:
                        best_score = sim
                        best_label = data.get("label", label.split("::")[-1])
                
        if best_score >= threshold and best_label is not None:
            return (best_label, round(best_score, 4))
        return (None, round(max(0.0, best_score), 4))

    def _match_bboxes_score(self, a: List[float], b: List[float]) -> Tuple[float, bool]:
        """
        Calculates robust match score and boolean validity between 2 normalized bboxes [x, y, w, h].
        """
        ax, ay, aw, ah = a
        bx, by, bw, bh = b

        acx, acy = ax + aw / 2.0, ay + ah / 2.0
        bcx, bcy = bx + bw / 2.0, by + bh / 2.0

        a_in_b = (bx <= acx <= bx + bw) and (by <= acy <= by + bh)
        b_in_a = (ax <= bcx <= ax + aw) and (ay <= bcy <= ay + ah)

        ix1, iy1 = max(ax, bx), max(ay, by)
        ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area_a = max(1e-6, aw * ah)
        area_b = max(1e-6, bw * bh)
        union = area_a + area_b - inter

        iou = inter / union if union > 1e-6 else 0.0
        cov_a = inter / area_a
        cov_b = inter / area_b

        dist = float(np.hypot(acx - bcx, acy - bcy))
        prox = max(0.0, (0.35 - dist) / 0.35)

        containment = 0.50 if (a_in_b or b_in_a) else 0.0
        overlap = max(iou * 1.5, cov_a * 0.8, cov_b * 0.8)
        score = containment + (0.35 * overlap) + (0.25 * prox)

        is_valid = (
            a_in_b or
            b_in_a or
            (cov_a >= 0.20) or
            (cov_b >= 0.20) or
            (iou >= 0.06) or
            (dist <= 0.22 and prox > 0.35)
        )
        return (score if is_valid else 0.0, is_valid)

    def assign_label_for_detection(
        self,
        cam_id: str,
        global_id: Optional[int],
        local_id: Optional[int],
        bbox: Optional[List[float]],
        feature: Optional[Any] = None,
        floor_pos: Optional[Tuple[float, float]] = None,
        det_class: Optional[str] = None,
        threshold: float = 0.65
    ) -> Optional[str]:
        """
        Returns a persistent user label for a live detection using true multi-angle 512-dim Re-ID matching.
        Priority:
          1. Multi-Angle 512-dim Re-ID Cosine Similarity Match (Appearance Matching against all viewpoint crops >= 65%),
          2. Active Track Lock (verified with Re-ID feature continuity >= 65%),
          3. FMS Ground Truth (Strict: Distance < 1.8m + Not Human + Re-ID verified >= 65%),
          4. Immediate Crop Bbox Match (only within <= 12s of user crop with Re-ID >= 65%).
        """
        if not self.targets:
            return None

        now = time.time()
        self._cleanup_expired_assignments(max_age=6.0)
        det_bbox = self._normalize_bbox(bbox)
        class_name = (det_class or "").lower()
        is_human = any(h in class_name for h in ("person", "human", "worker", "man", "woman"))

        # 0. Compute Multi-Angle Re-ID Cosine Similarity for the current detection feature against all target viewpoint galleries
        reid_scores: Dict[str, float] = {}
        best_reid_label = None
        best_reid_score = -1.0

        if feature is not None:
            try:
                q_vec = np.array(feature, dtype=np.float32)
                q_norm = float(np.linalg.norm(q_vec))
                if q_norm > 1e-6:
                    q_vec = q_vec / q_norm
                    for key, data in self.targets.items():
                        emb_type = data.get("embedding_type", "legacy_hsv")
                        # Live detector metadata is NvDCF Re-ID. CLIP and Re-ID
                        # vectors are not interchangeable despite both being 512D.
                        if emb_type != "reid_512":
                            continue
                        t_cam = data.get("cam_id") or data.get("last_cam")
                        if cam_id and t_cam and t_cam != cam_id and cam_id not in (data.get("samples_by_cam") or {}):
                            continue
                        
                        # Compare against all angle vectors in this target's gallery
                        gallery = data.get("vectors") or [data["vector"]]
                        best_gallery_sim = -1.0
                        for t_vec in gallery:
                            t_norm = float(np.linalg.norm(t_vec))
                            if t_norm > 1e-6:
                                sim = float(np.dot(q_vec, t_vec / t_norm))
                                if sim > best_gallery_sim:
                                    best_gallery_sim = sim
                                    
                        if best_gallery_sim > -1.0:
                            lbl = data.get("label", key.split("::")[-1])
                            reid_scores[lbl] = round(best_gallery_sim, 4)
                            if best_gallery_sim > best_reid_score:
                                best_reid_score = best_gallery_sim
                                best_reid_label = lbl
            except Exception:
                pass

        # 1. PRIMARY: Multi-Angle Re-ID Feature Cosine Similarity Match (Threshold = 65%)
        if best_reid_label and best_reid_score >= threshold:
            # Look up target category
            matched_data = next((d for k, d in self.targets.items() if (d.get("label") == best_reid_label or k == best_reid_label)), {})
            target_cat = (matched_data.get("category") or "").lower()
            is_robot_or_rack = any(k in target_cat or best_reid_label.lower().startswith(k) for k in ("robot", "rack", "shelf"))
            
            # Guardrail: Never assign a robot or rack target to a person detection
            if is_robot_or_rack and is_human:
                pass
            else:
                self._remember_assignment(best_reid_label, cam_id, global_id, local_id, live_bbox=det_bbox)
                return best_reid_label

        # 2. Track Lock & Continuity (Threshold = 65%)
        if cam_id and local_id is not None:
            existing = self.local_track_labels.get((cam_id, int(local_id)))
            existing_target = self._find_target_data(existing, cam_id) if existing else None
            if existing_target:
                target_cat = (existing_target.get("category") or "").lower()
                is_robot_or_rack = any(k in target_cat or existing.lower().startswith(k) for k in ("robot", "rack", "shelf"))
                if is_robot_or_rack and is_human:
                    self.local_track_labels.pop((cam_id, int(local_id)), None)
                else:
                    cur_sim = reid_scores.get(existing, 0.0)
                    if cur_sim >= 0.65:
                        self._remember_assignment(existing, cam_id, global_id, local_id, live_bbox=det_bbox)
                        return existing
                    else:
                        self.local_track_labels.pop((cam_id, int(local_id)), None)

        if global_id is not None:
            existing_g = self.global_track_labels.get(int(global_id))
            existing_target = self._find_target_data(existing_g, cam_id) if existing_g else None
            if existing_target:
                target_cat = (existing_target.get("category") or "").lower()
                is_robot_or_rack = any(k in target_cat or existing_g.lower().startswith(k) for k in ("robot", "rack", "shelf"))
                if is_robot_or_rack and is_human:
                    self.global_track_labels.pop(int(global_id), None)
                else:
                    cur_sim = reid_scores.get(existing_g, 0.0)
                    if cur_sim >= 0.65:
                        self._remember_assignment(existing_g, cam_id, global_id, local_id, live_bbox=det_bbox)
                        return existing_g
                    else:
                        self.global_track_labels.pop(int(global_id), None)

        # 3. FMS Ground Truth Spatial Match (Strict: Distance < 1.8m + Not Human + Re-ID >= 0.65)
        if floor_pos is not None and not is_human:
            try:
                from core.fms_bridge import fms_bridge
                fx, fy = float(floor_pos[0]), float(floor_pos[1])
                best_fms_label = None
                best_fms_dist = 1.8  # meters
                for label, data in self.targets.items():
                    if data.get("embedding_type", "legacy_hsv") != "reid_512":
                        continue
                    if (data.get("category") or "").lower() != "robot" and not label.lower().startswith("robot"):
                        continue
                    robot_id = data.get("fms_robot_id") or robot_number_from_label(label)
                    robot_key = str(robot_id) if robot_id is not None else label.replace("Robot_", "")
                    fms_robot = None
                    for k in (label, robot_key, f"Robot_{robot_key}"):
                        if k in fms_bridge.robot_states:
                            fms_robot = fms_bridge.robot_states[k]
                            break
                    if fms_robot and fms_robot.get("position"):
                        pos = fms_robot["position"]
                        dist = float(np.hypot(fx - float(pos[0]), fy - float(pos[2])))
                        fms_sim = reid_scores.get(label, 0.0)
                        if dist < best_fms_dist and fms_sim >= 0.65:
                            best_fms_dist = dist
                            best_fms_label = label
                if best_fms_label:
                    self._remember_assignment(best_fms_label, cam_id, global_id, local_id, live_bbox=det_bbox)
                    return best_fms_label
            except Exception:
                pass

        # 4. Immediate Crop Match (ONLY within 12 seconds of manual crop AND Re-ID similarity >= 0.65)
        if det_bbox is not None and not is_human:
            for label, data in self.targets.items():
                if data.get("embedding_type", "legacy_hsv") != "reid_512":
                    continue
                created_at = data.get("created_at", 0)
                if (now - created_at) > 12.0:
                    continue
                crop_sim = reid_scores.get(label, 0.0)
                if crop_sim < 0.65:
                    continue
                candidate_boxes = []
                cam_sample = (data.get("samples_by_cam") or {}).get(cam_id)
                if cam_sample and cam_sample.get("bbox"):
                    candidate_boxes.append(cam_sample.get("bbox"))
                if data.get("last_cam") == cam_id and data.get("bbox"):
                    candidate_boxes.append(data.get("bbox"))
                for candidate_bbox in candidate_boxes:
                    target_bbox = self._normalize_bbox(candidate_bbox)
                    if target_bbox is None:
                        continue
                    score, is_valid = self._match_bboxes_score(det_bbox, target_bbox)
                    if is_valid and score > 0.0:
                        self._remember_assignment(label, cam_id, global_id, local_id, live_bbox=det_bbox)
                        return label

        return None

    def get_all_targets(self, cam_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Returns list of registered targets for API/UI, optionally filtered by camera."""
        result = []
        for key, data in self.targets.items():
            t_cam = data.get("cam_id") or data.get("last_cam") or "unknown"
            if cam_id and t_cam != cam_id:
                if cam_id not in (data.get("samples_by_cam") or {}):
                    continue

            category = data.get("category", "object")
            label = data.get("label") or key.split("::")[-1]
            fms_robot_id = data.get("fms_robot_id")
            if fms_robot_id is None and ((category or "").lower() == "robot" or label.lower().startswith("robot")):
                fms_robot_id = robot_number_from_label(label)
                
            vectors_count = len(data.get("vectors", [data["vector"]])) if data.get("vectors") is not None else 1
            crops_list = data.get("crops", [])
            
            result.append({
                "key": key,
                "label": label,
                "cam_id": t_cam,
                "created_at": data["created_at"],
                "last_seen": data["last_seen"],
                "last_cam": t_cam,
                "last_pos": data.get("last_pos", (0.0, 0.0)),
                "samples_count": data.get("samples_count", vectors_count),
                "crops_count": len(crops_list) if crops_list else (1 if data.get("crop_image") else 0),
                "category": category,
                "embedding_type": data.get("embedding_type", "legacy_hsv"),
                "fms_robot_id": fms_robot_id,
                "bbox": data.get("bbox"),
                "has_crop_image": bool(data.get("crop_image")),
                "mask_samples_count": sum(bool(crop.get("mask")) for crop in crops_list),
                "samples_by_cam": {
                    cid: {
                        "cam_id": sample.get("cam_id", cid),
                        "bbox": sample.get("bbox"),
                        "has_crop_image": bool(sample.get("crop_image")),
                        "updated_at": sample.get("updated_at"),
                    }
                    for cid, sample in (data.get("samples_by_cam") or {}).items()
                }
            })
        return result

    def remove_target(self, label: str, cam_id: Optional[str] = None) -> bool:
        """
        Removes a target independently on a specific camera, or globally if no cam_id is provided.
        Deleting on one camera never affects another camera.
        Only deleted when user explicitly triggers removal.
        """
        deleted = False
        if cam_id:
            target_key = self._target_key(label, cam_id)
            if target_key in self.targets:
                del self.targets[target_key]
                deleted = True
            # Also remove matching key/labels for this camera
            for k, data in list(self.targets.items()):
                t_cam = data.get("cam_id") or data.get("last_cam")
                t_lbl = data.get("label") or k.split("::")[-1]
                if t_cam == cam_id and t_lbl == label:
                    del self.targets[k]
                    deleted = True
                elif t_lbl == label and cam_id in (data.get("samples_by_cam") or {}):
                    del data["samples_by_cam"][cam_id]
                    deleted = True
            for (c, l), lbl in list(self.local_track_labels.items()):
                if c == cam_id and lbl == label:
                    self.local_track_labels.pop((c, l), None)
            print(f"🗑️ Removed target: {label} on camera {cam_id}")
        else:
            # Global delete
            for k, data in list(self.targets.items()):
                t_lbl = data.get("label") or k.split("::")[-1]
                if t_lbl == label or k == label:
                    del self.targets[k]
                    deleted = True
            self.local_track_labels = {k: v for k, v in self.local_track_labels.items() if v != label}
            self.global_track_labels = {k: v for k, v in self.global_track_labels.items() if v != label}
            print(f"🗑️ Removed target globally: {label}")

        if deleted:
            self.save_to_disk()
            return True
        return False

    def remove_camera_sample(self, label: str, cam_id: str) -> bool:
        """Alias for camera-scoped target removal"""
        return self.remove_target(label, cam_id=cam_id)

    def save_to_disk(self):
        """Serializes targets to JSON file with atomic write to prevent corruption on reset"""
        try:
            serializable = {}
            for key, data in self.targets.items():
                vectors_list = [v.tolist() for v in data.get("vectors", [data["vector"]])]
                serializable[key] = {
                    "key": key,
                    "label": data.get("label", key.split("::")[-1]),
                    "cam_id": data.get("cam_id") or data.get("last_cam", "unknown"),
                    "vector": data["vector"].tolist(),
                    "vectors": vectors_list,
                    "crops": data.get("crops", []),
                    "created_at": data["created_at"],
                    "last_seen": data["last_seen"],
                    "last_cam": data.get("last_cam", "unknown"),
                    "last_pos": data.get("last_pos", (0.0, 0.0)),
                    "samples_count": data.get("samples_count", len(vectors_list)),
                    "category": data.get("category", "object"),
                    "embedding_type": data.get("embedding_type", "legacy_hsv"),
                    "fms_robot_id": data.get("fms_robot_id"),
                    "bbox": data.get("bbox"),
                    "crop_image": data.get("crop_image"),
                    "samples_by_cam": data.get("samples_by_cam", {})
                }
            tmp_path = self.persist_path.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(serializable, f, indent=2)
            tmp_path.replace(self.persist_path)
        except Exception as e:
            print(f"Error saving targets registry: {e}")

    def load_from_disk(self):
        """Loads registered targets from JSON file and restores all multi-angle feature vectors"""
        if not self.persist_path.exists() or self.persist_path.stat().st_size == 0:
            return
        try:
            with open(self.persist_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for raw_key, item in data.items():
                label = item.get("label") or raw_key.split("::")[-1]
                cam_id = item.get("cam_id") or item.get("last_cam") or "unknown"
                category = item.get("category", "object")
                # Files created before GPU NvDCF Re-ID was enabled are marked
                # legacy so they cannot accidentally match live embeddings.
                embedding_type = str(item.get("embedding_type") or "legacy_hsv").strip().lower()
                fms_robot_id = item.get("fms_robot_id")
                if fms_robot_id is None and ((category or "").lower() == "robot" or label.lower().startswith("robot")):
                    fms_robot_id = robot_number_from_label(label)
                
                target_key = item.get("key") or self._target_key(label, cam_id if cam_id != "unknown" else None)
                
                raw_vectors = item.get("vectors")
                if raw_vectors and isinstance(raw_vectors, list) and len(raw_vectors) > 0:
                    vectors = [np.array(v, dtype=np.float32) for v in raw_vectors]
                else:
                    vectors = [np.array(item["vector"], dtype=np.float32)]

                main_vector = np.array(item["vector"], dtype=np.float32)

                self.targets[target_key] = {
                    "key": target_key,
                    "label": label,
                    "cam_id": cam_id,
                    "vector": main_vector,
                    "vectors": vectors,
                    "crops": item.get("crops", []),
                    "created_at": item.get("created_at", time.time()),
                    "last_seen": item.get("last_seen", time.time()),
                    "last_cam": cam_id,
                    "last_pos": item.get("last_pos", (0.0, 0.0)),
                    "samples_count": item.get("samples_count", len(vectors)),
                    "category": category,
                    "embedding_type": embedding_type,
                    "fms_robot_id": fms_robot_id,
                    "bbox": self._normalize_bbox(item.get("bbox")),
                    "crop_image": item.get("crop_image"),
                    "samples_by_cam": item.get("samples_by_cam", {})
                }
            print(f"📂 Loaded {len(self.targets)} registered target signatures from disk with multi-angle galleries")
        except Exception as e:
            print(f"Error loading targets registry: {e}")


# Singleton target registry instance
target_registry = TargetRegistry()
