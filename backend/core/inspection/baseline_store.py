"""Persistent empty-cell baselines (``backend/data/inspection_baselines``).

A station keeps several baselines - the empty cell under each lighting
condition it meets (morning sun, lamps only, night shift ...), captured live or
taken from the recorded video at a moment the operator names.  Every frame is
compared with the best-matching one (baseline_selector).  Per baseline three
files are written atomically in ``<cam>/<rule>/``:

* ``<id>.png`` – the rectified BEV of the empty cell (1 px = 1 mm), the image
  the plan specifies and the Building view shows;
* ``<id>_camera.png`` – the per-pixel median of the captured camera frames,
  cropped around the ROI.  Every analysis view of the baseline is derived from
  it with exactly the warp used on live frames, so both are sampled alike;
* ``<id>.json`` – metadata (label, capture moment, source, ROI points, frame
  size, crop origin).

Format 1 kept one baseline per rule (``<cam>/<rule>.png|_camera.png|.json``);
it is moved into the rule's folder the first time the rule is read.

The median over several frames removes sensor noise - the light
normalisation itself is relative and happens at comparison time
(illumination_normalizer).
"""

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import DEFAULT_CELL_MM, WORK_MM_PER_PX
from .geometry_cv_inspector import BaselineModel
from .homography_rectifier import InvalidRoiError, MetricRectifier, order_clockwise
from .image_quality import check_image_quality

FORMAT_VERSION = 2
MAX_BASELINES = 12
LABEL_MAX_CHARS = 48
_SAFE_ID = re.compile(r"[^A-Za-z0-9_.-]")


def default_root() -> Path:
    configured = os.getenv("INSPECTION_BASELINE_DIR")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[2] / "data" / "inspection_baselines"


def _safe(value: str) -> str:
    text = _SAFE_ID.sub("_", str(value or "")).strip(".")
    if not text:
        raise ValueError("ID không hợp lệ")
    return text[:96]


def geometry_matches(meta: dict, points: Sequence[Sequence[float]], cell_mm: Tuple[float, float], frame_shape) -> bool:
    """Was the baseline ``meta`` captured for this ROI, cell size and camera resolution?"""
    try:
        if [int(v) for v in meta["frame_shape"][:2]] != [int(v) for v in frame_shape[:2]]:
            return False
        cell = meta.get("cell_mm") or DEFAULT_CELL_MM
        if any(abs(float(a) - float(b)) > 0.5 for a, b in zip(cell, cell_mm)):
            return False
        return bool(np.allclose(order_clockwise(meta["points"]), order_clockwise(points), atol=2e-3))
    except (KeyError, TypeError, ValueError, InvalidRoiError):
        return False


class BaselineCaptureError(ValueError):
    pass


class BaselineLimitError(BaselineCaptureError):
    pass


class BaselineStore:
    def __init__(self, root: Optional[Path] = None):
        self.root = Path(root) if root is not None else default_root()
        self._models: Dict[Tuple[str, str, str], Tuple[int, BaselineModel]] = {}
        self._entries: Dict[Tuple[str, str], Tuple[int, List[dict]]] = {}
        self._lock = threading.RLock()

    # ── paths ────────────────────────────────────────────────────────────────
    def _camera_dir(self, cam_id: str) -> Path:
        return self.root / _safe(cam_id)

    def _rule_dir(self, cam_id: str, rule_id: str) -> Path:
        return self._camera_dir(cam_id) / _safe(rule_id)

    def _paths(self, cam_id: str, rule_id: str, baseline_id: str) -> Dict[str, Path]:
        folder = self._rule_dir(cam_id, rule_id)
        bid = _safe(baseline_id)
        return {"dir": folder, "bev": folder / f"{bid}.png", "camera": folder / f"{bid}_camera.png",
                "meta": folder / f"{bid}.json"}

    def _legacy_paths(self, cam_id: str, rule_id: str) -> Dict[str, Path]:
        folder, rule = self._camera_dir(cam_id), _safe(rule_id)
        return {"bev": folder / f"{rule}.png", "camera": folder / f"{rule}_camera.png", "meta": folder / f"{rule}.json"}

    def relative_path(self, path: Path) -> str:
        try:
            return str(path.relative_to(Path(__file__).resolve().parents[2]))
        except ValueError:
            return str(path)

    # ── atomic writes ────────────────────────────────────────────────────────
    @staticmethod
    def _write_png(path: Path, image: np.ndarray) -> None:
        ok, data = cv2.imencode(".png", image)
        if not ok:
            raise OSError(f"Không mã hoá được {path.name}")
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(data.tobytes())
        os.replace(tmp, path)

    @staticmethod
    def _write_json(path: Path, data: dict) -> None:
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)

    def _new_id(self, cam_id: str, rule_id: str, moment: float) -> str:
        base = f"b{int(round(moment * 1000))}"
        candidate, suffix = base, 2
        while self._paths(cam_id, rule_id, candidate)["meta"].exists():
            candidate, suffix = f"{base}_{suffix}", suffix + 1
        return candidate

    def _migrate(self, cam_id: str, rule_id: str) -> None:
        """Move a format-1 baseline of the rule into its folder."""
        legacy = self._legacy_paths(cam_id, rule_id)
        if not legacy["meta"].is_file():
            return
        with self._lock:
            if not (legacy["meta"].is_file() and legacy["bev"].is_file() and legacy["camera"].is_file()):
                return
            try:
                meta = json.loads(legacy["meta"].read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return
            moment = float(meta.get("timestamp") or time.time())
            bid = self._new_id(cam_id, rule_id, moment)
            paths = self._paths(cam_id, rule_id, bid)
            paths["dir"].mkdir(parents=True, exist_ok=True)
            os.replace(legacy["bev"], paths["bev"])
            os.replace(legacy["camera"], paths["camera"])
            meta.update(id=bid, label=meta.get("label") or "", source=meta.get("source") or "live",
                        captured_at=meta.get("captured_at") or moment, format_version=FORMAT_VERSION,
                        image_path=self.relative_path(paths["bev"]))
            self._write_json(paths["meta"], meta)
            legacy["meta"].unlink()

    # ── capture ──────────────────────────────────────────────────────────────
    def save(self, cam_id: str, rule_id: str, frames: Sequence[np.ndarray], points: Sequence[Sequence[float]],
             cell_mm: Tuple[float, float] = DEFAULT_CELL_MM, label: str = "", source: str = "live",
             captured_at: Optional[float] = None, extra: Optional[dict] = None) -> dict:
        """Add the median of ``frames`` as one more baseline of the rule.

        Baselines of another ROI, cell size or camera resolution can never be
        used again and are replaced (``replaced`` in the result).  Raises
        :class:`BaselineCaptureError` on unusable input.
        """
        frames = [f for f in frames if f is not None and getattr(f, "size", 0)]
        if not frames:
            raise BaselineCaptureError("Không lấy được khung hình từ camera")
        shape = frames[0].shape
        frames = [f for f in frames if f.shape == shape]
        rect = MetricRectifier(points, cell_mm)
        rect.matrix(shape)
        quality = check_image_quality(frames[-1], rect.warp(frames[-1], WORK_MM_PER_PX))
        if not quality.ok:
            raise BaselineCaptureError(f"Ảnh nền không đạt chất lượng: {quality.message}")
        x0, y0, x1, y1 = rect.camera_bounds(shape)
        crops = np.stack([f[y0:y1, x0:x1] for f in frames])
        crop = np.median(crops, axis=0).astype(np.uint8) if len(frames) > 1 else crops[0].copy()
        bev = rect.warp_crop(crop, (x0, y0), shape, 1.0)
        timestamp = time.time()
        moment = float(captured_at) if captured_at is not None else timestamp
        cell = (float(rect.cell_mm[0]), float(rect.cell_mm[1]))

        with self._lock:
            existing = self.entries(cam_id, rule_id)
            stale = [meta for meta in existing if not geometry_matches(meta, points, cell, shape)]
            if len(existing) - len(stale) >= MAX_BASELINES:
                raise BaselineLimitError(f"Đã có {MAX_BASELINES} ảnh nền cho ô này – hãy xoá bớt ảnh không cần")
            bid = self._new_id(cam_id, rule_id, moment)
            paths = self._paths(cam_id, rule_id, bid)
            paths["dir"].mkdir(parents=True, exist_ok=True)
            self._write_png(paths["bev"], bev)
            self._write_png(paths["camera"], crop)
            meta = {
                "id": bid, "cam_id": cam_id, "rule_id": rule_id, "label": str(label or "").strip()[:LABEL_MAX_CHARS],
                "source": source, "captured_at": moment, "timestamp": timestamp,
                "points": [[float(x), float(y)] for x, y in order_clockwise(points)],
                "frame_shape": [int(shape[0]), int(shape[1])], "crop_origin": [int(x0), int(y0)],
                "frames_used": len(frames), "brightness": round(quality.brightness, 2),
                "laplacian_var": round(quality.laplacian_var, 2), "format_version": FORMAT_VERSION,
                "image_path": self.relative_path(paths["bev"]), "size_px": list(rect.size(1.0)),
                "cell_mm": list(cell), **(extra or {}),
            }
            self._write_json(paths["meta"], meta)
            for old in stale:
                self.delete(cam_id, rule_id, old["id"])
            self._entries.pop((cam_id, rule_id), None)
        return {**meta, "count": len(existing) - len(stale) + 1, "replaced": len(stale)}

    # ── read ─────────────────────────────────────────────────────────────────
    def entries(self, cam_id: str, rule_id: str) -> List[dict]:
        """Metadata of every complete baseline of the rule, oldest capture moment first."""
        self._migrate(cam_id, rule_id)
        folder = self._rule_dir(cam_id, rule_id)
        try:
            stamp = folder.stat().st_mtime_ns
        except OSError:
            return []
        key = (cam_id, rule_id)
        with self._lock:
            cached = self._entries.get(key)
            if cached and cached[0] == stamp:
                return [dict(meta) for meta in cached[1]]
        metas = []
        for meta_path in folder.glob("*.json"):
            bid = meta_path.stem
            if not ((folder / f"{bid}.png").is_file() and (folder / f"{bid}_camera.png").is_file()):
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            meta["id"] = bid
            meta.setdefault("captured_at", meta.get("timestamp"))
            meta.setdefault("label", "")
            meta.setdefault("source", "live")
            metas.append(meta)
        metas.sort(key=lambda m: (float(m.get("captured_at") or 0.0), m["id"]))
        with self._lock:
            self._entries[key] = (stamp, metas)
        return [dict(meta) for meta in metas]

    def info(self, cam_id: str, rule_id: str) -> Optional[dict]:
        """The most recently saved baseline, with ``count`` = how many the rule has."""
        metas = self.entries(cam_id, rule_id)
        if not metas:
            return None
        latest = max(metas, key=lambda m: float(m.get("timestamp") or 0.0))
        return {**latest, "count": len(metas)}

    def _model(self, cam_id: str, rule_id: str, meta: dict) -> Optional[BaselineModel]:
        paths = self._paths(cam_id, rule_id, meta["id"])
        try:
            stamp = paths["camera"].stat().st_mtime_ns
        except OSError:
            return None
        key = (cam_id, rule_id, meta["id"])
        with self._lock:
            cached = self._models.get(key)
            if cached and cached[0] == stamp:
                return cached[1]
        crop = cv2.imread(str(paths["camera"]), cv2.IMREAD_COLOR)
        if crop is None:
            return None
        try:
            rect = MetricRectifier(meta["points"], tuple(meta.get("cell_mm") or DEFAULT_CELL_MM))
            model = BaselineModel.from_camera(crop, tuple(meta["crop_origin"]), tuple(meta["frame_shape"]), rect,
                                              float(meta.get("timestamp") or 0.0), meta["points"], meta["id"],
                                              str(meta.get("label") or ""), float(meta.get("captured_at") or 0.0))
        except (KeyError, TypeError, ValueError, InvalidRoiError):
            return None
        with self._lock:
            self._models[key] = (stamp, model)
        return model

    def load_all(self, cam_id: str, rule_id: str) -> List[BaselineModel]:
        models = [self._model(cam_id, rule_id, meta) for meta in self.entries(cam_id, rule_id)]
        return [model for model in models if model is not None]

    def load(self, cam_id: str, rule_id: str) -> Optional[BaselineModel]:
        """The most recently saved baseline (single-baseline callers)."""
        meta = self.info(cam_id, rule_id)
        return self._model(cam_id, rule_id, meta) if meta else None

    def find(self, cam_id: str, points: Sequence[Sequence[float]]) -> Optional[str]:
        """Rule id of baselines captured for exactly these ROI points, if any."""
        folder = self._camera_dir(cam_id)
        if not folder.is_dir():
            return None
        target = order_clockwise(points)
        for rule_id in self.list_rules(cam_id):
            for meta in self.entries(cam_id, rule_id):
                try:
                    if np.allclose(order_clockwise(meta["points"]), target, atol=2e-3):
                        return str(meta.get("rule_id") or rule_id)
                except (KeyError, InvalidRoiError):
                    continue
        return None

    def bev_jpeg(self, cam_id: str, rule_id: str, baseline_id: Optional[str] = None, quality: int = 90) -> Optional[bytes]:
        meta = self.info(cam_id, rule_id) if baseline_id is None else next(
            (m for m in self.entries(cam_id, rule_id) if m["id"] == baseline_id), None)
        if meta is None:
            return None
        path = self._paths(cam_id, rule_id, meta["id"])["bev"]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR) if path.is_file() else None
        if image is None:
            return None
        ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return data.tobytes() if ok else None

    # ── delete ───────────────────────────────────────────────────────────────
    def delete(self, cam_id: str, rule_id: str, baseline_id: Optional[str] = None) -> int:
        """Delete one baseline (or all of the rule); returns how many were removed."""
        with self._lock:
            ids = [m["id"] for m in self.entries(cam_id, rule_id)] if baseline_id is None else [baseline_id]
            removed = 0
            for bid in ids:
                paths = self._paths(cam_id, rule_id, bid)
                existed = paths["meta"].exists()
                for name in ("meta", "bev", "camera"):
                    if paths[name].exists():
                        paths[name].unlink()
                self._models.pop((cam_id, rule_id, bid), None)
                removed += int(existed)
            if baseline_id is None:
                for path in self._legacy_paths(cam_id, rule_id).values():
                    if path.exists():
                        path.unlink()
                try:
                    self._rule_dir(cam_id, rule_id).rmdir()
                except OSError:
                    pass
            self._entries.pop((cam_id, rule_id), None)
            return removed

    def list_rules(self, cam_id: str) -> List[str]:
        folder = self._camera_dir(cam_id)
        if not folder.is_dir():
            return []
        for legacy in folder.glob("*.json"):
            self._migrate(cam_id, legacy.stem)
        return sorted(p.name for p in folder.iterdir() if p.is_dir() and any(p.glob("*.json")))
