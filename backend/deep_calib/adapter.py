"""Adapter around the upstream DeepCalib SingleNet model.

The legacy model is loaded lazily by the API worker, never by the realtime
DeepStream path. Its intrinsic estimate is used to rectify calibration points
before fitting the authoritative Camera↔FMS floor homography.
"""

from __future__ import annotations

import base64
import importlib.util
import os
from pathlib import Path
import sys
import threading
import cv2
import numpy as np
from deep_calib.geometry import (profile_is_ready, undistort_normalized_points,
                                make_rectification_profile, validate_visible_rectified_points)

ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / "upstream"
DEFAULT_WEIGHTS = ROOT / "weights" / "weights_06_5.61.h5"
DEFAULT_RUNTIME = ROOT / ".runtime"
_MODEL = None
_MODEL_DEVICE = None
_MODEL_ERROR = None
_MODEL_LOCK = threading.Lock()
REPOSITORY_URL = "https://github.com/alexvbogdan/DeepCalib"
UPSTREAM_REVISION = "a04d8e0c4d4fdc362ebee016196eee5971506cb0"


def _runtime_path() -> Path:
    return Path(os.getenv("DEEP_CALIB_RUNTIME", str(DEFAULT_RUNTIME))).resolve()


def _runtime_available() -> bool:
    runtime = _runtime_path()
    return (runtime / "keras" / "__init__.py").is_file() and importlib.util.find_spec("torch") is not None


def _load_model():
    global _MODEL, _MODEL_DEVICE, _MODEL_ERROR
    if _MODEL is not None:
        return _MODEL, _MODEL_DEVICE
    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL, _MODEL_DEVICE
        runtime = _runtime_path()
        if runtime.is_dir() and str(runtime) not in sys.path:
            sys.path.insert(0, str(runtime))
        os.environ.setdefault("KERAS_BACKEND", "torch")
        try:
            import torch
            torch.backends.cudnn.enabled = False
            import keras

            backbone = keras.applications.InceptionV3(include_top=False, weights=None, input_shape=(299, 299, 3))
            features = keras.layers.Flatten(name="phi-flattened")(backbone.output)
            model = keras.Model(backbone.input, [
                keras.layers.Dense(47, activation="softmax", name="output_focal")(features),
                keras.layers.Dense(61, activation="softmax", name="output_distortion")(features),
            ])
            model.load_weights(str(Path(os.getenv("DEEP_CALIB_WEIGHTS", str(DEFAULT_WEIGHTS))).resolve()))
            if not torch.cuda.is_available():
                raise RuntimeError("DeepCalib yêu cầu CUDA; không chạy model trên CPU.")
            device = "cuda"
            model.to(device)
            model.eval()
            _MODEL, _MODEL_DEVICE, _MODEL_ERROR = model, device, None
        except Exception as error:
            _MODEL_ERROR = str(error)
            raise
    return _MODEL, _MODEL_DEVICE


def status() -> dict:
    weights = Path(os.getenv("DEEP_CALIB_WEIGHTS", str(DEFAULT_WEIGHTS)))
    return {
        "name": "DeepCalib SingleNet",
        "repository": "backend/deep_calib/upstream",
        "repository_url": REPOSITORY_URL,
        "revision": UPSTREAM_REVISION,
        "weights": str(weights),
        "weights_available": weights.is_file(),
        "legacy_runtime_available": _runtime_available(),
        "runtime": str(_runtime_path()),
        "mode": "rectified_image_then_fms_homography",
    }


def _decode_image(encoded: str) -> np.ndarray:
    if len(encoded) > 24_000_000:
        raise ValueError("Snapshot DeepCalib vượt giới hạn 18 MB.")
    raw = base64.b64decode(encoded.split(",", 1)[-1], validate=True)
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError("DeepCalib không đọc được ảnh snapshot.")
    if image.shape[0] * image.shape[1] > 16_777_216 or min(image.shape[:2]) < 32:
        raise ValueError("Snapshot phải có kích thước từ 32 pixel đến 16 megapixel.")
    return image


def estimate(encoded: str) -> dict:
    """Run SingleNet with the preprocessing used by the upstream training generator."""
    image = _decode_image(encoded)
    result = status()
    result.update(width=int(image.shape[1]), height=int(image.shape[0]), available=False)
    if not result["weights_available"]:
        result["reason"] = "Chưa có weights DeepCalib."
        return result
    if not result["legacy_runtime_available"]:
        result["reason"] = "Chưa có runtime Keras/Torch của DeepCalib; vẫn dùng được homography Camera↔FMS."
        return result
    try:
        model, device = _load_model()
        import torch
        from torch.nn import functional
        with torch.inference_mode():
            tensor = torch.as_tensor(image, device=device).permute(2, 0, 1).unsqueeze(0).float()
            tensor = functional.interpolate(tensor, size=(299, 299), mode="bilinear", align_corners=False)
            tensor = tensor / 127.5 - 1
            tensor = tensor.permute(0, 2, 3, 1) - torch.tensor([103.939, 116.779, 123.68], device=device)
            focal_scores, distortion_scores = model(tensor, training=False)
        focal_index = int(torch.argmax(focal_scores[0]).item())
        distortion_index = int(torch.argmax(distortion_scores[0]).item())
        model_focal = float(40 + focal_index * 10)
        focal = model_focal * image.shape[1] / 299.0
        distortion = float(distortion_index / 50.0)
        result.update(
            available=True,
            device=str(device),
            focal_length_model_px=round(model_focal, 4),
            focal_length_px=round(focal, 4),
            distortion_xi=round(distortion, 6),
            image_width=int(image.shape[1]),
            image_height=int(image.shape[0]),
            preprocessing="upstream_rgb_minus1_plus1_then_imagenet_caffe",
            focal_confidence=round(float(torch.max(focal_scores[0]).item()), 6),
            distortion_confidence=round(float(torch.max(distortion_scores[0]).item()), 6),
            reason="Đã suy luận intrinsic bằng DeepCalib SingleNet trên snapshot.",
        )
    except Exception as error:
        result["reason"] = f"DeepCalib không chạy được: {error}"
        result["error"] = _MODEL_ERROR or str(error)
    return result


def prepare_config(calibrator, source, destination, map_id, frame, metadata=None,
                   length_constraints=None, intrinsic=None, points_space="raw"):
    """Rectify clicked points, fit and persist the authoritative FMS homography."""
    from core.manual_calibration import prepare_manual_calibration

    profile = dict(intrinsic or {})
    if profile.get("image_width") is None:
        profile["image_width"] = (metadata or {}).get("image_width")
    if profile.get("image_height") is None:
        profile["image_height"] = (metadata or {}).get("image_height")
    if not profile_is_ready(profile):
        raise ValueError("Hãy chạy DeepCalib và xem ảnh hiệu chỉnh trước khi lưu.")
    if points_space not in {"raw", "rectified"}:
        raise ValueError("Miền tọa độ ảnh không hợp lệ.")
    if points_space == "raw" and "geometry_version" not in profile:
        profile = make_rectification_profile(profile)
    deep_status = dict(status(), available=True, intrinsic_profile=profile,
                       rectification="spherical_image_and_points", rectified_point_count=len(source))
    enriched = dict(metadata or {})
    constraints = list(length_constraints or [])
    if points_space == "rectified":
        fit_source = np.asarray(source, dtype=np.float64)
        raw_source = validate_visible_rectified_points(fit_source, profile)
        fit_constraints = constraints
        raw_constraints = [dict(item, points=validate_visible_rectified_points(item["points"], profile).tolist())
                           for item in constraints]
    else:
        raw_source = np.asarray(source, dtype=np.float64)
        fit_source = undistort_normalized_points(raw_source, profile)
        raw_constraints = constraints
        fit_constraints = [dict(item, points=undistort_normalized_points(item["points"], profile).tolist())
                           for item in constraints]
    enriched.update(
        deepcalib=deep_status,
        intrinsic_profile=profile,
        calibration_pipeline="DeepCalib → Camera/FMS metric homography",
    )
    config = prepare_manual_calibration(
        calibrator, fit_source.tolist(), destination, map_id, frame, enriched,
        length_constraints=fit_constraints,
    )
    rectified_coverage = config.get("coverage_polygon")
    config["src_points"] = raw_source.tolist()
    config["rectified_src_points"] = fit_source.tolist()
    config["length_constraints"] = raw_constraints
    config["rectified_length_constraints"] = fit_constraints
    config["raw_src_points"] = raw_source.tolist()
    config["raw_coverage_polygon"] = cv2.convexHull(raw_source.astype(np.float32)).reshape(-1, 2).tolist()
    config["coverage_polygon"] = config["raw_coverage_polygon"]
    config["rectified_coverage_polygon"] = rectified_coverage
    config["deepcalib_points_rectified"] = True
    config["input_points_space"] = points_space
    config["homography_input_space"] = "rectified"
    config.update(
        method="deepcalib_camera_fms",
        active_method="deepcalib_camera_fms",
        input_method="deepcalib_points_and_metric_lengths",
        calibration_role="primary_camera_fms_map",
    )
    return config
