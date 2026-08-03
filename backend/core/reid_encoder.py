"""Shared crop encoder for Label registration.

Live vectors come from NvDCF's GPU Re-ID module.  This small fallback encoder
uses the exact same ONNX weights and preprocessing when a user registers a
crop before a live tracker vector is available.
"""

import os
import threading
from typing import List, Optional

import cv2
import numpy as np


_session = None
_session_lock = threading.Lock()
_inference_lock = threading.Lock()


def _get_session():
    global _session
    if _session is not None:
        return _session
    with _session_lock:
        if _session is not None:
            return _session
        import onnxruntime as ort

        model_path = os.getenv("REID_ONNX_FILE", "/app/models/reid_512.onnx")
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)
        available = set(ort.get_available_providers())
        providers = [
            p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
            if p in available
        ] or ["CPUExecutionProvider"]
        _session = ort.InferenceSession(model_path, providers=providers)
        print(f"[ReID] Crop encoder ready: {model_path} ({providers[0]})", flush=True)
        return _session


def encode_crop(image: np.ndarray) -> Optional[List[float]]:
    """Encode one BGR crop into the normalized 512-D gallery vector."""
    if image is None or image.size == 0:
        return None
    session = _get_session()
    resized = cv2.resize(image, (128, 256), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    # Match the NvDCF ReID config (pixel - offsets) * scale. Keeping this
    # preprocessing identical is more important than which execution provider
    # handles this one-off registration request.
    rgb = rgb * 255.0
    rgb = (rgb - np.asarray([123.675, 116.280, 103.530], dtype=np.float32)) * 0.01735207
    tensor = np.transpose(rgb, (2, 0, 1))[None, ...]
    input_name = session.get_inputs()[0].name
    with _inference_lock:
        output = session.run(None, {input_name: tensor})[0]
    vector = np.asarray(output, dtype=np.float32).reshape(-1)
    if vector.size != 512:
        raise ValueError(f"Re-ID model returned {vector.size} values; expected 512")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-6:
        return None
    return (vector / norm).tolist()
