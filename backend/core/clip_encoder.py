"""
CLIP ViT-B/32 Crop Encoder for Robot & Rack Registration.

Dùng cho việc encode ảnh crop từ UI vào 512-dim vector
rồi so sánh cosine similarity khi track live.

Pipeline:
  Web UI Crop → encode_crop_clip() → 512-D L2-normalized vector
                                       ↓ store in TargetRegistry
  DeepStream Probe → live crop → encode_crop_clip() → cosine match

Note:
  - cuDNN không tương thích với container này nên CLIP chạy trên CPU.
  - Inference ~85ms/frame trên CPU – đủ nhanh cho registration (one-shot),
    KHÔNG dùng cho per-frame real-time tracking.
  - Preprocessing dùng cv2 thuần (không cần PIL/torchvision).
"""

import threading
import numpy as np
import cv2
from typing import Optional, List

_clip_model = None
_clip_lock = threading.Lock()
# Force CPU – cuDNN mismatch trong container DeepStream
_CLIP_DEVICE = "cpu"


def _get_clip_model():
    """Lazy-load CLIP ViT-B/32 trên CPU (thread-safe)."""
    global _clip_model
    if _clip_model is not None:
        return _clip_model

    with _clip_lock:
        if _clip_model is not None:
            return _clip_model

        try:
            import torch
            from transformers import CLIPModel

            # Load CLIP ViT-B/32 – CPU only (cuDNN mismatch trong container)
            _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            _clip_model = _clip_model.to(_CLIP_DEVICE)
            _clip_model.eval()

            print(
                f"[CLIPEncoder] CLIP ViT-B/32 loaded on {_CLIP_DEVICE} (CPU) "
                f"(projection_dim={_clip_model.config.projection_dim})",
                flush=True,
            )
        except Exception as e:
            print(f"[CLIPEncoder] Failed to load CLIP: {e}", flush=True)
            _clip_model = None

    return _clip_model


def encode_crop_clip(image_bgr: np.ndarray) -> Optional[List[float]]:
    """
    Encode một ảnh BGR crop thành vector 512-D L2-normalized dùng CLIP ViT-B/32.

    Args:
        image_bgr: numpy array (H, W, 3) BGR (từ cv2)

    Returns:
        List[float] 512-dim vector đã L2-normalize, hoặc None nếu lỗi.
        Dùng cho registration (web UI crop) → lưu vào TargetRegistry.
    """
    if image_bgr is None or image_bgr.size == 0:
        return None

    model = _get_clip_model()
    if model is None:
        return _fallback_encode(image_bgr)

    return _encode_without_pil(image_bgr, model)


def _encode_without_pil(image_bgr: np.ndarray, model) -> Optional[List[float]]:
    """CLIP preprocess bằng cv2 thuần (không cần PIL/torchvision)."""
    try:
        import torch

        # CLIP manual preprocessing: resize 224x224, normalize
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
        img_f = resized.astype(np.float32) / 255.0

        # CLIP normalization constants
        mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
        std  = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
        img_f = (img_f - mean) / std

        # HWC → CHW → NCHW tensor trên CPU
        tensor = torch.tensor(img_f.transpose(2, 0, 1)[None], dtype=torch.float32)

        with torch.no_grad():
            vision_outputs = model.vision_model(pixel_values=tensor)
            projected = model.visual_projection(vision_outputs.pooler_output)  # (1, 512)

        vec = projected.squeeze().numpy().astype(np.float32)
        norm = float(np.linalg.norm(vec))
        if norm < 1e-6:
            return None
        return (vec / norm).tolist()

    except Exception as e:
        print(f"[CLIPEncoder] encode error: {e}", flush=True)
        return _fallback_encode(image_bgr)


def _fallback_encode(image_bgr: np.ndarray) -> Optional[List[float]]:
    """
    Last-resort fallback: HSV histogram + HOG → 512-D pseudo-embedding.
    Kém hơn CLIP nhưng không cần bất kỳ model nào.
    """
    try:
        resized = cv2.resize(image_bgr, (128, 128))
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

        # HSV histogram 3 channels × 64 bins = 192 features
        h_hist = cv2.calcHist([hsv], [0], None, [64], [0, 180]).flatten()
        s_hist = cv2.calcHist([hsv], [1], None, [64], [0, 256]).flatten()
        v_hist = cv2.calcHist([hsv], [2], None, [64], [0, 256]).flatten()
        hist = np.concatenate([h_hist, s_hist, v_hist])  # 192-D

        # Gradient magnitude histogram (160 features từ 8×8 cell trên 128x128)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx**2 + gy**2)
        cells_x, cells_y = 8, 8
        step_x, step_y = mag.shape[1] // cells_x, mag.shape[0] // cells_y
        grad_feat = []
        for iy in range(cells_y):
            for ix in range(cells_x):
                cell = mag[iy*step_y:(iy+1)*step_y, ix*step_x:(ix+1)*step_x]
                grad_feat.append(float(cell.mean()))
        grad_feat = np.array(grad_feat, dtype=np.float32)  # 64-D

        # Concat → pad/truncate đến 512-D
        combined = np.concatenate([hist, grad_feat])  # 256-D
        if combined.size < 512:
            combined = np.pad(combined, (0, 512 - combined.size))
        else:
            combined = combined[:512]

        norm = float(np.linalg.norm(combined))
        if norm < 1e-6:
            return None
        return (combined / norm).tolist()

    except Exception as e:
        print(f"[CLIPEncoder] fallback_encode error: {e}", flush=True)
        return None


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Tính cosine similarity giữa 2 vector 512-D đã normalize."""
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    return float(np.dot(a / na, b / nb))
