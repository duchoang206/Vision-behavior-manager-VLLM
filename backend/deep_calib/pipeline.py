"""One-shot GPU previews with camera-bound geometry receipts."""

import base64
import copy
import hashlib
import threading
import time
import uuid
from collections import OrderedDict

import cv2

from deep_calib.adapter import _decode_image, estimate, REPOSITORY_URL, UPSTREAM_REVISION
from deep_calib.geometry import make_rectification_profile, rectify_image_cuda

_PREVIEWS = OrderedDict()
_LOCK = threading.Lock()
_GPU_LOCK = threading.Lock()


def preview_profile(camera_id, preview_id):
    with _LOCK:
        item = _PREVIEWS.get(preview_id)
        if not item or item["camera_id"] != camera_id or time.monotonic() - item["created"] > 86400:
            raise ValueError("Ảnh hiệu chỉnh đã hết hạn hoặc không thuộc camera này. Hãy mở lại preview trước khi lưu.")
        _PREVIEWS.move_to_end(preview_id)
        return copy.deepcopy(item["profile"])


def preview(camera_id, encoded, profile=None):
    if not _GPU_LOCK.acquire(blocking=False):
        raise RuntimeError("DeepCalib đang xử lý một ảnh khác. Hãy thử lại sau vài giây.")
    try:
        image = _decode_image(encoded)
        result = None
        if profile is None:
            result = estimate(encoded)
            if not result.get("available"):
                raise RuntimeError(result.get("reason", "DeepCalib không suy luận được intrinsic."))
            profile = make_rectification_profile({key: result[key] for key in (
                "image_width", "image_height", "focal_length_px", "distortion_xi",
                "focal_confidence", "distortion_confidence", "preprocessing")})
            profile.update(repository_url=REPOSITORY_URL, revision=UPSTREAM_REVISION,
                           estimated_at=time.time(), model="SingleNet weights_06_5.61.h5")
        profile = copy.deepcopy(profile)
        rectified = rectify_image_cuda(image, profile)
        success, buffer = cv2.imencode(".jpg", rectified, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not success:
            raise RuntimeError("Không mã hóa được ảnh DeepCalib.")
        receipt = uuid.uuid4().hex
        profile["snapshot_sha256"] = hashlib.sha256(base64.b64decode(encoded.split(",", 1)[-1])).hexdigest()
        with _LOCK:
            _PREVIEWS[receipt] = dict(camera_id=camera_id, created=time.monotonic(), profile=profile)
            while len(_PREVIEWS) > 256:
                _PREVIEWS.popitem(last=False)
        return dict(camera_id=camera_id, preview_id=receipt, profile=profile,
                    rectified_image="data:image/jpeg;base64," + base64.b64encode(buffer).decode("ascii"),
                    deepcalib=result, device="cuda", reused_profile=result is None)
    finally:
        _GPU_LOCK.release()
