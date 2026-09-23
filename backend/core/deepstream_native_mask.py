"""Convert NvDsObjectMeta.mask_params to the dashboard mask contract."""

from __future__ import annotations


def polygon_contract_iou(binary, polygons):
    """Measure raster IoU after polygon serialization, before Label matching."""
    import cv2
    import numpy as np

    source = np.asarray(binary, dtype=np.uint8)
    height, width = source.shape[:2]
    reconstructed = np.zeros((height, width), dtype=np.uint8)
    contours = [
        np.rint(np.asarray(ring, dtype=np.float32) * [width - 1, height - 1]).astype(np.int32)
        for ring in polygons
    ]
    if contours:
        cv2.fillPoly(reconstructed, contours, 1)
    source = source > 0
    reconstructed = reconstructed > 0
    union = np.logical_or(source, reconstructed).sum()
    return float(np.logical_and(source, reconstructed).sum() / union) if union else 1.0

def mask_from_object_meta(obj_meta, pyds):
    """Return a bbox-relative polygon mask or ``None``.

    DeepStream owns the SAM2 inference and stores the result in
    ``NvDsObjectMeta.mask_params``.  Only the compact polygon conversion is
    performed here so the existing WebSocket contract remains unchanged.
    """

    params = getattr(obj_meta, "mask_params", None)
    if params is None or int(getattr(params, "width", 0)) <= 0 or int(getattr(params, "height", 0)) <= 0:
        return None
    try:
        import numpy as np
        from core.registered_target_mask import mask_payload

        bitmap = np.asarray(params.get_mask_array())
        width, height = int(params.width), int(params.height)
        if bitmap.size < width * height:
            return None
        bitmap = bitmap.reshape(height, width)
        threshold = float(getattr(params, "threshold", 0.5))
        binary = (bitmap >= threshold).astype(np.uint8)
        if not binary.any():
            return None
        result = mask_payload(binary, source="deepstream_masktracker", confidence=1.0)
        if not result.get("polygons"):
            return None
        result["contract_iou"] = round(polygon_contract_iou(binary, result["polygons"]), 6)
        return result
    except Exception:
        return None
