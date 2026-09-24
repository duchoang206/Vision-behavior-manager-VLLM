"""Coordinate transforms for metadata emitted by DeepStream probes.

The dashboard contract is normalized camera coordinates only.  nvinfer returns
``NvDsObjectMeta.rect_params`` in stream coordinates, while raw tensor parsers
may return network coordinates that include symmetric letterbox padding.  Both
paths terminate in these helpers so pixel-space values cannot escape in JSON.
"""

from __future__ import annotations

import math


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _letterbox(stream_width: float, stream_height: float, network_width: float, network_height: float):
    if min(stream_width, stream_height, network_width, network_height) <= 0:
        raise ValueError("Kích thước stream/model phải dương.")
    scale = min(network_width / stream_width, network_height / stream_height)
    valid_width, valid_height = stream_width * scale, stream_height * scale
    return valid_width, valid_height, (network_width - valid_width) / 2.0, (network_height - valid_height) / 2.0


def unletterbox_point(x, y, stream_width, stream_height, network_width, network_height):
    """Map one network-pixel point to normalized, padding-free camera space."""
    valid_width, valid_height, pad_x, pad_y = _letterbox(stream_width, stream_height, network_width, network_height)
    return _clamp((float(x) - pad_x) / valid_width), _clamp((float(y) - pad_y) / valid_height)


def unletterbox_bbox(x, y, width, height, stream_width, stream_height, network_width, network_height):
    """Map a network-pixel bbox to `[x, y, w, h]` in `[0, 1]`."""
    left, top = unletterbox_point(x, y, stream_width, stream_height, network_width, network_height)
    right, bottom = unletterbox_point(float(x) + float(width), float(y) + float(height), stream_width, stream_height, network_width, network_height)
    return [left, top, max(0.0, right - left), max(0.0, bottom - top)]


def unletterbox_polygon(points, stream_width, stream_height, network_width, network_height):
    """Map all vertices of a network-pixel contour to normalized camera space."""
    return [list(unletterbox_point(point[0], point[1], stream_width, stream_height, network_width, network_height))
            for point in points if isinstance(point, (list, tuple)) and len(point) >= 2]


def normalize_stream_bbox(x, y, width, height, stream_width, stream_height):
    """Normalize canonical stream-space metadata, clipping it to the camera."""
    if not all(math.isfinite(float(value)) for value in (x, y, width, height)):
        raise ValueError("Bounding box không hữu hạn.")
    left, top = _clamp(float(x) / float(stream_width)), _clamp(float(y) / float(stream_height))
    right = _clamp((float(x) + float(width)) / float(stream_width))
    bottom = _clamp((float(y) + float(height)) / float(stream_height))
    return [left, top, max(0.0, right - left), max(0.0, bottom - top)]


def clamp_normalized_polygon(points):
    return [[_clamp(point[0]), _clamp(point[1])] for point in points
            if isinstance(point, (list, tuple)) and len(point) >= 2]
