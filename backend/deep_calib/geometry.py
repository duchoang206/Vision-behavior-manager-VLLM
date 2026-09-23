"""Bidirectional geometry for the DeepCalib unified spherical camera model."""

from __future__ import annotations

import numpy as np


def _profile_geometry(profile):
    try:
        values = tuple(float(profile[key]) for key in
                       ("image_width", "image_height", "focal_length_px", "distortion_xi"))
    except (KeyError, TypeError, ValueError):
        return None
    width, height, focal, distortion = values
    if not np.isfinite(values).all() or min(width, height, focal) <= 0 or not 0 <= distortion <= 1.2:
        return None
    return values


def profile_is_ready(profile):
    return _profile_geometry(profile or {}) is not None


def _geometry(profile):
    source = _profile_geometry(profile)
    if source is None:
        raise ValueError("Profile DeepCalib thiếu hoặc có intrinsic không hợp lệ.")
    width, height, focal, distortion = source
    try:
        output = tuple(float(profile.get(key, default)) for key, default in (
            ("rectified_width", width), ("rectified_height", height),
            ("rectified_focal_px", focal), ("rectified_cx", width / 2), ("rectified_cy", height / 2)))
    except (TypeError, ValueError) as error:
        raise ValueError("Hình học ảnh hiệu chỉnh không hợp lệ.") from error
    if not np.isfinite(output).all() or min(output[:3]) <= 0:
        raise ValueError("Hình học ảnh hiệu chỉnh không hợp lệ.")
    return source, output


def _points(points):
    source = np.asarray(points, dtype=np.float64)
    if source.ndim != 2 or source.shape[1:] != (2,) or not np.isfinite(source).all():
        raise ValueError("Điểm ảnh DeepCalib không hợp lệ.")
    return source


def undistort_normalized_points(points, profile):
    source = _points(points)
    (width, height, focal, distortion), (out_width, out_height, out_focal, center_x, center_y) = _geometry(profile)
    camera = (source * [width, height] - [width / 2, height / 2]) / focal
    radius_sq = np.sum(camera * camera, axis=1)
    radicand = 1 + (1 - distortion * distortion) * radius_sq
    if (radicand <= 0).any():
        raise ValueError("Điểm nằm ngoài miền chiếu spherical của camera.")
    alpha = (distortion + np.sqrt(radicand)) / (radius_sq + 1)
    depth = alpha - distortion
    if (depth <= 1e-8).any():
        raise ValueError("Điểm nằm trên hoặc phía sau chân trời của ảnh hiệu chỉnh.")
    rectified = (camera * (alpha / depth)[:, None] * out_focal + [center_x, center_y]) / [out_width, out_height]
    if not np.isfinite(rectified).all():
        raise ValueError("Không thể hiệu chỉnh tọa độ điểm ảnh.")
    if profile.get("geometry_version", 1) < 2:
        return np.clip(rectified, 0, 1)
    return rectified


def distort_normalized_points(points, profile):
    rectified = _points(points)
    (width, height, focal, distortion), (out_width, out_height, out_focal, center_x, center_y) = _geometry(profile)
    rays = (rectified * [out_width, out_height] - [center_x, center_y]) / out_focal
    denominator = 1 + distortion * np.sqrt(1 + np.sum(rays * rays, axis=1))
    return (rays * (focal / denominator)[:, None] + [width / 2, height / 2]) / [width, height]


def validate_visible_rectified_points(points, profile):
    rectified = _points(points)
    raw = distort_normalized_points(rectified, profile)
    if (rectified < 0).any() or (rectified > 1).any() or (raw < 0).any() or (raw > 1).any():
        raise ValueError("Điểm nằm trong viền trống hoặc ngoài ảnh. Hãy chấm trên mặt sàn nhìn thấy được.")
    if not np.allclose(undistort_normalized_points(raw, profile), rectified, atol=1e-6):
        raise ValueError("Điểm không thuộc miền chiếu nhìn thấy của camera.")
    return raw


def make_rectification_profile(profile):
    geometry = _profile_geometry(profile)
    if geometry is None:
        raise ValueError("Không có intrinsic DeepCalib để hiệu chỉnh ảnh.")
    width, height, focal, distortion = geometry
    axis = np.linspace(0, 1, 65)
    grid_x, grid_y = np.meshgrid(axis, axis)
    camera = (np.column_stack((grid_x.ravel(), grid_y.ravel())) * [width, height] - [width / 2, height / 2]) / focal
    radius_sq = np.sum(camera * camera, axis=1)
    radicand = 1 + (1 - distortion * distortion) * radius_sq
    alpha = (distortion + np.sqrt(np.maximum(0, radicand))) / (radius_sq + 1)
    depth = alpha - distortion
    valid = (radicand > 0) & (depth > 1e-5)
    extent = np.max(np.abs(camera[valid] * (alpha[valid] / depth[valid])[:, None]), axis=0)
    output_focal = max(focal / 3, min(focal, width / (2 * max(extent[0], 1e-6)), height / (2 * max(extent[1], 1e-6))))
    return dict(profile, geometry_version=2, rectified_width=int(width), rectified_height=int(height),
                rectified_focal_px=float(output_focal), rectified_cx=width / 2, rectified_cy=height / 2)


def rectify_normalized_point(point, profile):
    return undistort_normalized_points([point], profile)[0].tolist()


def rectify_image_cuda(image, profile):
    import torch
    from torch.nn import functional

    if not torch.cuda.is_available():
        raise RuntimeError("DeepCalib cần CUDA để hiệu chỉnh ảnh; không chuyển xử lý ảnh sang CPU.")
    (width, height, focal, distortion), (out_width, out_height, out_focal, center_x, center_y) = _geometry(profile)
    if image.shape[:2] != (int(height), int(width)):
        raise ValueError("Độ phân giải snapshot khác profile đã lưu. Hãy chạy lại DeepCalib và chấm lại điểm.")
    out_width, out_height = int(out_width), int(out_height)
    if out_width * out_height > 16_777_216:
        raise ValueError("Ảnh hiệu chỉnh vượt giới hạn 16 megapixel.")
    with torch.inference_mode():
        tensor = torch.as_tensor(image, device="cuda").permute(2, 0, 1).unsqueeze(0).float()
        horizontal = (torch.arange(out_width, device="cuda", dtype=torch.float32) - center_x) / out_focal
        output = np.empty((out_height, out_width, 3), dtype=np.uint8)
        for start in range(0, out_height, 256):
            end = min(out_height, start + 256)
            vertical = (torch.arange(start, end, device="cuda", dtype=torch.float32) - center_y) / out_focal
            ray_y, ray_x = torch.meshgrid(vertical, horizontal, indexing="ij")
            denominator = 1 + distortion * torch.sqrt(1 + ray_x.square() + ray_y.square())
            pixels_x = ray_x * focal / denominator + width / 2
            pixels_y = ray_y * focal / denominator + height / 2
            grid = torch.stack(((pixels_x + .5) * 2 / width - 1, (pixels_y + .5) * 2 / height - 1), dim=-1)
            sampled = functional.grid_sample(tensor, grid.unsqueeze(0), mode="bilinear", padding_mode="zeros", align_corners=False)
            output[start:end] = sampled[0].permute(1, 2, 0).round().clamp(0, 255).byte().cpu().numpy()
    return output
