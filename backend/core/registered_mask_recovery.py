"""GPU feature proposals for lost targets; proposals never assign an identity."""

import math

import torch
import torch.nn.functional as functional


@torch.inference_mode()
def appearance_candidates(identity, label, predictor, image_shape, box_sizes, limit=6):
    with identity.lock:
        gallery = identity.gallery.get(label)
    if gallery is None or not box_sizes:
        return []
    feature_height, feature_width = predictor.feat_sizes[-1]
    features = predictor.vision_feats[-1].permute(1, 2, 0)[0].float()
    channels = features.shape[0]
    references = functional.normalize(gallery[:, :channels].float(), dim=1)
    similarities = (references @ functional.normalize(features, dim=0)).max(dim=0).values
    scores = similarities.reshape(1, 1, feature_height, feature_width)
    image_height, image_width = image_shape[:2]
    input_height, input_width = predictor.imgsz
    scale = min(input_height / image_height, input_width / image_width)
    valid_height = max(1, min(feature_height, math.floor(round(image_height * scale) * feature_height / input_height)))
    valid_width = max(1, min(feature_width, math.floor(round(image_width * scale) * feature_width / input_width)))
    scores = scores[:, :, :valid_height, :valid_width]
    proposals = []
    for box_width, box_height in box_sizes[-8:]:
        kernel_width = max(1, min(valid_width, round(box_width * valid_width)))
        kernel_height = max(1, min(valid_height, round(box_height * valid_height)))
        pooled = functional.avg_pool2d(scores, (kernel_height, kernel_width), stride=1)[0, 0]
        values, indices = pooled.flatten().topk(min(limit * 4, pooled.numel()))
        for score, index in zip(values.cpu().tolist(), indices.cpu().tolist()):
            if score < 0.55:
                continue
            row, column = divmod(index, pooled.shape[1])
            proposals.append((score, [column / valid_width, row / valid_height,
                                      kernel_width / valid_width, kernel_height / valid_height]))
    candidates = []
    for score, bbox in sorted(proposals, key=lambda item: item[0], reverse=True):
        center_x, center_y = bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2
        if any(abs(center_x - (other[0] + other[2] / 2)) < min(bbox[2], other[2]) * 0.6
               and abs(center_y - (other[1] + other[3] / 2)) < min(bbox[3], other[3]) * 0.6
               for other in candidates):
            continue
        candidates.append(bbox)
        if len(candidates) >= limit:
            break
    return candidates
