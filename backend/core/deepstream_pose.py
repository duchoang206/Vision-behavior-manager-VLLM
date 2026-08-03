import ctypes

import cv2
import numpy as np


def decode_pose_tensor(values, frame_width, frame_height, network_width=640, network_height=640, threshold=0.35):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 3 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 2 or values.shape[0] != 56:
        raise ValueError(f"Expected YOLOv8-pose [56, anchors], got {values.shape}")
    candidates = values[:, np.isfinite(values).all(axis=0) & (values[4] >= threshold)].T
    if not len(candidates):
        return []
    scale = min(network_width / frame_width, network_height / frame_height)
    padding_x = (network_width - round(frame_width * scale)) / 2
    padding_y = (network_height - round(frame_height * scale)) / 2
    boxes = candidates[:, :4].copy()
    boxes[:, :2] -= boxes[:, 2:] / 2
    keep = cv2.dnn.NMSBoxes(boxes.tolist(), candidates[:, 4].tolist(), threshold, 0.45)
    output = []
    for index in np.asarray(keep).reshape(-1):
        candidate = candidates[index]
        left, top, width, height = boxes[index]
        normalized_left = np.clip((left - padding_x) / scale / frame_width, 0, 1)
        normalized_top = np.clip((top - padding_y) / scale / frame_height, 0, 1)
        normalized_right = np.clip((left + width - padding_x) / scale / frame_width, 0, 1)
        normalized_bottom = np.clip((top + height - padding_y) / scale / frame_height, 0, 1)
        if normalized_right <= normalized_left or normalized_bottom <= normalized_top:
            continue
        points = candidate[5:].reshape(17, 3).copy()
        points[:, 0] = (points[:, 0] - padding_x) / scale / frame_width
        points[:, 1] = (points[:, 1] - padding_y) / scale / frame_height
        outside = (points[:, :2] < 0).any(axis=1) | (points[:, :2] > 1).any(axis=1)
        points[outside, 2] = 0
        points[:, :2] = np.clip(points[:, :2], 0, 1)
        output.append({
            "bbox": [float(normalized_left), float(normalized_top), float(normalized_right - normalized_left), float(normalized_bottom - normalized_top)],
            "confidence": float(candidate[4]),
            "keypoints": points.round(5).tolist(),
        })
    return output


def frame_poses(frame_meta, frame_width, frame_height):
    import pyds

    metadata = frame_meta.frame_user_meta_list
    while metadata is not None:
        user_meta = pyds.NvDsUserMeta.cast(metadata.data)
        if user_meta.base_meta.meta_type == pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
            tensor = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
            if tensor.unique_id == 1:
                layer = pyds.get_nvds_LayerInfo(tensor, 0)
                dimensions = tuple(layer.inferDims.d[index] for index in range(layer.inferDims.numDims))
                if int(layer.dataType) != 0:
                    raise ValueError("Pose output must be FP32")
                pointer = ctypes.cast(pyds.get_ptr(layer.buffer), ctypes.POINTER(ctypes.c_float))
                values = np.ctypeslib.as_array(pointer, shape=(int(np.prod(dimensions)),)).reshape(dimensions)
                return decode_pose_tensor(values, frame_width, frame_height)
        try:
            metadata = metadata.next
        except StopIteration:
            break
    return []


def attach_poses(detections, poses):
    pairs = []
    for detection_index, detection in enumerate(detections):
        left, top, width, height = detection["bbox"]
        for pose_index, pose in enumerate(poses):
            pose_left, pose_top, pose_width, pose_height = pose["bbox"]
            overlap = max(0, min(left + width, pose_left + pose_width) - max(left, pose_left)) * max(0, min(top + height, pose_top + pose_height) - max(top, pose_top))
            union = width * height + pose_width * pose_height - overlap
            score = overlap / union if union > 0 else 0
            if score >= 0.2:
                pairs.append((score, detection_index, pose_index))
        detection["keypoints"] = []
        detection["tracking_state"] = "predicted"
    assigned_detections, assigned_poses = set(), set()
    for score, detection_index, pose_index in sorted(pairs, reverse=True):
        if detection_index in assigned_detections or pose_index in assigned_poses:
            continue
        detections[detection_index].update(
            keypoints=poses[pose_index]["keypoints"],
            confidence=poses[pose_index]["confidence"],
            tracking_state="tracked",
            pose_source="deepstream_yolov8x_pose_tensorrt",
        )
        assigned_detections.add(detection_index)
        assigned_poses.add(pose_index)
