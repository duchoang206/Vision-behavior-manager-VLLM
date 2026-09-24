import ast
import hashlib
import json
import re


def parse_labels(value):
    if isinstance(value, str):
        if len(value) > 32768:
            raise ValueError("Danh sách nhãn quá lớn.")
        try:
            value = json.loads(value)
        except ValueError:
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError, RecursionError):
                raise ValueError("Metadata names phải là danh sách hoặc từ điển class ID.")
    if isinstance(value, dict):
        try:
            indexed = {int(key): name for key, name in value.items()}
            if len(indexed) != len(value) or set(indexed) != set(range(len(indexed))):
                raise ValueError()
            value = [indexed[index] for index in range(len(indexed))]
        except (TypeError, ValueError):
            raise ValueError("Class ID phải liên tục từ 0.")
    if not isinstance(value, list) or not 1 <= len(value) <= 100:
        raise ValueError("Cần 1–100 nhãn đúng thứ tự class ID.")
    if any(not isinstance(name, str) or not name.strip() or len(name) > 120
           or any(ord(char) < 32 for char in name) or ';' in name for name in value):
        raise ValueError("Tên lớp không hợp lệ.")
    names = [name.strip() for name in value]
    if len({name.casefold() for name in names}) != len(names):
        raise ValueError("Tên lớp bị trùng.")
    return names


def inspect_onnx(path, labels=None):
    import onnx
    from onnx.external_data_helper import _get_all_tensors

    model = onnx.load(str(path), load_external_data=False)
    if any(tensor.data_location == onnx.TensorProto.EXTERNAL or tensor.external_data
           for tensor in _get_all_tensors(model)):
        raise ValueError("Cần ONNX self-contained; không nhận file external data hoặc đường dẫn bên ngoài.")
    onnx.checker.check_model(model)
    metadata = {item.key: item.value for item in model.metadata_props}
    task = str(metadata.get("task", "detect")).strip().lower()
    task_aliases = {"seg": "segment", "segmentation": "segment", "classification": "classify"}
    task = task_aliases.get(task, task)
    if task not in {"detect", "segment", "pose", "obb", "classify"}:
        raise ValueError("task ONNX phải là detect, segment, pose, obb hoặc classify.")
    names = parse_labels(labels or metadata.get("names", []))
    inputs = [item for item in model.graph.input if item.name not in {item.name for item in model.graph.initializer}]
    outputs = list(model.graph.output)
    expected_outputs = 2 if task == "segment" else 1
    if len(inputs) != 1 or len(outputs) != expected_outputs:
        suffix = "hai output (detection + prototype mask)" if task == "segment" else "một output YOLO thô"
        raise ValueError(f"Cần một input ảnh và {suffix} (export nms=False).")
    for tensor in inputs + outputs:
        if tensor.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
            raise ValueError("Export ONNX half=False; backend sẽ tối ưu engine FP16.")
        if not re.fullmatch(r"[A-Za-z0-9_./-]{1,120}", tensor.name):
            raise ValueError("Tên tensor không được chứa ký tự cấu hình đặc biệt.")
    shape = [dimension.dim_value for dimension in inputs[0].type.tensor_type.shape.dim]
    output = [dimension.dim_value for dimension in outputs[0].type.tensor_type.shape.dim]
    if len(shape) != 4 or shape[:2] != [1, 3] or any(size < 128 or size > 1920 or size % 32 for size in shape[2:]):
        raise ValueError("Export batch=1, dynamic=False, RGB NCHW; chiều ảnh 128–1920, chia hết cho 32.")
    if task == "detect":
        if len(output) != 3 or output[0] != 1 or output[1] != 4 + len(names) or not 256 < output[2] <= 100000:
            raise ValueError("Output detect phải là [1, 4 + số lớp, anchors] của YOLOv8/YOLO11, nms=False.")
    elif task == "segment":
        prototype = [dimension.dim_value for dimension in outputs[1].type.tensor_type.shape.dim]
        if (len(output) != 3 or output[0] != 1 or output[1] <= 4 + len(names) or not 256 < output[2] <= 100000
                or len(prototype) != 4 or prototype[0] != 1 or prototype[1] < 8 or min(prototype[2:]) < 8):
            raise ValueError("Output segmentation phải gồm detection [1, 4 + classes + mask_coeffs, anchors] và prototype [1, mask_coeffs, H, W].")
    elif task == "pose":
        if len(names) != 1 or len(output) != 3 or output[0] != 1 or output[1] != 56 or not 256 < output[2] <= 100000:
            raise ValueError("Pose cần đúng một nhãn người và output [1, 56, anchors] (17 COCO keypoints), nms=False.")
    elif task == "obb":
        if len(output) != 3 or output[0] != 1 or output[1] != 5 + len(names) or not 256 < output[2] <= 100000:
            raise ValueError("Output OBB phải là [1, 5 + số lớp, anchors], nms=False.")
    elif task == "classify":
        if len(output) != 2 or output[0] != 1 or output[1] != len(names):
            raise ValueError("Output classifier phải là [1, số lớp].")
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"labels": names, "input": inputs[0].name, "output": outputs[0].name,
            "outputs": [item.name for item in outputs], "shape": shape, "output_shape": output,
            "sha256": digest.hexdigest(), "task": task, "model_type": task,
            "contract": f"yolo_raw_{task}_v8_v11", "precision": "fp16",
            "labels_source": "manual" if labels else "onnx_metadata"}


def infer_config(directory, metadata, parser):
    task = metadata.get("model_type") or metadata.get("task", "detect")
    task = {"seg": "segment", "segmentation": "segment"}.get(str(task).lower(), str(task).lower())
    if task == "segment":
        return f"""[property]
gie-unique-id=1
gpu-id=0
net-scale-factor=0.00392156862745098
model-color-format=0
model-engine-file={directory}/detector.engine
labelfile-path={directory}/labels.txt
batch-size=1
infer-dims=3;{metadata['shape'][2]};{metadata['shape'][3]}
num-detected-classes={len(metadata['labels'])}
network-mode=2
network-type=3
process-mode=1
interval=0
cluster-mode=4
maintain-aspect-ratio=1
symmetric-padding=1
output-tensor-meta=0
parse-bbox-instance-mask-func-name=NvDsInferParseYoloSeg
custom-lib-path={parser}
output-instance-mask=1
segmentation-threshold=0.5

[class-attrs-all]
pre-cluster-threshold=0.25
"""
    if task == "pose":
        return f"""[property]
gie-unique-id=1
gpu-id=0
net-scale-factor=0.00392156862745098
model-color-format=0
model-engine-file={directory}/detector.engine
labelfile-path={directory}/labels.txt
batch-size=1
infer-dims=3;{metadata['shape'][2]};{metadata['shape'][3]}
num-detected-classes=1
network-mode=2
network-type=0
process-mode=1
interval=0
cluster-mode=2
maintain-aspect-ratio=1
symmetric-padding=1
output-tensor-meta=1
output-blob-names={metadata['output']}
custom-lib-path={parser}
parse-bbox-func-name=NvDsInferParseYoloV8Pose

[class-attrs-all]
pre-cluster-threshold=0.25
nms-iou-threshold=0.45
topk=200
"""
    if task != "detect":
        raise ValueError(f"Chưa có DeepStream parser production cho model loại '{task}'.")
    return f"""[property]
gie-unique-id=1
gpu-id=0
net-scale-factor=0.00392156862745098
model-color-format=0
model-engine-file={directory}/detector.engine
labelfile-path={directory}/labels.txt
batch-size=1
infer-dims=3;{metadata['shape'][2]};{metadata['shape'][3]}
num-detected-classes={len(metadata['labels'])}
network-mode=2
network-type=0
process-mode=1
interval=0
cluster-mode=2
maintain-aspect-ratio=1
symmetric-padding=1
output-tensor-meta=0
output-blob-names={metadata['output']}
custom-lib-path={parser}
parse-bbox-func-name=NvDsInferParseYolo

[class-attrs-all]
pre-cluster-threshold=0.25
nms-iou-threshold=0.45
topk=200
"""


if __name__ == "__main__":
    import sys
    from pathlib import Path

    request = json.loads(Path(sys.argv[1]).read_text())
    result = inspect_onnx(request["path"], request.get("labels"))
    Path(sys.argv[2]).write_text(json.dumps(result, ensure_ascii=False))
