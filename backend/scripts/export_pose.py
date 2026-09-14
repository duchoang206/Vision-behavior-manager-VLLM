import argparse
import os
import shutil
import subprocess
from pathlib import Path


def validate_pose_onnx(path, batch):
    import onnx

    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    expected = {
        "images": [batch, 3, 640, 640],
        "output0": [batch, 56, 8400],
    }
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise ValueError("DeepStream pose requires exactly one input and one output")
    for value in [model.graph.input[0], model.graph.output[0]]:
        dimensions = [dimension.dim_value for dimension in value.type.tensor_type.shape.dim]
        if value.type.tensor_type.elem_type != onnx.TensorProto.FLOAT or dimensions != expected.get(value.name):
            raise ValueError(f"Unexpected pose tensor {value.name}: {dimensions}; expected FP32 {expected}")


def main():
    parser = argparse.ArgumentParser(description="Export COCO-17 pose for the existing DeepStream parser")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--build-engine", action="store_true")
    args = parser.parse_args()
    if not args.weights.is_file() or args.batch < 1:
        parser.error("Provide an existing pose checkpoint and a positive batch size")

    os.environ["YOLO_AUTOINSTALL"] = "false"
    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    head = model.model.model[-1]
    if model.task != "pose" or head.nc != 1 or list(head.kpt_shape) != [17, 3]:
        raise ValueError("Expected a one-class COCO-17 pose model, not a detection model")
    onnx_path = Path(model.export(
        format="onnx", imgsz=640, batch=args.batch, dynamic=False,
        simplify=False, opset=17, device="cpu", nms=False,
    ))
    validate_pose_onnx(onnx_path, args.batch)
    print(f"Validated FP32 pose tensors: {onnx_path}", flush=True)
    if args.build_engine:
        trtexec = shutil.which("trtexec")
        if trtexec is None:
            raise RuntimeError("Run inside the DeepStream container with trtexec installed")
        engine_path = onnx_path.with_name(f"{onnx_path.stem}_b{args.batch}_gpu0_fp16.engine")
        subprocess.run([
            trtexec, f"--onnx={onnx_path}", f"--saveEngine={engine_path}",
            "--fp16", "--inputIOFormats=fp32:chw", "--outputIOFormats=fp32:chw",
            "--memPoolSize=workspace:2048", "--skipInference",
        ], check=True)
        print(f"Built {engine_path}; validate inference before changing the running configuration", flush=True)


if __name__ == "__main__":
    main()
