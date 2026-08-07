#!/usr/bin/env python3
"""
Session 1: Export YOLOv8x Object Detector to TensorRT Engine (FP16 with dynamic batching).
Target classes: 'robot', 'rack' (or general warehouse object classes).
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def export_yolo():
    base_dir = Path(__file__).resolve().parent.parent
    models_dir = base_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    onnx_path = models_dir / "yolov8x.onnx"
    engine_path = models_dir / "yolov8x_fp16.engine"
    
    print("=" * 60)
    print("🚀 [Step 1] Loading YOLOv8x Model and Exporting to ONNX...")
    print("=" * 60)
    
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ Ultralytics not installed. Please install with `pip install ultralytics`")
        sys.exit(1)
        
    # Load YOLOv8x pretrained
    model = YOLO("yolov8x.pt")
    
    # Export to ONNX with dynamic batching and 640x640 resolution
    print(f"📦 Exporting ONNX to {onnx_path} with dynamic batching [1, 4, 8]...")
    exported_onnx = model.export(
        format="onnx",
        imgsz=640,
        dynamic=True,
        simplify=True,
        opset=12
    )
    
    # Ensure exported file is placed at target location
    if Path(exported_onnx).resolve() != onnx_path.resolve():
        shutil.copyfile(exported_onnx, onnx_path)
    print(f"✅ ONNX exported successfully: {onnx_path} ({onnx_path.stat().st_size / (1024*1024):.1f} MB)")
    
    print("\n" + "=" * 60)
    print("⚡ [Step 2] Building TensorRT FP16 Engine using trtexec...")
    print("=" * 60)
    
    trtexec_bin = shutil.which("trtexec") or "/usr/bin/trtexec"
    if not os.path.exists(trtexec_bin):
        print(f"❌ trtexec binary not found at {trtexec_bin}")
        sys.exit(1)
        
    cmd = [
        trtexec_bin,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--fp16",
        "--minShapes=images:1x3x640x640",
        "--optShapes=images:4x3x640x640",
        "--maxShapes=images:8x3x640x640",
        "--workspace=4096",
        "--buildOnly"
    ]
    
    print(f"Executing: {' '.join(cmd)}")
    ret = subprocess.run(cmd)
    
    if ret.returncode == 0 and engine_path.exists():
        print(f"\n🎉 Successfully built YOLOv8x TensorRT Engine: {engine_path} ({engine_path.stat().st_size / (1024*1024):.1f} MB)")
    else:
        print(f"\n❌ Failed to build TensorRT engine (Exit Code: {ret.returncode})")
        sys.exit(ret.returncode)

if __name__ == "__main__":
    export_yolo()
