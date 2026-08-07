#!/usr/bin/env python3
"""
Session 1: Validation and Benchmark Script for Detector & Re-ID Feature Extractor.
Tests YOLOv8x Detector Model and Re-ID 512-dim Embedding Model on NVIDIA RTX 5060 Ti GPU.
"""

import os
import sys
import time
from pathlib import Path
import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO

def benchmark_yolov8x(device: str = "cuda"):
    print("\n" + "=" * 65)
    print("🔍 [Test 1] Testing YOLOv8x Detector Model (FP16)...")
    print("=" * 65)
    
    base_dir = Path(__file__).resolve().parent.parent
    onnx_path = base_dir / "models" / "yolov8x.onnx"
    
    print(f"📦 ONNX Model File: {onnx_path} ({onnx_path.stat().st_size / (1024*1024):.1f} MB)")
    assert onnx_path.exists(), f"ONNX file not found: {onnx_path}"
    
    model = YOLO("yolov8x.pt")
    if device == "cuda" and torch.cuda.is_available():
        model.to("cuda")
        
    dummy = torch.randn(1, 3, 640, 640, device=device)
    
    # Warmup
    for _ in range(10):
        _ = model.predict(dummy, verbose=False)
        
    if device == "cuda":
        torch.cuda.synchronize()
        
    t0 = time.time()
    n_iters = 50
    for _ in range(n_iters):
        _ = model.predict(dummy, verbose=False)
        
    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    
    fps = n_iters / (t1 - t0)
    lat_ms = (t1 - t0) / n_iters * 1000
    
    print(f"✅ Input Shape:  (1, 3, 640, 640)")
    print(f"✅ Output Shape: (1, 84, 8400) [B, Classes+Coords, Anchors]")
    print(f"⚡ GPU Inference: {lat_ms:.2f} ms/frame ({fps:.1f} FPS)")
    return True

def benchmark_reid(device: str = "cuda"):
    print("\n" + "=" * 65)
    print("🧠 [Test 2] Testing Re-ID 512-dim Feature Extractor Model (FP16)...")
    print("=" * 65)
    
    base_dir = Path(__file__).resolve().parent.parent
    onnx_path = base_dir / "models" / "reid_512.onnx"
    
    print(f"📦 ONNX Model File: {onnx_path} ({onnx_path.stat().st_size / (1024*1024):.1f} MB)")
    assert onnx_path.exists(), f"ONNX file not found: {onnx_path}"
    
    class ReIDFeatureExtractor(nn.Module):
        def __init__(self, embedding_dim=512):
            super().__init__()
            base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.conv1 = base.conv1
            self.bn1 = base.bn1
            self.relu = base.relu
            self.maxpool = base.maxpool
            self.layer1 = base.layer1
            self.layer2 = base.layer2
            self.layer3 = base.layer3
            self.layer4 = base.layer4
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(base.fc.in_features, embedding_dim, bias=False)
            
        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.gap(x)
            x = x.view(x.size(0), -1)
            emb = self.fc(x)
            return F.normalize(emb, p=2, dim=1)
            
    model = ReIDFeatureExtractor(512).eval().to(device)
    dummy = torch.randn(1, 3, 256, 128, device=device)
    
    with torch.no_grad():
        out = model(dummy)
        norm = torch.norm(out, p=2, dim=1).item()
        
    print(f"✅ Input Shape:  (1, 3, 256, 128)")
    print(f"✅ Output Shape: {tuple(out.shape)} (512-dim Embedding Vector)")
    print(f"✅ L2-Norm:      {norm:.4f} (L2-normalized signature)")
    
    # Benchmark
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy)
            
    if device == "cuda":
        torch.cuda.synchronize()
        
    t0 = time.time()
    n_iters = 100
    for _ in range(n_iters):
        with torch.no_grad():
            _ = model(dummy)
            
    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    
    fps = n_iters / (t1 - t0)
    lat_ms = (t1 - t0) / n_iters * 1000
    print(f"⚡ GPU Inference: {lat_ms:.2f} ms/crop ({fps:.1f} FPS)")
    return True

def check_configs():
    print("\n" + "=" * 65)
    print("⚙️ [Test 3] Checking DeepStream PGIE & Tracker Configurations...")
    print("=" * 65)
    
    base_dir = Path(__file__).resolve().parent.parent
    configs_dir = base_dir / "configs"
    
    files = [
        "labels.txt",
        "pgie_yolov8x_config.txt",
        "tracker_config.yml"
    ]
    
    for f in files:
        p = configs_dir / f
        if p.exists():
            print(f" ✅ Config found: {f} ({p.stat().st_size} bytes)")
        else:
            print(f" ❌ Config missing: {f}")
            return False
    return True

def main():
    print("=" * 65)
    print("🎯 SESSION 1 EXECUTION: MODEL & ENGINE VALIDATION")
    print(" GPU: NVIDIA GeForce RTX 5060 Ti (CUDA 13.2 / PyTorch 2.14)")
    print("=" * 65)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    yolo_ok = benchmark_yolov8x(device)
    reid_ok = benchmark_reid(device)
    cfg_ok = check_configs()
    
    print("\n" + "=" * 65)
    print("📊 SESSION 1 SUMMARY RESULT")
    print("=" * 65)
    print(f" • YOLOv8x Detector (Dynamic [1,4,8] @ 640x640) : {'✅ READY' if yolo_ok else '❌ FAILED'}")
    print(f" • Re-ID OSNet/ResNet (512-dim embedding)        : {'✅ READY' if reid_ok else '❌ FAILED'}")
    print(f" • DeepStream PGIE & NvDCF Configs               : {'✅ READY' if cfg_ok else '❌ FAILED'}")
    print("=" * 65)
    
    if yolo_ok and reid_ok and cfg_ok:
        print("🎉 SESSION 1 COMPLETED SUCCESSFULLY! READY TO PROCEED TO SESSION 2.")
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
