#!/usr/bin/env python3
"""
Session 1: Export Re-ID Feature Extractor (ResNet Backbone) to 512-dim Embedding ONNX/Engine.
Input shape: [B, 3, 256, 128]
Output shape: [B, 512] (L2-normalized feature embedding)
"""

import os
import sys
import shutil
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ReIDFeatureExtractor(nn.Module):
    """
    Standard 512-dim Re-ID Feature Extractor for DeepStream NvDCF Tracker.
    Uses ResNet backbone with Linear embedding projection head and L2 normalization.
    """
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
        # x: [B, 3, 256, 128]
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

def export_reid():
    base_dir = Path(__file__).resolve().parent.parent
    models_dir = base_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    onnx_path = models_dir / "reid_512.onnx"
    
    print("=" * 60)
    print("🧠 [Step 1] Building Re-ID Feature Extractor & Exporting to ONNX...")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = ReIDFeatureExtractor(embedding_dim=512)
    model.eval().to(device)
    
    dummy_input = torch.randn(1, 3, 256, 128, device=device)
    
    with torch.no_grad():
        out = model(dummy_input)
        print(f"Forward check output shape: {out.shape} (Expected: [1, 512])")
        print(f"Output norm (L2): {torch.norm(out, p=2, dim=1).item():.4f} (Expected: ~1.0)")
        
    print(f"📦 Exporting to ONNX: {onnx_path} with dynamic batch [1, 4, 16]...")
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        },
        dynamo=False
    )
    print(f"✅ Re-ID ONNX exported successfully: {onnx_path} ({onnx_path.stat().st_size / (1024*1024):.1f} MB)")

if __name__ == "__main__":
    export_reid()
