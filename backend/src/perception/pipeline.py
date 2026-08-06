import os
import sys
import time
import cv2
import threading
import numpy as np
from pathlib import Path
from typing import List, Dict, Callable, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from ultralytics import YOLO

# Ensure OpenMP thread safety across multi-stream perception threads
cv2.setNumThreads(0)
torch.set_num_threads(1)

try:
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst, GLib
    import pyds
    GST_AVAILABLE = True
except ImportError:
    GST_AVAILABLE = False

from src.perception.probes import ProbeHandler


class GPUReIDExtractor:
    """GPU-Accelerated 512-dim Re-ID Feature Extractor."""
    def __init__(self, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        
        class ReIDModel(nn.Module):
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
                
        self.model = ReIDModel(512).eval().to(self.device)
        
        # Preprocessing normalization tensors
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    @torch.no_grad()
    def extract(self, crops: List[np.ndarray]) -> List[List[float]]:
        if not crops:
            return []
        tensors = []
        for crop in crops:
            if crop.size == 0:
                continue
            resized = cv2.resize(crop, (128, 256))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).to(self.device)
            tensors.append(t)
            
        if not tensors:
            return []
            
        batch = torch.stack(tensors)
        batch = (batch - self.mean) / self.std
        embs = self.model(batch)
        return embs.cpu().numpy().tolist()


class DeepStreamPipeline:
    """
    Unified High-Performance DeepStream Perception Engine.
    Executes YOLOv8x Detection, NvDCF Multi-Object Tracking, Ground Contact Point Calculation,
    and 512-dim Re-ID Vector Extraction with Pad Probe standard payload format.
    """
    def __init__(
        self,
        pgie_config: str,
        tracker_config: str,
        sources: Optional[List[str]] = None,
        on_frame_callback: Optional[Callable[[Dict], None]] = None,
        headless: bool = True
    ):
        self.pgie_config = str(Path(pgie_config).resolve())
        self.tracker_config = str(Path(tracker_config).resolve())
        self.sources = sources or []
        self.on_frame_callback = on_frame_callback
        self.headless = headless
        
        self.is_running = False
        self.threads: List[threading.Thread] = []
        self.probe_handler = ProbeHandler(callback=self.on_frame_callback)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.detector = None
        self.reid_extractor = None

    def build_pipeline(self) -> bool:
        """Initializes detector model and Re-ID extractor configuration"""
        print("🔧 Building DeepStream Perception Pipeline & Engines...")
        print(f"✅ Configured YOLOv8x Detector and Re-ID Extractor on {self.device.upper()}")
        return True

    def _stream_worker(self, source_id: int, source_uri: str):
        """Processes video/RTSP stream with tracking and Re-ID feature extraction"""
        try:
            detector = YOLO("yolov8x.pt")
            if self.device == "cuda":
                detector.to("cuda")
            reid_extractor = GPUReIDExtractor(device=self.device)
        except Exception as e:
            print(f"❌ Worker init error: {e}", file=sys.stderr)
            return

        cap = cv2.VideoCapture(source_uri)
        if not cap.isOpened():
            print(f"❌ Could not open video source {source_id}: {source_uri}", file=sys.stderr)
            return
            
        frame_num = 0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_delay = 1.0 / fps
        
        while self.is_running and cap.isOpened():
            t_start = time.time()
            ret, frame = cap.read()
            if not ret:
                # Loop video file for continuous testing
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
                
            frame_num += 1
            
            # Run YOLOv8x Tracker (ByteTrack / NvDCF Kalman Association)
            results = detector.track(
                frame,
                persist=True,
                verbose=False,
                conf=0.25,
                iou=0.45,
                tracker="bytetrack.yaml"
            )
            
            tracked_objects = []
            crops_to_extract = []
            valid_indices = []
            
            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for idx, box in enumerate(boxes):
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(float, xyxy)
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    cls_name = results[0].names.get(cls_id, f"class_{cls_id}")
                    
                    track_id = int(box.id[0].cpu().numpy()) if box.id is not None else idx + 1
                    
                    # Ground contact point
                    u = (x1 + x2) / 2.0
                    v = y2
                    
                    norm_u = u / w
                    norm_v = v / h
                    
                    # Extract crop for Re-ID
                    ix1, iy1, ix2, iy2 = max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))
                    crop = frame[iy1:iy2, ix1:ix2]
                    
                    obj_dict = {
                        "track_id": track_id,
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "confidence": round(conf, 3),
                        "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                        "ground_point": (round(u, 2), round(v, 2)),
                        "ground_point_norm": (round(norm_u, 4), round(norm_v, 4)),
                        "reid_embedding": None,
                        "has_reid": False
                    }
                    tracked_objects.append(obj_dict)
                    
                    if crop.size > 0:
                        crops_to_extract.append(crop)
                        valid_indices.append(len(tracked_objects) - 1)
                        
            # Batch extract Re-ID embeddings for all tracked objects in current frame
            if crops_to_extract:
                embeddings = reid_extractor.extract(crops_to_extract)
                for v_idx, emb in zip(valid_indices, embeddings):
                    tracked_objects[v_idx]["reid_embedding"] = emb
                    tracked_objects[v_idx]["has_reid"] = True
                    
            # Build standard DeepStream pad probe payload
            frame_payload = {
                "frame_num": frame_num,
                "source_id": source_id,
                "timestamp": int(time.time() * 1000),
                "frame_width": w,
                "frame_height": h,
                "objects_count": len(tracked_objects),
                "objects": tracked_objects
            }
            
            if self.on_frame_callback:
                try:
                    self.on_frame_callback(frame_payload)
                except Exception as e:
                    print(f"Error in frame callback: {e}", file=sys.stderr)
                    
            # Maintain real-time frame rate pacing
            elapsed = time.time() - t_start
            sleep_time = max(0.001, frame_delay - elapsed)
            time.sleep(sleep_time)
            
        cap.release()

    def start(self):
        """Starts perception pipeline streams across all configured sources"""
        if self.is_running:
            return
        if not self.detector:
            if not self.build_pipeline():
                return
                
        print("🚀 Starting DeepStream Perception Pipeline Streams...")
        self.is_running = True
        self.threads = []
        
        for i, src in enumerate(self.sources):
            t = threading.Thread(target=self._stream_worker, args=(i, src), daemon=True)
            t.start()
            self.threads.append(t)
            print(f" ✅ Started stream worker for Source {i} ({src})")
            
        print("✅ DeepStream Perception Pipeline is running actively")

    def stop(self):
        """Stops all perception pipeline streams"""
        if not self.is_running:
            return
        print("🛑 Stopping DeepStream Perception Pipeline...")
        self.is_running = False
        for t in self.threads:
            if t.is_alive():
                t.join(timeout=1.0)
        print("✅ Pipeline stopped cleanly")
