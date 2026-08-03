import os
import threading
from typing import List


class PoseEstimator:
    """YOLOv8 pose adapter producing the 17 COCO keypoints on CUDA."""

    def __init__(self):
        self.model = None
        self.lock = threading.Lock()
        self.path = os.getenv("POSE_MODEL_PATH", "/app/models/yolov8x-pose.pt")
        self.model_kind = None

    def _load(self):
        if self.model is not None:
            return self.model
        with self.lock:
            if self.model is None:
                try:
                    from ultralytics import YOLO
                    model_path = self.path
                    if not os.path.exists(model_path):
                        print(f"[PoseEstimator] Model not found: {model_path}", flush=True)
                        return self.model
                    self.model = YOLO(model_path)
                    self.model_kind = "pose"
                    import torch
                    if not torch.cuda.is_available():
                        raise RuntimeError("CUDA is not available for pose inference")
                    self.model.to("cuda")
                    print(f"[PoseEstimator] Loaded {model_path} ({self.model_kind})", flush=True)
                except Exception as exc:
                    print(f"[PoseEstimator] Disabled: {exc}", flush=True)
        return self.model

    def infer(self, frame) -> List[dict]:
        model = self._load()
        if model is None or frame is None:
            return []
        height, width = frame.shape[:2]
        try:
            results = model.predict(frame, verbose=False, conf=0.35, imgsz=640, classes=[0], device=0, half=True)
            output = []
            for result in results or []:
                if result.boxes is None:
                    continue
                boxes = result.boxes.xyxy.cpu().numpy()
                points = result.keypoints.xy.cpu().numpy() if result.keypoints is not None else None
                confidence = result.keypoints.conf.cpu().numpy() if result.keypoints is not None and result.keypoints.conf is not None else None
                for index, box in enumerate(boxes):
                    keypoints = []
                    if points is not None:
                        for point_index, point in enumerate(points[index]):
                            score = float(confidence[index][point_index]) if confidence is not None else 1.0
                            keypoints.append([round(float(point[0]) / width, 5), round(float(point[1]) / height, 5), round(score, 4)])
                    output.append({
                        "bbox": [float(box[0]) / width, float(box[1]) / height, float(box[2]) / width, float(box[3]) / height],
                        "keypoints": keypoints,
                    })
            return output
        except Exception as exc:
            print(f"[PoseEstimator] Inference error: {exc}", flush=True)
            return []


pose_estimator = PoseEstimator()
