import faulthandler
faulthandler.enable()

import os
import sys
import uuid
import json
import asyncio
import base64
import time
import math
import re
import requests
import cv2
import numpy as np
from typing import Dict, List, Set, Optional, Tuple, Any, Literal
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query, Response, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from core.database import db_manager
from core.deepstream_engine import deepstream_manager, sanitize_rtsp_url
from core.camera_calibrator import camera_calibrator
from core.behavior_analytics import behavior_engine
from core.reid_matcher import global_reid
from core.fms_bridge import fms_bridge
from core.person_tracker import person_tracker_manager
from core.template_identity_tracker import template_identity_tracker_manager
from core.identity_utils import identity_global_id, robot_number_from_label
from core.robot_spatial_identity import robot_spatial_identity
from core.registered_identity_metrics import registered_identity_metrics
from core.metadata_fusion import MetadataFusion
from core.metadata_broadcaster import LatestMetadataBroadcaster, close_metadata_socket
from core.mediamtx_client import mediamtx_client
from core.registered_target_mask import decode_frame, validate_mask, registered_target_mask_segmenter
from core.online_calibration import OnlineRobotCalibration
from core.recording_store import RecordingStore
from core.ffmpeg_recorder import FFmpegRecorder
from core.system_log_store import SystemLogHandler, SystemLogStore, SystemRequestLogMiddleware
from routers.analytics_storage import create_storage_router
from routers.system_logs import create_system_log_router
from routers.calibration_projection import create_calibration_projection_router
from core.workflow_definition import validate_definition
from core.workflow_store import WorkflowStore
from core.workflow_runtime import WorkflowRuntime
from routers.workflows import create_workflow_router
from routers.models import create_model_router
from core.model_registry import ModelRegistry
from core.custom_detector import CustomDetector
from core.monitor_model_policy import filter_monitor_metadata
from routers.legacy_labels import reject_legacy_label_registration
from core.model_label_store import ModelLabelStore
from routers.model_labels import create_model_label_router
from core.active_learning_store import ActiveLearningStore
from core.active_learning_worker import ActiveLearningWorker
from routers.active_learning import create_active_learning_router
from deep_calib.adapter import status as deepcalib_status

metadata_fusion = MetadataFusion(ttl=1.0, template_ttl=3.5)


app = FastAPI(title="RTC VMS (R-SkyView) - Real-time Decoupled Multi-Camera Analytics", version="3.0")
recording_store = RecordingStore(db_manager)
ffmpeg_recorder = FFmpegRecorder(recording_store)
system_log_store = SystemLogStore(db_manager)
app.include_router(create_storage_router(recording_store, ffmpeg_recorder))
app.include_router(create_system_log_router(system_log_store))
app.include_router(create_calibration_projection_router(camera_calibrator, lambda: cameras,
    lambda: {"origin_x": fms_bridge.origin_x, "origin_y": fms_bridge.origin_y, "layout_depth": fms_bridge.layout_depth}))
app.add_middleware(SystemRequestLogMiddleware, store=system_log_store)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi import Request
from fastapi.responses import JSONResponse
import logging

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logging.error(f"Global Exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal Server Error: {str(exc)}"}
    )

ENABLE_CPU_TRACKER_FALLBACK = os.getenv("ENABLE_CPU_TRACKER_FALLBACK", "1").lower() not in {"0", "false", "no"}


def legacy_deepstream_enabled() -> bool:
    """The bundled Pose sample is opt-in; uploaded deployments own inference."""
    return os.getenv("ENABLE_LEGACY_DEEPSTREAM_PIPELINE", "0").lower() in {"1", "true", "yes"}

# In-memory registry of active cameras
cameras: Dict[str, dict] = {}

def workflow_resources():
    from src.controller.registry import target_registry
    camera_rows = list(cameras.items())
    targets = list(target_registry.targets.values())
    try:
        model_rows = model_registry.list()
    except Exception:
        model_rows = []
    return {
        "cameras": [{"id": camera_id, "name": camera.get("name", camera_id), "status": camera.get("status", "offline"),
                     "calibrated": bool(camera_calibrator.get_config(camera_id))} for camera_id, camera in camera_rows],
        "labels": sorted({str(target.get("label")) for target in targets if target.get("label")}
                         | {label for model in model_rows if model["state"] == "ready" for label in model["labels"]}),
        "recording_enabled": ffmpeg_recorder.enabled,
        "deepstream_active": bool(deepstream_manager.pipeline is not None and deepstream_manager.is_running) or custom_detector.status()["state"] == "live",
        "detector_runtime": custom_detector.status(),
        "models": [{key: row.get(key) for key in ("id", "name", "filename", "size_bytes", "received_bytes", "state", "labels", "metadata", "error", "created_at", "updated_at")} for row in model_rows],
    }

def validate_workflow(definition):
    resources = workflow_resources()
    return validate_definition(definition, [camera["id"] for camera in resources["cameras"]], resources["labels"],
                               [camera["id"] for camera in resources["cameras"] if camera["calibrated"]], resources["recording_enabled"], resources.get("models", []))

def _clear_custom_detector_metadata(camera_id, model_id=None):
    source = f"custom_deepstream:{model_id}" if model_id else "custom_deepstream"
    metadata_fusion.remove_source(camera_id, source)
    latest_objects_by_cam[camera_id] = []
    latest_metadata_at_by_cam[camera_id] = time.time()
    metadata_broadcaster.publish({"source": "model_stopped", "timestamp": time.time() * 1000,
                                  "streams": [{"cam_id": camera_id, "objects": [], "model_id": model_id}]})

workflow_store = WorkflowStore(db_manager)
workflow_runtime = WorkflowRuntime(workflow_store, validate_workflow, lambda event: broadcast_event_sync(event))
model_registry = ModelRegistry(db_manager)
model_label_store = ModelLabelStore(db_manager)
custom_detector = CustomDetector(model_registry, workflow_store, lambda: set(cameras),
                                 lambda payload: broadcast_metadata_sync(payload),
                                 _clear_custom_detector_metadata,
                                 lambda camera_id: _stop_template_identity_tracker(camera_id))
app.include_router(create_workflow_router(workflow_store, workflow_runtime, workflow_resources, validate_workflow))
app.include_router(create_model_router(model_registry, custom_detector.status, lambda: set(cameras)))
app.include_router(create_model_label_router(model_label_store, model_registry, custom_detector))
active_learning_store = ActiveLearningStore(db_manager, model_registry)
active_learning_worker = ActiveLearningWorker(active_learning_store, model_registry, custom_detector)
custom_detector.before_launch = active_learning_worker.preempt_for_live

def _wait_for_gpu_build_window():
    active_learning_worker.preempt_for_live()
    deadline = time.monotonic() + float(os.getenv("DEEPSTREAM_BUILD_WAIT_TIMEOUT", "1800"))
    while time.monotonic() < deadline:
        if not model_registry.deployments(enabled_only=True) and not workflow_store.active() and not custom_detector.status().get("running"):
            return
        time.sleep(.5)
    raise TimeoutError("TensorRT build được hoãn vì DeepStream vẫn đang chạy; dừng deployment rồi build lại.")

model_registry.before_gpu_build = _wait_for_gpu_build_window
app.include_router(create_active_learning_router(active_learning_store, model_registry, custom_detector, active_learning_worker))

def _persist_online_calibration(cam_id, config):
    if cam_id not in cameras:
        raise RuntimeError("Camera không còn tồn tại.")
    config["save_id"] = uuid.uuid4().hex
    config["persisted_at"] = time.time()
    saved = db_manager.save_calibration(
        cam_id,
        config["src_points"],
        config["dst_points"],
        config["matrix"],
        config.get("cam_x"),
        config.get("cam_y"),
        config.get("cam_z"),
        config.get("yaw"),
        config.get("fov_polygon"),
        config=config,
    )
    if not saved:
        raise RuntimeError("Không lưu được hiệu chuẩn; giữ nguyên ma trận đang dùng.")
    try:
        confirmed = db_manager.get_calibration(cam_id)
    except Exception as error:
        raise RuntimeError("Chưa xác nhận được bản ghi PostgreSQL; tải lại Calibration để kiểm tra.") from error
    if not confirmed or confirmed.get("save_id") != config["save_id"]:
        raise RuntimeError("Bản ghi PostgreSQL không khớp lần Save này; tải lại trước khi tiếp tục.")
    if cam_id in cameras:
        cameras[cam_id]["calibration"] = config
        cameras[cam_id]["fov_polygon"] = config.get("fov_polygon")

online_robot_calibration = OnlineRobotCalibration(
    camera_calibrator,
    fms_bridge,
    persist=_persist_online_calibration,
)
calibration_task = None

async def _online_calibration_loop():
    while True:
        try:
            await asyncio.to_thread(online_robot_calibration.fit_pending)
        except Exception:
            logging.exception("Online calibration fitting failed")
        await asyncio.sleep(1.0)


def _is_person_object(obj: dict) -> bool:
    value = f"{obj.get('category', '')} {obj.get('class', '')}".lower()
    return any(term in value for term in ("person", "human", "worker"))

def _start_cpu_tracker_fallback(cam_id: str, rtsp_url: str):
    if not ENABLE_CPU_TRACKER_FALLBACK:
        return
    try:
        person_tracker_manager.add_camera(cam_id, sanitize_rtsp_url(rtsp_url))
    except Exception as e:
        print(f"[PersonTracker] Could not start fallback for {cam_id}: {e}", flush=True)

def _stop_cpu_tracker_fallback(cam_id: str):
    try:
        person_tracker_manager.remove_camera(cam_id)
    except Exception:
        pass

def _stop_template_identity_tracker(cam_id: str):
    metadata_fusion.remove_camera(cam_id)
    try:
        template_identity_tracker_manager.remove_camera(cam_id)
    except Exception:
        pass

def _decode_json_field(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value

def _camera_payload_from_db(row: dict, status: str = "offline") -> dict:
    return {
        "id": row["id"],
        "name": row["name"],
        "rtsp_url": sanitize_rtsp_url(row["rtsp_url"]),
        "calibration": _decode_json_field(row.get("calibration_points")),
        "cam_x": row.get("cam_x"),
        "cam_y": row.get("cam_y"),
        "cam_z": row.get("cam_z"),
        "yaw": row.get("yaw"),
        "fov_polygon": _decode_json_field(row.get("fov_polygon")),
        "status": status,
    }

def _canonical_identity_label(label: str, category: Optional[str] = None) -> str:
    raw = (label or "").strip()
    cat = (category or "").strip().lower()
    if not raw:
        return raw
    compact = re.sub(r"\s+", "_", raw)
    numbers = re.findall(r"\d+", compact)

    if cat == "robot":
        if numbers:
            return f"Robot_{numbers[-1]}"
        return compact if compact.lower().startswith("robot_") else f"Robot_{compact}"
    if cat == "rack":
        if compact.lower().startswith("rack_"):
            return compact
        return f"Rack_{compact}"
    if cat == "person":
        if compact.lower().startswith("person_"):
            return compact
        return f"Person_{compact}"
    return compact

from fastapi.responses import StreamingResponse
import httpx

class ChatMessage(BaseModel):
    text: str
    chat_history: Optional[str] = ""

# Google Gemini AI Studio
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AQ.Ab8RN6JZX1UcoVpiEwav3qrfUFVjhE77aUuWhUzdUytog7xNww")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL",   "gemini-3.6-flash")

def get_gemini_url():
    key = os.getenv("GEMINI_API_KEY") or GEMINI_API_KEY
    model = os.getenv("GEMINI_MODEL") or GEMINI_MODEL
    return f"https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?key={key}&alt=sse"

SYSTEM_PROMPT_TEMPLATE = """Bạn là Trợ lý Ảo AI chuyên trách hỗ trợ vận hành và hướng dẫn sử dụng hệ thống RTC VMS.
Nhiệm vụ của bạn là giải đáp thắc mắc, hướng dẫn người dùng thao tác giao diện hoặc cung cấp thông tin trạng thái hoạt động của hệ thống dựa CHÍNH XÁC vào các khối thông tin được cung cấp bên dưới.

### NGUYÊN TẮC BẮT BUỘC:
1. Độ ưu tiên thông tin:
   - Nếu câu hỏi liên quan đến số lượng camera, số cảnh báo, trạng thái runtime hoặc sự kiện vừa xảy ra: Hãy đọc và sử dụng thông tin trong mục [TRẠNG THÁI HỆ THỐNG THỜI GIAN THỰC].
   - Nếu câu hỏi là hướng dẫn thao tác, cách cấu hình, ý nghĩa các tab/nút bấm, thuật toán: Hãy đọc và sử dụng thông tin trong mục [TÀI LIỆU HƯỚNG DẪN KỸ THUẬT].
2. Tính trung thực & Giới hạn dữ liệu:
   - CHỈ trả lời dựa trên 2 nguồn dữ liệu được cấp. Tuyệt đối không tự suy diễn hoặc bịa đặt số liệu/tính năng không có trong tài liệu.
   - Nếu cả 2 nguồn đều không có thông tin để trả lời câu hỏi, hãy phản hồi: "Xin lỗi, hiện tôi không tìm thấy thông tin/dữ liệu tương ứng trong hệ thống. Vui lòng liên hệ quản trị viên."
3. Phong cách phản hồi:
   - Ngắn gọn, súc tích, đi thẳng vào câu trả lời (tối đa 2 - 4 câu hoặc dùng gạch đầu dòng rõ ràng).
   - Sử dụng tiếng Việt tự nhiên và giữ nguyên các thuật ngữ kỹ thuật trên giao diện (ví dụ: *Monitor*, *Building*, *Analytics*, *WHEP WebRTC*, *Homography 2D*, *MTMC Fusion*).

---
[TRẠNG THÁI HỆ THỐNG THỜI GIAN THỰC]:
{system_state_context}
---

[TÀI LIỆU HƯỚNG DẪN KỸ THUẬT]:
{retrieved_pdf_context}
---

[LỊCH SỬ HỘI THOẠI]:
{chat_history}

[CÂU HỎI CỦA NGƯỜI DÙNG]:
{user_query}

[TRẢ LỜI]:
"""

@app.post("/api/chat")
async def chat_with_bot(req: ChatMessage):
    user_query = req.text
    chat_history = req.chat_history if req.chat_history else "Không có"
    
    # 1. Routing nhẹ: Kiểm tra xem query có cần dữ liệu DB thời gian thực không
    realtime_keywords = ["mấy camera", "bao nhiêu cam", "cảnh báo", "trạng thái", "online", "sự kiện"]
    needs_realtime = any(kw in user_query.lower() for kw in realtime_keywords)
    
    # 2. Lấy dữ liệu động từ PostgreSQL / Memory
    if needs_realtime:
        stats = db_manager.get_dashboard_stats(len(cameras))
        system_state_context = f"- Số camera đang hoạt động: {len(cameras)}\n"
        system_state_context += f"- Tổng số cảnh báo hôm nay: {stats.get('total_alarms', 0)}\n"
        system_state_context += f"- Trạng thái AI Engine: Hoạt động (WHEP WebRTC Active)"
    else:
        system_state_context = "Không có yêu cầu kiểm tra trạng thái thời gian thực."

    # 3. Lấy dữ liệu tĩnh từ ChromaDB (PDF Chunks)
    from core.rag_manager import rag_engine
    retrieved_pdf_context = rag_engine.query_rag(user_query, top_k=2)

    # 4. Ghép hoàn chỉnh Prompt
    prompt = SYSTEM_PROMPT_TEMPLATE.format(
        system_state_context=system_state_context,
        retrieved_pdf_context=retrieved_pdf_context,
        chat_history=chat_history,
        user_query=user_query
    )
    
    async def generate_response():
        try:
            payload = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": prompt}]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.3,
                    "maxOutputTokens": 1024,
                    "topP": 0.8,
                },
                "safetySettings": [
                    {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH",       "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                ]
            }
            gemini_url = get_gemini_url()
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream("POST", gemini_url, json=payload,
                                         headers={"Content-Type": "application/json"}) as response:
                    if response.status_code != 200:
                        err = await response.aread()
                        logging.error(f"Gemini HTTP {response.status_code}: {err[:200]}")
                        yield f"Lỗi kết nối Gemini API (HTTP {response.status_code})."
                        return

                    # SSE stream: mỗi dòng có dạng "data: {...}"
                    async for line in response.aiter_lines():
                        line = line.strip()
                        if not line or not line.startswith("data:"):
                            continue
                        raw = line[5:].strip()  # bỏ prefix "data:"
                        if raw == "[DONE]":
                            break
                        try:
                            chunk = json.loads(raw)
                            # Lấy text từ candidates[0].content.parts[0].text
                            text = (
                                chunk
                                .get("candidates", [{}])[0]
                                .get("content", {})
                                .get("parts", [{}])[0]
                                .get("text", "")
                            )
                            if text:
                                yield text
                        except Exception:
                            pass
        except Exception as e:
            logging.error(f"Gemini Error: {str(e)}")
            yield "Xin lỗi, hiện tại tôi không thể kết nối tới Google Gemini AI. Vui lòng kiểm tra API key."

    return StreamingResponse(generate_response(), media_type="text/plain")
# Active WebSocket connections
metadata_broadcaster = LatestMetadataBroadcaster()
connected_event_ws: Set[WebSocket] = set()
loop: Optional[asyncio.AbstractEventLoop] = None
latest_objects_by_cam: Dict[str, List[dict]] = {}
latest_metadata_at_by_cam: Dict[str, float] = {}

# --- REQUEST MODELS ---
class CameraAddRequest(BaseModel):
    name: str
    rtsp_url: str

class CalibrationLengthConstraint(BaseModel):
    points: List[Tuple[float, float]] = Field(min_length=2, max_length=2)
    distance_m: float = Field(ge=.001, le=10000, allow_inf_nan=False)

class CalibrationRequest(BaseModel):
    src_points: List[List[float]]
    dst_points: List[List[float]]
    cam_x: Optional[float] = None
    cam_y: Optional[float] = None
    cam_z: Optional[float] = None
    yaw: Optional[float] = None
    method: Optional[str] = None
    map_id: Optional[str] = None
    fms_frame: Optional[Dict[str, float]] = None
    length_constraints: List[CalibrationLengthConstraint] = Field(default_factory=list)
    deepcalib: Optional[Dict[str, Any]] = None
    points_space: Literal["raw", "rectified"] = "raw"
    deepcalib_preview_id: Optional[str] = None

class DeepCalibPreviewRequest(BaseModel):
    image: str = Field(min_length=32, max_length=24_000_000)
    use_saved_profile: bool = False
    calibration_save_id: Optional[str] = None

class CalibrationMeasureRequest(BaseModel):
    points: List[Tuple[float, float]] = Field(min_length=2, max_length=256)
    points_space: Literal["raw", "rectified"] = "raw"
    calibration_save_id: Optional[str] = None

class AutoCalibrationStartRequest(BaseModel):
    auto_apply: bool = False
    time_offset_ms: float = Field(default=0, ge=-2000, le=2000, allow_inf_nan=False)
    reset: bool = False

class RuleItem(BaseModel):
    id: str
    type: str # intrusion, tripwire, dwell_time, density, occupancy
    name: str
    points: List[List[float]] # polygon or line coordinates
    camera_points: Optional[List[List[float]]] = None
    fms_points: Optional[List[List[float]]] = None
    target_objects: Optional[List[str]] = ["robot", "rack"]
    threshold: Optional[float] = 10.0
    direction: Optional[str] = "both"
    coordinate_space: Optional[str] = "camera"

class SaveRulesRequest(BaseModel):
    rules: List[RuleItem]

def _normalize_bbox(bbox: Optional[List[float]]) -> Optional[List[float]]:
    if not bbox or len(bbox) != 4:
        return None
    try:
        x, y, w, h = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None
    x = min(1.0, max(0.0, x))
    y = min(1.0, max(0.0, y))
    w = min(1.0 - x, max(0.0, w))
    h = min(1.0 - y, max(0.0, h))
    if w <= 0.0 or h <= 0.0:
        return None
    return [x, y, w, h]

def _bbox_iou(a: List[float], b: List[float]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 1e-6 else 0.0

def _center_distance(a: List[float], b: List[float]) -> float:
    import math
    return math.hypot((a[0] + a[2] / 2.0) - (b[0] + b[2] / 2.0), (a[1] + a[3] / 2.0) - (b[1] + b[3] / 2.0))

def _object_bbox(obj: dict) -> Optional[List[float]]:
    return _normalize_bbox([
        obj.get("x", 0.0),
        obj.get("y", 0.0),
        obj.get("w", 0.0),
        obj.get("h", 0.0),
    ])

def _match_crop_and_detection(crop: List[float], obj_bbox: List[float]) -> Tuple[float, dict]:
    cx, cy, cw, ch = crop
    ox, oy, ow, oh = obj_bbox
    
    c_cx, c_cy = cx + cw / 2.0, cy + ch / 2.0
    o_cx, o_cy = ox + ow / 2.0, oy + oh / 2.0
    
    crop_center_in_obj = (ox <= c_cx <= ox + ow) and (oy <= c_cy <= oy + oh)
    obj_center_in_crop = (cx <= o_cx <= cx + cw) and (cy <= o_cy <= cy + ch)
    
    ix1, iy1 = max(cx, ox), max(cy, oy)
    ix2, iy2 = min(cx + cw, ox + ow), min(cy + ch, oy + oh)
    inter_area = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    crop_area = max(1e-6, cw * ch)
    obj_area = max(1e-6, ow * oh)
    union_area = crop_area + obj_area - inter_area
    
    iou = inter_area / union_area if union_area > 1e-6 else 0.0
    crop_coverage = inter_area / crop_area
    obj_coverage = inter_area / obj_area
    
    import math
    dist = math.hypot(c_cx - o_cx, c_cy - o_cy)
    proximity = max(0.0, (0.35 - dist) / 0.35)
    
    containment_bonus = 0.50 if (crop_center_in_obj or obj_center_in_crop) else 0.0
    overlap_metric = max(iou * 1.5, crop_coverage * 0.8, obj_coverage * 0.8)
    score = containment_bonus + (0.35 * overlap_metric) + (0.25 * proximity)
    
    is_valid = (
        crop_center_in_obj or
        obj_center_in_crop or
        (crop_coverage >= 0.20) or
        (obj_coverage >= 0.20) or
        (iou >= 0.06) or
        (dist <= 0.22 and proximity > 0.35)
    )
    
    return (score if is_valid else 0.0, {
        "score": score,
        "iou": iou,
        "crop_coverage": crop_coverage,
        "dist": dist,
        "crop_center_in_obj": crop_center_in_obj,
        "is_valid": is_valid
    })

def _find_live_object_for_crop(cam_id: str, crop_bbox: Optional[List[float]]) -> Optional[dict]:
    crop = _normalize_bbox(crop_bbox)
    if crop is None:
        return None

    best_obj = None
    best_score = 0.0
    for obj in latest_objects_by_cam.get(cam_id, []):
        obj_bbox = _object_bbox(obj)
        if obj_bbox is None:
            continue
        score, _ = _match_crop_and_detection(crop, obj_bbox)
        if score > best_score:
            best_score = score
            best_obj = obj

    return best_obj if best_score > 0.0 else None

def _best_live_match_for_crop(cam_id: str, crop_bbox: Optional[List[float]]) -> dict:
    crop = _normalize_bbox(crop_bbox)
    if crop is None:
        return {"score": 0.0, "object": None, "reason": "target has no stored crop bbox"}

    best_obj = None
    best_score = 0.0
    best_meta = {}
    for obj in latest_objects_by_cam.get(cam_id, []):
        obj_bbox = _object_bbox(obj)
        if obj_bbox is None:
            continue
        score, meta = _match_crop_and_detection(crop, obj_bbox)
        if score > best_score:
            best_score = score
            best_meta = meta
            best_obj = obj

    if not latest_objects_by_cam.get(cam_id):
        reason = "no live objects from detector/tracker for this camera"
    elif best_score <= 0.0:
        reason = "live objects exist, but none overlap or contain the crop bbox"
    else:
        reason = "ready"

    return {
        "score": round(best_score, 4),
        "iou": round(best_meta.get("iou", 0.0), 4),
        "center_distance": round(best_meta.get("dist", 0.0), 4) if "dist" in best_meta else None,
        "object": best_obj,
        "reason": reason,
    }

# --- ASYNC EVENT & METADATA BROADCASTERS ---
def broadcast_metadata_sync(payload: dict):
    payload = filter_monitor_metadata(payload, custom_detector.accepts_stream)
    if payload is None:
        return
    now = time.time()
    if "streams" in payload and isinstance(payload["streams"], list):
        for st in payload["streams"]:
            cam_id = st.get("cam_id")
            objects = st.get("objects")
            if cam_id and isinstance(objects, list):
                cam_key = str(cam_id)
                incoming = [dict(obj) for obj in objects if isinstance(obj, dict)]
                registered_robots = {}
                if payload.get("source") == "identity_template" and online_robot_calibration.sessions.get(cam_key, {}).get("running"):
                    try:
                        from src.controller.registry import target_registry
                        for target in target_registry.get_all_targets(cam_id=cam_key):
                            if (target.get("category") or "").lower() != "robot":
                                continue
                            robot_id = target.get("fms_robot_id") or robot_number_from_label(target.get("label", ""))
                            if robot_id is not None:
                                registered_robots[str(target.get("label", "")).casefold()] = robot_id
                    except Exception:
                        registered_robots = {}
                if payload.get("source") == "custom_deepstream" and online_robot_calibration.sessions.get(cam_key, {}).get("running"):
                    registered_robots = {obj["label"].casefold(): robot_id for obj in incoming
                                         if obj.get("category") == "robot" and obj.get("label")
                                         and (robot_id := robot_number_from_label(obj["label"])) is not None}
                for obj in incoming:
                    obj.setdefault("category", (obj.get("class") or "object").lower())
                    obj.setdefault("observed_at", payload.get("timestamp", int(now * 1000)))
                    if payload.get("source") == "custom_deepstream":
                        obj.setdefault("generation", st.get("generation"))
                incoming, spatial_rejections = robot_spatial_identity.filter_objects(cam_key, incoming, now=now)
                st["identity_rejected_labels"] = sorted(set(st.get("identity_rejected_labels", []) + spatial_rejections))
                if registered_robots:
                    online_robot_calibration.observe(cam_key, incoming, registered_robots, now=now)
                fusion_source = payload.get("source", "deepstream")
                if fusion_source == "custom_deepstream" and st.get("model_id"):
                    fusion_source = f"custom_deepstream:{st['model_id']}"
                merged = metadata_fusion.update(cam_key, fusion_source, incoming)
                merged, merged_rejections = robot_spatial_identity.filter_objects(cam_key, merged, now=now)
                st["identity_rejected_labels"] = sorted(set(st["identity_rejected_labels"] + merged_rejections))
                for obj in merged:
                    ground = obj.get("ground_point") or [obj.get("x", 0) + obj.get("w", 0) / 2, obj.get("y", 0) + obj.get("h", 0)]
                    spatial = camera_calibrator.project_ground_point(cam_key, *ground)
                    obj.update(floor_x=spatial["x"], floor_y=spatial["z"],
                        world_position=[spatial["x"], spatial["y"], spatial["z"]], spatial_valid=spatial["valid"],
                        spatial_source=spatial["source"], spatial_confidence=spatial["confidence"],
                        inside_calibrated_area=spatial.get("inside_calibrated_area", False),
                        fms_world_position=fms_bridge.floor_to_fms(spatial["x"], spatial["z"]) if spatial["valid"] else None)
                latest_objects_by_cam[cam_key] = merged
                st["objects"] = merged
                latest_metadata_at_by_cam[str(cam_id)] = now
                if payload.get("source") == "custom_deepstream":
                    try:
                        triggered_events, tripwire_stats, roi_states = behavior_engine.process_frame(cam_key, merged)
                        alert_ids = {event.get("global_id") for event in triggered_events}
                        for obj in merged:
                            obj["alert"] = obj.get("id") in alert_ids
                        st["tripwire_stats"] = tripwire_stats
                        st["rois"] = roi_states
                        for event in triggered_events:
                            broadcast_event_sync(event)
                    except Exception:
                        logging.exception("Custom model behavior engine failed for %s", cam_key)
                if payload.get("source") == "identity_template":
                    from src.controller.registry import target_registry
                    for obj in incoming:
                        if obj.get("label") and obj.get("tracking_state") == "tracked":
                            target_registry.bind_label_to_track(
                                obj["label"], cam_key, obj["id"], obj["local_id"],
                                live_bbox=[obj["x"], obj["y"], obj["w"], obj["h"]],
                            )

    if not loop:
        return
        
    # Forward vision tracks to DigitalTwinBridge & enrich with real-time FMS telemetry
    try:
        from src.server.digital_twin_bridge import digital_twin_bridge
        telemetry = digital_twin_bridge.get_latest_telemetry_payload()
        robot_map = {r.get("id"): r for r in telemetry.get("robots", []) if isinstance(r, dict)}
        
        if "streams" in payload and isinstance(payload["streams"], list):
            for st in payload["streams"]:
                c_id = st.get("cam_id", "cam_default")
                for obj in st.get("objects", []):
                    t_id = obj.get("id", 0)
                    c_name = obj.get("label") or obj.get("class", "robot")
                    norm_u = obj.get("x", 0.5) + obj.get("w", 0.0) / 2.0
                    norm_v = obj.get("y", 0.5) + obj.get("h", 0.0)
                    bbox = [obj.get("x", 0), obj.get("y", 0), obj.get("x", 0) + obj.get("w", 0), obj.get("y", 0) + obj.get("h", 0)]
                    reid_vec = obj.get("reid_vector")
                    digital_twin_bridge.update_vision_track(
                        c_id, t_id, c_name, norm_u, norm_v, bbox, reid_vec,
                        object_data=obj,
                    )
                    
                    # Sync authoritative FMS telemetry to stream objects in real-time
                    matched_robot = robot_map.get(c_name)
                    if obj.get("model_id") and obj.get("fms_robot_id") is not None:
                        matched_robot = robot_map.get(f"Robot_{obj['fms_robot_id']}") or robot_map.get(str(obj["fms_robot_id"]))
                    if not matched_robot and c_name.startswith("Robot_"):
                        cand = c_name.replace("Robot_", "")
                        matched_robot = robot_map.get(cand) or robot_map.get(f"Robot_{cand}")
                    if matched_robot:
                        obj["fms_status"] = matched_robot.get("status")
                        obj["fms_battery"] = matched_robot.get("battery")
                        obj["fms_speed"] = matched_robot.get("velocity")
                        obj["carried_rack"] = matched_robot.get("carried_rack_id")
                        obj["cross_check"] = matched_robot.get("cross_check_status")
                        obj["delta_distance_m"] = matched_robot.get("delta_distance_m")
                        if matched_robot.get("position"):
                            obj["fms_pos"] = matched_robot["position"]
    except Exception as e:
        pass

    workflow_runtime.publish(payload)
    metadata_broadcaster.publish(custom_detector.monitor_payload(payload))

def broadcast_template_metadata_sync(payload: dict):
    """Publish registered crop identities alongside DeepStream tracks."""
    broadcast_metadata_sync(payload)

def broadcast_event_sync(event_payload: dict):
    severity = str(event_payload.get("severity", "info")).lower()
    system_log_store.emit(
        severity if severity in {"info", "warning", "error", "critical"} else "info",
        "analytics", str(event_payload.get("rule_type") or event_payload.get("type") or "event"),
        str(event_payload.get("description") or "Vision event"),
        camera_id=event_payload.get("cam_id"), details={
            key: event_payload.get(key) for key in ("rule_id", "global_id", "roi_status", "floor_pos", "bbox")
            if event_payload.get(key) is not None
        })
    if not connected_event_ws or not loop:
        return
    msg = json.dumps(event_payload)
    asyncio.run_coroutine_threadsafe(_broadcast_to_set(connected_event_ws, msg), loop)

async def _send_single_ws(ws: WebSocket, msg: str, disconnected: Set[WebSocket]):
    try:
        await asyncio.wait_for(ws.send_text(msg), timeout=0.10)
    except Exception:
        disconnected.add(ws)

async def _broadcast_to_set(target_set: Set[WebSocket], msg: str):
    if not target_set:
        return
    disconnected = set()
    ws_list = list(target_set)
    tasks = [_send_single_ws(ws, msg, disconnected) for ws in ws_list]
    await asyncio.gather(*tasks, return_exceptions=True)
    for ws in disconnected:
        target_set.discard(ws)
    await asyncio.gather(*(close_metadata_socket(ws) for ws in disconnected))

@app.on_event("startup")
async def startup_event():
    global loop, calibration_task
    loop = asyncio.get_running_loop()
    try:
        await asyncio.to_thread(system_log_store.initialize)
        root_logger = logging.getLogger()
        root_logger.setLevel(min(root_logger.level, logging.INFO))
        if not any(getattr(handler, "_rsky_system_log", False) for handler in root_logger.handlers):
            handler = SystemLogHandler(system_log_store)
            handler._rsky_system_log = True
            handler.setLevel(logging.INFO)
            root_logger.addHandler(handler)
        system_log_store.emit("info", "gateway", "startup", "R-SkyView backend started")
    except Exception:
        logging.exception("System log store startup failed; runtime continues without structured persistence")
    try:
        await asyncio.to_thread(model_registry.initialize)
        await asyncio.to_thread(model_label_store.initialize)
    except Exception:
        logging.exception("Model registry startup failed; existing tracking remains active")
    metadata_broadcaster.start(loop)
    try:
        await asyncio.to_thread(active_learning_store.initialize)
    except Exception:
        logging.exception("Active Learning startup failed; live tracking remains unchanged")
    calibration_task = asyncio.create_task(_online_calibration_loop())

    # Start FMS Realtime Bridge
    try:
        await fms_bridge.start()
    except Exception as e:
        logging.error(f"Failed to start FMS Bridge: {e}")

    # Start 3D Digital Twin Real-Time Broadcaster (15 Hz)
    try:
        from src.server.digital_twin_bridge import digital_twin_bridge
        digital_twin_bridge.start_broadcaster(loop, fps=15)
    except Exception as e:
        logging.error(f"Failed to start DigitalTwinBridge: {e}")

    # Khởi tạo RAG (Load PDF into ChromaDB trong background)
    def _init_rag():
        try:
            from core.rag_manager import rag_engine
            rag_pdf_path = os.path.join(os.path.dirname(__file__), "rag_data", "main-10.pdf")
            rag_engine.initialize_with_pdf(rag_pdf_path)
        except Exception as e:
            print(f"[RAG] Background init exception: {e}")

    import threading
    threading.Thread(target=_init_rag, daemon=True, name="rag-init").start()

    # Wire DeepStream / GPU PersonTracker to the metadata broadcast callback
    deepstream_manager.metadata_callback = broadcast_metadata_sync
    deepstream_manager.event_callback = broadcast_event_sync
    person_tracker_manager.metadata_callback = broadcast_metadata_sync
    person_tracker_manager.event_callback = broadcast_event_sync
    template_identity_tracker_manager.metadata_callback = None

    # Pre-populate persisted cameras from DB on startup.
    # A configured camera is only removed by the delete API, never by offline checks.
    db_cams = db_manager.get_all_cameras()
    for c in db_cams:
        cam_id = c["id"]
        cam = _camera_payload_from_db(c, status="offline")
        cameras[cam_id] = cam

        try:
            await asyncio.to_thread(mediamtx_client.ensure_path, cam_id, cam["rtsp_url"])
        except Exception as e:
            print(f"[MediaMTX] Startup proxy path registration failed for {cam_id}: {e}")

    await asyncio.to_thread(ffmpeg_recorder.start, list(cameras))
    initial_sources = []
    offline_cameras = []

    for cam_id, c in cameras.items():
        rtsp_url = c["rtsp_url"]

        calib_pts = c.get("calibration")
        if calib_pts and isinstance(calib_pts, dict):
            camera_calibrator.restore_config(cam_id, dict(calib_pts, cam_x=c.get("cam_x"),
                cam_y=c.get("cam_y"), cam_z=c.get("cam_z"), yaw=c.get("yaw")))
            online_settings = calib_pts.get("online") or {}
            if online_settings.get("enabled"):
                online_robot_calibration.start(cam_id, auto_apply=online_settings.get("auto_apply", False),
                                               time_offset_ms=online_settings.get("time_offset_ms", 0))

        db_manager.delete_invalid_occupancy_rules(cam_id)
        rules = db_manager.get_rules_by_camera(cam_id)
        if rules:
            behavior_engine.set_rules(cam_id, rules)

        is_reachable = False
        for attempt in range(2):
            try:
                from check_rtsp import is_rtsp_valid_async
                is_reachable = await is_rtsp_valid_async(rtsp_url, timeout=8)
                if is_reachable:
                    break
            except Exception:
                is_reachable = False

        if is_reachable:
            c["status"] = "online"
            await asyncio.to_thread(mediamtx_client.ensure_preview, cam_id)
            initial_sources.append((cam_id, rtsp_url))
            print(f"[Main] Camera {cam_id} reachable - will add to pipeline.", flush=True)
            _start_cpu_tracker_fallback(cam_id, rtsp_url)
        else:
            c["status"] = "offline"
            offline_cameras.append((cam_id, rtsp_url))
            print(f"[Main] Camera {cam_id} unreachable at startup - kept in DB and will retry.", flush=True)

    if legacy_deepstream_enabled():
        deepstream_manager.start(initial_sources=initial_sources if initial_sources else None)
    else:
        print("[Main] Legacy DeepStream Pose sample disabled; inference starts only after an uploaded model is deployed.", flush=True)

    if offline_cameras:
        asyncio.ensure_future(_retry_offline_cameras(offline_cameras))

    print(f"[Main] Startup complete. {len(initial_sources)} cameras online, {len(offline_cameras)} persisted offline/retrying.", flush=True)
    try:
        await asyncio.to_thread(workflow_runtime.start)
        custom_detector.start()
        if active_learning_store.ready:
            active_learning_worker.start()
    except Exception:
        logging.exception("Workflow startup failed; existing tracking remains active")


@app.on_event("shutdown")
async def shutdown_event():
    await asyncio.to_thread(active_learning_worker.stop)
    await asyncio.to_thread(custom_detector.stop)
    await asyncio.to_thread(model_registry.stop)
    await asyncio.to_thread(workflow_runtime.stop)
    await asyncio.to_thread(ffmpeg_recorder.stop)
    await asyncio.to_thread(mediamtx_client.stop_previews)
    await metadata_broadcaster.stop()
    if calibration_task:
        calibration_task.cancel()
        try:
            await calibration_task
        except asyncio.CancelledError:
            pass
    try:
        await fms_bridge.stop()
    except Exception as e:
        logging.error(f"Error stopping FMS Bridge: {e}")
    system_log_store.emit("info", "gateway", "shutdown", "R-SkyView backend stopping")
    await asyncio.to_thread(system_log_store.stop)


async def _retry_offline_cameras(offline_list: list, interval: int = 30):
    """Background task: retry adding offline cameras to the pipeline every `interval` seconds."""
    remaining = list(offline_list)
    while remaining:
        await asyncio.sleep(interval)
        still_offline = []
        for cam_id, rtsp_url in remaining:
            if cam_id not in cameras or cameras[cam_id]["rtsp_url"] != rtsp_url:
                continue

            try:
                await asyncio.to_thread(mediamtx_client.ensure_path, cam_id, rtsp_url)
                from check_rtsp import is_rtsp_valid_async
                is_reachable = await is_rtsp_valid_async(rtsp_url, timeout=5)
            except Exception:
                is_reachable = False

            if is_reachable:
                cameras[cam_id]["status"] = "online"
                await asyncio.to_thread(mediamtx_client.ensure_preview, cam_id)
                print(f"[Main] Camera {cam_id} is now reachable - adding to pipeline.", flush=True)
                if legacy_deepstream_enabled():
                    deepstream_manager.add_source(cam_id, rtsp_url)
                _start_cpu_tracker_fallback(cam_id, rtsp_url)
            else:
                cameras[cam_id]["status"] = "offline"
                still_offline.append((cam_id, rtsp_url))

        remaining = still_offline
        if remaining:
            print(f"[Main] Still waiting for {len(remaining)} offline camera(s) to come online.", flush=True)
        else:
            print("[Main] All cameras are now online.", flush=True)



# --- WEBSOCKET ENDPOINTS ---
@app.websocket("/ws/metadata")
async def websocket_metadata_endpoint(websocket: WebSocket):
    await websocket.accept()
    metadata_broadcaster.register(websocket)
    try:
        while True:
            await websocket.receive_text()
    except (WebSocketDisconnect, RuntimeError):
        pass
    except Exception:
        pass
    finally:
        await metadata_broadcaster.unregister(websocket)

@app.websocket("/ws/events")
async def websocket_events_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_event_ws.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        connected_event_ws.discard(websocket)
    except Exception:
        connected_event_ws.discard(websocket)

# --- CAMERA MANAGEMENT API ---
@app.post("/api/v1/streams/add")
@app.post("/api/camera/add")
async def add_camera(request: CameraAddRequest):
    cam_id = str(uuid.uuid4())[:8]
    clean_url = sanitize_rtsp_url(request.rtsp_url)
    
    # 1. Check RTSP asynchronously with 2.5s timeout (non-blocking)
    from check_rtsp import is_rtsp_valid_async
    is_valid = await is_rtsp_valid_async(clean_url, timeout=2.5)
    
    status = "online" if is_valid else "offline"

    # 2. Save to DB before runtime registration. If this fails, do not pretend the camera is persistent.
    if not db_manager.save_camera(cam_id, request.name, clean_url):
        raise HTTPException(status_code=500, detail="Không thể lưu camera vào PostgreSQL.")
        
    cameras[cam_id] = {
        "id": cam_id,
        "name": request.name,
        "rtsp_url": clean_url,
        "status": status
    }
    
    # 3. Register Camera Stream in MediaMTX for direct WebRTC/WHEP streaming (background / fast)
    try:
        await asyncio.to_thread(mediamtx_client.ensure_path, cam_id, clean_url)
    except Exception as e:
        print(f"[MediaMTX] Note: proxy path registration: {e}")

    ffmpeg_recorder.add_camera(cam_id)
    # 4. A configured camera is persisted even if the RTSP stream is temporarily offline.
    if is_valid:
        await asyncio.to_thread(mediamtx_client.ensure_preview, cam_id)
        if legacy_deepstream_enabled():
            deepstream_manager.add_source(cam_id, clean_url)
        _start_cpu_tracker_fallback(cam_id, clean_url)
    else:
        asyncio.ensure_future(_retry_offline_cameras([(cam_id, clean_url)]))
    
    return {
        "status": "success",
        "camera": cameras[cam_id],
        "warning": None if is_valid else "Camera đã được lưu nhưng RTSP hiện offline; hệ thống sẽ tự retry."
    }

class CameraUpdateRequest(BaseModel):
    name: Optional[str] = None
    rtsp_url: Optional[str] = None

@app.patch("/api/camera/{cam_id}")
async def update_camera(cam_id: str, request: CameraUpdateRequest):
    if cam_id not in cameras:
        persisted = next((c for c in db_manager.get_all_cameras() if c["id"] == cam_id), None)
        if not persisted:
            raise HTTPException(status_code=404, detail="Camera not found")
        cameras[cam_id] = _camera_payload_from_db(persisted, status="offline")
        
    cam = cameras[cam_id]
    if request.name:
        cam["name"] = request.name
    if request.rtsp_url:
        clean_url = sanitize_rtsp_url(request.rtsp_url)
        await asyncio.to_thread(mediamtx_client.stop_preview, cam_id)
        cam["rtsp_url"] = clean_url
        cam["status"] = "offline"
        try:
            await asyncio.to_thread(mediamtx_client.ensure_path, cam_id, clean_url)
        except Exception:
            pass
        deepstream_manager.delete_source(cam_id)
        _stop_cpu_tracker_fallback(cam_id)
        await asyncio.to_thread(_stop_template_identity_tracker, cam_id)

        try:
            from check_rtsp import is_rtsp_valid_async
            is_valid = await is_rtsp_valid_async(clean_url, timeout=2.5)
        except Exception:
            is_valid = False

        if is_valid:
            cam["status"] = "online"
            await asyncio.to_thread(mediamtx_client.ensure_preview, cam_id)
            if legacy_deepstream_enabled():
                deepstream_manager.add_source(cam_id, clean_url)
            _start_cpu_tracker_fallback(cam_id, clean_url)
        else:
            asyncio.ensure_future(_retry_offline_cameras([(cam_id, clean_url)]))

    if not db_manager.save_camera(cam_id, cam["name"], cam["rtsp_url"]):
        raise HTTPException(status_code=500, detail="Không thể lưu camera vào PostgreSQL.")
    return {"status": "success", "camera": cam}

@app.delete("/api/v1/streams/{cam_id}")
@app.delete("/api/camera/{cam_id}")
async def delete_camera(cam_id: str):
    persisted = any(c["id"] == cam_id for c in db_manager.get_all_cameras())
    if cam_id not in cameras and not persisted:
        raise HTTPException(status_code=404, detail="Camera not found")

    ffmpeg_recorder.remove_camera(cam_id)
    await asyncio.to_thread(mediamtx_client.stop_preview, cam_id)
    if cam_id in cameras:
        deepstream_manager.delete_source(cam_id)
        _stop_cpu_tracker_fallback(cam_id)
        await asyncio.to_thread(_stop_template_identity_tracker, cam_id)
    try:
        await asyncio.to_thread(mediamtx_client.delete_path, cam_id)
    except Exception:
        pass

    db_manager.delete_camera(cam_id)
    online_robot_calibration.remove_camera(cam_id)
    cameras.pop(cam_id, None)
    return {"status": "success", "deleted_id": cam_id}

@app.get("/api/v1/streams/list")
@app.get("/api/camera/list")
async def list_cameras():
    persisted = {}
    for row in await asyncio.to_thread(db_manager.get_all_cameras):
        cam = _camera_payload_from_db(row, status="offline")
        persisted[cam["id"]] = cam

    for cam_id, cam in cameras.items():
        persisted[cam_id] = {
            **persisted.get(cam_id, {}),
            **cam,
        }

    return {
        "status": "success",
        "cameras": list(persisted.values())
    }

@app.get("/api/camera/{cam_id}/snapshot")
async def get_camera_snapshot(cam_id: str):
    if cam_id not in cameras:
        raise HTTPException(status_code=404, detail="Camera not found")
        
    def _grab():
        rtsp_url = f"rtsp://localhost:8554/{cam_id}"
        cap = cv2.VideoCapture(rtsp_url)
        ret, frame = cap.read()
        cap.release()
        
        if not ret or frame is None:
            direct_url = cameras[cam_id].get("rtsp_url")
            if direct_url:
                cap = cv2.VideoCapture(direct_url)
                ret, frame = cap.read()
                cap.release()
                
        if not ret or frame is None:
            return None
            
        ret_encode, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret_encode:
            return None
        return buffer.tobytes()

    loop = asyncio.get_event_loop()
    img_bytes = await loop.run_in_executor(None, _grab)
    if img_bytes is None:
        raise HTTPException(status_code=500, detail="Không thể chụp snapshot từ camera")
        
    return Response(content=img_bytes, media_type="image/jpeg")

@app.post("/api/camera/{cam_id}/deepcalib/estimate")
async def estimate_camera_deepcalib(cam_id: str, request: DeepCalibPreviewRequest):
    if cam_id not in cameras:
        raise HTTPException(status_code=404, detail="Camera not found")

    from deep_calib.pipeline import preview
    profile = None
    if request.use_saved_profile:
        config = camera_calibrator.get_config(cam_id) or {}
        if not config.get("deepcalib_points_rectified") or not config.get("intrinsic_profile"):
            raise HTTPException(409, "Camera chưa lưu profile DeepCalib. Hãy chạy DeepCalib trước.")
        if request.calibration_save_id != config.get("save_id"):
            raise HTTPException(409, "Hiệu chuẩn đã thay đổi. Hãy tải lại Calibration trước khi mở profile đã lưu.")
        profile = config["intrinsic_profile"]
        if profile.get("geometry_version", 1) < 2:
            raise HTTPException(409, "Profile DeepCalib cũ chưa có miền ảnh hiệu chỉnh chuẩn. Hãy chạy DeepCalib và chấm lại điểm; calib hiện tại vẫn giữ nguyên.")
    try:
        result = await asyncio.to_thread(preview, cam_id, request.image, profile)
        return {"status": "success", **result}
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    except RuntimeError as error:
        raise HTTPException(503, str(error)) from error

# --- CAMERA CALIBRATION API (2D-to-Floor-Map) ---
@app.post("/api/camera/{cam_id}/calibration")
async def save_camera_calibration(cam_id: str, calib: CalibrationRequest):
    if cam_id not in cameras:
        raise HTTPException(status_code=404, detail="Camera not found")
        
    try:
        previous = camera_calibrator.get_config(cam_id) or {}
        metadata = {key: getattr(calib, key) if getattr(calib, key) is not None else previous.get(key, cameras[cam_id].get(key))
                    for key in ("cam_x", "cam_y", "cam_z", "yaw")}
        metadata["method"] = "homography"
        if calib.method in {"manual_camera_fms_click", "deepcalib_camera_fms"}:
            from core.manual_calibration import prepare_manual_calibration
            frame = {"origin_x": fms_bridge.origin_x, "origin_y": fms_bridge.origin_y, "layout_depth": fms_bridge.layout_depth}
            if calib.map_id != "TT" or not calib.fms_frame or any(
                not math.isfinite(calib.fms_frame.get(key, float("nan"))) or abs(calib.fms_frame.get(key, 0) - value) > 1e-6
                for key, value in frame.items()
            ):
                raise HTTPException(409, "Hệ tọa độ FMS đã thay đổi. Hãy tải lại Calibration trước khi lưu.")
            constraints = [item.model_dump() for item in calib.length_constraints]
            if calib.method == "deepcalib_camera_fms":
                from deep_calib.adapter import prepare_config as prepare_deepcalib_config
                from deep_calib.pipeline import preview_profile
                profile = calib.deepcalib or {}
                if calib.points_space == "rectified":
                    profile = preview_profile(cam_id, calib.deepcalib_preview_id)
                cfg = await asyncio.to_thread(
                    prepare_deepcalib_config, camera_calibrator, calib.src_points, calib.dst_points,
                    "TT", frame, metadata, constraints, profile, calib.points_space,
                )
            else:
                if calib.points_space != "raw":
                    raise ValueError("Điểm ảnh hiệu chỉnh phải được lưu bằng phương pháp DeepCalib.")
                cfg = await asyncio.to_thread(
                    prepare_manual_calibration, camera_calibrator, calib.src_points, calib.dst_points,
                    "TT", frame, metadata, constraints,
                )
        else:
            if calib.points_space != "raw":
                raise ValueError("Điểm ảnh hiệu chỉnh phải được lưu bằng phương pháp DeepCalib.")
            if calib.length_constraints:
                raise ValueError("Chiều dài phải đi kèm điểm neo trong phương pháp Camera ↔ FMS.")
            cfg = camera_calibrator.prepare_config(calib.src_points, calib.dst_points, metadata=metadata)
        await asyncio.to_thread(online_robot_calibration.save_manual, cam_id, cfg)
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    except RuntimeError as error:
        raise HTTPException(503, str(error)) from error
    return {"status": "success", "config": cfg, "persisted": True, "save_id": cfg.get("save_id")}

@app.get("/api/camera/{cam_id}/calibration")
async def get_camera_calibration(cam_id: str):
    if cam_id not in cameras:
        raise HTTPException(404, "Camera not found")
    try:
        saved = await asyncio.to_thread(db_manager.get_calibration, cam_id)
    except Exception as error:
        raise HTTPException(503, "Không đọc được calibration từ PostgreSQL.") from error
    cfg = camera_calibrator.get_config(cam_id)
    matrices_match = False
    if saved and cfg:
        try:
            matrices_match = np.allclose(
                np.asarray(saved.get("matrix"), dtype=float),
                np.asarray(cfg.get("matrix"), dtype=float),
                rtol=1e-7,
                atol=1e-9,
            )
        except (TypeError, ValueError):
            matrices_match = False
    return {"status": "success", "calibration": saved, "persisted": bool(saved),
            "active": bool(saved and cfg and matrices_match),
            "storage": "postgresql.public.cameras", "save_id": (saved or {}).get("save_id"),
            "fms_frame": {"origin_x": fms_bridge.origin_x, "origin_y": fms_bridge.origin_y, "layout_depth": fms_bridge.layout_depth}}

@app.get("/api/deepcalib/status")
async def get_deepcalib_status():
    return {"status": "success", "deepcalib": deepcalib_status()}

@app.post("/api/camera/{cam_id}/calibration/measure")
async def measure_camera_calibration(cam_id: str, request: CalibrationMeasureRequest):
    if cam_id not in cameras:
        raise HTTPException(status_code=404, detail="Camera not found")
    config = camera_calibrator.get_config(cam_id)
    if not config or not config.get("matrix"):
        raise HTTPException(status_code=409, detail="Camera này chưa được hiệu chuẩn. Hãy lưu Calibration trước khi đo.")
    if request.calibration_save_id and request.calibration_save_id != config.get("save_id"):
        raise HTTPException(409, "Hiệu chuẩn đã thay đổi. Hãy tải lại ảnh và thước đo.")
    try:
        from core.calibration_measurement import measure_calibrated_polyline
        return {"status": "success", "camera_id": cam_id, **measure_calibrated_polyline(config, request.points, request.points_space)}
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

@app.post("/api/camera/{cam_id}/calibration/auto/start")
async def start_auto_camera_calibration(cam_id: str, request: AutoCalibrationStartRequest):
    if cam_id not in cameras:
        raise HTTPException(status_code=404, detail="Camera not found")
    status = online_robot_calibration.start(
        cam_id,
        auto_apply=request.auto_apply,
        time_offset_ms=request.time_offset_ms,
        reset=request.reset,
    )
    return {"status": "success", "auto_calibration": status}

@app.post("/api/camera/{cam_id}/calibration/auto/stop")
async def stop_auto_camera_calibration(cam_id: str):
    if cam_id not in cameras:
        raise HTTPException(404, "Camera not found")
    return {"status": "success", "auto_calibration": await asyncio.to_thread(online_robot_calibration.stop, cam_id)}

@app.get("/api/camera/{cam_id}/calibration/auto/status")
async def get_auto_camera_calibration_status(cam_id: str):
    if cam_id not in cameras:
        raise HTTPException(404, "Camera not found")
    return {"status": "success", "auto_calibration": online_robot_calibration.status(cam_id)}

@app.post("/api/camera/{cam_id}/calibration/auto/apply")
async def apply_auto_camera_calibration(cam_id: str):
    if cam_id not in cameras:
        raise HTTPException(status_code=404, detail="Camera not found")
    try:
        config = await asyncio.to_thread(online_robot_calibration.apply, cam_id)
    except (ValueError, RuntimeError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"status": "success", "config": config, "auto_calibration": online_robot_calibration.status(cam_id)}

@app.get("/api/calibration/map-overview")
async def get_map_overview():
    calibrations = []
    for cam_id, cam in cameras.items():
        config = camera_calibrator.get_config(cam_id)
        if config:
            calibrations.append({"cam_id": cam_id, "name": cam["name"], "calibration": config})
    return {"status": "success", "calibrations": calibrations}

# --- BEHAVIOR RULES (ROI & TRIPWIRES) API ---
@app.post("/api/camera/{cam_id}/rules")
@app.post("/api/camera/{cam_id}/roi")
async def save_camera_rules(cam_id: str, req: SaveRulesRequest):
    if cam_id not in cameras:
        raise HTTPException(status_code=404, detail="Camera not found")
        
    rules_dict_list = []
    db_manager.delete_rules_by_camera(cam_id)
    for r in req.rules:
        r_dict = {
            "id": r.id,
            "cam_id": cam_id,
            "type": r.type,
            "name": r.name,
            "points": r.points,
            "camera_points": r.camera_points or [],
            "fms_points": r.fms_points or [],
            "target_objects": r.target_objects,
            "threshold": r.threshold,
            "direction": r.direction,
            "coordinate_space": r.coordinate_space or "camera"
        }
        db_manager.save_rule(r_dict)
        rules_dict_list.append(r_dict)
        
    behavior_engine.set_rules(cam_id, rules_dict_list)
    return {"status": "success", "rules_count": len(rules_dict_list)}

@app.get("/api/camera/{cam_id}/rules")
async def get_camera_rules(cam_id: str):
    db_manager.delete_invalid_occupancy_rules(cam_id)
    rules = db_manager.get_rules_by_camera(cam_id)
    return {"status": "success", "rules": rules}

@app.delete("/api/camera/{cam_id}/rules/{rule_id}")
async def delete_camera_rule(cam_id: str, rule_id: str):
    db_manager.delete_rule(rule_id)
    rules = db_manager.get_rules_by_camera(cam_id)
    behavior_engine.set_rules(cam_id, rules)
    return {"status": "success", "rules_count": len(rules)}

# --- TARGET REGISTRY & ON-DEMAND ROBOT IDENTIFICATION API ---
class RegisterTargetRequest(BaseModel):
    cam_id: str
    track_id: int = 0
    label: str
    category: Optional[str] = "robot"
    reid_vector: Optional[List[float]] = None

class RegisterCropTargetRequest(BaseModel):
    cam_id: str
    label: str
    category: Optional[str] = "robot"
    crop_image: str
    bbox: Optional[List[float]] = None
    mask: Optional[dict] = None
    frame_image: Optional[str] = None

class MaskPreviewRequest(BaseModel):
    cam_id: str
    label: str
    category: str
    frame_image: str
    bbox: List[float]
    points: List[List[float]] = []
    point_labels: List[int] = []

def _validate_mask_bbox(bbox):
    import math
    if not bbox or len(bbox) != 4 or not all(math.isfinite(value) for value in bbox):
        raise ValueError("BBox không hợp lệ.")
    left, top, width, height = bbox
    if min(left, top) < 0 or min(width, height) <= 0 or left + width > 1.00001 or top + height > 1.00001:
        raise ValueError("BBox phải nằm trong ảnh (0..1).")

@app.get("/api/registry/mask/status")
async def mask_status():
    return dict(registered_target_mask_segmenter.status(), enabled=False, mode="uploaded_model_only",
                reason="Label đã ngừng sử dụng; Monitor dùng YOLO-Seg native từ model đã deploy.",
                runtime=custom_detector.status(), identity_metrics=registered_identity_metrics.status())

@app.post("/api/registry/mask/preview", dependencies=[Depends(reject_legacy_label_registration)])
async def mask_preview(req: MaskPreviewRequest):
    import math
    if req.category not in {"robot", "rack"} or not req.label.strip():
        raise HTTPException(400, "Chỉ tạo mask cho nhãn robot/kệ đang đăng ký; không áp dụng cho người.")
    if req.cam_id not in cameras:
        raise HTTPException(404, "Camera không tồn tại.")
    try:
        _validate_mask_bbox(req.bbox)
        if len(req.points) != len(req.point_labels) or len(req.points) > 64:
            raise ValueError("Số điểm chỉnh mask không hợp lệ.")
        if any(len(point) != 2 or any(not math.isfinite(value) or not 0 <= value <= 1 for value in point) for point in req.points):
            raise ValueError("Điểm chỉnh mask phải nằm trong ảnh.")
        if any(value not in {0, 1} for value in req.point_labels):
            raise ValueError("Điểm chỉnh mask phải là thêm (1) hoặc bỏ (0).")
        frame = decode_frame(req.frame_image)
    except (ValueError, TypeError) as exc:
        raise HTTPException(400, str(exc))
    try:
        mask = await asyncio.to_thread(registered_target_mask_segmenter.preview, frame, req.bbox, req.points, req.point_labels)
    except Exception as exc:
        raise HTTPException(503, f"Không tạo được mask: {exc}")
    if not mask:
        raise HTTPException(422, "Chưa tách được vật. Hãy khoanh sát vật và thêm điểm chỉnh.")
    return {"mask": mask}

def _vector_from_crop_image(crop_image: str, category: Optional[str] = None) -> List[float]:
    """Encode crop image thành 512-D vector.

    - Robot / Rack → CLIP ViT-B/32 (hoạt động tốt hơn với góc nhìn dị biệt)
    - Person / default → NvDCF Re-ID ONNX (trửng hợp két live vector từ tracker)
    """
    payload = crop_image.split(",", 1)[1] if "," in crop_image else crop_image
    raw = base64.b64decode(payload)
    import numpy as np
    img_arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        raise ValueError("Invalid crop image")

    from core.reid_encoder import encode_crop
    vector = encode_crop(img)
    if not vector or len(vector) != 512:
        raise ValueError("Encoder did not return a 512-D vector")
    return vector

@app.post("/api/registry/register", dependencies=[Depends(reject_legacy_label_registration)])
async def register_target_api(req: RegisterTargetRequest):
    from src.controller.registry import target_registry
    label = _canonical_identity_label(req.label, req.category)
    vector = req.reid_vector
    if not vector or len(vector) != 512:
        # Check active live track vector from global_reid
        try:
            vector = global_reid.get_track_vector(req.track_id)
        except Exception:
            vector = None
    if not vector:
        raise HTTPException(
            status_code=409,
            detail="Chưa có vector Re-ID GPU cho track này. Hãy chờ DeepStream NvDCF xuất metadata rồi đăng ký lại nhãn."
        )
        
    success = target_registry.register_target(
        label,
        vector,
        cam_id=req.cam_id,
        category=req.category,
        embedding_type="reid_512",
    )
    return {"status": "success" if success else "failed", "label": label}

@app.post("/api/registry/register-crop", dependencies=[Depends(reject_legacy_label_registration)])
def register_crop_target_api(req: RegisterCropTargetRequest):
    from src.controller.registry import target_registry
    label = _canonical_identity_label(req.label, req.category)
    if not label:
        raise HTTPException(status_code=400, detail="Label không được để trống.")
    cam_info = cameras.get(req.cam_id)
    if not cam_info:
        persisted = next((camera for camera in db_manager.get_all_cameras() if camera["id"] == req.cam_id), None)
        cam_info = _camera_payload_from_db(persisted, status="offline") if persisted else None
    if not cam_info:
        raise HTTPException(status_code=404, detail="Camera không tồn tại.")
    if req.bbox is not None:
        try:
            _validate_mask_bbox(req.bbox)
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc))
    try:
        mediamtx_client.ensure_path(req.cam_id, cam_info["rtsp_url"])
    except requests.RequestException as exc:
        raise HTTPException(status_code=503, detail="Relay camera chưa sẵn sàng. Vui lòng thử đăng ký lại.") from exc
    if req.mask is not None:
        try:
            if req.category not in {"robot", "rack"} or not req.frame_image:
                raise ValueError("Mask chỉ áp dụng cho robot/kệ và cần frame gốc.")
            _validate_mask_bbox(req.bbox)
            req.mask = validate_mask(req.mask)
            decode_frame(req.frame_image)
        except (ValueError, TypeError) as exc:
            raise HTTPException(400, str(exc))
    is_mask_identity = req.category in {"robot", "rack"}
    if is_mask_identity and (req.mask is None or not req.frame_image):
        raise HTTPException(400, "Hãy tạo và xác nhận mask trên ảnh trước khi lưu góc nhìn robot/kệ.")
    live_obj = _find_live_object_for_crop(req.cam_id, req.bbox) if not is_mask_identity else None
    if live_obj and req.category == "person" and not _is_person_object(live_obj):
        live_obj = None
    live_vector = live_obj.get("reid_vector") if live_obj else None
    try:
        if is_mask_identity:
            vector = registered_target_mask_segmenter.describe(decode_frame(req.frame_image), req.mask)
            chosen_embedding_type = "sam2_mask_512"
        elif isinstance(live_vector, list) and len(live_vector) == 512:
            vector = live_vector
            chosen_embedding_type = "reid_512"
        else:
            vector = _vector_from_crop_image(req.crop_image, category=req.category)
            chosen_embedding_type = "reid_512"
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Không thể trích xuất đặc trưng từ ảnh đã chọn: {exc}")

    try:
        success = target_registry.register_target(
            label,
            vector,
            cam_id=req.cam_id,
            category=req.category,
            bbox=req.bbox,
            crop_image=req.crop_image,
            embedding_type=chosen_embedding_type,
            mask=req.mask,
            frame_image=req.frame_image if req.mask is not None else None,
        )
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    if not success:
        raise HTTPException(422, "Không lưu được mẫu đặc trưng hợp lệ; nhãn hiện tại không thay đổi.")

    assigned_track = None
    identity_status = None
    if success:
        if is_mask_identity:
            try:
                identity_status = registered_target_mask_segmenter.learn_identity(label, req.cam_id, req.frame_image, req.mask, vector)
            except Exception as exc:
                identity_status = {"state": "pending", "last_error": str(exc)}
        else:
            identity_status = registered_identity_metrics.status()
        if cam_info and req.bbox:
            template_identity_tracker_manager.add_target(
                req.cam_id,
                cam_info["rtsp_url"],
                label,
                req.category or "object",
                req.bbox,
                crop_image=req.crop_image,
                mask=req.mask,
                frame_image=req.frame_image if req.mask is not None else None,
                reid_vector=vector if not is_mask_identity else None,
                reid_vectors=[vector] if vector and not is_mask_identity else None,
                identity_vector=vector if is_mask_identity else None,
            )

        if live_obj:
            global_id = live_obj.get("id")
            local_id = live_obj.get("local_id", global_id)
            canonical_global_id = identity_global_id(label, req.category, global_id)
            det_box = _object_bbox(live_obj) or req.bbox
            if target_registry.bind_label_to_track(label, req.cam_id, canonical_global_id, local_id, live_bbox=det_box):
                live_obj["label"] = label
                if canonical_global_id is not None:
                    live_obj["id"] = canonical_global_id
                assigned_track = {
                    "id": canonical_global_id,
                    "local_id": local_id,
                    "class": live_obj.get("class"),
                    "bbox": det_box,
                }

    fms_id = robot_number_from_label(label) if (req.category or "").lower() == "robot" or label.lower().startswith("robot") else None
    target_key = target_registry._target_key(label, req.cam_id)
    target_data = target_registry.targets.get(target_key) or target_registry.targets.get(label)
    samples_cnt = target_data.get("samples_count", 1) if target_data else 1
    
    if samples_cnt > 1:
        msg = f"Đã bổ sung góc nhìn cho '{label}' ({samples_cnt} góc nhìn)."
    elif assigned_track:
        msg = f"Đã khóa nhãn '{label}' (1 góc nhìn) trực tiếp vào Live Track #{assigned_track.get('id') or assigned_track.get('local_id')}!"
    else:
        msg = f"Đã lưu nhãn '{label}' (1 góc nhìn) vào Registry. AI GPU Tracker đang quét frame tiếp theo để tự động bám vết!"

    return {
        "status": "success" if success else "failed",
        "label": label,
        "category": req.category,
        "fms_robot_id": fms_id,
        "bbox": req.bbox,
        "samples_count": samples_cnt,
        "assigned_track": assigned_track,
        "identity": identity_status,
        "message": msg + " Đã lưu mẫu trên SSD; đặc trưng được dùng kiểm tra danh tính và học triplet khi đủ mẫu cùng vật và khác vật."
    }

@app.get("/api/registry/targets")
async def get_registered_targets_api(cam_id: Optional[str] = None):
    from src.controller.registry import target_registry
    targets = target_registry.get_all_targets(cam_id=cam_id)
    return {"status": "success", "targets": targets, "identity": registered_target_mask_segmenter.status().get("identity"),
            "identity_metrics": registered_identity_metrics.status()}

@app.delete("/api/registry/target/{label}")
def delete_registered_target_api(label: str, cam_id: Optional[str] = None):
    from src.controller.registry import target_registry
    success = target_registry.remove_target(label, cam_id=cam_id)
    template_identity_tracker_manager.remove_target(label, cam_id=cam_id)
    registered_target_mask_segmenter.forget_identity(label, cam_id)
    return {"status": "success" if success else "failed"}

@app.delete("/api/camera/{cam_id}/registry/target/{label}")
@app.delete("/api/registry/target/{label}/camera/{cam_id}")
def delete_registered_target_camera_sample_api(cam_id: str, label: str):
    from src.controller.registry import target_registry
    success = target_registry.remove_target(label, cam_id=cam_id)
    template_identity_tracker_manager.remove_target(label, cam_id=cam_id)
    registered_target_mask_segmenter.forget_identity(label, cam_id)
    return {"status": "success" if success else "failed"}

# --- ON-DEMAND ANALYTICS & EVENTS API (PostgreSQL Storage) ---
@app.get("/api/analytics/dashboard")
async def get_dashboard_analytics():
    stats = await asyncio.to_thread(db_manager.get_dashboard_stats, active_cameras_count=len(cameras))
    stats["recent_events"] = await asyncio.to_thread(recording_store.attach_event_recordings, stats["recent_events"])
    return stats

@app.get("/api/analytics/classes")
@app.get("/api/model/classes")
async def get_model_classes():
    labels_file = os.path.join(os.path.dirname(__file__), "models_config", "labels.txt")
    classes = []
    try:
        with open(labels_file, "r") as f:
            classes = [line.strip() for line in f.readlines() if line.strip()]
    except Exception as e:
        import logging
        logging.error(f"Error reading labels.txt: {e}")
        classes = ["robot", "rack"]
    for model in await asyncio.to_thread(model_registry.list):
        if model["state"] == "ready":
            classes.extend(model["labels"])
    return {"status": "success", "classes": list(dict.fromkeys(classes))}

@app.get("/api/events/list")
async def list_events(
    limit: int = Query(50, ge=1, le=500),
    rule_type: Optional[str] = None,
    cam_id: Optional[str] = None
):
    events = db_manager.get_events_list(limit=limit, rule_type=rule_type, cam_id=cam_id)
    return {"status": "success", "events": events}

@app.get("/api/tracks/{global_id}/history")
async def get_track_history(global_id: int):
    # Try in-memory track history first, fallback to DB
    mem_history = global_reid.get_track_history(global_id)
    if mem_history:
        return {"status": "success", "source": "realtime", "data": mem_history}
        
    db_history = db_manager.get_global_track_journey(global_id)
    return {
        "status": "success",
        "source": "database",
        "data": {
            "global_id": global_id,
            "trajectory": db_history
        }
    }

@app.get("/api/tracks/deep-tracking")
async def get_deep_tracking_status():
    """Inspects Deep Tracking state, active tracks, fast-motion tracks, and dormant/recovered tracks."""
    return {
        "status": "success",
        "active_tracks": global_reid.get_active_tracks(),
        "dormant_tracks": global_reid.get_dormant_tracks(),
        "total_active": len(global_reid.get_active_tracks()),
        "total_dormant": len(global_reid.get_dormant_tracks()),
        "reid_sim_threshold": global_reid.sim_threshold,
        "max_time_gap": global_reid.max_time_gap
    }

@app.get("/api/debug/pipeline")
async def debug_pipeline():
    """Debug endpoint: inspect DeepStream pipeline state and source mappings."""
    manager = deepstream_manager
    return {
        "is_running": manager.is_running,
        "pipeline_exists": manager.pipeline is not None,
        "sources_count": len(manager.sources),
        "cam_id_to_source_id": dict(manager.cam_id_to_source_id),
        "source_id_to_cam_id": dict(manager.source_id_to_cam_id),
        "debug_frame_count": getattr(manager, '_debug_frame_count', 0),
        "cameras_in_memory": list(cameras.keys()),
        "cpu_tracker_fallback_enabled": ENABLE_CPU_TRACKER_FALLBACK,
        "detector_model": os.getenv("DEEPSTREAM_ONNX_FILE"),
        "detector_engine": os.getenv("DEEPSTREAM_ENGINE_FILE"),
        "detector_classes": ["person"],
        "pose_enabled": manager.is_running and manager.pipeline is not None,
        "pose_backend": "deepstream_tensorrt" if manager.pipeline is not None else "disabled",
        "pose_counts": getattr(manager, "_pose_counts", {}),
        "cpu_tracker_active_cameras": person_tracker_manager.active_cameras(),
        "template_identity_active_cameras": template_identity_tracker_manager.active_cameras(),
        "template_identity_state": template_identity_tracker_manager.debug_state(),
        "legacy_label_tracking_enabled": False,
        "sam2_mode": "uploaded_model_only",
        "custom_detector": custom_detector.status(),
        "metadata_transport": metadata_broadcaster.status(),
    }

@app.get("/api/debug/metadata")
async def debug_metadata(cam_id: Optional[str] = None):
    """Inspect latest live objects used by Label registration and Monitor overlay."""
    if cam_id:
        return {
            "status": "success",
            "cam_id": cam_id,
            "age_sec": round(time.time() - latest_metadata_at_by_cam.get(cam_id, 0), 3) if cam_id in latest_metadata_at_by_cam else None,
            "objects": latest_objects_by_cam.get(cam_id, []),
        }
    return {
        "status": "success",
        "cameras": {
            c_id: {
                "age_sec": round(time.time() - latest_metadata_at_by_cam.get(c_id, 0), 3),
                "objects_count": len(objects),
                "objects": objects,
            }
            for c_id, objects in latest_objects_by_cam.items()
        }
    }

@app.get("/api/debug/registry-bindings")
async def debug_registry_bindings():
    """Explain why each registered label is or is not attached to a live track."""
    from src.controller.registry import target_registry
    diagnostics = []
    for target in target_registry.get_all_targets():
        label = target["label"]
        target_cam = target.get("last_cam")
        match = _best_live_match_for_crop(target_cam, target.get("bbox")) if target_cam else {
            "score": 0.0,
            "object": None,
            "reason": "target has no camera",
        }
        assignments = target_registry.get_assignments_for_label(label)
        diagnostics.append({
            "label": label,
            "category": target.get("category"),
            "cam_id": target_cam,
            "samples_count": target.get("samples_count"),
            "stored_bbox": target.get("bbox"),
            "metadata_age_sec": round(time.time() - latest_metadata_at_by_cam.get(target_cam, 0), 3) if target_cam in latest_metadata_at_by_cam else None,
            "live_objects_count": len(latest_objects_by_cam.get(target_cam, [])) if target_cam else 0,
            "assignments": assignments,
            "best_live_match": match,
            "is_bound": bool(assignments["local"] or assignments["global"]),
        })
    return {"status": "success", "labels": diagnostics}

# --- 3D DIGITAL TWIN & MULTI-ENTITY REAL-TIME ENDPOINTS ---
@app.websocket("/ws/digital_twin")
async def websocket_digital_twin_endpoint(websocket: WebSocket):
    """WebSocket endpoint streaming 15Hz 3D Digital Twin multi-entity telemetry (Robots, Persons, Racks)"""
    from src.server.digital_twin_bridge import digital_twin_bridge
    from starlette.websockets import WebSocketDisconnect
    await digital_twin_bridge.register_client(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        digital_twin_bridge.unregister_client(websocket)
    except Exception:
        digital_twin_bridge.unregister_client(websocket)

@app.get("/api/digital_twin/telemetry")
async def get_digital_twin_telemetry():
    """Get instant snapshot of all 3D entities (Robots, Persons, Racks)"""
    from src.server.digital_twin_bridge import digital_twin_bridge
    return digital_twin_bridge.build_telemetry_payload()

# --- FMS ROBOT 3D BRIDGE ENDPOINTS ---
@app.websocket("/ws")
@app.websocket("/ws/fms")
async def websocket_fms_endpoint(websocket: WebSocket):
    """WebSocket endpoint streaming 10Hz robot fleet telemetry from FMS"""
    await fms_bridge.handle_websocket(websocket)

@app.get("/api/fms/status")
async def get_fms_status():
    """Get FMS Bridge connection and runtime status"""
    return fms_bridge.get_status()

@app.get("/api/fms/robots")
async def get_fms_robots():
    """Get list and real-time state of all AGV/AMR robots from FMS"""
    return fms_bridge.get_robots()

@app.post("/api/fms/config")
async def update_fms_config(config: dict):
    """Dynamically update FMS connection parameters (IP, Port, Coordinates origin)"""
    return fms_bridge.update_config(config)

@app.get("/api/fms/layout")
async def get_fms_layout(compact: bool = False):
    """Serve the 3D warehouse layout JSON"""
    layout_file = os.path.join(os.path.dirname(__file__), "warehouse_layout.json")
    if os.path.exists(layout_file):
        with open(layout_file, "r", encoding="utf-8") as f:
            layout = json.load(f)
            if compact:
                return {key: layout.get(key) for key in ("id", "name", "units", "size", "origin_world", "slam_map")}
            return layout
    return {"error": "layout file not found"}

@app.get("/health")
async def health_check():
    """System health check including FMS bridge"""
    return {
        "status": "ok",
        "cameras_count": len(cameras),
        "fms_bridge": fms_bridge.get_status()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
