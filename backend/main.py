import faulthandler
faulthandler.enable()

import os
import sys
import uuid
import json
import asyncio
import base64
import time
import re
import requests
import cv2
from typing import Dict, List, Set, Optional, Tuple, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.database import db_manager
from utils.circular_logger import app_logger
from core.deepstream_engine import deepstream_manager, sanitize_rtsp_url
from core.camera_calibrator import camera_calibrator
from core.behavior_analytics import behavior_engine
from core.reid_matcher import global_reid
from core.fms_bridge import fms_bridge
from core.person_tracker import person_tracker_manager
from core.template_identity_tracker import template_identity_tracker_manager
from core.identity_utils import identity_global_id, robot_number_from_label
from core.metadata_fusion import MetadataFusion
from core.registered_target_mask import decode_frame, validate_mask, registered_target_mask_segmenter

metadata_fusion = MetadataFusion()


app = FastAPI(title="RTC VMS (R-SkyView) - Real-time Decoupled Multi-Camera Analytics", version="3.0")

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

MEDIAMTX_API = os.getenv("MEDIAMTX_API", "http://127.0.0.1:9997/v3/config/paths")
ENABLE_CPU_TRACKER_FALLBACK = os.getenv("ENABLE_CPU_TRACKER_FALLBACK", "1").lower() not in {"0", "false", "no"}

# In-memory registry of active cameras
cameras: Dict[str, dict] = {}


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

def _start_template_identity_tracker(cam_id: str, rtsp_url: str):
    try:
        template_identity_tracker_manager.add_camera(cam_id, sanitize_rtsp_url(rtsp_url))
    except Exception as e:
        print(f"[IdentityTemplate] Could not start tracker for {cam_id}: {e}", flush=True)

def _restore_template_targets_for_camera(cam_id: str, rtsp_url: str):
    try:
        from src.controller.registry import target_registry
        for target in target_registry.get_all_targets(cam_id=cam_id):
            samples = target_registry.get_template_samples(target["label"], cam_id)
            for sample in samples:
                template_identity_tracker_manager.add_target(
                    cam_id,
                    sanitize_rtsp_url(rtsp_url),
                    target["label"],
                    sample.get("category") or target.get("category", "object"),
                    sample["bbox"],
                    crop_image=sample.get("crop_image"),
                    mask=sample.get("mask"),
                    frame_image=sample.get("frame_image"),
                )
    except Exception as e:
        print(f"[IdentityTemplate] Could not restore targets for {cam_id}: {e}", flush=True)

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
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL",   "gemini-3.6-flash")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:streamGenerateContent?key={GEMINI_API_KEY}&alt=sse"

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
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream("POST", GEMINI_URL, json=payload,
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
connected_metadata_ws: Set[WebSocket] = set()
connected_event_ws: Set[WebSocket] = set()
loop: Optional[asyncio.AbstractEventLoop] = None
latest_objects_by_cam: Dict[str, List[dict]] = {}
latest_metadata_at_by_cam: Dict[str, float] = {}

# --- REQUEST MODELS ---
class CameraAddRequest(BaseModel):
    name: str
    rtsp_url: str

class CalibrationRequest(BaseModel):
    src_points: List[List[float]] # 4 points normalized [[x,y]...]
    dst_points: List[List[float]] # 4 points floor map [[X,Y]...]
    cam_x: Optional[float] = None
    cam_y: Optional[float] = None
    cam_z: Optional[float] = None
    yaw: Optional[float] = None

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
    now = time.time()
    if "streams" in payload and isinstance(payload["streams"], list):
        for st in payload["streams"]:
            cam_id = st.get("cam_id")
            objects = st.get("objects")
            if cam_id and isinstance(objects, list):
                cam_key = str(cam_id)
                incoming = [dict(obj) for obj in objects if isinstance(obj, dict)]
                for obj in incoming:
                    category = (obj.get("category") or obj.get("class") or "object").lower()
                    obj.setdefault("category", category)
                    if "world_position" not in obj:
                        try:
                            bbox = [obj.get("x", 0.0), obj.get("y", 0.0), obj.get("w", 0.0), obj.get("h", 0.0)]
                            spatial = camera_calibrator.project_ground_point(
                                cam_key, bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3]
                            )
                            obj["floor_x"] = spatial["x"]
                            obj["floor_y"] = spatial["z"]
                            obj["world_position"] = [spatial["x"], spatial["y"], spatial["z"]]
                            obj["spatial_valid"] = spatial["valid"]
                            obj["spatial_source"] = spatial["source"]
                            obj["spatial_confidence"] = spatial["confidence"]
                        except Exception:
                            pass
                merged = metadata_fusion.update(cam_key, payload.get("source", "deepstream"), incoming)
                latest_objects_by_cam[cam_key] = merged
                st["objects"] = merged
                latest_metadata_at_by_cam[str(cam_id)] = now
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
        telemetry = digital_twin_bridge.build_telemetry_payload()
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

    if not connected_metadata_ws:
        return
    msg = json.dumps(payload)
    asyncio.run_coroutine_threadsafe(_broadcast_to_set(connected_metadata_ws, msg), loop)


def broadcast_template_metadata_sync(payload: dict):
    """Publish registered crop identities alongside DeepStream tracks."""
    broadcast_metadata_sync(payload)

def broadcast_event_sync(event_payload: dict):
    if not connected_event_ws or not loop:
        return
    msg = json.dumps(event_payload)
    asyncio.run_coroutine_threadsafe(_broadcast_to_set(connected_event_ws, msg), loop)

async def _broadcast_to_set(target_set: Set[WebSocket], msg: str):
    disconnected = set()
    for ws in list(target_set):
        try:
            await ws.send_text(msg)
        except Exception:
            disconnected.add(ws)
    for ws in disconnected:
        target_set.discard(ws)

@app.on_event("startup")
async def startup_event():
    global loop
    loop = asyncio.get_running_loop()

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
    template_identity_tracker_manager.metadata_callback = broadcast_template_metadata_sync

    # Pre-populate persisted cameras from DB on startup.
    # A configured camera is only removed by the delete API, never by offline checks.
    db_cams = db_manager.get_all_cameras()
    for c in db_cams:
        cam_id = c["id"]
        cam = _camera_payload_from_db(c, status="offline")
        cameras[cam_id] = cam

        try:
            res = requests.post(f"{MEDIAMTX_API}/add/{cam_id}", json={
                "source": cam["rtsp_url"],
                "sourceOnDemand": False,
                "rtspTransport": "tcp"
            }, timeout=2)
            if res.status_code not in (200, 201):
                requests.post(f"{MEDIAMTX_API}/patch/{cam_id}", json={"source": cam["rtsp_url"]}, timeout=2)
        except Exception as e:
            print(f"[MediaMTX] Startup proxy path registration failed for {cam_id}: {e}")

    initial_sources = []
    offline_cameras = []

    for cam_id, c in cameras.items():
        rtsp_url = c["rtsp_url"]

        calib_pts = c.get("calibration")
        if calib_pts and isinstance(calib_pts, dict):
            camera_calibrator.set_calibration(
                cam_id,
                calib_pts.get("src_points", []),
                calib_pts.get("dst_points", []),
                c.get("cam_x"),
                c.get("cam_y"),
                c.get("cam_z"),
                c.get("yaw")
            )

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
            initial_sources.append((cam_id, rtsp_url))
            print(f"[Main] Camera {cam_id} reachable - will add to pipeline.", flush=True)
            _start_cpu_tracker_fallback(cam_id, rtsp_url)
            # add_target starts the camera tracker only when trusted label crops exist.
            _restore_template_targets_for_camera(cam_id, rtsp_url)
        else:
            c["status"] = "offline"
            offline_cameras.append((cam_id, rtsp_url))
            print(f"[Main] Camera {cam_id} unreachable at startup - kept in DB and will retry.", flush=True)

    deepstream_manager.start(initial_sources=initial_sources if initial_sources else None)

    if offline_cameras:
        asyncio.ensure_future(_retry_offline_cameras(offline_cameras))

    print(f"[Main] Startup complete. {len(initial_sources)} cameras online, {len(offline_cameras)} persisted offline/retrying.", flush=True)


@app.on_event("shutdown")
async def shutdown_event():
    try:
        await fms_bridge.stop()
    except Exception as e:
        logging.error(f"Error stopping FMS Bridge: {e}")


async def _retry_offline_cameras(offline_list: list, interval: int = 30):
    """Background task: retry adding offline cameras to the pipeline every `interval` seconds."""
    remaining = list(offline_list)
    while remaining:
        await asyncio.sleep(interval)
        still_offline = []
        for cam_id, rtsp_url in remaining:
            if cam_id not in cameras:
                continue

            try:
                from check_rtsp import is_rtsp_valid_async
                is_reachable = await is_rtsp_valid_async(rtsp_url, timeout=5)
            except Exception:
                is_reachable = False

            if is_reachable:
                cameras[cam_id]["status"] = "online"
                print(f"[Main] Camera {cam_id} is now reachable - adding to pipeline.", flush=True)
                deepstream_manager.add_source(cam_id, rtsp_url)
                _start_cpu_tracker_fallback(cam_id, rtsp_url)
                _start_template_identity_tracker(cam_id, rtsp_url)
                _restore_template_targets_for_camera(cam_id, rtsp_url)
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
    connected_metadata_ws.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        connected_metadata_ws.discard(websocket)
    except Exception:
        connected_metadata_ws.discard(websocket)

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
        res = requests.post(f"{MEDIAMTX_API}/add/{cam_id}", json={
            "source": clean_url,
            "sourceOnDemand": False,
            "rtspTransport": "tcp"
        }, timeout=1.5)
        if res.status_code not in (200, 201):
            requests.post(f"{MEDIAMTX_API}/patch/{cam_id}", json={"source": clean_url}, timeout=1.0)
    except Exception as e:
        print(f"[MediaMTX] Note: proxy path registration: {e}")

    # 4. A configured camera is persisted even if the RTSP stream is temporarily offline.
    if is_valid:
        deepstream_manager.add_source(cam_id, clean_url)
        _start_cpu_tracker_fallback(cam_id, clean_url)
        _start_template_identity_tracker(cam_id, clean_url)
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
        cam["rtsp_url"] = clean_url
        cam["status"] = "offline"
        try:
            requests.post(f"{MEDIAMTX_API}/patch/{cam_id}", json={"source": clean_url}, timeout=2)
        except Exception:
            pass
        deepstream_manager.delete_source(cam_id)
        _stop_cpu_tracker_fallback(cam_id)
        _stop_template_identity_tracker(cam_id)

        try:
            from check_rtsp import is_rtsp_valid_async
            is_valid = await is_rtsp_valid_async(clean_url, timeout=2.5)
        except Exception:
            is_valid = False

        if is_valid:
            cam["status"] = "online"
            deepstream_manager.add_source(cam_id, clean_url)
            _start_cpu_tracker_fallback(cam_id, clean_url)
            _start_template_identity_tracker(cam_id, clean_url)
            _restore_template_targets_for_camera(cam_id, clean_url)
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

    if cam_id in cameras:
        deepstream_manager.delete_source(cam_id)
        _stop_cpu_tracker_fallback(cam_id)
        _stop_template_identity_tracker(cam_id)
    try:
        requests.post(f"{MEDIAMTX_API}/delete/{cam_id}", timeout=2)
    except Exception:
        pass

    db_manager.delete_camera(cam_id)
    cameras.pop(cam_id, None)
    return {"status": "success", "deleted_id": cam_id}

@app.get("/api/v1/streams/list")
@app.get("/api/camera/list")
async def list_cameras():
    persisted = {}
    for row in db_manager.get_all_cameras():
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

# --- CAMERA CALIBRATION API (2D-to-Floor-Map) ---
@app.post("/api/camera/{cam_id}/calibration")
async def save_camera_calibration(cam_id: str, calib: CalibrationRequest):
    if cam_id not in cameras:
        raise HTTPException(status_code=404, detail="Camera not found")
        
    success = camera_calibrator.set_calibration(cam_id, calib.src_points, calib.dst_points, calib.cam_x, calib.cam_y, calib.cam_z, calib.yaw)
    if not success:
        raise HTTPException(status_code=400, detail="Không thể tính ma trận biến đổi từ các điểm đã chọn")
        
    cfg = camera_calibrator.get_config(cam_id)
    if cfg:
        db_manager.save_calibration(cam_id, calib.src_points, calib.dst_points, cfg["matrix"], calib.cam_x, calib.cam_y, calib.cam_z, calib.yaw, cfg.get("fov_polygon"))
        cameras[cam_id]["calibration"] = cfg
        cameras[cam_id]["cam_x"] = calib.cam_x
        cameras[cam_id]["cam_y"] = calib.cam_y
        cameras[cam_id]["cam_z"] = calib.cam_z
        cameras[cam_id]["yaw"] = calib.yaw
        cameras[cam_id]["fov_polygon"] = cfg.get("fov_polygon")
        
    return {"status": "success", "config": cfg}

@app.get("/api/camera/{cam_id}/calibration")
async def get_camera_calibration(cam_id: str):
    cfg = camera_calibrator.get_config(cam_id)
    return {"status": "success", "calibration": cfg}

@app.get("/api/calibration/map-overview")
async def get_map_overview():
    calibrations = []
    for cam_id, cam in cameras.items():
        if "calibration" in cam and cam["calibration"]:
            calibrations.append({
                "cam_id": cam_id,
                "name": cam["name"],
                "calibration": cam["calibration"]
            })
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
    return registered_target_mask_segmenter.status()

@app.post("/api/registry/mask/preview")
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

@app.post("/api/registry/register")
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

@app.post("/api/registry/register-crop")
async def register_crop_target_api(req: RegisterCropTargetRequest):
    from src.controller.registry import target_registry
    label = _canonical_identity_label(req.label, req.category)
    if not label:
        raise HTTPException(status_code=400, detail="Label không được để trống.")
    if req.mask is not None:
        try:
            if req.category not in {"robot", "rack"} or not req.frame_image:
                raise ValueError("Mask chỉ áp dụng cho robot/kệ và cần frame gốc.")
            _validate_mask_bbox(req.bbox)
            req.mask = validate_mask(req.mask)
            decode_frame(req.frame_image)
        except (ValueError, TypeError) as exc:
            raise HTTPException(400, str(exc))
    # Prefer the vector emitted by the active DeepStream/NvDCF track. This is
    # the exact GPU embedding that will be compared on subsequent frames.
    live_obj = _find_live_object_for_crop(req.cam_id, req.bbox)
    live_vector = live_obj.get("reid_vector") if live_obj else None

    # The live detector/tracker emits NvDCF Re-ID embeddings. Prefer that exact
    # vector for every mobile/static target; never compare CLIP and Re-ID merely
    # because both happen to have 512 dimensions.
    try:
        if isinstance(live_vector, list) and len(live_vector) == 512:
            vector = live_vector
            chosen_embedding_type = "reid_512"
        else:
            vector = _vector_from_crop_image(req.crop_image, category=req.category)
            chosen_embedding_type = "reid_512"
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Không thể tạo vector Re-ID 512-D từ ảnh crop: {exc}")

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

    assigned_track = None
    if success:
        cam_info = cameras.get(req.cam_id)
        if not cam_info:
            persisted = next((c for c in db_manager.get_all_cameras() if c["id"] == req.cam_id), None)
            cam_info = _camera_payload_from_db(persisted, status="offline") if persisted else None
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
        "message": msg + (" Đã lưu mask và frame để bám vật theo các góc nhìn đã xác nhận." if req.mask is not None else "")
    }

@app.get("/api/registry/targets")
async def get_registered_targets_api(cam_id: Optional[str] = None):
    from src.controller.registry import target_registry
    targets = target_registry.get_all_targets(cam_id=cam_id)
    return {"status": "success", "targets": targets}

@app.delete("/api/registry/target/{label}")
async def delete_registered_target_api(label: str, cam_id: Optional[str] = None):
    from src.controller.registry import target_registry
    success = target_registry.remove_target(label, cam_id=cam_id)
    template_identity_tracker_manager.remove_target(label, cam_id=cam_id)
    return {"status": "success" if success else "failed"}

@app.delete("/api/camera/{cam_id}/registry/target/{label}")
@app.delete("/api/registry/target/{label}/camera/{cam_id}")
async def delete_registered_target_camera_sample_api(cam_id: str, label: str):
    from src.controller.registry import target_registry
    success = target_registry.remove_target(label, cam_id=cam_id)
    template_identity_tracker_manager.remove_target(label, cam_id=cam_id)
    return {"status": "success" if success else "failed"}

# --- ON-DEMAND ANALYTICS & EVENTS API (PostgreSQL Storage) ---
@app.get("/api/analytics/dashboard")
async def get_dashboard_analytics():
    stats = db_manager.get_dashboard_stats(active_cameras_count=len(cameras))
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
    return {"status": "success", "classes": classes}

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
        "pose_backend": "deepstream_tensorrt",
        "pose_counts": getattr(manager, "_pose_counts", {}),
        "cpu_tracker_active_cameras": person_tracker_manager.active_cameras(),
        "template_identity_active_cameras": template_identity_tracker_manager.active_cameras(),
        "template_identity_state": template_identity_tracker_manager.debug_state(),
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
async def get_fms_layout():
    """Serve the 3D warehouse layout JSON"""
    layout_file = os.path.join(os.path.dirname(__file__), "warehouse_layout.json")
    if os.path.exists(layout_file):
        with open(layout_file, "r", encoding="utf-8") as f:
            return json.load(f)
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
