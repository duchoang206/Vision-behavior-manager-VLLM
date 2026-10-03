"""REST API of the Building view inspection stations (plan §5.2)."""

import asyncio
import time
from typing import Any, Callable, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, Field, ValidationError

from core.inspection.baseline_store import MAX_BASELINES, BaselineCaptureError, BaselineLimitError
from core.inspection.config import OCC_CARFULL, InspectionConfig
from core.inspection.homography_rectifier import InvalidRoiError, order_clockwise
from core.inspection.live_detections import cell_detections
from core.inspection.occupancy import GOODS_ROLES
from core.inspection.recorded_frames import RecordedFrames, RecordingNotFound
from core.inspection.runtime import InspectionRuntime


class CaptureBackgroundRequest(BaseModel):
    rule_id: str = Field(min_length=1, max_length=96)
    points: List[List[float]] = Field(min_length=4, max_length=4)
    # Real size of the outlined floor cell (edge 1→2, edge 2→3); 1 m x 1 m by default.
    roi_width_mm: float = Field(1000.0, ge=200, le=10000, allow_inf_nan=False)
    roi_height_mm: float = Field(1000.0, ge=200, le=10000, allow_inf_nan=False)
    # One more empty-cell baseline: its lighting condition ("Sáng", "Đèn ca đêm" ...) and,
    # instead of the live stream, a moment of the recorded video (epoch seconds).
    label: str = Field("", max_length=48)
    at: Optional[float] = Field(None, gt=0, allow_inf_nan=False)
    target_objects: Optional[List[str]] = None
    force: bool = False


class InspectionTestRequest(BaseModel):
    points: List[List[float]] = Field(min_length=4, max_length=4)
    config: Dict[str, Any] = Field(default_factory=dict)
    rule_id: Optional[str] = Field(default=None, max_length=96)
    target_objects: Optional[List[str]] = None


class OperatorDecisionRequest(BaseModel):
    approve: bool
    operator: str = Field(default="", max_length=64)


def _size(frame) -> str:
    return f"{frame.shape[1]}×{frame.shape[0]}"


def create_inspection_router(runtime: InspectionRuntime, camera_exists: Callable[[str], bool],
                             recordings: Optional[RecordedFrames] = None) -> APIRouter:
    router = APIRouter(tags=["inspection"])

    def require_camera(cam_id: str) -> None:
        if not camera_exists(cam_id):
            raise HTTPException(status_code=404, detail="Camera not found")

    def require_roi(points: List[List[float]]) -> None:
        try:
            order_clockwise(points)
        except InvalidRoiError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    def occupied_warning(cam_id: str, request: CaptureBackgroundRequest, frame, cell) -> Optional[str]:
        """Against the baselines already captured, does image processing see an object on this frame?"""
        existing = runtime.store.load_all(cam_id, request.rule_id)
        if not existing:
            return None
        config = InspectionConfig(mode="CV_ONLY", roi_width_mm=cell[0], roi_height_mm=cell[1])
        occupancy = runtime.pipeline.run(frame, request.points, config, existing).get("occupancy") or {}
        if occupancy.get("state") != OCC_CARFULL:
            return None
        size = occupancy.get("cv_size_mm") or ["?", "?"]
        return (f"So với {len(existing)} ảnh nền đã có, xử lý ảnh thấy vùng khác biệt ~{size[0]}×{size[1]} mm. "
                "Nếu ô thật sự trống (chỉ khác ánh sáng) thì ảnh nền này giúp hết báo nhầm; "
                "nếu lúc đó ô có vật, hãy xoá ảnh nền này.")

    @router.post("/api/camera/{cam_id}/inspection/capture-background")
    async def capture_background(cam_id: str, request: CaptureBackgroundRequest):
        require_camera(cam_id)
        require_roi(request.points)
        cell = (request.roi_width_mm, request.roi_height_mm)
        extra: Dict[str, Any] = {}
        if request.at is not None:
            if recordings is None:
                raise HTTPException(status_code=503, detail="Hệ thống chưa bật ghi hình – không lấy được ảnh từ video")
            try:
                frames, found = await asyncio.to_thread(recordings.frames_at, cam_id, request.at, 5)
            except RecordingNotFound as exc:
                raise HTTPException(status_code=404, detail=str(exc))
            live = await asyncio.to_thread(runtime.frames.latest, cam_id, 5.0, 3.0)
            if live is not None and live[0].shape[:2] != frames[0].shape[:2]:
                raise HTTPException(status_code=422, detail=f"Video ghi hình {_size(frames[0])} khác độ phân giải "
                                                            f"luồng trực tiếp {_size(live[0])} – không dùng làm ảnh nền được")
            source, captured_at, extra["recording"] = "recording", found["at"], found
        else:
            frames = await asyncio.to_thread(runtime.frames.frames, cam_id, 5)
            if not frames:
                raise HTTPException(status_code=503, detail="Không lấy được khung hình từ camera")
            source, captured_at = "live", None
            if not request.force:
                # The camera's model sees one of the goods classes on the cell: it is not empty.
                try:
                    rect = runtime.pipeline.rectifier(request.points, cell)
                    goods = [d for d in cell_detections(runtime.live(cam_id), rect, frames[-1].shape, request.target_objects)
                             if d.role in GOODS_ROLES]
                except InvalidRoiError as exc:
                    raise HTTPException(status_code=422, detail=str(exc))
                if goods:
                    names = ", ".join(dict.fromkeys(f"{d.class_name} {d.confidence:.0%}" for d in goods))
                    raise HTTPException(status_code=409, detail=f"AI đang thấy {names} trong ô – ảnh nền phải chụp "
                                                                "khi ô trống hoàn toàn")
        try:
            warning = await asyncio.to_thread(occupied_warning, cam_id, request, frames[-1], cell)
            meta = await asyncio.to_thread(runtime.store.save, cam_id, request.rule_id, frames, request.points,
                                           cell, request.label, source, captured_at, extra)
        except BaselineLimitError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        except (BaselineCaptureError, InvalidRoiError) as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        return {"status": "success", "id": meta["id"], "label": meta["label"], "source": meta["source"],
                "captured_at": meta["captured_at"], "image_path": meta["image_path"], "timestamp": meta["timestamp"],
                "size_px": meta["size_px"], "cell_mm": meta["cell_mm"], "frames_used": meta["frames_used"],
                "brightness": meta["brightness"], "laplacian_var": meta["laplacian_var"],
                "count": meta["count"], "replaced": meta["replaced"], "warning": warning}

    @router.get("/api/camera/{cam_id}/inspection/baselines/{rule_id}")
    async def list_baselines(cam_id: str, rule_id: str):
        entries = await asyncio.to_thread(runtime.store.entries, cam_id, rule_id)
        coverage = await asyncio.to_thread(recordings.coverage, cam_id) if recordings is not None else None
        return {"status": "success", "max": MAX_BASELINES,
                "baselines": [{key: meta.get(key) for key in (
                    "id", "label", "source", "captured_at", "timestamp", "brightness", "laplacian_var",
                    "frames_used", "size_px", "cell_mm", "points", "frame_shape", "recording")} for meta in entries],
                "recordings": {"from": coverage[0], "to": coverage[1]} if coverage else None}

    @router.get("/api/camera/{cam_id}/inspection/baseline/{rule_id}")
    async def get_baseline(cam_id: str, rule_id: str, optional: bool = False, bid: Optional[str] = None):
        info = await asyncio.to_thread(runtime.store.info, cam_id, rule_id)
        image = await asyncio.to_thread(runtime.store.bev_jpeg, cam_id, rule_id, bid) if info else None
        if not image:
            if optional:
                # The Building view probes before the first capture: "none yet" is not an error.
                return Response(status_code=204, headers={"Cache-Control": "no-store"})
            raise HTTPException(status_code=404, detail="Chưa có ảnh nền cho quy tắc này")
        return Response(content=image, media_type="image/jpeg", headers={
            "X-Baseline-Timestamp": str(info.get("timestamp")), "X-Baseline-Size": "x".join(map(str, info["size_px"])),
            "X-Baseline-Count": str(info.get("count", 1)), "Cache-Control": "no-store",
            "Access-Control-Expose-Headers": "X-Baseline-Timestamp, X-Baseline-Size, X-Baseline-Count"})

    @router.delete("/api/camera/{cam_id}/inspection/baseline/{rule_id}/{baseline_id}")
    async def delete_baseline(cam_id: str, rule_id: str, baseline_id: str):
        try:
            removed = await asyncio.to_thread(runtime.store.delete, cam_id, rule_id, baseline_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        if not removed:
            raise HTTPException(status_code=404, detail="Không tìm thấy ảnh nền")
        remaining = await asyncio.to_thread(runtime.store.entries, cam_id, rule_id)
        return {"status": "success", "deleted_id": baseline_id, "count": len(remaining)}

    @router.post("/api/camera/{cam_id}/inspection/test")
    async def test_inspection(cam_id: str, request: InspectionTestRequest):
        started = time.perf_counter()
        require_camera(cam_id)
        require_roi(request.points)
        try:
            config = InspectionConfig.model_validate(request.config or {})
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors(include_url=False, include_context=False))
        rule_id = request.rule_id or await asyncio.to_thread(runtime.store.find, cam_id, request.points)
        baselines = await asyncio.to_thread(runtime.store.load_all, cam_id, rule_id) if rule_id else []
        packet = await asyncio.to_thread(runtime.frames.latest, cam_id, 1.0, 6.0)
        frame, frame_at = (packet[0], packet[1]) if packet else (None, None)
        report = await asyncio.to_thread(runtime.pipeline.run, frame, request.points, config, baselines, True,
                                         request.target_objects, runtime.live(cam_id))
        info = runtime.store.info(cam_id, rule_id) if rule_id else None
        report["baseline"] = ({"rule_id": rule_id, "timestamp": info.get("timestamp"), "count": info.get("count", 1)}
                              if info else None)
        report["frame"] = ({"width": int(frame.shape[1]), "height": int(frame.shape[0]),
                            "age_ms": round((time.time() - frame_at) * 1000.0, 1)} if frame is not None else None)
        report["config"] = config.model_dump()
        report["targets"] = request.target_objects or []
        report["server_latency_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
        return report

    @router.get("/api/inspection/states")
    async def inspection_states(cam_id: Optional[str] = None):
        return {"status": "success", "stations": runtime.states(cam_id),
                "published": list(runtime.published)[-20:]}

    @router.post("/api/camera/{cam_id}/inspection/{rule_id}/decision")
    async def operator_decision(cam_id: str, rule_id: str, request: OperatorDecisionRequest):
        try:
            station = runtime.apply_decision(cam_id, rule_id, request.approve, request.operator)
        except KeyError:
            raise HTTPException(status_code=404, detail="Không tìm thấy trạm kiểm định")
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        return {"status": "success", "station": station}

    return router
