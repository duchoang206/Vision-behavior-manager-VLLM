import asyncio
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Path
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from core.dashboard_auth import require_dashboard_user
from core.object_logic import infer_category


class SnapshotRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    camera_id: str = Field(min_length=1, max_length=128)


class PreviewRequest(SnapshotRequest):
    snapshot_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    bbox: list[float] = Field(min_length=4, max_length=4)
    points: list[list[float]] = Field(default_factory=list, max_length=32)
    point_labels: list[Literal[0, 1]] = Field(default_factory=list, max_length=32)

    @field_validator("bbox")
    @classmethod
    def valid_box(cls, value):
        left, top, width, height = value
        if min(left, top) < 0 or min(width, height) <= 0 or left + width > 1 or top + height > 1:
            raise ValueError("Khoanh vật trong ảnh.")
        return value

    @field_validator("points")
    @classmethod
    def valid_points(cls, value):
        if any(len(point) != 2 or any(not 0 <= coordinate <= 1 for coordinate in point) for point in value):
            raise ValueError("Điểm phải nằm trong ảnh.")
        return value


class SampleRequest(SnapshotRequest):
    preview_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    label: str = Field(min_length=1, max_length=120)
    class_name: str = Field(min_length=1, max_length=120)
    negative: bool = False


class SettingsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    require_labels: bool


def create_model_label_router(store, registry, detector):
    router = APIRouter(prefix="/api/models/{model_id}/labels", tags=["Model labels"], dependencies=[Depends(require_dashboard_user)])

    def model(model_id):
        try:
            return registry.get(model_id)
        except (ValueError, KeyError) as error:
            raise HTTPException(404, str(error)) from error

    async def command(model_id, action, request):
        try:
            return await asyncio.to_thread(detector.command, model_id, action, request)
        except (RuntimeError, TimeoutError, ValueError) as error:
            raise HTTPException(409, str(error)) from error

    @router.get("")
    def list_labels(model_id: str):
        model(model_id)
        return dict(store.list(model_id), runtime=detector.status())

    @router.post("/snapshot")
    async def snapshot(model_id: str, request: SnapshotRequest):
        model(model_id)
        return await command(model_id, "snapshot", request.model_dump())

    @router.post("/preview")
    async def preview(model_id: str, request: PreviewRequest):
        model(model_id)
        if len(request.points) != len(request.point_labels):
            raise HTTPException(422, "Số điểm và loại điểm không khớp.")
        return await command(model_id, "preview", request.model_dump())

    @router.post("/samples", status_code=201)
    async def save(model_id: str, request: SampleRequest, actor=Depends(require_dashboard_user)):
        row = model(model_id)
        if request.class_name not in row["labels"]:
            raise HTTPException(422, "Chọn đúng lớp trong model đã upload.")
        sample = await command(model_id, "sample", {"camera_id": request.camera_id, "preview_id": request.preview_id})
        category = infer_category(None, request.class_name)
        try:
            result = await asyncio.to_thread(store.save, model_id, request.label, request.class_name,
                                             category, request.negative, sample, actor)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        if "revision" in result:
            activation = None if request.negative else dict(camera_id=request.camera_id, sample_id=result['sample_id'])
            result["runtime_sync"] = await asyncio.to_thread(detector.reload_labels, model_id, result["revision"], activation)
        return result

    @router.get("/samples")
    def list_samples(model_id: str, label: str = Query(min_length=1, max_length=120),
                     limit: int = Query(default=12, ge=1, le=48), offset: int = Query(default=0, ge=0)):
        model(model_id)
        return store.samples(model_id, label, limit, offset)

    @router.get("/samples/{sample_id}/image")
    def sample_image(model_id: str, sample_id: str = Path(pattern=r"^[a-f0-9]{32}$")):
        model(model_id)
        try:
            return FileResponse(store.sample_image(model_id, sample_id), media_type="image/jpeg", headers={"Cache-Control": "no-store"})
        except KeyError as error:
            raise HTTPException(404, str(error)) from error

    @router.delete("/samples/{sample_id}")
    def delete_sample(model_id: str, sample_id: str = Path(pattern=r"^[a-f0-9]{32}$"), actor=Depends(require_dashboard_user)):
        model(model_id)
        try:
            result = store.delete_sample(model_id, sample_id, actor)
            if "revision" in result:
                result["runtime_sync"] = detector.reload_labels(model_id, result["revision"])
            return result
        except KeyError as error:
            raise HTTPException(404, str(error)) from error

    @router.delete("")
    def delete(model_id: str, label: str = Query(min_length=1, max_length=120), actor=Depends(require_dashboard_user)):
        model(model_id)
        result = store.delete(model_id, label, actor)
        if "revision" in result:
            result["runtime_sync"] = detector.reload_labels(model_id, result["revision"])
        return result

    @router.put("/settings")
    def settings(model_id: str, request: SettingsRequest, actor=Depends(require_dashboard_user)):
        model(model_id)
        result = store.settings(model_id, request.require_labels, actor)
        if "revision" in result:
            result["runtime_sync"] = detector.reload_labels(model_id, result["revision"])
        return result

    return router
