import asyncio
from typing import Literal
from zoneinfo import ZoneInfoNotFoundError

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from core.dashboard_auth import require_dashboard_user
from core.workflow_store import WorkflowConflict


class CaptureRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    camera_id: str = Field(min_length=1, max_length=128)


class Annotation(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    id: str = Field(min_length=1, max_length=128)
    class_name: str = Field(min_length=1, max_length=120)
    label: str = Field(default="", max_length=120)
    polygons: list[list[tuple[float, float]]] = Field(min_length=1, max_length=32)


class ReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    annotations: list[Annotation] = Field(default_factory=list, max_length=128)
    feedback: Literal["correct", "corrected", "reject"]
    complete: bool = False


class LearningSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    enabled: bool = False
    threshold: int = Field(default=500, ge=10, le=100000)
    daily_hour: int | None = Field(default=0, ge=0, le=23)
    timezone: str = Field(default="Asia/Bangkok", max_length=80)
    epochs: int = Field(default=5, ge=1, le=10)
    learning_rate: float = Field(default=.0001, ge=.000001, le=.001)
    batch: int = Field(default=2, ge=1, le=8)
    minimum_gain: float = Field(default=.001, ge=0, le=.1)
    max_class_drop: float = Field(default=.02, ge=0, le=.05)
    auto_promote: bool = False
    base_dataset: str = Field(default="base/data.yaml", min_length=1, max_length=250)


class WeightsUpload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    filename: str = Field(min_length=1, max_length=255)
    size_bytes: int = Field(ge=1, le=512 * 1024**2)
    trusted: Literal[True]


def create_active_learning_router(store, registry, detector, worker):
    router = APIRouter(prefix="/api/active-learning/models/{model_id}", tags=["Active learning"],
                       dependencies=[Depends(require_dashboard_user)])

    def execute(callback):
        if not store.ready:
            raise HTTPException(503, "Kho Active Learning chưa sẵn sàng.")
        try:
            return callback()
        except KeyError as error:
            raise HTTPException(404, str(error)) from error
        except (WorkflowConflict, RuntimeError, TimeoutError) as error:
            raise HTTPException(409, str(error)) from error
        except (ValueError, ZoneInfoNotFoundError) as error:
            raise HTTPException(422, str(error)) from error

    @router.get("")
    def status(model_id: str):
        return execute(lambda: dict(store.status(model_id), worker=worker.status()))

    @router.put("/settings")
    def configure(model_id: str, request: LearningSettings, actor=Depends(require_dashboard_user)):
        return execute(lambda: store.configure(model_id, request.model_dump(), actor))

    @router.post("/snapshots", status_code=201)
    async def capture(model_id: str, request: CaptureRequest, actor=Depends(require_dashboard_user)):
        def save():
            registry.get(model_id)
            store.cleanup_drafts()
            snapshot = detector.command(model_id, "snapshot", request.model_dump())
            return store.capture(model_id, snapshot, actor)
        return await asyncio.to_thread(execute, save)

    @router.get("/samples")
    def samples(model_id: str, status: Literal["draft", "approved", "rejected"] | None = None,
                limit: int = Query(24, ge=1, le=48), offset: int = Query(0, ge=0)):
        return execute(lambda: store.list_samples(model_id, status, limit, offset))

    @router.post("/samples/{sample_id}/review")
    def review(model_id: str, sample_id: str, request: ReviewRequest, actor=Depends(require_dashboard_user)):
        return execute(lambda: store.review(model_id, sample_id, [item.model_dump() for item in request.annotations],
                                             request.feedback, request.complete, actor))

    @router.get("/samples/{sample_id}/image")
    def image(model_id: str, sample_id: str):
        def response():
            store.get_sample(model_id, sample_id)
            path = store.asset(model_id, sample_id, "image")
            if not path.is_file():
                raise KeyError("Không tìm thấy ảnh trên SSD.")
            return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "no-store"})
        return execute(response)

    @router.delete("/samples/{sample_id}")
    def delete(model_id: str, sample_id: str):
        return execute(lambda: store.delete_sample(model_id, sample_id))

    @router.post("/weights", status_code=201)
    def weights(model_id: str, request: WeightsUpload, actor=Depends(require_dashboard_user)):
        return execute(lambda: store.create_weights(model_id, request.filename, request.size_bytes, actor))

    @router.put("/weights/{weights_id}/file")
    async def append(model_id: str, weights_id: str, request: Request, offset: int = Query(ge=0)):
        content = bytearray()
        async for part in request.stream():
            if len(content) + len(part) > store.chunk_size:
                raise HTTPException(413, "Mỗi chunk tối đa 2 MiB.")
            content.extend(part)
        return await asyncio.to_thread(execute, lambda: store.append_weights(model_id, weights_id, offset, content))

    @router.post("/weights/{weights_id}/complete")
    def complete(model_id: str, weights_id: str):
        return execute(lambda: store.finish_weights(model_id, weights_id))

    @router.post("/jobs", status_code=202)
    def enqueue(model_id: str, actor=Depends(require_dashboard_user)):
        return execute(lambda: store.enqueue(model_id, actor))

    @router.post("/jobs/{job_id}/cancel")
    def cancel(model_id: str, job_id: str):
        return execute(lambda: store.cancel(model_id, job_id))

    @router.post("/jobs/{job_id}/promote")
    def promote(model_id: str, job_id: str, actor=Depends(require_dashboard_user)):
        return execute(lambda: worker.promote(model_id, job_id, actor))

    @router.get("/jobs/{job_id}/log")
    def log(model_id: str, job_id: str):
        def read():
            store.get_job(model_id, job_id)
            path = store.directory(model_id) / "jobs" / job_id / "train.log"
            text = "Chưa chạy training."
            if path.is_file():
                with path.open("rb") as source:
                    source.seek(max(0, path.stat().st_size - 12000))
                    text = source.read().decode("utf-8", errors="replace")
            return Response(text, media_type="text/plain", headers={"Cache-Control": "no-store"})
        return execute(read)

    return router
