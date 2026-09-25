import asyncio
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from core.dashboard_auth import require_dashboard_user
from core.workflow_store import WorkflowConflict


class ModelUpload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(default="", max_length=120)
    filename: str = Field(min_length=1, max_length=255)
    size_bytes: int = Field(ge=1, le=512 * 1024 * 1024)
    labels: list[str] = Field(default_factory=list, max_length=100)


class ModelBuild(BaseModel):
    model_config = ConfigDict(extra="forbid")
    labels: list[str] = Field(default_factory=list, max_length=100)


class MonitorDeployment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    camera_ids: list[str] = Field(default_factory=list, max_length=256)
    # Never broaden a deployment silently when a client omits this field.
    all_cameras: bool
    confidence_thresholds: dict = Field(default_factory=dict)


class DeploymentCreate(MonitorDeployment):
    model_id: str = Field(min_length=32, max_length=32)


class DeploymentEnabled(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool

def create_model_router(registry, live_status, cameras=lambda: []):
    router = APIRouter(prefix="/api/models", tags=["TensorRT models"], dependencies=[Depends(require_dashboard_user)])

    def execute(callback):
        try:
            return callback()
        except KeyError as error:
            raise HTTPException(404, str(error)) from error
        except WorkflowConflict as error:
            raise HTTPException(409, str(error)) from error
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    @router.get("")
    def list_models():
        return {"models": registry.list(), "runtime": live_status(), "chunk_size": registry.chunk_size,
                "max_size": registry.max_size, "deployment": registry.deployment(),
                "deployments": registry.deployments()}

    @router.get("/deployments")
    def list_deployments():
        return {"deployments": registry.deployments()}

    @router.post("/deployments")
    def create_deployment(request: DeploymentCreate, actor=Depends(require_dashboard_user)):
        return execute(lambda: registry.deploy(request.model_id, request.camera_ids, request.all_cameras, cameras(), actor, request.confidence_thresholds))

    @router.patch("/deployments/{deployment_id}")
    def set_deployment_enabled(deployment_id: str, request: DeploymentEnabled, actor=Depends(require_dashboard_user)):
        return execute(lambda: registry.set_deployment_enabled(deployment_id, request.enabled, actor))

    @router.delete("/deployments/{deployment_id}")
    def remove_deployment(deployment_id: str, actor=Depends(require_dashboard_user)):
        return execute(lambda: registry.stop_deployment(actor, deployment_id))

    @router.delete("/deployment")
    def stop_deployment(actor=Depends(require_dashboard_user)):
        return execute(lambda: registry.stop_deployment(actor))

    @router.post("/{model_id}/deploy")
    def deploy(model_id: str, request: MonitorDeployment, actor=Depends(require_dashboard_user)):
        return execute(lambda: registry.deploy(model_id, request.camera_ids, request.all_cameras, cameras(), actor, request.confidence_thresholds))

    @router.post("", status_code=201)
    def upload(request: ModelUpload, actor=Depends(require_dashboard_user)):
        return execute(lambda: registry.create(request.name, request.filename, request.size_bytes, request.labels, actor))

    @router.put("/{model_id}/file")
    async def chunk(model_id: str, request: Request, offset: int = Query(ge=0)):
        content = bytearray()
        async for part in request.stream():
            if len(content) + len(part) > registry.chunk_size:
                raise HTTPException(413, "Mỗi phần upload tối đa 2 MiB.")
            content.extend(part)
        return await asyncio.to_thread(execute, lambda: registry.append(model_id, offset, content))

    @router.post("/{model_id}/build", status_code=202)
    def build(model_id: str, request: ModelBuild):
        return execute(lambda: registry.enqueue(model_id, request.labels))

    return router
