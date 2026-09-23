from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from core.dashboard_auth import require_dashboard_user
from core.workflow_definition import WorkflowDefinition, block_catalog, connector_settings
from core.workflow_store import WorkflowConflict


class WorkflowSave(BaseModel):
    model_config = ConfigDict(extra="forbid")
    definition: WorkflowDefinition
    revision: int | None = Field(default=None, ge=1)


class WorkflowDeploy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    revision: int = Field(ge=1)
    start_paused: bool = False


class WorkflowAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["pause", "resume", "stop"]


def create_workflow_router(store, runtime, resources, validator):
    router = APIRouter(prefix="/api/workflows", tags=["Workflow pipelines"], dependencies=[Depends(require_dashboard_user)])

    def execute(callback):
        try:
            return callback()
        except KeyError as error:
            raise HTTPException(404, str(error)) from error
        except WorkflowConflict as error:
            raise HTTPException(409, str(error)) from error

    @router.get("/catalog")
    def catalog():
        return {"blocks": block_catalog(), "connectors": list(connector_settings()), **resources(),
                "execution": "shared_live_metadata", "max_active": 32}

    @router.post("/validate")
    def validate(request: WorkflowDefinition):
        return validator(request.model_dump())

    @router.get("")
    def list_workflows():
        result = store.list()
        for deployment in result["deployments"]:
            deployment["runtime"] = runtime.status(deployment["id"])
        return {**result, "runtime": runtime.status()}

    @router.post("", status_code=201)
    def create(request: WorkflowSave, user=Depends(require_dashboard_user)):
        return store.save(request.definition.model_dump(), user)

    @router.put("/{pipeline_id}")
    def save(pipeline_id: str, request: WorkflowSave, user=Depends(require_dashboard_user)):
        if request.revision is None:
            raise HTTPException(422, "Cần revision của bản nháp đang sửa.")
        return execute(lambda: store.save(request.definition.model_dump(), user, pipeline_id, request.revision))

    @router.delete("/{pipeline_id}")
    def delete(pipeline_id: str, revision: int = Query(ge=1), user=Depends(require_dashboard_user)):
        execute(lambda: store.delete(pipeline_id, revision, user))
        runtime.refresh()
        return {"deleted_id": pipeline_id}

    @router.post("/{pipeline_id}/deploy", status_code=201)
    def deploy(pipeline_id: str, request: WorkflowDeploy, user=Depends(require_dashboard_user)):
        if not runtime.status()["healthy"]:
            raise HTTPException(503, "Runtime workflow chưa sẵn sàng; xem trạng thái kết nối backend.")
        result = execute(lambda: store.deploy(pipeline_id, request.revision, user, validator, request.start_paused))
        runtime.refresh()
        return result

    @router.post("/deployments/{deployment_id}/action")
    def action(deployment_id: str, request: WorkflowAction, user=Depends(require_dashboard_user)):
        result = execute(lambda: store.transition(deployment_id, request.action, user, validator))
        runtime.refresh()
        return result

    @router.get("/deployments/{deployment_id}/logs")
    def logs(deployment_id: str, before: int | None = Query(default=None, ge=1)):
        return {**store.logs(deployment_id, before), "runtime": runtime.status(deployment_id)}

    return router
