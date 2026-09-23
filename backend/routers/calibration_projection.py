from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from core.calibration_projection import calibration_footprint, project_calibration
from core.dashboard_auth import require_dashboard_user


class ProjectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    points: list[tuple[float, float]] = Field(min_length=1, max_length=256)
    source: Literal["image", "floor", "fms"] = "floor"
    save_id: str | None = None


def create_calibration_projection_router(calibrator, cameras, frame):
    router = APIRouter(tags=["Camera projection"], dependencies=[Depends(require_dashboard_user)])

    @router.get("/api/calibration/cameras")
    def overview():
        rows = []
        for camera_id, camera in list(cameras().items()):
            config = calibrator.get_config(camera_id)
            if not config:
                continue
            try:
                rows.append(dict(camera_id=camera_id, name=camera.get("name", camera_id),
                                 **calibration_footprint(config, frame())))
            except (ValueError, TypeError, IndexError):
                rows.append(dict(camera_id=camera_id, name=camera.get("name", camera_id),
                                 footprint=[], error="Hiệu chuẩn không còn phù hợp với hệ tọa độ FMS hiện tại."))
        return {"cameras": rows, "coordinate_space": "fms_floor_metric", "fms_frame": frame()}

    @router.post("/api/camera/{camera_id}/calibration/project")
    def project(camera_id: str, request: ProjectionRequest):
        if camera_id not in cameras():
            raise HTTPException(404, "Camera không tồn tại.")
        config = calibrator.get_config(camera_id)
        if not config:
            raise HTTPException(409, "Camera chưa hiệu chuẩn.")
        if request.save_id and request.save_id != config.get("save_id"):
            raise HTTPException(409, "Hiệu chuẩn đã thay đổi; hãy tải lại camera trên bản đồ.")
        try:
            return dict(camera_id=camera_id, **project_calibration(config, request.points, request.source, frame()))
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    return router
