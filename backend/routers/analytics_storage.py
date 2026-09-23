from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from core.dashboard_auth import require_dashboard_user


class HistoryDeletion(BaseModel):
    before: datetime
    confirmation: str


def create_storage_router(store, recorder):
    router = APIRouter(prefix="/api", tags=["Analytics storage"])

    @router.get("/recordings")
    def list_recordings(camera_id: str = None, limit: int = Query(50, ge=1, le=300), offset: int = Query(0, ge=0),
                        user=Depends(require_dashboard_user)):
        return {**store.list(camera_id, limit, offset), "recorder": recorder.status()}

    @router.get("/recordings/{record_id}/video")
    def recording_video(record_id: str, download: bool = False, user=Depends(require_dashboard_user)):
        record = store.get(record_id)
        if not record:
            raise HTTPException(404, "Không tìm thấy video hoặc video đã bị xóa.")
        if record["status"] not in {"ready", "interrupted"}:
            raise HTTPException(409, "Video đang ghi hoặc không hoàn chỉnh, chưa sẵn sàng phát.")
        path = recorder.safe_path(record["path"])
        if not path.is_file():
            raise HTTPException(404, "Video đã hết hạn hoặc không còn trên SSD.")
        return FileResponse(path, media_type="video/mp4", filename=path.name,
                            content_disposition_type="attachment" if download else "inline",
                            headers={"Cache-Control": "private, no-store"})

    @router.delete("/recordings/{record_id}")
    def delete_recording(record_id: str, user=Depends(require_dashboard_user)):
        try:
            recorder.delete_recording(record_id, user)
        except FileNotFoundError as error:
            raise HTTPException(404, str(error)) from error
        except RuntimeError as error:
            raise HTTPException(409, str(error)) from error
        except (ValueError, OSError) as error:
            raise HTTPException(400, "Không xóa được file video trên SSD.") from error
        return {"status": "success", "deleted_id": record_id}

    @router.delete("/events/{event_id}")
    def delete_event(event_id: int, user=Depends(require_dashboard_user)):
        if not store.delete_event(event_id, user):
            raise HTTPException(404, "Sự kiện không còn tồn tại.")
        return {"status": "success", "deleted_id": event_id}

    @router.delete("/analytics/history")
    def clear_history(request: HistoryDeletion, user=Depends(require_dashboard_user)):
        if request.confirmation != "DELETE_ANALYTICS_HISTORY" or request.before.tzinfo is None:
            raise HTTPException(400, "Cần xác nhận xóa và mốc thời gian có múi giờ.")
        if request.before > datetime.now(timezone.utc):
            raise HTTPException(400, "Không thể xóa dữ liệu của thời điểm tương lai.")
        return {"status": "success", "deleted": store.clear_history(request.before, user)}

    return router
