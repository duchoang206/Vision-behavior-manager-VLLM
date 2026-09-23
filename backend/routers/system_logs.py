import gzip
import json
import re
from datetime import datetime
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse

from core.dashboard_auth import require_dashboard_user


def create_system_log_router(store):
    router = APIRouter(prefix="/api/system-logs", tags=["System logs"], dependencies=[Depends(require_dashboard_user)])

    @router.get("")
    def list_system_logs(level: str | None = None, source: str | None = None, search: str | None = None,
                         camera_id: str | None = None, limit: int = Query(100, ge=1, le=500),
                         before: int | None = Query(None, ge=1), channel: Literal['system', 'workflow', 'audit'] = 'system',
                         since: datetime | None = None, until: datetime | None = None, download: bool = False):
        if any(value is not None and value.tzinfo is None for value in (since, until)) or (since and until and since > until):
            raise HTTPException(422, "Chọn khoảng thời gian có múi giờ, bắt đầu không sau kết thúc.")
        result = store.list(level=level, source=source, search=search, camera_id=camera_id, limit=limit, before=before,
                            channel=channel, since=since, until=until)
        if download:
            lines = '\n'.join(json.dumps(item, ensure_ascii=False, separators=(',', ':')) for item in jsonable_encoder(result['logs'])) + '\n'
            return Response(gzip.compress(lines.encode(), compresslevel=6), media_type='application/gzip',
                            headers={'Content-Disposition': 'attachment; filename="logs-query.jsonl.gz"', 'Cache-Control': 'no-store'})
        return result

    @router.get("/status")
    def system_log_status():
        return store.status()

    @router.get('/archives')
    def archives():
        return {'archives': store.archives()}

    @router.get('/archives/{name}')
    def archive(name: str):
        if not re.fullmatch(r'system-\d{4}-\d{2}-\d{2}(?:-\d{4,})?\.jsonl\.gz', name):
            raise HTTPException(404, 'Không tìm thấy log.')
        path = store.log_dir / name
        if path.is_symlink() or not path.is_file():
            raise HTTPException(404, 'Log đã hết hạn hoặc không tồn tại.')
        return FileResponse(path, media_type='application/gzip', filename=name, headers={'Cache-Control': 'no-store'})

    return router
