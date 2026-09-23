import gzip
import json
import logging
import os
import queue
import re
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from psycopg2.extras import Json, RealDictCursor, execute_values


class SystemLogStore:
    """Bounded, structured system log sink for PostgreSQL and compact JSONL.GZ files."""

    def __init__(self, database, log_dir=None):
        self.database = database
        self.log_dir = Path(log_dir or os.getenv("SYSTEM_LOG_DIR", str(Path(__file__).resolve().parents[1] / "data/system_logs")))
        self.events = queue.Queue(maxsize=max(1, int(os.getenv("SYSTEM_LOG_QUEUE_SIZE", "8192"))))
        self.flush_seconds = max(0.1, float(os.getenv("SYSTEM_LOG_FLUSH_SECONDS", "0.5")))
        self.batch_size = max(10, int(os.getenv("SYSTEM_LOG_BATCH_SIZE", "100")))
        self.stop_event = threading.Event()
        self.thread = None
        self.dropped = 0
        self._drop_lock = threading.Lock()
        self.database_error = self.archive_error = None
        self.written = 0
        self.retention_days = max(1, int(os.getenv("SYSTEM_LOG_RETENTION_DAYS", "7")))
        self.archive_days = max(self.retention_days, int(os.getenv("SYSTEM_LOG_ARCHIVE_DAYS", "30")))
        self.archive_limit = max(16, int(os.getenv("SYSTEM_LOG_ARCHIVE_MB", "512"))) * 1024 ** 2
        self.last_cleanup = 0
        self.archive_chunk_bytes = min(16 * 1024 ** 2, self.archive_limit)
        self.active_archive = None

    @contextmanager
    def transaction(self):
        connection = self.database._get_connection()
        try:
            with connection:
                with connection.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute("SET LOCAL statement_timeout = '3s'")
                    yield cursor
        finally:
            connection.close()

    def initialize(self):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        with self.transaction() as cursor:
            cursor.execute("CREATE SCHEMA IF NOT EXISTS audit")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit.system_logs (
                    id BIGSERIAL PRIMARY KEY,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    level TEXT NOT NULL,
                    source TEXT NOT NULL,
                    event TEXT NOT NULL,
                    message TEXT NOT NULL,
                    camera_id TEXT,
                    request_id TEXT,
                    status_code INTEGER,
                    duration_ms DOUBLE PRECISION,
                    details JSONB NOT NULL DEFAULT '{}'
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS system_logs_time ON audit.system_logs(created_at DESC, id DESC)")
            cursor.execute("CREATE INDEX IF NOT EXISTS system_logs_level_time ON audit.system_logs(level, created_at DESC, id DESC)")
            cursor.execute("CREATE INDEX IF NOT EXISTS system_logs_source_time ON audit.system_logs(source, created_at DESC, id DESC)")
            cursor.execute("CREATE INDEX IF NOT EXISTS system_logs_camera_time ON audit.system_logs(camera_id, created_at DESC, id DESC)")
        if not self.thread or not self.thread.is_alive():
            self.stop_event.clear()
            self.thread = threading.Thread(target=self._run, name="system-log-writer", daemon=True)
            self.thread.start()

    @staticmethod
    def _json_safe(value):
        try:
            json.dumps(value, allow_nan=False)
            return value
        except (TypeError, ValueError):
            return str(value)

    @staticmethod
    def redact(value):
        if isinstance(value, dict):
            return {str(key): "[REDACTED]" if re.search(r"password|secret|token|api.?key|authorization|cookie", str(key), re.I)
                    else SystemLogStore.redact(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [SystemLogStore.redact(item) for item in value]
        if isinstance(value, str):
            value = re.sub(r"(?i)((?:rtsp|rtsps|https?)://)[^/\s@]+@", r"\1[REDACTED]@", value)
            value = re.sub(r"(?i)([?&](?:key|api_key|token|password)=)[^&\s]+", r"\1[REDACTED]", value)
            value = re.sub(r"\b(?:AQ\.|AIza|sk-)[A-Za-z0-9_.-]{12,}", "[REDACTED]", value)
            value = re.sub(r"(?i)(Bearer\s+)[A-Za-z0-9._~+/=-]+", r"\1[REDACTED]", value)
        return value

    def emit(self, level="info", source="system", event="log", message="", *, camera_id=None,
             request_id=None, status_code=None, duration_ms=None, details=None):
        record = {
            "id": uuid.uuid4().hex,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "level": str(level).lower()[:16],
            "source": str(source)[:64],
            "event": str(event)[:96],
            "message": str(message),
            "camera_id": str(camera_id)[:128] if camera_id else None,
            "request_id": str(request_id)[:128] if request_id else None,
            "status_code": int(status_code) if status_code is not None else None,
            "duration_ms": round(float(duration_ms), 3) if duration_ms is not None else None,
            "details": self._json_safe(details or {}),
        }
        record = self.redact(record)
        try:
            self.events.put_nowait(record)
        except queue.Full:
            with self._drop_lock:
                self.dropped += 1

    def _run(self):
        pending = []
        deadline = time.monotonic() + self.flush_seconds
        while not self.stop_event.is_set() or not self.events.empty() or pending:
            try:
                pending.append(self.events.get(timeout=max(.001, deadline - time.monotonic())))
            except queue.Empty:
                pass
            while len(pending) < self.batch_size:
                try:
                    pending.append(self.events.get_nowait())
                except queue.Empty:
                    break
            if len(pending) >= self.batch_size or self.stop_event.is_set() or time.monotonic() >= deadline:
                batch, pending = pending, []
                if batch:
                    self._flush(batch)
                deadline = time.monotonic() + self.flush_seconds
            if time.monotonic() - self.last_cleanup > 300:
                self.last_cleanup = time.monotonic()
                self._cleanup()

    def _flush(self, batch):
        rows = []
        for item in batch:
            try:
                created_at = datetime.fromisoformat(item["created_at"])
            except (TypeError, ValueError):
                created_at = datetime.now(timezone.utc)
            rows.append((created_at, item["level"], item["source"], item["event"], item["message"],
                         item["camera_id"], item["request_id"], item["status_code"], item["duration_ms"],
                         Json(item["details"])))
        try:
            with self.transaction() as cursor:
                execute_values(cursor, """
                    INSERT INTO audit.system_logs
                    (created_at,level,source,event,message,camera_id,request_id,status_code,duration_ms,details)
                    VALUES %s
                """, rows)
            self.database_error = None
        except Exception as error:
            self.database_error = type(error).__name__
        try:
            by_day = {}
            for item in batch:
                day = item["created_at"][:10]
                by_day.setdefault(day, []).append(json.dumps(item, ensure_ascii=False, separators=(",", ":")))
            for day, lines in by_day.items():
                path = self.log_dir / f"system-{day}.jsonl.gz"
                part = 0
                while path.is_symlink() or (path.exists() and path.stat().st_size >= self.archive_chunk_bytes):
                    part += 1
                    path = self.log_dir / f"system-{day}-{part:04d}.jsonl.gz"
                self.active_archive = path
                with gzip.open(path, "at", encoding="utf-8", compresslevel=6) as stream:
                    stream.write("\n".join(lines) + "\n")
            self.archive_error = None
            self.written += len(batch)
        except Exception as error:
            self.archive_error = type(error).__name__

    def _cleanup(self):
        try:
            with self.transaction() as cursor:
                cursor.execute("DELETE FROM audit.system_logs WHERE created_at < NOW() - %s * INTERVAL '1 day'", (self.retention_days,))
                cursor.execute("DELETE FROM audit.system_logs WHERE id < (SELECT id FROM audit.system_logs ORDER BY id DESC OFFSET 199999 LIMIT 1)")
        except Exception as error:
            self.database_error = type(error).__name__
        try:
            files = sorted((path for path in self.log_dir.glob("system-*.jsonl.gz") if path.is_file() and not path.is_symlink()), key=lambda path: path.stat().st_mtime)
            total = sum(path.stat().st_size for path in files)
            for path in files:
                if path == self.active_archive:
                    continue
                size = path.stat().st_size
                if total > self.archive_limit or time.time() - path.stat().st_mtime > self.archive_days * 86400:
                    path.unlink()
                    total -= size
        except OSError as error:
            self.archive_error = type(error).__name__

    def list(self, *, level=None, source=None, search=None, camera_id=None, limit=100, before=None,
             channel="system", since=None, until=None):
        selections = {
            "system": "SELECT id,created_at,level,source,event,message,camera_id,request_id,status_code,duration_ms,details FROM audit.system_logs",
            "workflow": "SELECT id,created_at,level,'workflow'::text AS source,'pipeline'::text AS event,message,details->>'camera_id' AS camera_id, deployment_id AS request_id,NULL::integer AS status_code,NULL::float AS duration_ms,details FROM workflow.logs",
            "audit": "SELECT id,created_at,'info'::text AS level,'audit'::text AS source,action AS event,actor || ': ' || action AS message,details->>'camera_id' AS camera_id,NULL::text AS request_id,NULL::integer AS status_code,NULL::float AS duration_ms,details FROM audit.actions",
        }
        if channel not in selections:
            raise ValueError("Kênh log không hợp lệ.")
        clauses = []
        params = []
        if level:
            clauses.append("level = %s")
            params.append(level)
        if source:
            clauses.append("source = %s")
            params.append(source)
        if camera_id:
            clauses.append("camera_id = %s")
            params.append(camera_id)
        if search:
            clauses.append("(message ILIKE %s OR event ILIKE %s OR source ILIKE %s)")
            term = f"%{search[:120]}%"
            params.extend([term, term, term])
        if before:
            clauses.append("id < %s")
            params.append(before)
        if since:
            clauses.append("created_at >= %s")
            params.append(since)
        if until:
            clauses.append("created_at <= %s")
            params.append(until)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(min(max(int(limit), 1), 500) + 1)
        with self.transaction() as cursor:
            cursor.execute(f"""
                SELECT id, created_at, level, source, event, message, camera_id,
                       request_id, status_code, duration_ms, details
                FROM ({selections[channel]}) entries {where}
                ORDER BY id DESC LIMIT %s
            """, tuple(params))
            rows = [dict(row) for row in cursor.fetchall()]
        return {"logs": self.redact(rows[:limit]), "has_more": len(rows) > limit, "dropped": self.dropped,
                "channel": channel, "storage": "PostgreSQL audit.system_logs + SSD JSONL.GZ"}

    def status(self):
        return {"queue": self.events.qsize(), "queue_capacity": self.events.maxsize, "dropped": self.dropped,
                "directory": str(self.log_dir), "format": "PostgreSQL + JSONL.GZ", "flush_seconds": self.flush_seconds,
                "written": self.written, "database_error": self.database_error, "archive_error": self.archive_error,
                "retention_days": self.retention_days, "archive_days": self.archive_days,
                "archive_limit_bytes": self.archive_limit}

    def archives(self):
        return [{"name": path.name, "size_bytes": path.stat().st_size} for path in
                sorted(self.log_dir.glob("system-*.jsonl.gz"), reverse=True) if path.is_file() and not path.is_symlink()]

    def stop(self):
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=max(7.0, self.flush_seconds * 4))


class SystemLogHandler(logging.Handler):
    def __init__(self, store):
        super().__init__()
        self.store = store

    def emit(self, record):
        try:
            details = {"module": record.module, "function": record.funcName, "line": record.lineno}
            if record.exc_info:
                details["exception"] = self.format(record)
            self.store.emit(record.levelname, record.name, "application_log", record.getMessage(), details=details)
        except Exception:
            pass


class SystemRequestLogMiddleware:
    """Non-blocking ASGI access logger; records only API requests, not media/WebSocket frames."""
    def __init__(self, app, store):
        self.app = app
        self.store = store

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        if not path.startswith("/api/") or path.startswith("/api/system-logs") or path in {"/api/health", "/api/debug/metadata"}:
            await self.app(scope, receive, send)
            return
        started = time.monotonic()
        status_code = 500
        request_id = next((value.decode("latin1") for key, value in scope.get("headers", []) if key == b"x-request-id"), None) or uuid.uuid4().hex
        parts = path.split("/")
        camera_id = next((parts[index + 1] for index, part in enumerate(parts) if part == "camera" and index + 1 < len(parts)), None)

        async def send_logged(message):
            nonlocal status_code
            if message.get("type") == "http.response.start":
                status_code = int(message.get("status", 500))
            await send(message)

        try:
            await self.app(scope, receive, send_logged)
        except Exception:
            self.store.emit("error", "gateway", "http_request", f"{scope.get('method', 'GET')} {path}",
                            camera_id=camera_id, request_id=request_id, status_code=500,
                            duration_ms=(time.monotonic() - started) * 1000)
            raise
        self.store.emit("info" if status_code < 400 else "warning", "gateway", "http_request",
                        f"{scope.get('method', 'GET')} {path}", camera_id=camera_id, request_id=request_id,
                        status_code=status_code, duration_ms=(time.monotonic() - started) * 1000)
