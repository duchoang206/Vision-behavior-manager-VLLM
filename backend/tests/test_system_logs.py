import asyncio
import gzip
import json
import logging
import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.dashboard_auth import require_dashboard_user
from core.system_log_store import SystemLogHandler, SystemLogStore, SystemRequestLogMiddleware
from routers.system_logs import create_system_log_router


class SystemLogTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = SystemLogStore(MagicMock(), self.directory.name)
        self.cursor = MagicMock()

        @contextmanager
        def transaction():
            yield self.cursor

        self.store.transaction = transaction

    def flush(self):
        records = []
        while not self.store.events.empty():
            records.append(self.store.events.get_nowait())
        with patch("core.system_log_store.execute_values"):
            self.store._flush(records)
        return records

    def test_unicode_long_log_is_losslessly_compressed(self):
        message = "Mask của người dùng · ghi nhận realtime\n" * 1000
        self.store.emit(message=message)
        records = self.flush()
        archive = Path(self.directory.name) / self.store.archives()[0]["name"]
        raw = gzip.decompress(archive.read_bytes())
        self.assertEqual(json.loads(raw)["message"], message)
        self.assertEqual(json.loads(raw), records[0])
        self.assertLess(archive.stat().st_size, len(raw) / 5)

    def test_credentials_redacted_before_database_and_archive(self):
        self.store.emit(message="rtsp://someone:password@host/live?token=secret-token",
                        details={"api_key": "sensitive", "nested": [{"password": "sensitive"}], "good": "keep"})
        record = self.flush()[0]
        self.assertNotIn("secret-token", record["message"])
        self.assertNotIn("someone:password", record["message"])
        self.assertEqual(record["details"]["api_key"], "[REDACTED]")
        self.assertEqual(record["details"]["good"], "keep")

    def test_queue_is_bounded_and_reports_overflow(self):
        with patch.dict(os.environ, {"SYSTEM_LOG_QUEUE_SIZE": "2"}):
            store = SystemLogStore(MagicMock(), self.directory.name)
        for index in range(4):
            store.emit(message=str(index))
        self.assertEqual(store.status()["queue"], 2)
        self.assertEqual(store.status()["dropped"], 2)

    def test_archive_rotation_and_cleanup_include_same_day_parts(self):
        self.store.archive_chunk_bytes = 1
        for index in range(3):
            self.store.emit(message=str(index))
            self.flush()
        self.assertEqual(len(self.store.archives()), 3)
        self.store.archive_limit = 1
        self.store._cleanup()
        self.assertEqual(len(self.store.archives()), 1)
        self.assertTrue(self.store.active_archive.exists())

    def test_database_failure_still_keeps_archive(self):
        self.store.emit(message="persist fallback")
        with patch("core.system_log_store.execute_values", side_effect=RuntimeError("database unavailable")):
            self.store._flush([self.store.events.get_nowait()])
        self.assertEqual(self.store.status()["database_error"], "RuntimeError")
        self.assertEqual(self.store.status()["written"], 1)

    def test_query_uses_parameters_and_bounds(self):
        self.cursor.fetchall.return_value = []
        self.store.list(search="' OR 1=1--", camera_id="cam", before=100, limit=10)
        statement, params = self.cursor.execute.call_args.args
        self.assertNotIn("' OR 1=1--", statement)
        self.assertIn("%' OR 1=1--%", params)
        self.assertEqual(params[-1], 11)
        with self.assertRaises(ValueError):
            self.store.list(channel="arbitrary_table")

    def test_application_handler_enqueues_no_disk_io(self):
        handler = SystemLogHandler(self.store)
        handler.emit(logging.LogRecord("detector", logging.WARNING, "runtime.py", 12, "frame %d", (20,), None))
        self.assertEqual(self.store.events.get_nowait()["message"], "frame 20")
        self.assertEqual(self.store.archives(), [])

    def test_http_logger_never_reads_query_or_body(self):
        async def app(scope, receive, send):
            await send({"type": "http.response.start", "status": 201})
            await send({"type": "http.response.body", "body": b""})

        async def send(message):
            pass

        scope = {"type": "http", "method": "POST", "path": "/api/camera/cam/calibration", "headers": [],
                 "query_string": b"api_key=must-not-be-logged"}
        asyncio.run(SystemRequestLogMiddleware(app, self.store)(scope, None, send))
        record = self.store.events.get_nowait()
        self.assertEqual(record["camera_id"], "cam")
        self.assertEqual(record["status_code"], 201)
        self.assertNotIn("must-not-be-logged", json.dumps(record))

    def test_api_auth_limits_gzip_and_archive_names(self):
        app = FastAPI()
        self.store.list = MagicMock(return_value={"logs": [{"id": 1, "message": "kiểm tra"}], "has_more": False})
        app.include_router(create_system_log_router(self.store))
        client = TestClient(app)
        self.assertEqual(client.get("/api/system-logs").status_code, 401)
        app.dependency_overrides[require_dashboard_user] = lambda: "admin"
        self.assertEqual(client.get("/api/system-logs?limit=501").status_code, 422)
        self.assertEqual(client.get("/api/system-logs?since=2026-01-01T00:00:00").status_code, 422)
        response = client.get("/api/system-logs?download=true")
        self.assertEqual(json.loads(gzip.decompress(response.content))["message"], "kiểm tra")
        self.assertEqual(client.get("/api/system-logs/archives/private.env").status_code, 404)


if __name__ == "__main__":
    unittest.main()
