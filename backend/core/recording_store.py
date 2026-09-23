import hashlib
from contextlib import contextmanager
from datetime import datetime, timezone

from psycopg2.extras import RealDictCursor, Json


class RecordingStore:
    def __init__(self, database):
        self.database = database

    @contextmanager
    def transaction(self):
        connection = self.database._get_connection()
        try:
            with connection:
                with connection.cursor(cursor_factory=RealDictCursor) as cursor:
                    yield cursor
        finally:
            connection.close()

    def initialize(self):
        with self.transaction() as cursor:
            cursor.execute("ALTER TABLE events ADD COLUMN IF NOT EXISTS evidence_requested BOOLEAN NOT NULL DEFAULT TRUE")
            cursor.execute("CREATE SCHEMA IF NOT EXISTS archive; CREATE SCHEMA IF NOT EXISTS audit")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS archive.recordings (
                    id TEXT PRIMARY KEY, camera_id TEXT NOT NULL, path TEXT UNIQUE NOT NULL,
                    started_at TIMESTAMPTZ NOT NULL, ended_at TIMESTAMPTZ,
                    duration_seconds DOUBLE PRECISION NOT NULL DEFAULT 0,
                    size_bytes BIGINT NOT NULL DEFAULT 0, codec TEXT,
                    status TEXT NOT NULL, reason TEXT, updated_at TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS recordings_camera_time ON archive.recordings(camera_id, started_at DESC);
                CREATE INDEX IF NOT EXISTS recordings_status_time ON archive.recordings(status, started_at DESC);
                CREATE TABLE IF NOT EXISTS audit.actions (
                    id BIGSERIAL PRIMARY KEY, actor TEXT NOT NULL, action TEXT NOT NULL,
                    details JSONB NOT NULL, created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            cursor.execute("""
                UPDATE events SET video_file = NULL
                WHERE video_file IS NOT NULL
                  AND NOT EXISTS (
                    SELECT 1 FROM archive.recordings recording
                    WHERE recording.path = events.video_file AND recording.status <> 'deleted'
                  )
            """)

    @staticmethod
    def recording_id(path):
        return hashlib.sha256(path.encode()).hexdigest()[:32]

    def save(self, camera_id, path, started_at, status, size_bytes, duration=0, codec=None, reason=None):
        record_id = self.recording_id(path)
        ended_at = started_at + duration if status != "recording" else None
        with self.transaction() as cursor:
            cursor.execute("""
                INSERT INTO archive.recordings(id,camera_id,path,started_at,ended_at,duration_seconds,size_bytes,codec,status,reason)
                VALUES (%s,%s,%s,to_timestamp(%s),to_timestamp(%s),%s,%s,%s,%s,%s)
                ON CONFLICT(path) DO UPDATE SET ended_at=EXCLUDED.ended_at,duration_seconds=EXCLUDED.duration_seconds,
                    size_bytes=EXCLUDED.size_bytes,codec=EXCLUDED.codec,status=EXCLUDED.status,reason=EXCLUDED.reason,updated_at=NOW()
                WHERE archive.recordings.status NOT IN ('deleted','deleting')
                RETURNING id
            """, (record_id, camera_id, path, started_at, ended_at, duration, size_bytes, codec, status, reason))
            if cursor.fetchone() and status in {"ready", "interrupted"}:
                cursor.execute("""
                    UPDATE events SET video_file=%s WHERE cam_id=%s AND video_file IS NULL AND evidence_requested
                    AND timestamp >= to_timestamp(%s) AT TIME ZONE 'UTC'
                    AND timestamp < to_timestamp(%s) AT TIME ZONE 'UTC'
                """, (path, camera_id, started_at, ended_at))
        return record_id

    def get(self, record_id):
        with self.transaction() as cursor:
            cursor.execute("SELECT * FROM archive.recordings WHERE id=%s AND status <> 'deleted'", (record_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def list(self, camera_id=None, limit=50, offset=0):
        with self.transaction() as cursor:
            cursor.execute("""
                SELECT recording.*, COALESCE(camera.name, recording.camera_id) AS camera_name
                FROM archive.recordings recording LEFT JOIN cameras camera ON camera.id=recording.camera_id
                WHERE recording.status NOT IN ('deleted','deleting') AND (%s IS NULL OR recording.camera_id=%s)
                ORDER BY recording.started_at DESC, recording.id LIMIT %s OFFSET %s
            """, (camera_id, camera_id, limit + 1, offset))
            rows = [dict(row) for row in cursor.fetchall()]
            return {"recordings": rows[:limit], "has_more": len(rows) > limit}

    def pending_recovery(self):
        with self.transaction() as cursor:
            cursor.execute("SELECT * FROM archive.recordings WHERE status IN ('recording','deleting')")
            return [dict(row) for row in cursor.fetchall()]

    def mark_deleting(self, record_id):
        with self.transaction() as cursor:
            cursor.execute("UPDATE archive.recordings SET status='deleting',updated_at=NOW() WHERE id=%s", (record_id,))

    def mark_deleted(self, record_id, actor, reason):
        with self.transaction() as cursor:
            cursor.execute("UPDATE archive.recordings SET status='deleted',updated_at=NOW(),reason=%s WHERE id=%s RETURNING path", (reason, record_id))
            row = cursor.fetchone()
            if row:
                cursor.execute("UPDATE events SET video_file=NULL WHERE video_file=%s", (row["path"],))
                self.audit(cursor, actor, "delete_recording", {"id": record_id, "reason": reason})

    @staticmethod
    def audit(cursor, actor, action, details):
        cursor.execute("INSERT INTO audit.actions(actor,action,details) VALUES(%s,%s,%s)", (actor, action, Json(details)))

    def delete_event(self, event_id, actor):
        with self.transaction() as cursor:
            cursor.execute("DELETE FROM events WHERE id=%s RETURNING id", (event_id,))
            deleted = bool(cursor.fetchone())
            if deleted:
                self.audit(cursor, actor, "delete_event", {"event_id": event_id})
            return deleted

    def clear_history(self, before, actor):
        cutoff = before.astimezone(timezone.utc).replace(tzinfo=None)
        counts = {}
        with self.transaction() as cursor:
            for table in ("events", "detections", "global_tracks_log"):
                cursor.execute(f"DELETE FROM {table} WHERE timestamp <= %s", (cutoff,))
                counts[table] = cursor.rowcount
            cursor.execute("DELETE FROM tripwire_counts WHERE last_updated <= %s", (cutoff,))
            counts["tripwire_counts"] = cursor.rowcount
            self.audit(cursor, actor, "clear_analytics_history", {"before": before.isoformat(), "counts": counts})
        return counts

    def attach_event_recordings(self, events):
        paths = list({event.get("video_file") for event in events if event.get("video_file")})
        matches = {}
        if paths:
            with self.transaction() as cursor:
                cursor.execute("SELECT id,path FROM archive.recordings WHERE path=ANY(%s) AND status IN ('ready','interrupted')", (paths,))
                matches = {row["path"]: row["id"] for row in cursor.fetchall()}
        for event in events:
            event["recording_id"] = matches.get(event.get("video_file"))
            if not event["recording_id"]:
                event["video_file"] = None
        return events
