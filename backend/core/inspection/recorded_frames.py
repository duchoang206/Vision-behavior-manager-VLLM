"""Empty-cell frames from the recorded video, at a moment the operator names.

The evidence recorder (core/ffmpeg_recorder.py) keeps every camera as MP4
segments ``<RECORDINGS_DIR>/<cam>/<YYYYmmddTHHMMSSZ>_<session>.mp4`` - UTC
start in the name, fragmented so the segment still being written is readable
too.  ``RecordedFrames.frames_at`` opens the segment holding the moment, seeks
to it and returns a few frames spread over about a second for the baseline
median.  The operator can so give "the cell was empty at 06:10 and at 22:40"
without standing at the screen at those hours.
"""

import calendar
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

SEGMENT_NAME = re.compile(r"^(\d{8}T\d{6}Z)_[a-f0-9]{12}\.mp4$")    # ffmpeg_recorder.RECORDING_NAME
SAFE_CAMERA = re.compile(r"^[A-Za-z0-9_-]{1,100}$")
SEEK_TOLERANCE_SEC = 2.0


class RecordingNotFound(LookupError):
    pass


@dataclass
class Segment:
    path: Path
    start: float


def _format(epoch: float) -> str:
    return time.strftime("%H:%M %d/%m/%Y", time.localtime(epoch))


class RecordedFrames:
    def __init__(self, root_resolver: Callable[[], Path]):
        self.root_resolver = root_resolver

    def segments(self, cam_id: str) -> List[Segment]:
        if not SAFE_CAMERA.fullmatch(str(cam_id)):
            return []
        try:
            root = Path(self.root_resolver()).resolve()
            folder = (root / cam_id).resolve()
            if not folder.is_relative_to(root) or not folder.is_dir():
                return []
            found = []
            for path in folder.iterdir():
                match = SEGMENT_NAME.fullmatch(path.name)
                if match and path.is_file() and not path.is_symlink():
                    found.append(Segment(path, float(calendar.timegm(time.strptime(match[1], "%Y%m%dT%H%M%SZ")))))
        except OSError:
            return []
        return sorted(found, key=lambda s: s.start)

    def coverage(self, cam_id: str) -> Optional[Tuple[float, float]]:
        """``(first, last)`` moment covered by the camera's recordings, if any."""
        segments = self.segments(cam_id)
        if not segments:
            return None
        try:
            end = segments[-1].path.stat().st_mtime
        except OSError:
            end = segments[-1].start
        return segments[0].start, max(end, segments[-1].start)

    def frames_at(self, cam_id: str, at: float, count: int = 5, spacing_sec: float = 0.2) -> Tuple[List[np.ndarray], dict]:
        """``count`` frames from ``at`` (epoch seconds) on, ``spacing_sec`` apart."""
        if at > time.time() + 1.0:
            raise RecordingNotFound("Thời điểm đã chọn ở tương lai")
        segments = self.segments(cam_id)
        covered = self.coverage(cam_id)
        where = (f" (video ghi hình hiện có từ {_format(covered[0])} đến {_format(covered[1])})" if covered
                 else " (camera này chưa có video ghi hình nào)")
        candidates = [s for s in segments if s.start <= at]
        if not candidates:
            raise RecordingNotFound(f"Không có video ghi hình tại {_format(at)}{where}")
        segment = candidates[-1]
        offset = at - segment.start
        capture = cv2.VideoCapture(str(segment.path), cv2.CAP_FFMPEG)
        if not capture.isOpened():
            raise RecordingNotFound(f"Không mở được video ghi hình {segment.path.name}")
        frames: List[np.ndarray] = []
        first_position: Optional[float] = None
        try:
            fps = capture.get(cv2.CAP_PROP_FPS)
            fps = fps if 0.0 < fps < 240.0 else 25.0
            step = max(1, int(round(spacing_sec * fps)))
            capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, offset) * 1000.0)
            index = 0
            while len(frames) < count and index < count * step + int(fps * 2):
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                if index % step == 0:
                    if first_position is None:
                        first_position = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    frames.append(frame)
                index += 1
        finally:
            capture.release()
        if not frames or first_position is None or abs(first_position - offset) > SEEK_TOLERANCE_SEC:
            raise RecordingNotFound(f"Video ghi hình không có dữ liệu tại {_format(at)}{where}")
        return frames, {"file": segment.path.name, "segment_start": segment.start,
                        "offset_sec": round(first_position, 3), "at": round(segment.start + first_position, 3)}
