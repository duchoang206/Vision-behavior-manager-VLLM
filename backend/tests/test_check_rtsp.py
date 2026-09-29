import json
import sys
import unittest
from pathlib import Path
from unittest import mock


BACKEND_DIR = str(Path(__file__).resolve().parents[1])
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from check_rtsp import is_rtsp_valid_async


class RtspProbeTests(unittest.IsolatedAsyncioTestCase):
    async def _probe(self, streams, returncode=0):
        process = mock.Mock()
        process.returncode = returncode
        process.communicate = mock.AsyncMock(return_value=(json.dumps({"streams": streams}).encode(), b""))
        with mock.patch("check_rtsp.asyncio.create_subprocess_exec", new=mock.AsyncMock(return_value=process)):
            return await is_rtsp_valid_async("rtsp://127.0.0.1:8554/camera", timeout=1)

    async def test_requires_a_real_video_shape(self):
        self.assertTrue(await self._probe([{"codec_type": "video", "width": 1280, "height": 720}]))
        self.assertFalse(await self._probe([{"codec_type": "video", "width": 0, "height": 0}]))

    async def test_rejects_probe_failure_and_non_video_stream(self):
        self.assertFalse(await self._probe([{"codec_type": "audio", "width": 0, "height": 0}]))
        self.assertFalse(await self._probe([], returncode=1))


if __name__ == "__main__":
    unittest.main()
