import tempfile
from pathlib import Path
from unittest import TestCase, mock

from core.ffmpeg_recorder import FFmpegRecorder


class RecorderRecoveryTests(TestCase):
    def test_recovery_skips_files_already_in_archive_index(self):
        with tempfile.TemporaryDirectory() as root:
            camera_dir = Path(root) / "cam1"
            camera_dir.mkdir()
            known = camera_dir / "20260925T010203Z_aaaaaaaaaaaa.mp4"
            unknown = camera_dir / "20260925T010204Z_bbbbbbbbbbbb.mp4"
            known.touch()
            unknown.touch()

            store = mock.Mock()
            store.pending_recovery.return_value = []
            store.indexed_paths.return_value = {f"cam1/{known.name}"}
            recorder = FFmpegRecorder(store, root=root)

            with mock.patch.object(recorder, "_finalize") as finalize:
                recorder._recover()

            finalize.assert_called_once_with(
                "cam1", unknown.resolve(), interrupted=True
            )
            self.assertIn(f"cam1/{known.name}", recorder.finalized)
