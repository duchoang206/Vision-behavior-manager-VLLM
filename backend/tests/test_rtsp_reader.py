import sys
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

BACKEND_DIR = str(Path(__file__).resolve().parents[1])
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from core.rtsp_reader import RTSPLatestFrameReader


class LatestFrameReaderTests(unittest.TestCase):
    def test_default_reader_keeps_original_decoder_configuration(self):
        with mock.patch('core.rtsp_reader.cv2.VideoCapture') as capture, \
                mock.patch('core.rtsp_reader.threading.Thread'):
            reader = RTSPLatestFrameReader('rtsp://test')
            capture.assert_called_once_with('rtsp://test')
            reader.stop()

    def test_low_latency_options_are_used_on_open_and_reconnect(self):
        with mock.patch('core.rtsp_reader.cv2.VideoCapture') as capture, \
                mock.patch('core.rtsp_reader.threading.Thread'), \
                mock.patch('core.rtsp_reader.time.sleep') as sleep:
            reader = RTSPLatestFrameReader('rtsp://test', decoder_threads=1)
            expected = mock.call('rtsp://test', cv2.CAP_FFMPEG, [
                cv2.CAP_PROP_N_THREADS, 1,
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC, 2000,
            ])
            capture.return_value.isOpened.return_value = False
            sleep.side_effect = lambda duration: setattr(reader, 'running', False)
            reader._capture_frames()
            self.assertEqual([expected, expected], capture.call_args_list)
            reader.stop()

    def test_latest_frame_and_decode_timestamp_stay_paired(self):
        first = np.zeros((20, 20, 3), dtype=np.uint8)
        latest = np.ones_like(first)
        with mock.patch('core.rtsp_reader.cv2.VideoCapture') as capture, \
                mock.patch('core.rtsp_reader.threading.Thread'), \
                mock.patch('core.rtsp_reader.time.time', side_effect=[100.0, 100.04]):
            reader = RTSPLatestFrameReader('rtsp://test', decoder_threads=1)
            capture.return_value.isOpened.return_value = True
            frames = iter([first, latest])

            def read_frame():
                frame = next(frames)
                if frame is latest:
                    reader.running = False
                return True, frame

            capture.return_value.read.side_effect = read_frame
            reader._capture_frames()
            available, frame, decoded_at = reader.get_latest_frame_packet()
            self.assertTrue(available)
            self.assertIs(latest, frame)
            self.assertEqual(100.04, decoded_at)
            self.assertEqual((False, None, None), reader.get_latest_frame_packet())
            reader.frame_queue.put((first, 101.0))
            available, frame = reader.get_latest_frame()
            self.assertTrue(available)
            self.assertIs(first, frame)
            self.assertEqual((False, None), reader.get_latest_frame())
            reader.stop()


if __name__ == '__main__':
    unittest.main()
