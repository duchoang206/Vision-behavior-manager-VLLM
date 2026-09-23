import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from core.registered_target_mask import MaskRuntime


class MaskPromptShapeTests(unittest.TestCase):
    def test_registered_mask_keeps_channel_for_unresized_letterbox(self):
        from ultralytics.data.augment import LetterBox

        for height, width in [(360, 640), (640, 640), (720, 1280)]:
            with self.subTest(shape=(height, width)):
                runtime = MaskRuntime.__new__(MaskRuntime)
                runtime.cameras = {}
                runtime.options = {}
                runtime.preview = SimpleNamespace(model=object())
                runtime.identity = mock.Mock()
                runtime._identity_descriptor = mock.Mock(return_value=mock.Mock())
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                predictor = mock.Mock()
                predictor.memory_bank = []
                predictor.obj_idx_set = set()
                predictor.obj_id_to_idx = {0: 0}
                predictor.obj_idx_to_id = {0: 0}
                predictor._max_obj_num = 1

                def seed(**prompts):
                    masks = prompts["masks"]
                    self.assertEqual((1, height, width, 1), masks.shape)
                    for bitmap in masks:
                        transformed = LetterBox((640, 640), auto=False, center=False)(image=bitmap)
                        self.assertEqual((640, 640, 1), transformed.shape)

                predictor.side_effect = seed
                with mock.patch("core.sam2_live_predictor.LiveSAM2Predictor", return_value=predictor), \
                        mock.patch("core.registered_target_mask.decode_frame", return_value=frame):
                    runtime.configure("new-camera", [{"label": "Rack_A", "samples": [{
                        "mask": {"polygons": [[[.1, .1], [.3, .1], [.3, .4]]]}, "frame_image": "encoded"
                    }]}])
                predictor.assert_called_once()
                self.assertFalse(runtime.cameras["new-camera"]["objects"]["Rack_A"]["seeded"])
                self.assertEqual([], predictor.memory_bank)


if __name__ == "__main__":
    unittest.main()
