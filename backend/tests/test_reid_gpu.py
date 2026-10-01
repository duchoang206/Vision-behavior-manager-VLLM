import os
import unittest

import numpy as np

MODEL = os.getenv("REID_ONNX_FILE", "/app/models/reid_512.onnx")


def cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


@unittest.skipUnless(os.path.exists(MODEL) and cuda_available(), "needs the ReID ONNX file and a CUDA device")
class GpuReidEncoderTests(unittest.TestCase):
    def test_gpu_vectors_match_the_cpu_encoder(self):
        from core.reid_encoder import encode_crop
        from core.reid_gpu import GpuReidEncoder

        rng = np.random.RandomState(0)
        images = [(rng.rand(180 + 20 * i, 90 + 10 * i, 3) * 255).astype("uint8") for i in range(4)]
        gpu = GpuReidEncoder().encode_batch(images + [None])
        self.assertIsNone(gpu[-1])
        for image, vector in zip(images, gpu):
            self.assertAlmostEqual(1.0, float(np.linalg.norm(vector)), places=3)
            self.assertGreater(float(np.dot(vector, encode_crop(image))), .995)


if __name__ == "__main__":
    unittest.main()
