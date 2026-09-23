import cv2
import unittest
import numpy as np

from core.deepstream_native_mask import mask_from_object_meta, polygon_contract_iou


class _SyntheticMaskParams:
    def __init__(self, mask_array, threshold=0.5):
        self.height, self.width = mask_array.shape[:2]
        self._array = mask_array.astype(np.float32)
        self.threshold = threshold

    def get_mask_array(self):
        return self._array.flatten()


class _SyntheticObject:
    def __init__(self, mask_params):
        self.mask_params = mask_params


class NativeMaskContractTests(unittest.TestCase):
    def test_native_mask_is_bbox_relative_polygon(self):
        bitmap = np.zeros((24, 32), dtype=np.float32)
        bitmap[4:20, 8:24] = 1.0
        result = mask_from_object_meta(_SyntheticObject(_SyntheticMaskParams(bitmap)), None)
        self.assertIsNotNone(result)
        self.assertEqual(result["source"], "deepstream_masktracker")
        self.assertTrue(result["polygons"])
        self.assertTrue(all(0 <= value <= 1 for ring in result["polygons"] for point in ring for value in point))
        self.assertIn("contract_iou", result)
        self.assertGreaterEqual(result["contract_iou"], 0.98)

    def test_raster_to_polygon_iou_circle(self):
        mask = np.zeros((128, 128), dtype=np.float32)
        cv2.circle(mask, (64, 64), 40, 1.0, -1)
        result = mask_from_object_meta(_SyntheticObject(_SyntheticMaskParams(mask)), None)
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result["contract_iou"], 0.98)

    def test_raster_to_polygon_iou_robot_irregular_shape(self):
        mask = np.zeros((200, 200), dtype=np.float32)
        pts = np.array([[30, 40], [170, 35], [190, 160], [150, 190], [50, 175]], np.int32)
        cv2.fillPoly(mask, [pts], 1.0)
        result = mask_from_object_meta(_SyntheticObject(_SyntheticMaskParams(mask)), None)
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result["contract_iou"], 0.99)

    def test_empty_or_subthreshold_mask_returns_none(self):
        empty_mask = np.zeros((64, 64), dtype=np.float32)
        self.assertIsNone(mask_from_object_meta(_SyntheticObject(_SyntheticMaskParams(empty_mask)), None))

        subthreshold_mask = np.full((64, 64), 0.3, dtype=np.float32)
        self.assertIsNone(mask_from_object_meta(_SyntheticObject(_SyntheticMaskParams(subthreshold_mask, threshold=0.5)), None))

    def test_missing_or_invalid_params_returns_none_safely(self):
        self.assertIsNone(mask_from_object_meta(None, None))
        self.assertIsNone(mask_from_object_meta(object(), None))


if __name__ == "__main__":
    unittest.main()

