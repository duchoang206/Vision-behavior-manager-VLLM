import logging
import re
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as functional


def polygon_bitmap(polygons, shape, device):
    height, width = shape
    horizontal = (torch.arange(width, device=device, dtype=torch.float32) + .5) / width
    vertical = (torch.arange(height, device=device, dtype=torch.float32) + .5) / height
    inside = torch.zeros((height, width), dtype=torch.bool, device=device)
    for ring in polygons:
        vertices = torch.as_tensor(ring, dtype=torch.float32, device=device)
        following = torch.roll(vertices, -1, 0)
        for offset in range(0, len(ring), 32):
            start = vertices[offset:offset + 32]
            end = following[offset:offset + 32]
            delta = end[:, 1] - start[:, 1]
            safe_delta = torch.where(delta.abs() > 1e-10, delta, torch.ones_like(delta))
            crossings = ((start[:, 1, None, None] > vertical[None, :, None])
                         != (end[:, 1, None, None] > vertical[None, :, None]))
            edge_x = ((end[:, 0] - start[:, 0])[:, None, None]
                      * (vertical[None, :, None] - start[:, 1, None, None])
                      / safe_delta[:, None, None] + start[:, 0, None, None])
            inside ^= (crossings & (horizontal[None, None, :] < edge_x)).sum(0).remainder(2).bool()
    return inside


class LabelReferenceMemory:
    def __init__(self, directory, capacity=48):
        self.directory = Path(directory) if directory else None
        self.capacity = capacity
        self.entries = OrderedDict()
        self.failures = {}
        self.encoded = 0

    def _encode(self, runtime, sample):
        from torchvision.io import decode_jpeg

        identifier = sample['id']
        if not self.directory or not re.fullmatch(r'[a-f0-9]{32}', identifier):
            raise ValueError('Invalid Label reference identifier')
        path = self.directory / 'samples' / f'{identifier}.jpg'
        if path.is_symlink():
            raise ValueError('Invalid Label reference path')
        encoded = torch.frombuffer(bytearray(path.read_bytes()), dtype=torch.uint8)
        image = decode_jpeg(encoded, device=runtime.base.device).permute(1, 2, 0)
        tensor, backbone, resized, shape = runtime._image(image)
        predictor = runtime._predictor(backbone, shape)
        predictor.get_im_features(tensor)
        mask = polygon_bitmap(sample['mask']['polygons'], resized, tensor.device)[None, None].float()
        mask = functional.pad(mask, (0, runtime.size - resized[1], 0, runtime.size - resized[0]))
        predictor.seed_reference_mask(mask)
        self.encoded += 1
        return predictor.memory_bank[0]

    def get(self, runtime, sample):
        identifier = sample['id']
        if identifier in self.entries:
            self.entries.move_to_end(identifier)
            return self.entries[identifier]
        if identifier in self.failures:
            return None
        try:
            memory = self._encode(runtime, sample)
        except (OSError, ValueError, RuntimeError) as error:
            if isinstance(error, torch.cuda.OutOfMemoryError):
                raise
            self.failures[identifier] = f'{type(error).__name__}: {str(error)[:240]}'
            logging.warning('Label reference %s could not be encoded: %s', identifier, self.failures[identifier])
            return None
        self.entries[identifier] = memory
        while len(self.entries) > self.capacity:
            self.entries.popitem(last=False)
        return memory

    def prune(self, active):
        for records in (self.entries, self.failures):
            for identifier in list(records):
                if identifier not in active:
                    del records[identifier]

    def clear(self):
        self.entries.clear()
        self.failures.clear()
