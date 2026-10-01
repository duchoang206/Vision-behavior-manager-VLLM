"""GPU ReID encoder: runs the same reid_512.onnx weights through PyTorch/CUDA.

The container's onnxruntime is CPU-only and costs ~65 ms of CPU per crop. The network is a plain ResNet-50
(Conv/Relu/Add/pool + a small normalisation tail), so the ONNX graph is replayed op by op on the GPU. Output
vectors match core.reid_encoder.encode_crop (L2-normalised 512-D) within fp16 rounding.
"""

import os
import threading

import cv2
import numpy as np

_MEAN = np.asarray([123.675, 116.280, 103.530], dtype=np.float32)
_SCALE = 0.01735207


class GpuReidEncoder:
    def __init__(self, model_path=None, device="cuda:0", half=True):
        import onnx
        import torch
        from onnx import numpy_helper

        self.torch = torch
        if os.getenv("REID_CUDNN", "0") != "1":
            torch.backends.cudnn.enabled = False  # cuDNN sub-libraries fail to load in this container
        self.device = torch.device(device)
        self.half = half
        model = onnx.load(model_path or os.getenv("REID_ONNX_FILE", "/app/models/reid_512.onnx"))
        self.graph = model.graph
        dtype = torch.float16 if half else torch.float32
        self.weights = {}
        for init in self.graph.initializer:
            tensor = torch.from_numpy(np.array(numpy_helper.to_array(init)))
            # integer initializers are shape arithmetic: keep them on the CPU
            self.weights[init.name] = tensor.to(self.device, dtype) if tensor.is_floating_point() else tensor
        self.input_name = next(item.name for item in self.graph.input if item.name not in self.weights)
        self.lock = threading.Lock()
        self.ops = {"Conv": self._conv, "Relu": self._relu, "Add": self._add, "MaxPool": self._maxpool,
                    "GlobalAveragePool": self._gap, "Shape": self._shape, "Gather": self._gather,
                    "Unsqueeze": self._unsqueeze, "Concat": self._concat, "Reshape": self._reshape,
                    "MatMul": self._matmul, "ReduceL2": self._reduce_l2, "Clip": self._clip, "Expand": self._expand,
                    "Div": self._div, "Constant": self._constant}
        unknown = {node.op_type for node in self.graph.node} - set(self.ops)
        if unknown:
            raise NotImplementedError(f"ReID graph uses unsupported ops: {sorted(unknown)}")

    # ---- op implementations -------------------------------------------------------------------------------
    @staticmethod
    def _attrs(node):
        from onnx import helper
        return {attribute.name: helper.get_attribute_value(attribute) for attribute in node.attribute}

    def _conv(self, node, x):
        functional = self.torch.nn.functional
        a = self._attrs(node)
        pads = a.get("pads", [0, 0, 0, 0])
        image = x[0]
        if pads[0] != pads[2] or pads[1] != pads[3]:
            image = functional.pad(image, (pads[1], pads[3], pads[0], pads[2]))
            padding = 0
        else:
            padding = (pads[0], pads[1])
        return functional.conv2d(image, x[1], x[2] if len(x) > 2 else None, stride=tuple(a.get("strides", [1, 1])),
                                 padding=padding, dilation=tuple(a.get("dilations", [1, 1])), groups=a.get("group", 1))

    def _relu(self, node, x):
        return self.torch.relu(x[0])

    def _add(self, node, x):
        return x[0] + x[1]

    def _maxpool(self, node, x):
        a = self._attrs(node)
        pads = a.get("pads", [0, 0, 0, 0])
        return self.torch.nn.functional.max_pool2d(x[0], tuple(a["kernel_shape"]), tuple(a.get("strides", [1, 1])),
                                                   (pads[0], pads[1]), ceil_mode=bool(a.get("ceil_mode", 0)))

    def _gap(self, node, x):
        return x[0].mean(dim=(2, 3), keepdim=True)

    def _shape(self, node, x):
        return self.torch.tensor(list(x[0].shape), dtype=self.torch.int64)

    def _gather(self, node, x):
        index = x[1]
        return x[0][index] if index.dim() else x[0][int(index)]

    def _unsqueeze(self, node, x):
        axes = x[1].tolist() if len(x) > 1 else self._attrs(node)["axes"]
        value = x[0]
        for axis in sorted(axes):
            value = value.unsqueeze(axis)
        return value

    def _concat(self, node, x):
        axis = self._attrs(node).get("axis", 0)
        return self.torch.cat([item.reshape(-1) if item.dim() == 0 else item for item in x], dim=axis)

    def _reshape(self, node, x):
        shape = [int(v) for v in x[1].tolist()]
        return x[0].reshape([x[0].shape[i] if v == 0 else v for i, v in enumerate(shape)])

    def _matmul(self, node, x):
        return x[0] @ x[1]

    def _reduce_l2(self, node, x):
        a = self._attrs(node)
        axes = tuple(a.get("axes", [-1]))
        return self.torch.sqrt((x[0].float() ** 2).sum(dim=axes, keepdim=bool(a.get("keepdims", 1))))

    def _clip(self, node, x):
        low = x[1] if len(x) > 1 and x[1] is not None else None
        high = x[2] if len(x) > 2 and x[2] is not None else None
        return x[0].clamp(min=float(low) if low is not None else None, max=float(high) if high is not None else None)

    def _expand(self, node, x):
        return x[0].expand(*[int(v) for v in x[1].tolist()])

    def _div(self, node, x):
        return x[0].float() / x[1].float()

    def _constant(self, node, x):
        from onnx import numpy_helper
        value = self._attrs(node)["value"]
        tensor = self.torch.from_numpy(np.array(numpy_helper.to_array(value)))
        return tensor.to(self.device) if tensor.is_floating_point() else tensor

    # ---- inference ----------------------------------------------------------------------------------------
    def _forward(self, batch):
        values = dict(self.weights)
        values[self.input_name] = batch
        for node in self.graph.node:
            args = [values[name] if name else None for name in node.input]
            values[node.output[0]] = self.ops[node.op_type](node, args)
        return values[self.graph.output[0].name]

    def encode_batch(self, images):
        """images: list of BGR uint8 crops -> list of unit 512-D float lists (None for an unusable crop)."""
        torch = self.torch
        usable = [(index, image) for index, image in enumerate(images) if image is not None and image.size]
        results = [None] * len(images)
        if not usable:
            return results
        batch = np.stack([self._prepare(image) for _, image in usable])
        with self.lock, torch.inference_mode():
            tensor = torch.from_numpy(batch).to(self.device, torch.float16 if self.half else torch.float32)
            output = self._forward(tensor).float()
            output = torch.nn.functional.normalize(output.reshape(output.shape[0], -1), dim=1)
        vectors = output.cpu().numpy()
        for (index, _), vector in zip(usable, vectors):
            if np.isfinite(vector).all() and float(np.linalg.norm(vector)) > 1e-6:
                results[index] = vector.astype(np.float32).tolist()
        return results

    @staticmethod
    def _prepare(image):
        resized = cv2.resize(image, (128, 256), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        return np.transpose((rgb - _MEAN) * _SCALE, (2, 0, 1))
