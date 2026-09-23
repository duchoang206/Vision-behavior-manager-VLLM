import sys


def network_structure(path):
    import onnx
    model = onnx.load(path, load_external_data=False)
    initializers = {item.name: (item.data_type, tuple(item.dims)) for item in model.graph.initializer}
    result = []
    for node in model.graph.node:
        if node.op_type not in {"Conv", "ConvTranspose", "Gemm", "MatMul"}:
            continue
        shapes = tuple(initializers.get(name) for name in node.input[1:])
        result.append((node.domain, node.op_type, shapes,
                       tuple(attribute.SerializeToString() for attribute in sorted(node.attribute, key=lambda item: item.name))))
    if not result:
        raise ValueError("Không xác minh được kiến trúc trọng số ONNX.")
    return result


if __name__ == "__main__":
    if network_structure(sys.argv[1]) != network_structure(sys.argv[2]):
        raise ValueError("Kiến trúc ONNX khác baseline. Cần cùng kiến trúc/exporter cho hot-reload, không tự thay runtime.")
