"""Native DeepStream MaskTracker configuration and metadata helpers.

The native tracker is opt-in because it requires the SAM2 TensorRT assets to
be present in the runtime image.  Keeping the switch explicit lets the
existing Python SAM2 path remain a safe fallback during rollout.
"""

from __future__ import annotations

import os
import re
from pathlib import Path


NATIVE_TRACKER_ENV = "DEEPSTREAM_NATIVE_MASKTRACKER"


def native_masktracker_enabled() -> bool:
    return os.getenv(NATIVE_TRACKER_ENV, "0").strip().lower() in {"1", "true", "yes", "on"}


def _required_assets(model_dir: Path) -> tuple[Path, ...]:
    return tuple(model_dir / name for name in (
        "image_encoder.onnx",
        "image_encoder.engine",
        "mask_decoder.onnx",
        "mask_decoder.engine",
        "memory_attention.onnx",
        "memory_attention.engine",
        "memory_encoder.onnx",
        "memory_encoder.engine",
    ))


def prepare_native_tracker_config(model_dir: str | os.PathLike[str] | None = None) -> str:
    """Create a project-local MaskTracker config using the mounted SAM2 assets."""

    model_root = Path(model_dir or os.getenv("DEEPSTREAM_SAM2_MODEL_DIR", "/app/models/sam2_tracker")).resolve()
    source_root = Path(
        os.getenv(
            "DEEPSTREAM_MASKTRACKER_CONFIG_ROOT",
            "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app",
        )
    )
    tracker_template = model_root / "config_tracker_MaskTracker.yml"
    if not tracker_template.is_file():
        tracker_template = source_root / "config_tracker_MaskTracker.yml"
    segmenter_template = model_root / "config_tracker_module_Segmenter.yml"
    if not segmenter_template.is_file():
        segmenter_template = source_root / "config_tracker_module_Segmenter.yml"
    
    # Check if segmenter has memory modules enabled
    seg_text_check = segmenter_template.read_text() if segmenter_template.is_file() else ""
    has_mem_attention = bool(re.search(r"^\s*MemoryAttention:\s*$", seg_text_check, re.MULTILINE))
    has_mem_encoder = bool(re.search(r"^\s*MemoryEncoder:\s*$", seg_text_check, re.MULTILINE))

    required_names = ["image_encoder.onnx", "image_encoder.engine", "mask_decoder.onnx", "mask_decoder.engine"]
    if has_mem_attention:
        required_names.extend(["memory_attention.onnx", "memory_attention.engine"])
    if has_mem_encoder:
        required_names.extend(["memory_encoder.onnx", "memory_encoder.engine"])

    missing = [str(model_root / name) for name in required_names if not (model_root / name).is_file() and not name.endswith(".engine")]
    if not tracker_template.is_file():
        missing.append(str(tracker_template))
    if not segmenter_template.is_file():
        missing.append(str(segmenter_template))
    if missing:
        raise FileNotFoundError("Native DeepStream MaskTracker thiếu tài sản: " + ", ".join(missing))

    model_root.mkdir(parents=True, exist_ok=True)
    segmenter_path = model_root / "config_tracker_module_Segmenter.yml"
    tracker_path = model_root / "config_tracker_MaskTracker.yml"
    text = segmenter_template.read_text()
    network_files = {
        "ImageEncoder": (model_root / "image_encoder.onnx", model_root / "image_encoder.engine"),
        "MaskDecoder": (model_root / "mask_decoder.onnx", model_root / "mask_decoder.engine"),
        "MemoryAttention": (model_root / "memory_attention.onnx", model_root / "memory_attention.engine"),
        "MemoryEncoder": (model_root / "memory_encoder.onnx", model_root / "memory_encoder.engine"),
    }
    section = None
    rewritten = []
    for line in text.splitlines(keepends=True):
        match = re.match(r"^(ImageEncoder|MaskDecoder|MemoryAttention|MemoryEncoder):\s*$", line.strip())
        if match:
            section = match.group(1)
        elif re.match(r"^#\s*(ImageEncoder|MaskDecoder|MemoryAttention|MemoryEncoder):\s*$", line.strip()):
            section = None  # Commented out module
        if section and "onnxFile:" in line:
            line = re.sub(r'("?)[^"\s]+\1(\s+#.*)?$', lambda m: f'"{network_files[section][0]}"{m.group(2) or ""}', line)
        elif section and "modelEngineFile:" in line:
            line = re.sub(r'("?)[^"\s]+\1(\s+#.*)?$', lambda m: f'"{network_files[section][1]}"{m.group(2) or ""}', line)
        rewritten.append(line)
    text = "".join(rewritten)
    if not segmenter_path.exists() or segmenter_path.read_text() != text:
        segmenter_path.write_text(text)

    validate_native_segmenter_config(segmenter_path, model_root)

    tracker_text = tracker_template.read_text()
    tracker_text = tracker_text.replace(
        "segmenterConfigPath: \"/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_module_Segmenter.yml\"",
        f"segmenterConfigPath: \"{segmenter_path}\"",
    )
    tracker_text = tracker_text.replace(
        "/opt/nvidia/deepstream/deepstream-8.0/samples/configs/deepstream-app/config_tracker_module_Segmenter.yml",
        str(segmenter_path),
    )
    if not tracker_path.exists() or tracker_path.read_text() != tracker_text:
        tracker_path.write_text(tracker_text)
    validate_native_tracker_config(tracker_path, segmenter_path)
    return str(tracker_path)


def validate_native_segmenter_config(path: str | os.PathLike[str], model_root: Path) -> None:
    text = Path(path).read_text()
    required = [
        "inferDims: [3, 1024, 1024]",
        "ImageEncoder:",
        "MaskDecoder:",
        "offsets:",
        "netScaleFactor:",
    ]
    # Check if memory modules are enabled
    has_mem_attention = bool(re.search(r"^\s*MemoryAttention:\s*$", text, re.MULTILINE))
    has_mem_encoder = bool(re.search(r"^\s*MemoryEncoder:\s*$", text, re.MULTILINE))
    if has_mem_attention:
        required.append("MemoryAttention:")
    if has_mem_encoder:
        required.append("MemoryEncoder:")
    missing = [entry for entry in required if entry not in text]
    if missing:
        raise ValueError("Native Segmenter config thiếu trường chuẩn: " + ", ".join(missing))
    if "enableReAssoc: 0" in text:
        raise ValueError("Native Segmenter config chứa override enableReAssoc=0 không hợp lệ.")
    check_assets = ["image_encoder.onnx", "image_encoder.engine", "mask_decoder.onnx"]
    if has_mem_attention:
        check_assets.extend(["memory_attention.onnx", "memory_attention.engine"])
    if has_mem_encoder:
        check_assets.extend(["memory_encoder.onnx", "memory_encoder.engine"])
    if any(not (model_root / name).is_file() for name in check_assets):
        raise FileNotFoundError("Native Segmenter config trỏ tới SAM2 asset không tồn tại.")


def validate_native_tracker_config(path: str | os.PathLike[str], segmenter_path: Path) -> None:
    text = Path(path).read_text()
    required = (
        "BaseConfig:",
        "TargetManagement:",
        "DataAssociator:",
        "StateEstimator:",
        "Segmenter:",
        "segmenterType: 1",
    )
    missing = [entry for entry in required if entry not in text]
    if missing:
        raise ValueError("Native MaskTracker config thiếu trường chuẩn: " + ", ".join(missing))
    if str(segmenter_path) not in text:
        raise ValueError("MaskTracker không trỏ đúng config Segmenter project-local.")
    if "enableReAssoc: 0" in text:
        raise ValueError("Native MaskTracker chứa override enableReAssoc=0 không hợp lệ.")
