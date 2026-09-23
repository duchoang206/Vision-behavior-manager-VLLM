import hashlib
import json
from pathlib import Path
import shutil

from core.active_learning_data import atomic_write, checksum, detection_lines, json_bytes
from core.model_contract import parse_labels


def local_path(root, value):
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Dataset không được tham chiếu đường dẫn bên ngoài base/.")
    return path


def dataset_images(root, entries):
    entries = entries if isinstance(entries, list) else [entries]
    images = set()
    for entry in entries:
        if not isinstance(entry, str):
            raise ValueError("train/val phải là thư mục hoặc danh sách ảnh cục bộ.")
        path = local_path(root, entry)
        if path.is_dir():
            paths = path.rglob("*")
        elif path.is_file() and path.suffix == ".txt":
            paths = [local_path(root, line.strip()) for line in path.read_text().splitlines() if line.strip()]
        else:
            paths = [path]
        for image in paths:
            image = image.resolve()
            if image.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            if not image.is_relative_to(root.resolve()) or not image.is_file():
                raise ValueError("Ảnh dataset không hợp lệ hoặc nằm ngoài base/.")
            images.add(image)
    return sorted(images)


def image_label(image, root):
    relative = image.relative_to(root)
    parts = list(relative.parts)
    if "images" not in parts:
        raise ValueError("Dataset gốc cần cấu trúc images/ và labels/ của YOLO.")
    parts[parts.index("images")] = "labels"
    label = local_path(root, str(Path(*parts).with_suffix(".txt")))
    if not label.is_file():
        raise ValueError(f"Thiếu label {label.name}; ảnh nền cần file .txt rỗng, không được ngầm coi thiếu nhãn là nền.")
    return label


def validate_detection_text(text, labels):
    classes = set()
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError("Dataset nền hiện phải là YOLO detection (class cx cy w h). Mask correction vẫn được giữ riêng.")
        class_id = int(parts[0])
        center_x, center_y, width, height = map(float, parts[1:])
        if not 0 <= class_id < len(labels) or not all(0 <= value <= 1 for value in (center_x, center_y, width, height)):
            raise ValueError("Label dataset ngoài miền hợp lệ.")
        if width <= 0 or height <= 0 or not (width / 2 <= center_x <= 1 - width / 2 + 1e-6
                                         and height / 2 <= center_y <= 1 - height / 2 + 1e-6):
            raise ValueError("BBox dataset không nằm trong ảnh.")
        classes.add(class_id)
    return classes


def prepare_dataset(store, job, destination):
    import yaml

    model_id, labels = job["model_id"], job["inputs"]["labels"]
    root = (store.directory(model_id) / "base").resolve()
    config_file = local_path(root, str(Path(job["config"]["base_dataset"]).relative_to("base")))
    configuration = yaml.safe_load(config_file.read_text())
    if not isinstance(configuration, dict) or parse_labels(configuration.get("names")) != labels:
        raise ValueError("names dataset gốc phải khớp đúng thứ tự lớp model.")
    if configuration.get("download"):
        raise ValueError("Không chạy download/script từ dataset YAML.")
    data_root = local_path(root, str(configuration.get("path", ".")))
    train = dataset_images(data_root, configuration.get("train"))
    validation = dataset_images(data_root, configuration.get("val"))
    if len(train) < 2 or len(validation) < 2:
        raise ValueError("Cần ít nhất 2 ảnh train gốc và 2 ảnh validation riêng biệt.")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    records, split_hashes, validation_classes = [], {"train": set(), "val": set()}, set()
    required_bytes = sum(image.stat().st_size for image in set(train + validation))
    if shutil.disk_usage(destination).free < required_bytes + 2 * 1024**3:
        raise ValueError("SSD không đủ chỗ tạo dataset cố định cho job.")
    for split, images in (("val", validation), ("train", train)):
        for image in images:
            digest = checksum(image)
            if digest in split_hashes["val" if split == "train" else "train"]:
                raise ValueError("Train và validation có ảnh trùng nội dung; cần tách lại để tránh rò rỉ dữ liệu.")
            if digest in split_hashes[split]:
                continue
            split_hashes[split].add(digest)
            text = image_label(image, data_root).read_text()
            classes = validate_detection_text(text, labels)
            if split == "val":
                validation_classes.update(classes)
            name = "base_" + digest
            target = destination / "images" / split / (name + image.suffix.lower())
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(image, target)
            atomic_write(destination / "labels" / split / (name + ".txt"), text)
            records.append(dict(source="base", split=split, sha256=digest,
                                labels_sha256=hashlib.sha256(text.encode()).hexdigest()))
    if validation_classes != set(range(len(labels))):
        raise ValueError("Tập validation phải có ground truth cho mọi lớp của model.")
    for item in job["inputs"]["samples"]:
        with store.lock:
            row = store.get_sample(model_id, item["id"])
            if row["status"] != "approved" or not row["complete"] or row["annotation_hash"] != item["hash"]:
                raise ValueError("Mẫu đã bị xóa/thay đổi kể từ khi tạo job.")
            image = store.asset(model_id, row["id"], "image")
            digest = checksum(image)
            name = "correction_" + row["id"]
            if digest not in split_hashes["val"] and digest not in split_hashes["train"]:
                shutil.copyfile(image, destination / "images/train" / (name + ".jpg"))
                atomic_write(destination / "labels/train" / (name + ".txt"), detection_lines(row["annotations"]))
                atomic_write(destination / "masks" / (name + ".json"), json_bytes(row["annotations"]))
        if digest != row["frame"]["image_sha256"]:
            raise ValueError("Ảnh correction không khớp checksum đã lưu.")
        if digest in split_hashes["val"]:
            raise ValueError("Correction trùng ảnh validation cố định.")
        if digest in split_hashes["train"]:
            continue
        split_hashes["train"].add(digest)
        records.append(dict(source="correction", split="train", id=row["id"], camera_id=row["camera_id"],
                            frame=row["frame"], sha256=digest, annotation_hash=item["hash"]))
    validation_manifest = [row for row in records if row["split"] == "val"]
    validation_hash = hashlib.sha256(json_bytes(validation_manifest)).hexdigest()
    atomic_write(destination / "manifest.jsonl", "\n".join(json.dumps(row, ensure_ascii=False) for row in records) + "\n")
    atomic_write(destination / "data.yaml", yaml.safe_dump(dict(path=str(destination.resolve()), train="images/train", val="images/val",
                                                                names=dict(enumerate(labels))), allow_unicode=True))
    return dict(dataset=str(destination / "data.yaml"), validation_sha256=validation_hash,
                train_images=len(split_hashes["train"]), validation_images=len(split_hashes["val"]),
                manifest_sha256=checksum(destination / "manifest.jsonl"))
