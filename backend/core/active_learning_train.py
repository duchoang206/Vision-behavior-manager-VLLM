import csv
import gc
import json
import os
from pathlib import Path
import shutil
import sys

from core.active_learning_data import atomic_write, json_bytes, quality_gate
from core.model_contract import parse_labels


def metrics_payload(metrics, names):
    return dict(map50_95=float(metrics.box.map), map50=float(metrics.box.map50),
                per_class={name: float(metrics.box.maps[index]) for index, name in enumerate(names)})


def run(request):
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
    import torch
    from ultralytics import YOLO

    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("Retrain cần GPU riêng hoặc chờ GPU live rảnh; không chạy fallback CPU.")
    output = Path(request["output"])
    output.mkdir(parents=True, exist_ok=True)
    names, config = request["labels"], request["config"]
    common = dict(data=request["dataset"], device=0, imgsz=request["imgsz"], batch=1,
                  workers=0, plots=False, verbose=False, save_json=False, project=str(output))
    if request["stage"] == "engine_validation":
        baseline_model = YOLO(request["baseline_engine"], task="detect")
        baseline = metrics_payload(baseline_model.val(**common, name="baseline_engine_validation"), names)
        del baseline_model
        gc.collect()
        torch.cuda.empty_cache()
        candidate_model = YOLO(request["engine"], task="detect")
        candidate = metrics_payload(candidate_model.val(**common, name="candidate_engine_validation"), names)
        accepted, reason = quality_gate(baseline, candidate, config["minimum_gain"], config["max_class_drop"])
        atomic_write(output / "engine_metrics.json", json_bytes(dict(baseline=baseline, candidate=candidate,
                                                                    accepted=accepted, reason=reason)))
        return
    model = YOLO(request["weights"])
    if model.task != "detect" or parse_labels(model.names) != names:
        raise ValueError("Checkpoint cần là YOLO detection cùng kiến trúc và cùng thứ tự lớp model đang chạy.")
    model.train(data=request["dataset"], device=0, imgsz=request["imgsz"], epochs=config["epochs"],
                batch=config["batch"], lr0=config["learning_rate"], lrf=.1, optimizer="AdamW",
                workers=0, cache=False, amp=False, resume=False, pretrained=True, seed=42,
                project=str(output), name="finetune", exist_ok=False, plots=False, save=True,
                close_mosaic=0, mosaic=0, mixup=0, copy_paste=0, patience=config["epochs"])
    best = Path(model.trainer.best)
    if not best.is_file():
        raise RuntimeError("Training không xuất best.pt.")
    shutil.copyfile(best, output / "best.pt")
    history_file = Path(model.trainer.save_dir) / "results.csv"
    with history_file.open(newline="") as source:
        history = [{key.strip(): value.strip() for key, value in row.items()} for row in csv.DictReader(source)]
    del model
    gc.collect()
    torch.cuda.empty_cache()
    candidate_model = YOLO(str(output / "best.pt"))
    candidate = metrics_payload(candidate_model.val(**common, name="candidate_validation"), names)
    exported = candidate_model.export(format="onnx", imgsz=request["imgsz"], batch=1, dynamic=False,
                                      half=False, nms=False, simplify=False, opset=17, device=0)
    if Path(exported).resolve() != (output / "model.onnx").resolve():
        shutil.copyfile(exported, output / "model.onnx")
    atomic_write(output / "training_metrics.json", json_bytes(dict(candidate_pt=candidate, history=history,
                                                                  metric="bbox_mAP50_95", sam2_trained=False)))


if __name__ == "__main__":
    run(json.loads(Path(sys.argv[1]).read_text()))
