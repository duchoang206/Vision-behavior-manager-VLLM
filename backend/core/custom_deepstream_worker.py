import json
import hashlib
import math
import os
from pathlib import Path
import sys
import threading
import time

from core.model_track_masks import model_identity
from core.deepstream_native_mask import mask_from_object_meta
from core.deepstream_geometry import clamp_normalized_polygon, normalize_stream_bbox, unletterbox_bbox


def limit_one_object_per_class(objects):
    """Keep the most confident object for each model label in one camera frame."""
    best = {}
    for obj in objects:
        label = str(obj.get("class") or obj.get("label") or "").casefold()
        if not label:
            continue
        try:
            score = float(obj.get("confidence", 0) or 0)
            area = float(obj.get("w", 0) or 0) * float(obj.get("h", 0) or 0)
        except (TypeError, ValueError):
            score, area = -math.inf, -math.inf
        if not math.isfinite(score):
            score = -math.inf
        if not math.isfinite(area):
            area = -math.inf
        previous = best.get(label)
        try:
            previous_score = float(previous.get("confidence", 0) or 0) if previous else -math.inf
            previous_area = (float(previous.get("w", 0) or 0) * float(previous.get("h", 0) or 0)
                             if previous else -math.inf)
        except (TypeError, ValueError):
            previous_score, previous_area = -math.inf, -math.inf
        if not math.isfinite(previous_score):
            previous_score = -math.inf
        if not math.isfinite(previous_area):
            previous_area = -math.inf
        if previous is None or (score, area) > (previous_score, previous_area):
            best[label] = obj
    selected = list(best.values())
    return selected, len(objects) - len(selected)


def run(settings):
    from core.phase0_baseline import initialize
    baseline = initialize()
    import gi
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst, GLib
    import pyds

    Gst.init(None)
    main_loop = GLib.MainLoop()
    pipeline = Gst.Pipeline.new("custom-detector")
    mailbox = {}
    mutex = threading.Lock()
    metadata_jobs = {}
    metadata_mutex = threading.Lock()
    metadata_wake = threading.Event()
    wake = threading.Event()
    stopped = threading.Event()
    cameras = settings["cameras"]
    model_id = settings["model_id"]
    labels = settings["labels"]
    categories = settings.get("categories", [label.lower() for label in labels])
    model_type = str(settings.get("model_type", "detect")).lower()
    stream_width = float(settings.get("stream_width", 1280))
    stream_height = float(settings.get("stream_height", 720))
    network_shape = settings.get("network_shape", [3, 640, 640])
    network_width, network_height = float(network_shape[-1]), float(network_shape[-2])
    metadata_coordinates = str(settings.get("metadata_coordinates", "stream")).lower()
    confidence_cache = {}
    output_lock = threading.Lock()
    from core.model_track_gate import ModelTrackGate
    from core.deepstream_pose import attach_poses_to_tracks, frame_poses
    gate = ModelTrackGate(confidence_thresholds=settings.get("confidence_thresholds", {}))

    def emit(prefix, payload):
        if baseline and baseline.enabled() and prefix == 'RSKY_META ':
            payload['_phase0'] = dict(emit_ns=time.perf_counter_ns(), epoch_ms=time.time() * 1000)
            for stream in payload.get('streams', []):
                baseline.mark(stream['cam_id'], stream['frame_id'], 'metadata_emit')
        encoded = prefix + json.dumps(payload, separators=(",", ":"))
        with output_lock:
            print(encoded, flush=True)

    def attach_masks(camera_id, objects, frame_id, captured_at):
        """Attach YOLO-Seg contours emitted by nvinfer to the stable Web contract."""
        for obj in objects:
            native = obj.pop("native_mask", None)
            if not native:
                continue
            x, y, width, height = (float(obj[name]) for name in ("x", "y", "w", "h"))
            polygons = [clamp_normalized_polygon([[x + point[0] * width, y + point[1] * height] for point in ring])
                        for ring in native["polygons"]]
            obj["mask"] = dict(native, polygons=polygons, frame_id=frame_id,
                                observed_at=captured_at, attached_at=captured_at,
                                source="deepstream_yolo_seg")
            obj["mask_stale"] = False
        return objects

    def segmentation_status():
        return {
            "ready": True,
            "backend": "yolo_seg_native" if model_type == "segment" else "none",
            "mask_backend": "yolo_seg_native" if model_type == "segment" else "none",
            "model_type": model_type,
        }
    reload_pending = {}
    reload_lock = threading.Lock()

    def reload_timeout(identifier):
        with reload_lock:
            pending = reload_pending.pop(identifier, None)
        if pending:
            emit("RSKY_REPLY ", dict(id=identifier, error="nvinfer không xác nhận model update trong 60 giây; runtime cần khôi phục."))
            main_loop.quit()
        return False

    def reload_engine(request):
        try:
            engine = Path(request["engine"]).resolve()
            root = Path(settings.get("model_directory", Path(settings["config"]).parent)).resolve()
            if not engine.is_relative_to(root) or not engine.is_file() or engine.suffix != ".engine":
                raise ValueError("Engine không thuộc thư mục model.")
            with reload_lock:
                if reload_pending:
                    raise ValueError("Một engine khác đang được cập nhật.")
                reload_pending[request["id"]] = dict(request, engine=str(engine))
            detector.set_property("model-engine-file", str(engine))
            GLib.timeout_add_seconds(60, reload_timeout, request["id"])
        except Exception as error:
            with reload_lock:
                reload_pending.pop(request["id"], None)
            emit("RSKY_REPLY ", dict(id=request["id"], error=str(error)))
        return False

    def receive():
        for line in sys.stdin:
            try:
                request = json.loads(line)
                if request.get("action") == "reload_model":
                    GLib.idle_add(reload_engine, request)
                    continue
                if request.get("action") == "reload_labels":
                    # Native YOLO-Seg has no Python prompt/gallery worker. Keep
                    # the command protocol non-breaking for label clients.
                    emit("RSKY_REPLY ", dict(id=request["id"], result={
                        "mode": "native_yolo_seg", "revision": request.get("revision"), "activation": None,
                    }))
                    continue
                if request.get("camera_id") not in cameras:
                    emit("RSKY_REPLY ", dict(id=request.get("id"), error="Camera không thuộc deployment."))
                    continue
                emit("RSKY_REPLY ", dict(id=request.get("id"), error="YOLO-Seg native không nhận prompt mask; hãy train/deploy model segmentation."))
            except (ValueError, TypeError):
                print("Invalid Label command", file=sys.stderr, flush=True)

    def element(kind, name, properties):
        value = Gst.ElementFactory.make(kind, name)
        if value is None:
            raise RuntimeError(f"DeepStream plugin không khả dụng: {kind}")
        for key, setting in properties.items():
            value.set_property(key, setting)
        pipeline.add(value)
        return value

    muxer = element("nvstreammux", "mux", {"batch-size": 1, "width": 1280, "height": 720,
                    "live-source": True, "batched-push-timeout": 10000, "sync-inputs": False,
                    "nvbuf-memory-type": 2})
    detector = element("nvinfer", "detector", {"config-file-path": settings["config"]})

    def model_updated(element, status, engine):
        with reload_lock:
            pending = next((request for request in reload_pending.values()
                            if Path(str(engine)).resolve() == Path(request["engine"])), None)
            if pending:
                reload_pending.pop(pending["id"], None)
        if pending:
            if int(status) == 0:
                settings["model_version"] = pending["version"]
                settings["engine"] = pending["engine"]
                emit("RSKY_REPLY ", dict(id=pending["id"], result=dict(mode="hot_reload", version=pending["version"])))
            else:
                emit("RSKY_REPLY ", dict(id=pending["id"], error=f"nvinfer từ chối engine: {status}"))

    detector.connect("model-updated", model_updated)
    tracker_props = {
        "ll-lib-file": "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so",
        "ll-config-file": settings["tracker"],
        "tracker-width": 640,
        "tracker-height": 384,
        "gpu-id": 0,
    }
    tracker = element("nvtracker", "tracker", tracker_props)
    sink = element("fakesink", "sink", {"sync": False, "async": False, "enable-last-sample": False})
    converter = element("nvvideoconvert", "rgba-gpu", {"nvbuf-memory-type": 2})
    rgba = element("capsfilter", "rgba", {"caps": Gst.Caps.from_string("video/x-raw(memory:NVMM),format=RGBA")})
    if not all((muxer.link(detector), detector.link(tracker), tracker.link(converter), converter.link(rgba), rgba.link(sink))):
        raise RuntimeError("Không nối được GPU pipeline.")

    def link_source(source, pad, target):
        caps = pad.get_current_caps() or pad.query_caps(None)
        if caps and caps.get_structure(0).get_name().startswith("video/"):
            if not caps.get_features(0).contains("memory:NVMM"):
                raise RuntimeError("Decoder không dùng GPU NVMM.")
            if not target.is_linked():
                pad.link(target)

    for source_id, camera_id in enumerate(cameras):
        source = element("nvurisrcbin", f"camera-{source_id}", {"uri": f"rtsp://127.0.0.1:8554/{camera_id}",
                         "source-id": source_id, "rtsp-reconnect-interval": 5, "latency": 80})
        for key, value in (("drop-on-latency", True), ("disable-audio", True), ("cudadec-memtype", 0)):
            if source.find_property(key):
                source.set_property(key, value)
        latest = element("queue", f"latest-{source_id}", {"leaky": 2, "max-size-buffers": 1, "max-size-bytes": 0, "max-size-time": 0})
        sink_pad = muxer.request_pad_simple(f"sink_{source_id}")
        if latest.get_static_pad("src").link(sink_pad) != Gst.PadLinkReturn.OK:
            raise RuntimeError("Không nối được camera vào muxer.")
        source.connect("pad-added", link_source, latest.get_static_pad("sink"))

    def probe(pad, info, data):
        try:
            buffer = info.get_buffer()
            if not buffer:
                return Gst.PadProbeReturn.OK
            batch = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
            frame_list = batch.frame_meta_list if batch else None
            now = time.time()
            while frame_list:
                frame = pyds.NvDsFrameMeta.cast(frame_list.data)
                camera_id = cameras[int(frame.source_id)]
                if baseline:
                    baseline.mark(camera_id, int(frame.frame_num), 'rgba_probe')
                objects = []
                object_list = frame.obj_meta_list
                while object_list:
                    obj = pyds.NvDsObjectMeta.cast(object_list.data)
                    class_id = int(obj.class_id)
                    track_id = int(obj.object_id)
                    confidence = float(obj.confidence)
                    key = (camera_id, track_id, class_id)
                    if confidence >= 0:
                        confidence_cache[key] = (confidence, now)
                    cached = confidence_cache.get(key, (0, 0))
                    observed_at = cached[1]
                    if 0 <= class_id < len(labels) and track_id != 2**64 - 1 and now - observed_at < .3:
                        rect = obj.rect_params
                        if metadata_coordinates == "network":
                            left, top, width, height = unletterbox_bbox(
                                rect.left, rect.top, rect.width, rect.height,
                                stream_width, stream_height, network_width, network_height,
                            )
                        else:
                            # NvDsObjectMeta.rect_params has already been transformed
                            # back from the model's letterbox into muxer/stream space.
                            left, top, width, height = normalize_stream_bbox(
                                rect.left, rect.top, rect.width, rect.height, stream_width, stream_height,
                            )
                        score = cached[0] if confidence < 0 else confidence
                        if all(math.isfinite(value) for value in (left, top, width, height, score)) and width > 0 and height > 0:
                            identity = f"{model_id}:{settings['generation']}:{camera_id}:{track_id}"
                            identifier = 2**40 + int.from_bytes(hashlib.blake2b(identity.encode(), digest_size=6).digest(), 'big')
                            item = {"id": identifier,
                                            "local_id": str(track_id), "class": labels[class_id], "class_id": class_id,
                                            "category": categories[class_id], "label": model_identity(labels[class_id], categories[class_id]), "model_id": model_id,
                                            "x": left, "y": top, "w": width, "h": height, "confidence": score,
                                            "tracking_state": "tracked" if confidence >= 0 else "predicted", "observed_at": now * 1000,
                                            "detected_at": observed_at * 1000,
                                            "frame_width": int(stream_width), "frame_height": int(stream_height), "keypoints": []}
                            if model_type == "segment":
                                native_mask = mask_from_object_meta(obj, pyds)
                                if native_mask:
                                    item["native_mask"] = native_mask
                            objects.append(item)
                    try:
                        object_list = object_list.next
                    except StopIteration:
                        break
                if model_type == "pose" and objects:
                    poses = frame_poses(frame, stream_width, stream_height, network_width, network_height)
                    attach_poses_to_tracks(objects, poses)
                objects, rejected_by_class = limit_one_object_per_class(objects)
                detector_candidates = [{"track_id": obj["local_id"], "class_name": obj["class"],
                                        "confidence": round(obj["confidence"], 4)} for obj in objects[:32]]
                objects, rejected = gate.filter(camera_id, objects, now * 1000)
                if rejected_by_class:
                    rejected["duplicate_class"] = rejected_by_class
                verification_candidates = sum(bool(obj.get("requires_label_verification")) for obj in objects)
                stream = {"cam_id": camera_id, "frame_id": int(frame.frame_num), "frame_pts_ns": int(frame.buf_pts),
                          "source_id": int(frame.source_id), "generation": settings["generation"],
                          "model_version": settings.get("model_version", "original"), "model_id": model_id,
                          "objects": [dict(obj) for obj in objects]}
                with metadata_mutex:
                    metadata_jobs[camera_id] = (stream, now * 1000, rejected, detector_candidates,
                                                 verification_candidates)
                metadata_wake.set()
                try:
                    frame_list = frame_list.next
                except StopIteration:
                    break
            if len(confidence_cache) > 512:
                for key in list(confidence_cache):
                    if now - confidence_cache[key][1] > 2:
                        del confidence_cache[key]
        except Exception as error:
            print(f"Metadata error: {error}", file=sys.stderr, flush=True)
        return Gst.PadProbeReturn.OK

    def send():
        while not stopped.is_set():
            wake.wait(.005)
            wake.clear()
            with mutex:
                batch = list(mailbox.values())
                mailbox.clear()
            for payload in batch:
                if time.time() * 1000 - payload["timestamp"] <= 300:
                    emit("RSKY_META ", payload)

    def publish_metadata():
        while not stopped.is_set():
            metadata_wake.wait(.02)
            metadata_wake.clear()
            with metadata_mutex:
                jobs = list(metadata_jobs.values())
                metadata_jobs.clear()
            for stream, captured_at, rejected, detector_candidates, verification_candidates in jobs:
                camera_id = stream["cam_id"]
                objects = stream["objects"]
                stream["objects"] = attach_masks(camera_id, objects, stream["frame_id"], captured_at)
                stream["segmentation"] = segmentation_status()
                stream["segmentation"]["detector_rejected"] = rejected
                stream["segmentation"]["detector_candidates"] = detector_candidates
                stream["segmentation"]["verification_candidates"] = verification_candidates
                stream["segmentation"]["visible_labels"] = []
                with mutex:
                    mailbox[camera_id] = {"source": "custom_deepstream", "timestamp": captured_at, "streams": [stream]}
                wake.set()

    if baseline:
        def tracker_probe(pad, info, data):
            buffer = info.get_buffer()
            if buffer:
                batch = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
                frames = batch.frame_meta_list if batch else None
                while frames:
                    frame = pyds.NvDsFrameMeta.cast(frames.data)
                    baseline.tracker(cameras[int(frame.source_id)], int(frame.frame_num), int(frame.num_obj_meta), int(frame.buf_pts))
                    try:
                        frames = frames.next
                    except StopIteration:
                        break
            return Gst.PadProbeReturn.OK
        tracker.get_static_pad('src').add_probe(Gst.PadProbeType.BUFFER, tracker_probe, None)
    rgba.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, probe, None)
    thread = threading.Thread(target=send, daemon=True)
    thread.start()
    metadata_thread = threading.Thread(target=publish_metadata, daemon=True, name="metadata-attach-worker")
    metadata_thread.start()
    failed = []

    def bus_message(bus, message):
        if message.type == Gst.MessageType.ERROR:
            error, debug = message.parse_error()
            print(f"DeepStream error: {error}; {debug}", file=sys.stderr, flush=True)
            failed.append(str(error))
            main_loop.quit()
        elif message.type == Gst.MessageType.EOS:
            main_loop.quit()

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_message)
    threading.Thread(target=receive, daemon=True, name="label-commands").start()
    try:
        if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("Không khởi động được DeepStream.")
        main_loop.run()
    finally:
        stopped.set()
        wake.set()
        metadata_wake.set()
        pipeline.set_state(Gst.State.NULL)
        metadata_thread.join(timeout=1)
    if failed:
        raise RuntimeError(failed[-1])


if __name__ == "__main__":
    os.environ["GIO_USE_PROXY"] = "dummy"
    run(json.loads(Path(sys.argv[1]).read_text()))
