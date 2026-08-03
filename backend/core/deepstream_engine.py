import os
os.environ["GIO_USE_PROXY"] = "dummy"
os.environ["GIO_USE_VFS"] = "local"
import sys
import time
import threading
import queue
import urllib.parse
import math
import re
from typing import Dict, List, Callable, Optional

try:
    import pyds
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst, GLib
except ImportError as e:
    print(f"Warning: Failed to import GStreamer/pyds on host (Expected when running outside Docker): {e}")

from core.reid_matcher import global_reid
from core.camera_calibrator import camera_calibrator
from core.behavior_analytics import behavior_engine
from core.database import db_manager
from core.identity_utils import identity_global_id, robot_number_from_label
from core.deepstream_pose import frame_poses, attach_poses
from core.person_analytics import person_ground_point
try:
    from src.perception.probes import extract_reid_from_user_meta
except Exception:
    extract_reid_from_user_meta = None

FMS_ROBOT_MATCH_THRESHOLD_M = float(os.getenv("FMS_ROBOT_MATCH_THRESHOLD_M", "1.5"))

def sanitize_rtsp_url(url: str) -> str:
    """Removes all whitespace and invalid characters from RTSP URL"""
    import re
    url = re.sub(r'\s+', '', url).replace("%20", "")
    if not url.startswith("rtsp://"):
        return url
    try:
        prefix = "rtsp://"
        rest = url[len(prefix):]
        if "@" in rest:
            last_at_idx = rest.rfind("@")
            userinfo = rest[:last_at_idx]
            hostpath = rest[last_at_idx+1:]
            if ":" in userinfo:
                u, p = userinfo.split(":", 1)
                u_enc = urllib.parse.quote(urllib.parse.unquote(u), safe="")
                p_enc = urllib.parse.quote(urllib.parse.unquote(p), safe="")
                return f"{prefix}{u_enc}:{p_enc}@{hostpath}"
    except Exception as e:
        print(f"Error sanitizing RTSP URL: {e}")
    return url

def _match_registered_robot_by_fms(floor_x: float, floor_y: float) -> Optional[str]:
    try:
        from core.fms_bridge import fms_bridge
        from src.controller.registry import target_registry

        best_label = None
        best_dist = FMS_ROBOT_MATCH_THRESHOLD_M
        for target in target_registry.get_all_targets():
            if (target.get("category") or "").lower() != "robot":
                continue
            label = target.get("label") or ""
            robot_id = target.get("fms_robot_id") or robot_number_from_label(label)
            robot_key = str(robot_id) if robot_id is not None else label.replace("Robot_", "")
            candidates = [label, robot_key, f"Robot_{robot_key}"]

            fms_robot = None
            for key in candidates:
                if key in fms_bridge.robot_states:
                    fms_robot = fms_bridge.robot_states[key]
                    break
            if not fms_robot or not fms_robot.get("position"):
                continue

            pos = fms_robot["position"]
            dist = math.hypot(float(floor_x) - float(pos[0]), float(floor_y) - float(pos[2]))
            if dist < best_dist:
                best_dist = dist
                best_label = label
        return best_label
    except Exception:
        return None

class DeepStreamManager:
    """
    Unified Multi-Stream Headless DeepStream Engine with MTMC Fusion & Behavior Analytics.
    - Manages dynamic runtime RTSP source add/delete on a single nvstreammux.
    - Zero GPU NVENC bottleneck by using fakesink (video streamed decoupled via MediaMTX).
    - Extracts Person Detection + Tracking metadata.
    - Performs Multi-Camera Global Re-ID Association & Floor Map Homography projection.
    - Runs Behavior Analytics (Intrusion, Tripwires, Dwell Time, Crowd Density).
    - Dispatches structured events to Database and WebSockets.
    """
    def __init__(self, metadata_callback: Optional[Callable[[dict], None]] = None, event_callback: Optional[Callable[[dict], None]] = None):
        self.metadata_callback = metadata_callback
        self.event_callback = event_callback
        self.sources: Dict[int, dict] = {}
        self.cam_id_to_source_id: Dict[str, int] = {}
        self.source_id_to_cam_id: Dict[int, str] = {}
        self.next_source_id = 0
        self.lock = threading.RLock()
        self.is_running = False
        self._is_playing = False  # Track if pipeline has been set to PLAYING
        self.pipeline = None
        self.context = None
        self.loop = None
        self._gst_available = False
        self._position_log_queue = queue.Queue(maxsize=2048)
        self._position_log_last_at = {}
        self._position_log_thread = threading.Thread(
            target=self._position_log_worker,
            daemon=True,
            name="deepstream-position-log",
        )
        self._position_log_thread.start()

        try:
            if os.getenv("DISABLE_DEEPSTREAM_GST", "0").lower() in {"1", "true", "yes"}:
                print("[DeepStreamManager] DeepStream pipeline disabled by DISABLE_DEEPSTREAM_GST", flush=True)
            elif 'Gst' in globals() and 'GLib' in globals():
                self._gst_available = True
                print("[DeepStreamManager] GStreamer/pyds available, pipeline will build on start()", flush=True)
            else:
                print("[DeepStreamManager] GStreamer not available, running in stub mode", flush=True)
        except Exception as error:
            print(f"[DeepStreamManager] GStreamer check failed: {error}", flush=True)

    def _position_log_worker(self):
        while True:
            item = self._position_log_queue.get()
            if item is None:
                return
            try:
                db_manager.log_track_position(*item)
            except Exception:
                pass

    def _enqueue_position_log(self, global_id, cam_id, floor_x, floor_y):
        key = (cam_id, global_id)
        now = time.monotonic()
        if now - self._position_log_last_at.get(key, 0.0) < 1.0:
            return
        self._position_log_last_at[key] = now
        try:
            self._position_log_queue.put_nowait((global_id, cam_id, floor_x, floor_y))
        except queue.Full:
            pass

    def start(self, initial_sources: Optional[List[tuple]] = None):
        """Build pipeline, add all initial sources (while NULL), then start GLib loop.
        
        KEY INSIGHT: Sources MUST be added to the pipeline BEFORE set_state(PLAYING).
        Adding sources dynamically to a PAUSED/PLAYING pipeline with nvinfer causes
        a C-level SIGABRT in DeepStream. The safe pattern is:
          1. Build pipeline (NULL state)
          2. Add all sources (still NULL)
          3. Start GLib loop → set_state(PLAYING) once, with all sources present
        
        For runtime dynamic add (user adds camera after startup), add_source()
        uses context.invoke_full which is safe only on an already-PLAYING pipeline.
        """
        if self.is_running:
            return
        if not self._gst_available:
            print("[DeepStreamManager] Skipping pipeline start - GStreamer not available", flush=True)
            return
        try:
            if not Gst.is_initialized():
                Gst.init(None)
            self.context = GLib.MainContext.new()
            self.loop = GLib.MainLoop.new(self.context, False)
            initial_source_count = len(initial_sources) if initial_sources else 0
            self._build_pipeline(initial_source_count=initial_source_count)

            # Add all initial sources BEFORE starting the GLib loop (pipeline is NULL here)
            if initial_sources:
                for cam_id, rtsp_url in initial_sources:
                    self._add_source_static(cam_id, sanitize_rtsp_url(rtsp_url))
                print(f"[DeepStreamManager] Added {len(initial_sources)} sources before loop start", flush=True)

            self.thread = threading.Thread(target=self._run_loop, daemon=True, name="deepstream-glib-loop")
            self.thread.start()
            print("[DeepStreamManager] Pipeline thread started", flush=True)
        except Exception as e:
            print(f"[DeepStreamManager] PIPELINE BUILD FAILED: {e}", flush=True)
            import traceback
            traceback.print_exc()
            self.pipeline = None  # Ensure None so add_source no-ops safely


    def _build_pipeline(self, tracker_config_path: str = "/app/models_config/dstest2_tracker_config.txt", initial_source_count: int = 0):
        # 1. Dynamically determine batch size based on the number of initial sources
        # We need a minimum of 1.
        num_cameras = max(1, initial_source_count, len(self.sources))
        # A custom export may have a static-batch TensorRT engine. In that
        # case, process one frame per mux batch while still servicing every
        # camera, instead of making TensorRT build a new engine for the camera
        # count on every restart.
        try:
            configured_batch_size = int(os.getenv("DEEPSTREAM_BATCH_SIZE", "0"))
        except (TypeError, ValueError):
            configured_batch_size = 0
        pipeline_batch_size = configured_batch_size if configured_batch_size > 0 else num_cameras
        
        # 2. Update config_infer_primary.txt dynamically to match the number of cameras
        try:
            config_path = "/app/models_config/config_infer_primary.txt"
            with open(config_path, "r") as f:
                config_content = f.read()
            import re
            onnx_override = os.getenv("DEEPSTREAM_ONNX_FILE")
            engine_override = os.getenv("DEEPSTREAM_ENGINE_FILE")
            labels_override = os.getenv("DEEPSTREAM_LABELS_FILE")
            if onnx_override:
                config_content = re.sub(r'^onnx-file=.*$', f'onnx-file={onnx_override}', config_content, count=1, flags=re.MULTILINE)
            if engine_override:
                config_content = re.sub(r'^model-engine-file=.*$', f'model-engine-file={engine_override}', config_content, count=1, flags=re.MULTILINE)
            if labels_override:
                config_content = re.sub(r'^labelfile-path=.*$', f'labelfile-path={labels_override}', config_content, count=1, flags=re.MULTILINE)
            # Replace batch-size=... in the [property] section
            config_content = re.sub(r'batch-size=\d+', f'batch-size={pipeline_batch_size}', config_content, count=1)
            if onnx_override and not engine_override:
                root, _ = os.path.splitext(onnx_override)
                # TensorRT keeps the .onnx suffix when deriving its default
                # engine filename (e.g. yolov8x.onnx_b2_gpu0_fp16.engine).
                generated_engine = f"{onnx_override}_b{num_cameras}_gpu0_fp16.engine"
                config_content = re.sub(
                    r'^model-engine-file=.*$',
                    f'model-engine-file={generated_engine}',
                    config_content,
                    count=1,
                    flags=re.MULTILINE
                )
                print(f"[DeepStreamManager] Using ONNX override: {onnx_override} -> {generated_engine}", flush=True)
            elif not engine_override:
                config_content = re.sub(
                    r'(model-engine-file=.*?_b)\d+(_gpu\d+_[^.]+\.engine)',
                    rf'\g<1>{pipeline_batch_size}\2',
                    config_content,
                    count=1
                )
            with open(config_path, "w") as f:
                f.write(config_content)
            print(f"[DeepStreamManager] Dynamically set nvinfer batch-size to {pipeline_batch_size}", flush=True)
        except Exception as e:
            print(f"[DeepStreamManager] Failed to update nvinfer config batch-size: {e}", flush=True)

        self.pipeline = Gst.Pipeline(name="rtc-vms-pipeline")

        # 1. nvstreammux - batch up to 32 streams
        self.muxer = Gst.ElementFactory.make("nvstreammux", "unified-muxer")
        if not self.muxer:
            raise RuntimeError("[DeepStreamManager] Failed to create nvstreammux - DeepStream plugin missing")
        self.muxer.set_property("batch-size", pipeline_batch_size)
        self.muxer.set_property("width", 1280)
        self.muxer.set_property("height", 720)
        self.muxer.set_property("batched-push-timeout", 40000)
        self.muxer.set_property("live-source", 1)
        if self.muxer.find_property("sync-inputs"):
            self.muxer.set_property("sync-inputs", False)
        self.pipeline.add(self.muxer)

        # 2. nvinfer - Primary YOLO inference (reads .engine file)
        self.pgie = Gst.ElementFactory.make("nvinfer", "primary-yolo-detector")
        if not self.pgie:
            raise RuntimeError("[DeepStreamManager] Failed to create nvinfer - DeepStream plugin missing")
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models_config", "config_infer_primary.txt")
        if not os.path.exists(config_path):
            raise RuntimeError(f"[DeepStreamManager] nvinfer config not found: {config_path}")
        self.pgie.set_property("config-file-path", config_path)
        self.pipeline.add(self.pgie)

        # 3. nvtracker - NvDCF/ByteTracker per-stream tracking
        self.tracker = Gst.ElementFactory.make("nvtracker", "nvtracker-engine")
        if not self.tracker:
            raise RuntimeError("[DeepStreamManager] Failed to create nvtracker")
        tracker_config_path = os.getenv(
            "DEEPSTREAM_TRACKER_CONFIG",
            "/app/models_config/tracker_config.yml",
        )
        if not os.path.exists(tracker_config_path):
            raise RuntimeError(f"[DeepStreamManager] tracker config not found: {tracker_config_path}")
        tracker_lib = "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so"
        if not os.path.exists(tracker_lib):
            raise RuntimeError(f"[DeepStreamManager] nvtracker lib not found: {tracker_lib}")
        self.tracker.set_property("ll-config-file", tracker_config_path)
        self.tracker.set_property("ll-lib-file", tracker_lib)
        self.tracker.set_property("tracker-width", 640)
        self.tracker.set_property("tracker-height", 384)
        self.tracker.set_property("gpu-id", 0)
        self.pipeline.add(self.tracker)

        # 4. fakesink - headless (no display, no NVENC encode)
        self.sink = Gst.ElementFactory.make("fakesink", "headless-sink")
        if not self.sink:
            raise RuntimeError("[DeepStreamManager] Failed to create fakesink")
        self.sink.set_property("sync", False)
        self.sink.set_property("async", False)
        self.pipeline.add(self.sink)

        # Link: muxer -> pgie -> tracker -> fakesink
        if not self.muxer.link(self.pgie):
            raise RuntimeError("[DeepStreamManager] Failed to link muxer -> pgie")
        if not self.pgie.link(self.tracker):
            raise RuntimeError("[DeepStreamManager] Failed to link pgie -> tracker")
        if not self.tracker.link(self.sink):
            raise RuntimeError("[DeepStreamManager] Failed to link tracker -> fakesink")

        # Attach metadata probe to tracker src pad
        tracker_src_pad = self.tracker.get_static_pad("src")
        if not tracker_src_pad:
            raise RuntimeError("[DeepStreamManager] Could not get tracker src pad")
        tracker_src_pad.add_probe(Gst.PadProbeType.BUFFER, self._metadata_probe, 0)
        print(
            f"[DeepStreamManager] Pipeline built successfully: muxer->pgie->nvtracker->fakesink "
            f"(tracker={tracker_config_path})",
            flush=True,
        )
        # NOTE: Bus watch is set up in _run_loop() within the GLib main context thread.
        # Do NOT call bus.add_signal_watch() here (wrong thread context -> segfault).


    def _bus_call(self, bus, message):
        t = message.type
        if t == Gst.MessageType.EOS:
            print("[DeepStreamManager] End-of-stream reached", flush=True)
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"[DeepStreamManager] Pipeline Error: {err}: {debug}", flush=True)
            
            # Isolate the source of the error to prevent full pipeline abort
            src_element = message.src
            if src_element:
                name = src_element.get_name()
                print(f"[DeepStreamManager] Error originated from element: {name}", flush=True)
                if name.startswith("uri-decode-bin-"):
                    source_id_str = name.replace("uri-decode-bin-", "")
                    try:
                        source_id = int(source_id_str)
                        cam_id = self.source_id_to_cam_id.get(source_id)
                        if cam_id:
                            print(f"[DeepStreamManager] Isolating and removing faulty source {cam_id}...", flush=True)
                            GLib.idle_add(self._delete_source_glib, cam_id, source_id)
                    except ValueError:
                        pass
        elif t == Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            print(f"[DeepStreamManager] Pipeline Warning: {err}: {debug}", flush=True)
        return True

    def _run_loop(self):
        if self.pipeline:
            self.context.push_thread_default()
            # Re-attach bus watch from within the loop context
            bus = self.pipeline.get_bus()
            bus.add_watch(GLib.PRIORITY_DEFAULT, self._bus_call_context)

            ret = self.pipeline.set_state(Gst.State.PAUSED)
            print(f"[DeepStreamManager] Pipeline set_state(PAUSED), ret={ret}", flush=True)
            play_ret = self.pipeline.set_state(Gst.State.PLAYING)
            print(f"[DeepStreamManager] Pipeline set_state(PLAYING), ret={play_ret}", flush=True)
            self._is_playing = True

            self.is_running = True
            self.loop.run()
            print("[DeepStreamManager] GLib main loop exited", flush=True)

    def _bus_call_context(self, bus, message):
        """Bus callback for use with add_watch() inside GLib main context."""
        return self._bus_call(bus, message)

    def _add_source_static(self, cam_id: str, clean_url: str) -> bool:
        """
        Add a source to the pipeline while it is still in NULL state (before GLib loop).
        This is the SAFE path for startup sources. The pipeline state will be set to
        PLAYING by _run_loop() after all sources are added.
        """
        source_id = self.next_source_id
        self.next_source_id += 1

        source_bin = Gst.ElementFactory.make("nvurisrcbin", f"uri-decode-bin-{source_id}")
        if not source_bin:
            print(f"[DeepStreamManager] Failed to create nvurisrcbin for {cam_id}", flush=True)
            return False

        source_bin.set_property("uri", clean_url)
        source_bin.set_property("source-id", source_id)
        source_bin.set_property("rtsp-reconnect-interval", 5)
        if source_bin.find_property("latency"):
            source_bin.set_property("latency", 50)
        if source_bin.find_property("drop-on-latency"):
            source_bin.set_property("drop-on-latency", True)
        if source_bin.find_property("low-latency-mode"):
            source_bin.set_property("low-latency-mode", True)
        if source_bin.find_property("buffer-mode"):
            source_bin.set_property("buffer-mode", 0)
        if source_bin.find_property("max-size-buffers"):
            source_bin.set_property("max-size-buffers", 1)
        if source_bin.find_property("drop-frame-interval"):
            source_bin.set_property("drop-frame-interval", 0)

        source_bin.connect("pad-added", self._cb_newpad, source_id)
        self.pipeline.add(source_bin)

        # Request a sink pad from muxer now (while NULL) and link when pad-added fires
        # We pre-request here so it's ready when the src pad negotiates
        self._pending_sink_pads = getattr(self, '_pending_sink_pads', {})
        self._pending_sink_pads[source_id] = self.muxer.get_request_pad(f"sink_{source_id}")

        self.cam_id_to_source_id[cam_id] = source_id
        self.source_id_to_cam_id[source_id] = cam_id
        self.sources[source_id] = {
            "cam_id": cam_id,
            "url": clean_url,
            "bin": source_bin,
            "pad": None
        }
        print(f"[DeepStreamManager] Static-added camera {cam_id} as source_id {source_id} (pipeline NULL)", flush=True)
        return True

    def add_source(self, cam_id: str, rtsp_url: str) -> bool:
        """
        Dynamically adds an RTSP stream to the running nvstreammux at runtime.
        Only called for cameras added by user AFTER startup. Runs on the GLib main loop.
        """
        if not self.pipeline:
            print(f"[DeepStreamManager] Note: Adding source {cam_id} in standalone mode")
            return True

        with self.lock:
            if cam_id in self.cam_id_to_source_id:
                print(f"[DeepStreamManager] Camera {cam_id} is already in pipeline.")
                return True

            clean_url = sanitize_rtsp_url(rtsp_url)
            source_id = self.next_source_id
            self.next_source_id += 1

            if hasattr(self, 'context') and self.context:
                GLib.idle_add(self._add_source_glib, cam_id, clean_url, source_id)
            else:
                # Pipeline not yet initialized: add statically
                self._add_source_static(cam_id, clean_url)
            return True

    def _add_source_glib(self, cam_id, clean_url, source_id):
        print(f"[DeepStreamManager] Inside _add_source_glib for {cam_id}", flush=True)
        source_bin = Gst.ElementFactory.make("nvurisrcbin", f"uri-decode-bin-{source_id}")
        if not source_bin:
            print(f"[DeepStreamManager] Failed to create nvurisrcbin for {cam_id}")
            return False
            
        source_bin.set_property("uri", clean_url)
        source_bin.set_property("source-id", source_id)
        source_bin.set_property("rtsp-reconnect-interval", 5)
        if source_bin.find_property("latency"):
            source_bin.set_property("latency", 50)
        if source_bin.find_property("drop-on-latency"):
            source_bin.set_property("drop-on-latency", True)
        if source_bin.find_property("low-latency-mode"):
            source_bin.set_property("low-latency-mode", True)
        if source_bin.find_property("buffer-mode"):
            source_bin.set_property("buffer-mode", 0)
        if source_bin.find_property("max-size-buffers"):
            source_bin.set_property("max-size-buffers", 1)
        if source_bin.find_property("drop-frame-interval"):
            source_bin.set_property("drop-frame-interval", 0)
        
        source_bin.connect("pad-added", self._cb_newpad, source_id)
        
        self.pipeline.add(source_bin)
        # Set source to PAUSED first so it can negotiate caps without pipeline being PLAYING yet
        source_bin.set_state(Gst.State.PAUSED)
        
        with self.lock:
            self.cam_id_to_source_id[cam_id] = source_id
            self.source_id_to_cam_id[source_id] = cam_id
            self.sources[source_id] = {
                "cam_id": cam_id,
                "url": clean_url,
                "bin": source_bin,
                "pad": None
            }

        print(f"[DeepStreamManager] Added camera {cam_id} as source_id {source_id} ({clean_url})")
        return False

    def _cb_newpad(self, decodebin, decoder_src_pad, source_id):
        pad_name_src = decoder_src_pad.get_name()
        print(f"[DeepStreamManager] pad-added signal received for source {source_id}: {pad_name_src}", flush=True)
        if pad_name_src.startswith("audio"):
            return

        with self.lock:
            # Use pre-requested pad (static startup) or request a new one (dynamic add)
            pending = getattr(self, '_pending_sink_pads', {})
            sink_pad = pending.pop(source_id, None) or self.muxer.get_request_pad(f"sink_{source_id}")

            if sink_pad and not sink_pad.is_linked():
                ret = decoder_src_pad.link(sink_pad)
                if source_id in self.sources:
                    self.sources[source_id]["pad"] = sink_pad
                print(f"[DeepStreamManager] Successfully linked pad sink_{source_id} for source {source_id}, ret: {ret}", flush=True)

                # Now that pad is linked, set source to PLAYING
                src_info = self.sources.get(source_id)
                if src_info and src_info.get("bin"):
                    src_info["bin"].set_state(Gst.State.PLAYING)

                # If first source just linked, transition entire pipeline PAUSED → PLAYING
                if not self._is_playing:
                    self._is_playing = True
                    play_ret = self.pipeline.set_state(Gst.State.PLAYING)
                    print(f"[DeepStreamManager] First pad linked - pipeline set_state(PLAYING) ret={play_ret}", flush=True)

    def delete_source(self, cam_id: str) -> bool:
        """
        Dynamically removes an RTSP stream from the running nvstreammux at runtime.
        """
        if not self.pipeline:
            return True
            
        with self.lock:
            if cam_id not in self.cam_id_to_source_id:
                return False
                
            source_id = self.cam_id_to_source_id[cam_id]
            if hasattr(self, 'context') and self.context:
                GLib.idle_add(self._delete_source_glib, cam_id, source_id)
            else:
                self._delete_source_glib(cam_id, source_id)
            return True

    def _delete_source_glib(self, cam_id, source_id):
        with self.lock:
            source_info = self.sources.get(source_id)
            if not source_info:
                return False
                
            source_bin = source_info["bin"]
            sink_pad = source_info.get("pad")
            
            if source_bin:
                source_bin.set_state(Gst.State.NULL)
                self.pipeline.remove(source_bin)
                
            if sink_pad:
                self.muxer.release_request_pad(sink_pad)
                
            self.sources.pop(source_id, None)
            self.cam_id_to_source_id.pop(cam_id, None)
            self.source_id_to_cam_id.pop(source_id, None)
            print(f"[DeepStreamManager] Successfully deleted source {cam_id} (source_id: {source_id})")
        return False

    def _metadata_probe(self, pad, info, u_data):
        try:
            gst_buffer = info.get_buffer()
            if not gst_buffer:
                return Gst.PadProbeReturn.OK

            batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
            if not batch_meta:
                return Gst.PadProbeReturn.OK
            l_frame = batch_meta.frame_meta_list

            timestamp_ms = int(time.time() * 1000)
            streams_payload = []

            while l_frame is not None:
                try:
                    frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
                except StopIteration:
                    break

                source_id = frame_meta.source_id
                cam_id = self.source_id_to_cam_id.get(source_id, f"cam_{source_id}")

                # DeepStream nvinfer rect_params are in muxer resolution (1280x720)
                frame_w = 1280
                frame_h = 720
                mux_w = 1280.0
                mux_h = 720.0

                raw_detections = []
                l_obj = frame_meta.obj_meta_list

                while l_obj is not None:
                    try:
                        obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                    except StopIteration:
                        break

                    # Extract class name from label or class_id
                    class_id = obj_meta.class_id
                    class_name = obj_meta.obj_label or f"class_{class_id}"
                    class_name = class_name.lower()

                    if class_id != 0:
                        try:
                            l_obj = l_obj.next
                        except StopIteration:
                            break
                        continue
                    class_name = "person"

                    rect = obj_meta.rect_params
                    # object_id is 64-bit uint; mask to 32-bit for JS/JSON safety
                    local_id = int(obj_meta.object_id) & 0xFFFFFFFF

                    # NvDCF can publish the same 512-D Re-ID tensor it uses
                    # internally. Pass it through unchanged so the registry
                    # compares a live crop against the Label gallery, rather
                    # than trying to infer identity from the detector class.
                    reid_feature = None
                    if extract_reid_from_user_meta is not None:
                        try:
                            reid_feature = extract_reid_from_user_meta(obj_meta)
                        except Exception:
                            reid_feature = None

                    left   = max(0.0, float(rect.left))
                    top    = max(0.0, float(rect.top))
                    width  = max(1.0, float(rect.width))
                    height = max(1.0, float(rect.height))

                    norm_x = min(1.0, max(0.0, left / mux_w))
                    norm_y = min(1.0, max(0.0, top / mux_h))
                    norm_w = min(1.0 - norm_x, max(0.0, width / mux_w))
                    norm_h = min(1.0 - norm_y, max(0.0, height / mux_h))
                    raw_detections.append({
                        "local_id": local_id,
                        "class": class_name,
                        "bbox": [round(norm_x, 4), round(norm_y, 4),
                                 round(norm_w, 4), round(norm_h, 4)],
                        "confidence": float(getattr(obj_meta, 'confidence', 0.9)),
                        "feature": reid_feature,
                        "frame_width": frame_w,
                        "frame_height": frame_h,
                        "frame_num": int(getattr(frame_meta, "frame_num", 0)),
                    })

                    try:
                        l_obj = l_obj.next
                    except StopIteration:
                        break

                # Debug heartbeat every ~5 seconds (approx every 150 frames at 30fps)
                poses = frame_poses(frame_meta, frame_w, frame_h)
                attach_poses(raw_detections, poses)
                if not hasattr(self, "_pose_counts"):
                    self._pose_counts = {}
                self._pose_counts[cam_id] = sum(bool(item.get("keypoints")) for item in raw_detections)
                if not hasattr(self, '_debug_frame_count'):
                    self._debug_frame_count = {}
                    self._debug_detect_count = {}
                cam_key = f"{cam_id}_{source_id}"
                self._debug_frame_count[cam_key] = self._debug_frame_count.get(cam_key, 0) + 1
                self._debug_detect_count[cam_key] = self._debug_detect_count.get(cam_key, 0) + len(raw_detections)
                
                fc = self._debug_frame_count[cam_key]
                if fc <= 10 or fc % 30 == 0:
                    print(f"[Probe] cam={cam_id} src={source_id} "
                          f"frame={frame_w}x{frame_h} "
                          f"objects_detected={len(raw_detections)} "
                          f"frame_number={fc}", flush=True)
                    if fc % 30 == 0:
                        self._debug_detect_count[cam_key] = 0
                # MTMC Fusion & Spatial Mapping
                objects_list = []
                tripwire_stats = {}
                roi_states = []

                if raw_detections:
                    try:
                        augmented_dets = global_reid.process_camera_detections(cam_id, raw_detections)
                    except Exception as reid_err:
                        print(f"[Probe] ReID error for {cam_id}: {reid_err}", flush=True)
                        augmented_dets = []
                        for d in raw_detections:
                            d2 = dict(d)
                            d2["global_id"] = d["local_id"]
                            d2["floor_x"] = d["bbox"][0] + d["bbox"][2] / 2.0
                            d2["floor_y"] = d["bbox"][1] + d["bbox"][3]
                            augmented_dets.append(d2)

                    # Import object logic (lazy)
                    try:
                        from core.object_logic import person_logic, rack_logic, robot_logic, infer_category
                        _obj_logic_available = True
                    except Exception:
                        _obj_logic_available = False

                    # Pass 1: label matching + thu thập robot floor positions
                    labeled_dets = []
                    robot_floor_positions = []

                    for d in augmented_dets:
                        gid = d["global_id"]
                        fx = d.get("floor_x")
                        fy = d.get("floor_y")
                        fx = float(fx) if fx is not None else 0.0
                        fy = float(fy) if fy is not None else 0.0
                        ground, ground_source = person_ground_point(d["bbox"], d.get("keypoints"))
                        spatial = camera_calibrator.project_ground_point(cam_id, *ground)
                        if spatial["valid"]:
                            fx, fy = spatial["x"], spatial["z"]

                        self._enqueue_position_log(gid, cam_id, fx, fy)
                        matched_label = None


                        # Infer category
                        obj_class_name = d.get("class", "object")
                        category = "object"
                        if _obj_logic_available:
                            try:
                                category = infer_category(matched_label, obj_class_name)
                            except Exception:
                                pass

                        # Collect robot positions for rack occupancy
                        if category == "robot":
                            robot_floor_positions.append((fx, fy, matched_label or f"robot_{gid}"))

                        labeled_dets.append({
                            "_d": d, "_gid": gid, "_fx": fx, "_fy": fy,
                            "_label": matched_label, "_category": category,
                            "_class": obj_class_name,
                            "_spatial": spatial,
                        })

                    # Pass 2: category-specific logic + build objects_list
                    for item in labeled_dets:
                        d = item["_d"]
                        gid = item["_gid"]
                        fx, fy = item["_fx"], item["_fy"]
                        matched_label = item["_label"]
                        category = item["_category"]
                        obj_class_name = item["_class"]
                        spatial = item["_spatial"]

                        extra_fields = {}
                        if _obj_logic_available:
                            try:
                                if category == "person":
                                    d["global_id"] = gid
                                    extra_fields = person_logic.process(d, (fx, fy), cam_id)
                                elif category == "rack":
                                    extra_fields = rack_logic.process(d, (fx, fy), robot_floor_positions)
                                elif category == "robot":
                                    try:
                                        from core.fms_bridge import fms_bridge as _fms
                                        fms_states = _fms.robot_states
                                    except Exception:
                                        fms_states = {}
                                    extra_fields = robot_logic.process(d, (fx, fy), matched_label, fms_states)
                            except Exception:
                                pass

                        if category == "person" and extra_fields.get("fall_event") and self.event_callback:
                            self.event_callback({
                                "cam_id": cam_id,
                                "global_id": gid,
                                "rule_type": "fall_detection",
                                "severity": "critical",
                                "description": f"Phát hiện người có khả năng bị ngã (track #{gid})",
                                "timestamp": int(time.time() * 1000),
                                "object": {"id": gid, "bbox": d["bbox"], "floor_x": fx, "floor_y": fy},
                            })

                        objects_list.append({
                            "id": gid,
                            "local_id": d["local_id"],
                            "class": obj_class_name,
                            "category": category,
                            "x": round(d["bbox"][0], 4),
                            "y": round(d["bbox"][1], 4),
                            "w": round(d["bbox"][2], 4),
                            "h": round(d["bbox"][3], 4),
                            "floor_x": round(fx, 4),
                            "floor_y": round(fy, 4),
                            "velocity": round(d.get("velocity", 0.0), 2),
                            "speed": round(d.get("velocity", 0.0), 2),
                            "track_state": d.get("track_state", "TRACKED"),
                            "reid_score": d.get("reid_score"),
                            "recovered_from_loss": d.get("recovered_from_loss", False),
                            "reid_vector": d.get("feature"),
                            "confidence": round(d.get("confidence", 0.9), 2),
                            "label": matched_label,
                            "world_position": [spatial["x"], spatial["y"], spatial["z"]],
                            "spatial_valid": spatial["valid"],
                            "spatial_source": spatial["source"],
                            "spatial_confidence": spatial["confidence"],
                            "keypoints": d.get("keypoints") or [],
                            "pose_source": d.get("pose_source"),
                            "tracking_state": d.get("tracking_state", "predicted"),
                            "frame_width": d.get("frame_width"),
                            "frame_height": d.get("frame_height"),
                            **extra_fields,
                        })

                # Always process behavior engine to update empty frames
                try:
                    triggered_events, tripwire_stats, roi_states = behavior_engine.process_frame(cam_id, objects_list)
                    for ev in triggered_events:
                        try:
                            db_manager.log_event(ev)
                        except Exception:
                            pass
                        if self.event_callback:
                            self.event_callback(ev)
                except Exception as be_err:
                    print(f"[Probe] BehaviorEngine error for {cam_id}: {be_err}", flush=True)

                streams_payload.append({
                    "cam_id": cam_id,
                    "objects": objects_list,
                    "tripwire_stats": tripwire_stats,
                    "rois": roi_states
                })

                try:
                    l_frame = l_frame.next
                except StopIteration:
                    break

            if streams_payload and self.metadata_callback:
                self.metadata_callback({
                    "source": "deepstream",
                    "timestamp": timestamp_ms,
                    "streams": streams_payload
                })

        except Exception as e:
            print(f"[Probe] CRITICAL error in _metadata_probe: {e}", flush=True)
            import traceback
            traceback.print_exc()

        return Gst.PadProbeReturn.OK

# Instantiate Singleton
deepstream_manager = DeepStreamManager()
