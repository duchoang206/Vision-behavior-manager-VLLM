"""
Session 2: DeepStream Pad Probe Module for Tracking and Re-ID Metadata Extraction.
Extracts:
- frame_num, source_id, timestamp
- track_id, class_id, class_name, confidence
- 2D Bounding Box [x1, y1, x2, y2]
- Grounding Contact Point [u = left + width/2, v = top + height]
- 512-dim Re-ID Feature Vector from NvDsUserMeta / Re-ID Feature Extractor
"""

import sys
import ctypes
import numpy as np
from typing import Dict, List, Any, Callable, Optional

try:
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst
    import pyds
    PYDS_AVAILABLE = True
except (ImportError, ValueError):
    PYDS_AVAILABLE = False


def extract_reid_from_user_meta(obj_meta) -> Optional[List[float]]:
    """
    Extracts the 512-dimensional Re-ID feature embedding from DeepStream NvDCF UserMeta.
    """
    if not PYDS_AVAILABLE:
        return None
        
    try:
        l_user = obj_meta.obj_user_meta_list
        while l_user is not None:
            try:
                user_meta = pyds.NvDsUserMeta.cast(l_user.data)
            except StopIteration:
                break
                
            if not user_meta or not user_meta.user_meta_data:
                l_user = l_user.next
                continue

            # DeepStream 8 exposes NvDCF object embeddings as
            # NVDS_TRACKER_OBJ_REID_META.  The payload is NvDsObjReid, not a
            # raw float pointer; use the binding's host-vector accessor so the
            # registry compares exactly the vector produced by the tracker.
            tracker_meta_type = getattr(pyds, "NVDS_TRACKER_OBJ_REID_META", None)
            if tracker_meta_type is not None and user_meta.base_meta.meta_type == tracker_meta_type:
                reid_obj = pyds.NvDsObjReid.cast(user_meta.user_meta_data)
                vector = np.asarray(reid_obj.get_host_reid_vector(), dtype=np.float32).reshape(-1)
                if vector.size > 0:
                    norm = float(np.linalg.norm(vector))
                    if norm > 1e-6:
                        return (vector / norm).tolist()

            # Backwards compatibility with older DeepStream builds that used
            # the named NVIDIA.NVTRACKER.REID_META user-meta type.
            legacy_meta_type = pyds.nvds_get_user_meta_type("NVIDIA.NVTRACKER.REID_META")
            if user_meta.base_meta.meta_type == legacy_meta_type:
                ptr = user_meta.user_meta_data
                c_float_p = ctypes.POINTER(ctypes.c_float)
                float_array = ctypes.cast(ptr, c_float_p)
                reid_vec = [float(float_array[i]) for i in range(512)]
                norm = np.linalg.norm(reid_vec)
                if norm > 1e-6:
                    return (np.array(reid_vec) / norm).tolist()
            l_user = l_user.next
    except Exception as e:
        pass
    return None


class ProbeHandler:
    """
    High-performance Pad Probe Handler for DeepStream pipelines.
    """
    def __init__(self, callback: Optional[Callable[[Dict[str, Any]], None]] = None):
        self.callback = callback
        self.frame_count = 0
        
    def tracker_src_pad_probe(self, pad, info, u_data):
        if not PYDS_AVAILABLE:
            return 0  # Gst.PadProbeReturn.OK
            
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK
            
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        if not batch_meta:
            return Gst.PadProbeReturn.OK
            
        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break
                
            source_id = frame_meta.source_id
            frame_num = frame_meta.frame_num
            frame_width = frame_meta.source_frame_width or 1920
            frame_height = frame_meta.source_frame_height or 1080
            
            tracked_objects = []
            
            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break
                    
                track_id = int(obj_meta.object_id)
                class_id = int(obj_meta.class_id)
                class_name = str(obj_meta.obj_label) if obj_meta.obj_label else f"class_{class_id}"
                conf = float(obj_meta.confidence)
                
                rect = obj_meta.rect_params
                x1 = float(rect.left)
                y1 = float(rect.top)
                w = float(rect.width)
                h = float(rect.height)
                x2 = x1 + w
                y2 = y1 + h
                
                # Ground contact point (center bottom of bounding box)
                u = x1 + w / 2.0
                v = y2
                
                # Normalized coordinates (0.0 to 1.0)
                norm_u = u / frame_width
                norm_v = v / frame_height
                
                # Extract Re-ID vector
                reid_emb = extract_reid_from_user_meta(obj_meta)
                
                obj_data = {
                    "track_id": track_id,
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": conf,
                    "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "ground_point": (round(u, 2), round(v, 2)),
                    "ground_point_norm": (round(norm_u, 4), round(norm_v, 4)),
                    "reid_embedding": reid_emb,
                    "has_reid": reid_emb is not None
                }
                tracked_objects.append(obj_data)
                
                l_obj = l_obj.next
                
            frame_payload = {
                "frame_num": frame_num,
                "source_id": source_id,
                "timestamp": frame_meta.ntp_timestamp or int(frame_meta.buf_pts / 1000000),
                "frame_width": frame_width,
                "frame_height": frame_height,
                "objects_count": len(tracked_objects),
                "objects": tracked_objects
            }
            
            if self.callback:
                try:
                    self.callback(frame_payload)
                except Exception as e:
                    print(f"Error in probe callback: {e}", file=sys.stderr)
                    
            l_frame = l_frame.next
            
        return Gst.PadProbeReturn.OK
