import asyncio
import json
import websockets
import time
import math
from collections import defaultdict

async def sample_rack_data(duration=10):
    uri_meta = "ws://localhost:8000/ws/metadata"
    uri_twin = "ws://localhost:8000/ws/digital_twin"
    
    print(f"Sampling Rack live detections & digital twin for {duration}s...")
    
    meta_records = []
    twin_records = []
    
    async def listen_meta():
        try:
            async with websockets.connect(uri_meta) as ws:
                start = time.time()
                while time.time() - start < duration:
                    msg = await asyncio.wait_for(ws.receive(), timeout=2.0)
                    data = json.loads(msg)
                    now = time.time()
                    for st in data.get("streams", []):
                        cam_id = st.get("cam_id")
                        for obj in st.get("objects", []):
                            cls = obj.get("class", "").lower()
                            lbl = obj.get("label", "").lower()
                            if "rack" in cls or "rack" in lbl:
                                meta_records.append({
                                    "t": now,
                                    "cam_id": cam_id,
                                    "obj": obj
                                })
        except Exception as e:
            print("listen_meta err:", e)

    async def listen_twin():
        try:
            async with websockets.connect(uri_twin) as ws:
                start = time.time()
                while time.time() - start < duration:
                    msg = await asyncio.wait_for(ws.receive(), timeout=2.0)
                    data = json.loads(msg)
                    now = time.time()
                    for rk in data.get("racks", []):
                        twin_records.append({
                            "t": now,
                            "rack": rk
                        })
        except Exception as e:
            print("listen_twin err:", e)
            
    await asyncio.gather(listen_meta(), listen_twin())
    
    print(f"\n--- COLLECTED {len(meta_records)} METADATA SAMPLES & {len(twin_records)} TWIN SAMPLES ---")
    
    # Analyze Metadata
    by_cam = defaultdict(list)
    for r in meta_records:
        by_cam[r["cam_id"]].append(r["obj"])
        
    for cam_id, objs in by_cam.items():
        print(f"\n[Camera {cam_id}]: {len(objs)} rack detections")
        track_ids = set(o.get("id") for o in objs)
        print(f"  Track IDs seen: {track_ids}")
        
        xs = [o.get("x", 0) for o in objs]
        ys = [o.get("y", 0) for o in objs]
        ws = [o.get("w", 0) for o in objs]
        hs = [o.get("h", 0) for o in objs]
        confs = [o.get("confidence", 0) for o in objs]
        tracking_states = [o.get("tracking_state") for o in objs]
        
        # Bottom center contact point
        u_pts = [o.get("x", 0) + o.get("w", 0)/2.0 for o in objs]
        v_pts = [o.get("y", 0) + o.get("h", 0) for o in objs]
        
        def stats(arr):
            if not arr: return "N/A"
            mean = sum(arr) / len(arr)
            std = math.sqrt(sum((x - mean)**2 for x in arr) / len(arr))
            return f"min={min(arr):.4f}, max={max(arr):.4f}, mean={mean:.4f}, std={std:.5f}, range={max(arr)-min(arr):.4f}"
            
        print(f"  X: {stats(xs)}")
        print(f"  Y: {stats(ys)}")
        print(f"  W: {stats(ws)}")
        print(f"  H: {stats(hs)}")
        print(f"  Foot U (center): {stats(u_pts)}")
        print(f"  Foot V (bottom): {stats(v_pts)}")
        print(f"  Confidence: {stats(confs)}")
        state_counts = defaultdict(int)
        for s in tracking_states: state_counts[s] += 1
        print(f"  Tracking states: {dict(state_counts)}")
        
        # Check first 5 consecutive frame deltas
        if len(objs) >= 5:
            print("  Consecutive sample deltas (first 5):")
            for i in range(1, 6):
                dx = abs(objs[i].get("x",0) - objs[i-1].get("x",0))
                dy = abs(objs[i].get("y",0) - objs[i-1].get("y",0))
                dw = abs(objs[i].get("w",0) - objs[i-1].get("w",0))
                dh = abs(objs[i].get("h",0) - objs[i-1].get("h",0))
                print(f"    frame {i}: conf={objs[i].get('confidence',0):.2f}, dx={dx:.4f}, dy={dy:.4f}, dw={dw:.4f}, dh={dh:.4f}")

    # Analyze Digital Twin positions
    print("\n--- DIGITAL TWIN RACK POSITIONS ---")
    by_rack_id = defaultdict(list)
    for r in twin_records:
        by_rack_id[r["rack"]["id"]].append(r["rack"])
        
    for rk_id, rks in by_rack_id.items():
        print(f"\n[Rack ID {rk_id}]: {len(rks)} twin updates")
        statuses = set(r.get("status") for r in rks)
        print(f"  Statuses: {statuses}")
        cams = set(r.get("cam_id") for r in rks)
        print(f"  Reporting cams: {cams}")
        pos_x = [r["position"][0] for r in rks]
        pos_z = [r["position"][2] for r in rks]
        def stats_m(arr):
            if not arr: return "N/A"
            mean = sum(arr) / len(arr)
            std = math.sqrt(sum((x - mean)**2 for x in arr) / len(arr))
            return f"min={min(arr):.3f}m, max={max(arr):.3f}m, mean={mean:.3f}m, std={std:.4f}m, delta={max(arr)-min(arr):.3f}m"
        print(f"  Position X: {stats_m(pos_x)}")
        print(f"  Position Z: {stats_m(pos_z)}")
        # Print transitions
        unique_pos = []
        for r in rks:
            p = (r["position"][0], r["position"][2])
            if not unique_pos or unique_pos[-1] != p:
                unique_pos.append(p)
        print(f"  Position jumps/changes ({len(unique_pos)} distinct points): {unique_pos[:10]}")

if __name__ == "__main__":
    asyncio.run(sample_rack_data(8))
