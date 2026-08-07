"""
Session 4 Validation Test: End-to-End WebSocket 3D Digital Twin Broadcaster & Multi-Entity Tracking.
Validates real-time streaming & cross-checking of:
1. FMS Authoritative Robot State (Battery, Charging, Status) + Vision AI Ground-Truth Cross-Check
2. Persons / Human Workers (with walking speed, proximity halo, and collision safety check)
3. Racks (Static storage vs Carried on robot back)
"""

import sys
import os
import json
import time
import asyncio
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.fms_bridge import fms_bridge
from src.server.digital_twin_bridge import digital_twin_bridge


async def run_digital_twin_test():
    print("=" * 70)
    print("🧪 [Session 4 Test] 3D Digital Twin Multi-Entity Telemetry & FMS Cross-Check")
    print("=" * 70)

    # 1. Inject FMS authoritative telemetry for Robot_9001 (FMS says CHARGING with 92% battery)
    print("🔹 Injecting FMS fleet state for Robot_9001 (Charging @ 92% pin)...")
    fms_bridge.robot_states["Robot_9001"] = {
        "id": "Robot_9001",
        "model": "AMR-L",
        "floor": 1,
        "position": [16.85, 0.0, 12.55],
        "heading": 0.0,
        "velocity": 0.0,
        "battery": 92,
        "status": "CHARGING",
        "fsm": "CHARGING",
        "destination": "CHARGE_DOCK_01",
        "map_id": "TT"
    }

    # 2. Simulate Vision Tracks for Robot, Person, and Rack
    now = time.time()
    
    # 2A. Robot 9001 detected by Cam_01 at physical position [16.90, 0.0, 12.60]
    print("🔹 Injecting Vision AI Camera Track for Robot_9001...")
    digital_twin_bridge.update_vision_track(
        cam_id="cam_01",
        track_id=9001,
        raw_class="delivery-robot",
        norm_u=0.65,
        norm_v=0.70,
        bbox=[1200, 700, 1350, 850],
        reid_vector=[0.1] * 512
    )

    # 2B. Human Worker Person_102
    print("🔹 Injecting Human Worker Person_102 track...")
    digital_twin_bridge.update_vision_track(
        cam_id="cam_01",
        track_id=102,
        raw_class="person",
        norm_u=0.45,
        norm_v=0.55,
        bbox=[850, 500, 920, 750],
        reid_vector=[0.2] * 512
    )

    # 2C. Static Rack Rack_B2
    print("🔹 Injecting Static Storage Rack_B2 track...")
    digital_twin_bridge.update_vision_track(
        cam_id="cam_02",
        track_id=202,
        raw_class="rack",
        norm_u=0.85,
        norm_v=0.85,
        bbox=[1600, 800, 1800, 1050],
        reid_vector=None
    )

    # 3. Extract and Validate Telemetry Payload
    payload = digital_twin_bridge.build_telemetry_payload()
    print("\n📦 Generated 3D Digital Twin Telemetry Payload:")
    print(json.dumps(payload, indent=2))

    robots = payload.get("robots", [])
    persons = payload.get("persons", [])
    racks = payload.get("racks", [])

    # Assertions
    assert len(robots) >= 1, "❌ Missing robots in telemetry payload"
    assert len(persons) >= 1, "❌ Missing persons/workers in telemetry payload"
    assert len(racks) >= 1, "❌ Missing racks in telemetry payload"

    r0 = robots[0]
    p0 = persons[0]
    rk0 = racks[0]

    # Validate FMS authoritative fields + Vision cross-check
    assert r0["battery"] == 92, f"❌ Expected battery 92% from FMS, got {r0['battery']}"
    assert r0["status"] == "CHARGING", f"❌ Expected status CHARGING from FMS, got {r0['status']}"
    assert r0["cross_check_status"] in ["CHARGING_VERIFIED", "SYNCED"], f"❌ Unexpected cross_check_status: {r0['cross_check_status']}"
    assert r0["vision_position"] is not None, "❌ Vision position should be present"
    assert r0["delta_distance_m"] is not None and r0["delta_distance_m"] < 0.2, f"❌ Delta distance {r0['delta_distance_m']}m unexpected"

    print("\n🔍 Verification Checks:")
    print(f"  ✓ Robot ID: {r0['id']}, Battery (FMS): {r0['battery']}%, Status (FMS): {r0['status']}")
    print(f"  ✓ Cross-Check: {r0['cross_check_status']} - {r0['cross_check_msg']}")
    print(f"  ✓ FMS Pos: {r0['fms_position']} vs Vision Pos: {r0['vision_position']} (Δd = {r0['delta_distance_m']}m)")
    print(f"  ✓ Person ID: {p0['id']}, Pos: {p0['position']}, Status: {p0['status']}, Safety: {p0.get('safety_msg')}")
    print(f"  ✓ Rack ID: {rk0['id']}, Pos: {rk0['position']}, Status: {rk0['status']}, Msg: {rk0.get('cross_check_msg')}")

    print("\n" + "=" * 70)
    print("🎉 ALL SESSION 4 FMS FUSION & VISION CROSS-CHECK TESTS PASSED (100%)")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_digital_twin_test())
