#!/usr/bin/env python3
"""
Session 3: Verification Script for Spatial Mapping, Re-ID Target Registry, and Rack Association.
Tests:
1. Homography Projective Geometry (pixel_to_world & world_to_pixel)
2. Target Registry & Cosine Similarity Signature Matching
3. Rack Association (Mobile carrier & Storage slot parking)
"""

import sys
import numpy as np
from pathlib import Path

# Add app root to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mapping.homography import HomographyMapper
from src.controller.registry import TargetRegistry
from src.controller.rack_association import RackAssociationEngine


def test_homography():
    print("\n" + "=" * 65)
    print("📐 [Test 1] Testing Homography Spatial Mapping...")
    print("=" * 65)
    
    # Define 4 point correspondences between camera pixel plane and warehouse floor (meters)
    img_pts = [(200.0, 150.0), (1720.0, 160.0), (1800.0, 950.0), (120.0, 920.0)]
    world_pts = [(2.0, 3.0), (22.0, 3.0), (22.0, 16.0), (2.0, 16.0)]
    
    mapper = HomographyMapper.from_point_correspondences(img_pts, world_pts)
    
    # Test point 1
    u1, v1 = img_pts[0]
    xw1, yw1 = mapper.pixel_to_world(u1, v1)
    print(f" 📍 Pixel ({u1}, {v1}) -> World: ({xw1:.3f}m, {yw1:.3f}m) [Expected: (2.000m, 3.000m)]")
    
    # Test inverse
    inv_u, inv_v = mapper.world_to_pixel(xw1, yw1)
    print(f" 🔄 World ({xw1:.3f}m, {yw1:.3f}m) -> Pixel: ({inv_u:.1f}, {inv_v:.1f}) [Expected: ({u1:.1f}, {v1:.1f})]")
    
    err_world = np.hypot(xw1 - 2.0, yw1 - 3.0)
    err_pixel = np.hypot(inv_u - u1, inv_v - v1)
    
    passed = err_world < 0.05 and err_pixel < 2.0
    print(f" ✅ Spatial Homography Accuracy: {'PASS' if passed else 'FAIL'} (World err: {err_world:.4f}m, Pixel err: {err_pixel:.2f}px)")
    return passed


def test_registry_and_reid():
    print("\n" + "=" * 65)
    print("🧠 [Test 2] Testing Target Registry & Re-ID Cosine Similarity Matcher...")
    print("=" * 65)
    
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp_registry_path = f.name
        
    registry = TargetRegistry(persistence_file=tmp_registry_path)
    
    # 1. Create distinct 512-dim signatures
    np.random.seed(42)
    sig_robot_9001 = np.random.randn(512).astype(np.float32)
    sig_robot_9001 /= np.linalg.norm(sig_robot_9001)
    
    sig_robot_2001 = np.random.randn(512).astype(np.float32)
    sig_robot_2001 /= np.linalg.norm(sig_robot_2001)
    
    # 2. Register targets
    reg1 = registry.register_target("Robot_9001", sig_robot_9001.tolist(), cam_id="Cam_1", world_pos=(14.5, 14.0))
    reg2 = registry.register_target("Robot_2001", sig_robot_2001.tolist(), cam_id="Cam_2", world_pos=(7.2, 9.8))
    
    print(f" 📝 Registered Robot_9001: {'PASS' if reg1 else 'FAIL'}")
    print(f" 📝 Registered Robot_2001: {'PASS' if reg2 else 'FAIL'}")
    
    # 3. Test matching with a slightly perturbed/noisy query vector (simulate camera lighting changes)
    noise = np.random.randn(512).astype(np.float32)
    noise /= np.linalg.norm(noise)
    noisy_query = 0.95 * sig_robot_9001 + 0.05 * noise
    noisy_query /= np.linalg.norm(noisy_query)
    
    matched_label, score = registry.match_reid(noisy_query.tolist(), threshold=0.75)
    print(f" 🔍 Query Robot_9001 (noisy) -> Matched: '{matched_label}' with Similarity: {score:.4f} [Expected: Robot_9001, > 0.85]")
    
    # 4. Test matching with unrelated random vector
    unrelated_vec = np.random.randn(512).astype(np.float32)
    unrelated_vec /= np.linalg.norm(unrelated_vec)
    unmatched_label, un_score = registry.match_reid(unrelated_vec.tolist(), threshold=0.75)
    print(f" 🔍 Query Unrelated Target    -> Matched: {unmatched_label} with Similarity: {un_score:.4f} [Expected: None]")
    
    passed = (matched_label == "Robot_9001") and (score >= 0.85) and (unmatched_label is None)
    print(f" ✅ Re-ID Cosine Matching: {'PASS' if passed else 'FAIL'}")
    return passed


def test_rack_association():
    print("\n" + "=" * 65)
    print("📦 [Test 3] Testing Mobile Rack Association & Storage Slot Parking...")
    print("=" * 65)
    
    engine = RackAssociationEngine()
    
    # Register a storage slot ROI (normalized 0..1 polygon)
    engine.register_storage_slot(
        slot_id="SLOT-A01",
        name="Kệ A · Ô 01",
        polygon=[[0.1, 0.1], [0.3, 0.1], [0.3, 0.4], [0.1, 0.4]],
        cam_id="cam_1"
    )
    
    # Scenario A: Robot 9001 is carrying Rack A1 on its back
    # Robot is at [500, 400, 700, 550] (Width 200, Height 150)
    # Rack is mounted on top at [510, 260, 690, 430] (Bottom overlaps robot top)
    tracked_frame_A = [
        {
            "track_id": 9001,
            "class_name": "robot",
            "bbox": [500.0, 400.0, 700.0, 550.0],
            "ground_point_norm": (0.312, 0.509)
        },
        {
            "track_id": 101,
            "class_name": "rack",
            "bbox": [510.0, 260.0, 690.0, 430.0],
            "ground_point_norm": (0.312, 0.398)
        }
    ]
    
    res_A = engine.process_frame_associations(tracked_frame_A, cam_id="cam_1")
    robot_A = [o for o in res_A if o["track_id"] == 9001][0]
    rack_A = [o for o in res_A if o["track_id"] == 101][0]
    
    print(f" 🤖 Robot 9001 Carrying Rack: has_rack={robot_A['has_rack']}, carried_rack_id={robot_A['carried_rack_id']}")
    print(f" 📦 Rack 101 Carrier Robot : is_carried={rack_A['is_carried']}, carrier_robot={rack_A['carrier_robot']}")
    
    # Scenario B: Static Rack parked in Slot A1
    tracked_frame_B = [
        {
            "track_id": 102,
            "class_name": "rack",
            "bbox": [200.0, 150.0, 350.0, 300.0],
            "ground_point_norm": (0.18, 0.25)  # Inside slot polygon
        }
    ]
    res_B = engine.process_frame_associations(tracked_frame_B, cam_id="cam_1")
    rack_B = res_B[0]
    print(f" 🏢 Static Rack in Slot     : stored_slot_id={rack_B['stored_slot_id']}, slot_name='{rack_B.get('slot_name')}'")
    
    passed = robot_A["has_rack"] and (rack_B["stored_slot_id"] == "SLOT-A01")
    print(f" ✅ Rack Association Logic: {'PASS' if passed else 'FAIL'}")
    return passed


def main():
    print("=" * 65)
    print("🎯 SESSION 3 EXECUTION: SPATIAL MAPPING & TARGET REGISTRY")
    print("=" * 65)
    
    t1 = test_homography()
    t2 = test_registry_and_reid()
    t3 = test_rack_association()
    
    print("\n" + "=" * 65)
    print("📊 SESSION 3 SUMMARY RESULT")
    print("=" * 65)
    print(f" • 2D-to-3D Homography Metric Projection : {'✅ PASS' if t1 else '❌ FAIL'}")
    print(f" • Re-ID Cosine Similarity Matcher       : {'✅ PASS' if t2 else '❌ FAIL'}")
    print(f" • Rack Association (Mobile + Storage)   : {'✅ PASS' if t3 else '❌ FAIL'}")
    print("=" * 65)
    
    if t1 and t2 and t3:
        print("🎉 SESSION 3 COMPLETED SUCCESSFULLY! READY FOR SESSION 4.")
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
