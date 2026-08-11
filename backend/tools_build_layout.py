import os
import json
import math
import struct
import base64
import psycopg2
from PIL import Image
import io

DB_HOST = "192.168.5.105"
DB_PORT = 5432
DB_NAME = "fms_db_v2"
DB_USER = "rtc"
DB_PASS = "rtc@1234"

# 3D World Origin for Map TT
ORIGIN_X = 185.0
ORIGIN_Y = 194.5
LAYOUT_WIDTH = 26.0
LAYOUT_DEPTH = 18.0
LAYOUT_HEIGHT = 4.0
ROAD_WIDTH = 0.7  # 70cm road width

# SLAM Image dimensions and sub-millimeter calibrated origin in FMS
SLAM_RES_X = 0.050024
SLAM_RES_Z = 0.049836
SLAM_IMG_OX = 186.170
SLAM_IMG_OY = 210.627
SLAM_IMG_W = 465
SLAM_IMG_H = 301

def world_to_layout(x, y):
    return [round(x - ORIGIN_X, 3), round(LAYOUT_DEPTH - (y - ORIGIN_Y), 3)]

def eval_cubic_bezier(p0, p1, p2, p3, t):
    it = 1.0 - t
    x = it**3 * p0[0] + 3 * it**2 * t * p1[0] + 3 * it * t**2 * p2[0] + t**3 * p3[0]
    y = it**3 * p0[1] + 3 * it**2 * t * p1[1] + 3 * it * t**2 * p2[1] + t**3 * p3[1]
    return [x, y]

def make_ribbon(points, width):
    if len(points) < 2:
        return []
    left = []
    right = []
    half_w = width / 2.0
    for i in range(len(points)):
        if i == 0:
            dx = points[1][0] - points[0][0]
            dy = points[1][1] - points[0][1]
        elif i == len(points) - 1:
            dx = points[-1][0] - points[-2][0]
            dy = points[-1][1] - points[-2][1]
        else:
            dx = points[i+1][0] - points[i-1][0]
            dy = points[i+1][1] - points[i-1][1]
        
        L = math.hypot(dx, dy)
        if L < 1e-6:
            nx, ny = 0, half_w
        else:
            nx = -dy / L * half_w
            ny = dx / L * half_w
            
        px, py = points[i]
        left.append([round(px + nx, 3), round(py + ny, 3)])
        right.append([round(px - nx, 3), round(py - ny, 3)])
        
    return left + right[::-1]

def build_fms_assets():
    print(f"Connecting to FMS DB at {DB_HOST}:{DB_PORT}...")
    conn = psycopg2.connect(f"host={DB_HOST} port={DB_PORT} dbname={DB_NAME} user={DB_USER} password={DB_PASS}")
    cur = conn.cursor()

    # Calculate exact SLAM image rect in 3D scene (100% overlay with LiDAR walls):
    slam_x1 = round(SLAM_IMG_OX - ORIGIN_X, 3)
    slam_x2 = round(SLAM_IMG_OX + SLAM_IMG_W * SLAM_RES_X - ORIGIN_X, 3)
    slam_z1 = round(LAYOUT_DEPTH - (SLAM_IMG_OY - ORIGIN_Y), 3)
    slam_z2 = round(slam_z1 + SLAM_IMG_H * SLAM_RES_Z, 3)

    slam_map = {
        "href": "/maps/fms_map.png",
        "source": "postgres.nav_map.data_map",
        "map_name": "TT",
        "rect": [slam_x1, slam_z1, slam_x2, slam_z2],
        "pixel_size": [SLAM_IMG_W, SLAM_IMG_H],
        "flip_y": False
    }

    # 1. Save Map Image to public directory and backend
    cur.execute("SELECT map_name, data_map, offset_map_list, raw_file FROM nav_map WHERE map_name='TT';")
    nav_row = cur.fetchone()
    if nav_row:
        slam_map["map_name"] = nav_row[0] or "TT"
        if nav_row[2]:
            slam_map["offset_map_list"] = nav_row[2]
            
    if nav_row and nav_row[1]:
        img_data = base64.b64decode(nav_row[1])
        img = Image.open(io.BytesIO(img_data))
        slam_map["pixel_size"] = [img.width, img.height]
        with open("/app/fms_map.png", "wb") as f:
            f.write(img_data)
        print("Saved map image to /app/fms_map.png")

    # 2. Extract SLAM LiDAR points & 3D Wall segments from raw_file (MRSLAM)
    slam_points_3d = []
    slam_walls = []
    if nav_row and nav_row[3]:
        raw = bytes(nav_row[3])
        n_pts = struct.unpack('<I', raw[35:39])[0]
        pts = [struct.unpack('<dd', raw[39 + i*16 : 39 + (i+1)*16]) for i in range(n_pts)]
        print(f"Extracted {len(pts)} SLAM raw LiDAR points from MRSLAM")
        
        for p in pts:
            lx = round(p[0] - ORIGIN_X, 3)
            lz = round(LAYOUT_DEPTH - (p[1] - ORIGIN_Y), 3)
            slam_points_3d.append([lx, lz])
            
        # Connect consecutive points into wall segments if dist < 0.25m
        for i in range(len(pts) - 1):
            p1 = pts[i]
            p2 = pts[i+1]
            dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            if dist < 0.25:
                x1 = round(p1[0] - ORIGIN_X, 3)
                z1 = round(LAYOUT_DEPTH - (p1[1] - ORIGIN_Y), 3)
                x2 = round(p2[0] - ORIGIN_X, 3)
                z2 = round(LAYOUT_DEPTH - (p2[1] - ORIGIN_Y), 3)
                slam_walls.append([[x1, z1], [x2, z2]])
        print(f"Built {len(slam_walls)} 3D continuous SLAM wall segments")

    # 3. Fetch all nodes for map_code_id = 110 (Map TT)
    cur.execute("SELECT id, node_name, node_x, node_y, node_type, node_sub_type FROM nodes WHERE map_code_id=110;")
    nodes = {}
    for r in cur.fetchall():
        nodes[r[0]] = {
            "id": str(r[0]),
            "name": r[1] or str(r[0]),
            "x": float(r[2]),
            "y": float(r[3]),
            "type": r[4] or "Passing",
            "sub_type": r[5] or "Passing"
        }
    print(f"Loaded {len(nodes)} nodes")

    # 4. Fetch all edges for map_code_id = 110
    cur.execute("SELECT id, edge_code, start_node, end_node, edge_type, control_point, distance FROM edges WHERE map_code_id=110;")
    edges = cur.fetchall()
    print(f"Loaded {len(edges)} edges")

    walkways = []
    locations = []
    charging_stations = []
    stations = []
    
    # Process edges into walkways
    for idx, e in enumerate(edges):
        eid, ecode, start_id, end_id, etype, cp, dist = e
        if start_id not in nodes or end_id not in nodes:
            continue
        n1 = nodes[start_id]
        n2 = nodes[end_id]
        
        p_start = [n1["x"], n1["y"]]
        p_end = [n2["x"], n2["y"]]
        
        pts = []
        if etype == "Nurbs" and cp and "controlPoints" in cp and len(cp["controlPoints"]) >= 4:
            cps = cp["controlPoints"]
            p0 = [cps[0]["x"], cps[0]["y"]]
            p1 = [cps[1]["x"], cps[1]["y"]]
            p2 = [cps[2]["x"], cps[2]["y"]]
            p3 = [cps[3]["x"], cps[3]["y"]]
            for step in range(11):
                t = step / 10.0
                pt = eval_cubic_bezier(p0, p1, p2, p3, t)
                pts.append(world_to_layout(pt[0], pt[1]))
        else:
            pts = [
                world_to_layout(p_start[0], p_start[1]),
                world_to_layout(p_end[0], p_end[1])
            ]
            
        poly = make_ribbon(pts, ROAD_WIDTH)
        if poly:
            walkways.append({
                "id": f"ROAD-{eid}",
                "polygon": poly,
                "robots_allowed": True,
                "speed_limit_mps": 1.5
            })

    # Process nodes into Charging Stations & Locations & Waypoints
    for nid, n in nodes.items():
        lx, lz = world_to_layout(n["x"], n["y"])
        
        if n["type"] == "Charging":
            charging_stations.append({
                "id": n["name"] or f"CHG-{nid}",
                "zone": "MAIN",
                "position": [lx, 0.0, lz],
                "heading": 0.0,
                "power_kw": 22.0,
                "access_point": [lx, lz]
            })
        elif n["type"] == "Working":
            stations.append({
                "id": n["name"] or f"STN-{nid}",
                "kind": "WORK",
                "zone": "MAIN",
                "rect": [round(lx - 0.4, 2), round(lz - 0.4, 2), round(lx + 0.4, 2), round(lz + 0.4, 2)],
                "access_point": [lx, lz]
            })
            
        locations.append({
            "id": n["name"] or str(nid),
            "kind": "CHARGING" if n["type"] == "Charging" else "SHELF",
            "zone": "MAIN",
            "floor": 1,
            "rack_id": None,
            "level_range": None,
            "access_point": [lx, lz]
        })

    # Main zone - fits the SLAM map bounds
    zones = [
        {
            "id": "MAIN",
            "name": "FMS Floor (Map TT)",
            "color": "#3b82f6",
            "floor": 1,
            "polygon": [
                [0.5, 0.5],
                [LAYOUT_WIDTH - 0.5, 0.5],
                [LAYOUT_WIDTH - 0.5, LAYOUT_DEPTH - 0.5],
                [0.5, LAYOUT_DEPTH - 0.5]
            ]
        }
    ]

    layout = {
        "schema_version": "1.0",
        "id": "fms_map_tt",
        "name": "FMS Warehouse (Map TT)",
        "units": "m",
        "origin_world": [ORIGIN_X, ORIGIN_Y],
        "coordinate_transform": {
            "source": "fms_ros_stage",
            "x": "world_x - origin_world_x",
            "z": "layout_depth - (world_y - origin_world_y)"
        },
        "slam_map": slam_map,
        "slam_walls": slam_walls,
        "slam_points": slam_points_3d,
        "size": {
            "width": int(LAYOUT_WIDTH),
            "depth": int(LAYOUT_DEPTH),
            "height": int(LAYOUT_HEIGHT)
        },
        "floors": [
            {
                "id": 1,
                "name": "Ground Floor",
                "elevation": 0.0
            }
        ],
        "columns": [],
        "lifts": [],
        "grid": {
            "cell_size": 0.5,
            "cols": int(LAYOUT_WIDTH / 0.5),
            "rows": int(LAYOUT_DEPTH / 0.5)
        },
        "zones": zones,
        "docks": [],
        "racks": [],
        "conveyors": [],
        "stations": stations,
        "charging_stations": charging_stations,
        "parking": [],
        "restricted_areas": [],
        "walkways": walkways,
        "cameras": [],
        "sensors": [],
        "locations": locations,
        "obstacles": [],
        "spawn": {"robots": []}
    }

    # Save layout
    with open("/app/warehouse_layout.json", "w") as f:
        json.dump(layout, f, indent=2)

    print("Successfully built /app/warehouse_layout.json!")
    print(f"- Walkways (Roads): {len(walkways)}")
    print(f"- Charging Stations: {len(charging_stations)}")
    print(f"- Stations: {len(stations)}")
    print(f"- Locations: {len(locations)}")
    print(f"- SLAM 3D Walls: {len(slam_walls)}")

if __name__ == "__main__":
    build_fms_assets()
