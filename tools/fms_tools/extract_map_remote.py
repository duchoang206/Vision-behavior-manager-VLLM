import json
import psycopg2

def extract_map():
    conn = psycopg2.connect("host=192.168.5.105 dbname=fms_db_v2 user=rtc password=rtc@1234")
    c = conn.cursor()
    
    # Extract zones
    c.execute("SELECT id, name, control_point, type FROM zones WHERE control_point IS NOT NULL;")
    fms_zones = c.fetchall()
    
    zones = []
    min_x, max_x, min_y, max_y = 0, 10, 0, 10
    
    colors = ["#3b82f6", "#a855f7", "#22c55e", "#f59e0b", "#14b8a6", "#ec4899", "#ef4444"]
    color_idx = 0
    
    for z in fms_zones:
        z_id, z_name, z_cp, z_type = z
        if not z_cp or not isinstance(z_cp, list):
            continue
            
        polygon = []
        for point_str in z_cp:
            try:
                if isinstance(point_str, dict):
                    point_data = point_str
                else:
                    point_data = json.loads(point_str)
                x = point_data.get("x", 0)
                y = point_data.get("y", 0)
                polygon.append([x, y])
                
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)
            except Exception as e:
                pass
                
        if polygon:
            zones.append({
                "id": str(z_id),
                "name": z_name,
                "color": colors[color_idx % len(colors)],
                "floor": 1,
                "polygon": polygon
            })
            color_idx += 1
            
    # Also extract nodes to see bounds
    c.execute("SELECT id, node_x, node_y, node_name FROM nodes;")
    fms_nodes = c.fetchall()
    nodes_data = []
    for n in fms_nodes:
        n_id, x, y, action = n
        if x is not None and y is not None:
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)
            nodes_data.append({"id": str(n_id), "x": x, "y": y, "action": action})

    width = math.ceil(max_x - min_x) + 10 if 'math' in globals() else int(max_x - min_x + 10)
    depth = math.ceil(max_y - min_y) + 10 if 'math' in globals() else int(max_y - min_y + 10)
    
    # Adjust coordinates to be positive if they are negative
    offset_x = -min_x + 5 if min_x < 0 else 5
    offset_y = -min_y + 5 if min_y < 0 else 5
    
    for z in zones:
        for p in z["polygon"]:
            p[0] += offset_x
            p[1] += offset_y
            
    # Create Layout JSON
    layout = {
        "schema_version": "1.0",
        "id": "fms_map",
        "name": "FMS Warehouse",
        "units": "m",
        "size": {
            "width": int(max_x - min_x + 20),
            "depth": int(max_y - min_y + 20),
            "height": 10
        },
        "floors": [{"id": 1, "name": "Floor 1", "elevation": 0.0}],
        "columns": [],
        "lifts": [],
        "grid": {
            "cell_size": 1.0,
            "cols": int(max_x - min_x + 20),
            "rows": int(max_y - min_y + 20)
        },
        "zones": zones,
        "docks": [],
        "racks": [],
        "conveyors": [],
        "stations": [],
        "charging_stations": [],
        "parking": [],
        "restricted_areas": [],
        "walkways": [],
        "cameras": [],
        "sensors": [],
        "locations": [],
        "obstacles": [],
        "spawn": { "robots": [] }
    }
    
    with open("backend/app/warehouse_layout.json", "w") as f:
        json.dump(layout, f, indent=2)
        
    print(f"Generated layout with {len(zones)} zones and bounds: width={layout['size']['width']}, depth={layout['size']['depth']}.")
    
if __name__ == "__main__":
    extract_map()
