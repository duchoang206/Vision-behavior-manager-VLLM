import asyncio
import json
import logging
import math
import os
import time
from typing import Dict, Set, Optional, Any

logger = logging.getLogger("FMSBridge")

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    logger.warning("paho-mqtt not installed, MQTT bridge will stay offline until real FMS MQTT is available")
    MQTT_AVAILABLE = False

try:
    import psycopg2
    PG_AVAILABLE = True
except ImportError:
    PG_AVAILABLE = False

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False


def canonical_fms_status(
    raw_status: Optional[str] = None,
    driving: bool = False,
    is_charging: bool = False,
    errors: list = None,
    e_stop: str = "NONE",
    conn_state: str = "ONLINE"
) -> str:
    """
    Standardizes robot status strictly to canonical FMS ground truth states:
    - 'OFFLINE'
    - 'ERROR'
    - 'CHARGING'
    - 'RUNNING'
    - 'IDLE'
    """
    raw = (raw_status or "").upper().strip()
    if conn_state in ["CONNECTIONBROKEN", "OFFLINE"] or raw == "OFFLINE":
        return "OFFLINE"
    if (errors and len(errors) > 0) or (e_stop not in ["NONE", "AUTOACK", ""]) or raw in ["ERROR", "ALARM", "EMERGENCY"]:
        return "ERROR"
    if is_charging or raw in ["CHARGING", "CHARGE"]:
        return "CHARGING"
    if raw in ["RUNNING", "ACTIVE", "NAVIGATING", "TASK", "WORKING", "MOVING"] or driving:
        return "RUNNING"
    if raw in ["IDLE", "STANDBY", "WAITING", "SUSPEND", "AUTOMATIC", "MANUAL", "SEMIAUTOMATIC", ""]:
        return "IDLE"
    return "IDLE"


class FMSBridge:
    def __init__(self):
        # Configuration (from env or defaults)
        self.fms_ip = os.getenv("FMS_IP", "192.168.5.105")
        self.mqtt_port = int(os.getenv("FMS_MQTT_PORT", "1883"))
        self.mqtt_user = os.getenv("FMS_MQTT_USER", "admin")
        self.mqtt_pass = os.getenv("FMS_MQTT_PASS", "rtc@12345")

        self.db_port = int(os.getenv("FMS_DB_PORT", "5432"))
        self.db_name = os.getenv("FMS_DB_NAME", "fms_db_v2")
        self.db_user = os.getenv("FMS_DB_USER", "rtc")
        self.db_pass = os.getenv("FMS_DB_PASS", "rtc@1234")

        # 3D Origin in global world coords (Map TT)
        self.origin_x = float(os.getenv("FMS_ORIGIN_X", "185.0"))
        self.origin_y = float(os.getenv("FMS_ORIGIN_Y", "194.5"))
        self.layout_depth = float(os.getenv("FMS_LAYOUT_DEPTH", "18.0"))

        # State tracking
        self.active_websockets: Set[Any] = set()
        self.robot_states: Dict[str, dict] = {}
        self.tick_counter: int = 0
        self.is_connected_mqtt: bool = False
        self.last_mqtt_packet_time: float = 0.0
        self.total_packets_received: int = 0
        self.mode: str = "CONNECTING"  # "LIVE" | "CONNECTING"
        self.live_pose_ttl_sec = float(os.getenv("FMS_LIVE_POSE_TTL_SEC", "15.0"))
        self.pose_jump_floor_m = float(os.getenv("FMS_POSE_JUMP_FLOOR_M", "2.5"))
        self.pose_jump_speed_factor = float(os.getenv("FMS_POSE_JUMP_SPEED_FACTOR", "4.0"))

        # Offset map dictionary
        self.offset_map_dict: Dict[str, dict] = {
            "TT": {"x": 0.0, "y": 0.0, "theta": 0.0},
            "AAA": {"x": 201.107365, "y": 201.985339, "theta": 3.124776},
            "showROOMRTC_2": {"x": 201.809297, "y": 199.830395, "theta": -0.035875},
            "showRoomRtc_1": {"x": 199.950105, "y": 198.699333, "theta": -0.001444}
        }

        # Background tasks & MQTT client
        self._mqtt_client = None
        self._broadcast_task: Optional[asyncio.Task] = None
        self._datasocket_task: Optional[asyncio.Task] = None
        self._is_running: bool = False

    def _pose_distance(self, a, b) -> float:
        try:
            return math.hypot(float(a[0]) - float(b[0]), float(a[2]) - float(b[2]))
        except Exception:
            return 0.0

    def _preserve_live_pose(self, existing: Optional[dict], now: float) -> bool:
        if not existing:
            return False
        source = existing.get("last_position_source")
        age = now - float(existing.get("last_position_at", 0.0) or 0.0)
        return source in ("datasocket", "mqtt") and age < self.live_pose_ttl_sec

    def _apply_pose_update(
        self,
        robot: dict,
        pos_3d: list,
        heading_3d: float,
        map_id: str,
        source: str,
        raw_fms: Optional[dict] = None,
        velocity: Optional[float] = None,
    ) -> bool:
        now = time.time()
        prev_pos = robot.get("position")
        prev_at = float(robot.get("last_position_at", 0.0) or 0.0)
        if prev_pos and prev_at > 0:
            dt = max(0.05, now - prev_at)
            max_speed = float(robot.get("max_speed", 1.8) or 1.8)
            allowed_jump = max(self.pose_jump_floor_m, max_speed * dt * self.pose_jump_speed_factor)
            dist = self._pose_distance(prev_pos, pos_3d)
            if source in ("datasocket", "mqtt") and dt < 3.0 and dist > allowed_jump:
                logger.warning(
                    "Rejected FMS pose jump for %s: %.2fm in %.2fs from %s",
                    robot.get("id"),
                    dist,
                    dt,
                    source,
                )
                return False

        robot["position"] = pos_3d
        robot["heading"] = heading_3d
        robot["map_id"] = map_id
        robot["last_position_at"] = now
        robot["last_position_source"] = source
        if raw_fms is not None:
            robot["raw_fms"] = raw_fms
        if velocity is not None:
            robot["velocity"] = velocity
        return True

    def load_offset_maps_from_db(self):
        """Load nav_map offsets from PostgreSQL"""
        if not PG_AVAILABLE:
            return
        try:
            conn = psycopg2.connect(
                host=self.fms_ip,
                port=self.db_port,
                dbname=self.db_name,
                user=self.db_user,
                password=self.db_pass,
                connect_timeout=3
            )
            cur = conn.cursor()
            cur.execute("SELECT map_name, offset_map_list FROM nav_map WHERE map_name='TT';")
            row = cur.fetchone()
            if row and row[1]:
                for m in row[1]:
                    m_name = m.get("mapName")
                    if m_name:
                        self.offset_map_dict[m_name] = {
                            "x": float(m.get("xOffset", 0.0)),
                            "y": float(m.get("yOffset", 0.0)),
                            "theta": float(m.get("thetaOffset", 0.0))
                        }
                logger.info(f"Loaded {len(self.offset_map_dict)} map offsets from DB: {list(self.offset_map_dict.keys())}")
            conn.close()
        except Exception as e:
            logger.warning(f"Could not load offsets from FMS DB ({self.fms_ip}:{self.db_port}): {e}. Using defaults.")

    def load_robots_from_db(self):
        """Pre-populate real robot inventory & last known states from FMS PostgreSQL database"""
        if not PG_AVAILABLE:
            return
        try:
            conn = psycopg2.connect(
                host=self.fms_ip,
                port=self.db_port,
                dbname=self.db_name,
                user=self.db_user,
                password=self.db_pass,
                connect_timeout=3
            )
            cur = conn.cursor()
            cur.execute("""
                SELECT 
                    rl.id, 
                    rl.robot_name, 
                    rl.robot_serial, 
                    rl.enable_robot,
                    rm.model_name,
                    rd.robot_connection,
                    rd.robot_visualization,
                    rd.robot_state
                FROM robot_list rl
                LEFT JOIN robot_model rm ON rl.robot_model_id = rm.id
                LEFT JOIN robot_data rd ON rd.robot_id = rl.id
                WHERE rl.enable_robot = True AND (110 = ANY(rl.use_map_code_id) OR rl.use_map_code_id IS NULL)
                ORDER BY rl.id;
            """)
            rows = cur.fetchall()
            for r in rows:
                rid, rname, rserial, renable, rmodel, rconn, rvis, rstate = r
                serial_str = str(rserial or rname or rid)
                
                # Check connection status
                conn_state = (rconn or {}).get("connectionState", "OFFLINE")
                is_online = (conn_state == "ONLINE")
                
                # Extract last known position
                pos = (rvis or {}).get("agvPosition") or (rstate or {}).get("agvPosition") or {}
                map_id = pos.get("mapId", "TT")
                x_local = float(pos.get("x", 0.0))
                y_local = float(pos.get("y", 0.0))
                theta_local = float(pos.get("theta", 0.0))
                
                pos_3d, heading_3d = self.to_3d_pose(x_local, y_local, theta_local, map_id)
                if pos_3d is None:
                    pos_3d = [0.0, 0.0, 0.0]
                    heading_3d = 0.0
                
                # Extract battery
                bat_state = (rstate or {}).get("batteryState") or {}
                b_charge = bat_state.get("batteryCharge")
                battery = int(b_charge) if b_charge is not None else 100
                is_charging = bat_state.get("charging", False)
                
                existing = self.robot_states.get(serial_str)
                now = time.time()
                if existing and existing.get("status") in ["OFFLINE", "RUNNING", "ACTIVE", "CHARGING", "ERROR", "IDLE"]:
                    # Preserve authoritative live FMS / MQTT status
                    current_status = canonical_fms_status(raw_status=existing["status"])
                    current_fsm = existing.get("fsm", current_status)
                    current_vel = existing.get("velocity", 0.0)
                else:
                    current_status = "OFFLINE" if not is_online else ("CHARGING" if is_charging else "IDLE")
                    current_fsm = current_status
                    current_vel = 0.0

                if self._preserve_live_pose(existing, now):
                    pos_3d = existing.get("position", pos_3d)
                    heading_3d = existing.get("heading", heading_3d)
                    map_id = existing.get("map_id", map_id)
                    position_source = existing.get("last_position_source")
                    position_at = existing.get("last_position_at")
                    raw_fms = existing.get("raw_fms")
                else:
                    position_source = "postgres"
                    position_at = now
                    raw_fms = {
                        "x": round(x_local, 2),
                        "y": round(y_local, 2),
                        "theta": round(theta_local, 2)
                    }

                self.robot_states[serial_str] = {
                    "id": serial_str,
                    "model": rmodel or "AMR-Vision",
                    "floor": 1,
                    "lift_id": None,
                    "lift_stage": None,
                    "position": pos_3d,
                    "heading": heading_3d,
                    "velocity": current_vel,
                    "max_speed": 1.8,
                    "battery": battery,
                    "status": current_status,
                    "fsm": current_fsm,
                    "load": {"current": 0, "max": 100},
                    "path": [],
                    "path_index": 0,
                    "destination": None,
                    "map_id": map_id,
                    "raw_fms": raw_fms,
                    "last_position_at": position_at,
                    "last_position_source": position_source,
                }
            conn.close()
            logger.info(f"Loaded {len(self.robot_states)} real robots from FMS DB: {list(self.robot_states.keys())}")
        except Exception as e:
            logger.warning(f"Could not load robots from FMS DB ({self.fms_ip}:{self.db_port}): {e}")

    def to_3d_pose(self, x_local: float, y_local: float, theta_local: float, map_id: str):
        """Convert local map coordinates to global 3D scene coordinates"""
        if map_id not in self.offset_map_dict and map_id != "TT":
            return None, 0.0
        off = self.offset_map_dict.get(map_id, {"x": 0.0, "y": 0.0, "theta": 0.0})
        ct = math.cos(off["theta"])
        st = math.sin(off["theta"])
        gx = x_local * ct - y_local * st + off["x"]
        gy = x_local * st + y_local * ct + off["y"]
        gtheta = theta_local + off["theta"]

        # Match FMS stage coordinates: ROS/FMS Y is inverted on the rendered map.
        pos_3d = [round(gx - self.origin_x, 3), 0.0, round(self.layout_depth - (gy - self.origin_y), 3)]
        heading_3d = -gtheta
        return pos_3d, heading_3d

    def get_fleet_kpi(self):
        total = len(self.robot_states)
        active = sum(1 for r in self.robot_states.values() if r.get("status") in ["RUNNING", "ACTIVE"])
        charging = sum(1 for r in self.robot_states.values() if r.get("status") == "CHARGING")
        idle = sum(1 for r in self.robot_states.values() if r.get("status") == "IDLE")
        error = sum(1 for r in self.robot_states.values() if r.get("status") == "ERROR")
        offline = sum(1 for r in self.robot_states.values() if r.get("status") == "OFFLINE")
        return {
            "total": total,
            "active": active,
            "running": active,
            "charging": charging,
            "idle": idle,
            "warning": 0,
            "error": error,
            "offline": offline
        }

    def create_full_message(self):
        return {
            "type": "FULL",
            "state": {
                "schema_version": "1.0",
                "layout_id": "fms_map_tt",
                "runtime": {
                    "tick": self.tick_counter,
                    "tick_ms": 100,
                    "speed": 0,
                    "mode": self.mode
                },
                "fms_meta": {
                    "server_ip": self.fms_ip,
                    "mqtt_connected": self.is_connected_mqtt,
                    "last_packet_time": self.last_mqtt_packet_time,
                    "total_packets": self.total_packets_received,
                    "mode": self.mode
                },
                "robots": self.robot_states,
                "tasks": {},
                "lifts": {},
                "zones": {},
                "conveyors": {},
                "cameras": {},
                "sensors": {},
                "people": {},
                "alerts": {},
                "recent_events": [],
                "recent_decisions": [],
                "kpi": {
                    "tick": self.tick_counter,
                    "fleet": self.get_fleet_kpi(),
                    "operation": {
                        "throughput_per_min": 0,
                        "completed_today": 0,
                        "completed_target": 0,
                        "pending": 0,
                        "ongoing": len(self.robot_states),
                        "avg_task_time_s": 0,
                        "on_time_rate": 0,
                        "avg_utilization": 0
                    },
                    "efficiency": {
                        "avg_travel_distance_m": 0,
                        "avg_wait_time_s": 0,
                        "congestion_index": 0,
                        "energy_kwh": 0
                    },
                    "throughput_series": [],
                    "lifts": {"trips": 0, "utilization": 0, "avg_wait_s": 0, "faults": 0}
                },
                "subsystems": {
                    "NETWORK": "NORMAL" if self.is_connected_mqtt else "OFFLINE"
                }
            }
        }

    def create_patch_message(self):
        self.tick_counter += 1
        return {
            "type": "PATCH",
            "base_tick": self.tick_counter - 1,
            "tick": self.tick_counter,
            "patch": {
                "robots": self.robot_states,
                "kpi": {
                    "tick": self.tick_counter,
                    "fleet": self.get_fleet_kpi(),
                    "operation": {
                        "throughput_per_min": 0,
                        "completed_today": 0,
                        "completed_target": 0,
                        "pending": 0,
                        "ongoing": len(self.robot_states),
                        "avg_task_time_s": 0,
                        "on_time_rate": 0,
                        "avg_utilization": 0
                    },
                    "efficiency": {
                        "avg_travel_distance_m": 0,
                        "avg_wait_time_s": 0,
                        "congestion_index": 0,
                        "energy_kwh": 0
                    }
                },
                "fms_meta": {
                    "server_ip": self.fms_ip,
                    "mqtt_connected": self.is_connected_mqtt,
                    "last_packet_time": self.last_mqtt_packet_time,
                    "total_packets": self.total_packets_received,
                    "mode": self.mode
                }
            },
            "events": []
        }

    def _on_mqtt_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logger.info(f"Successfully connected to FMS MQTT Broker ({self.fms_ip}:{self.mqtt_port})")
            self.is_connected_mqtt = True
            client.subscribe("VDA/+/+/+/visualization")
            client.subscribe("VDA/+/+/+/state")
            client.subscribe("VDA/+/+/+/connection")
        else:
            logger.error(f"Failed to connect to FMS MQTT broker, code: {rc}")
            self.is_connected_mqtt = False

    def _on_mqtt_disconnect(self, client, userdata, rc):
        logger.warning(f"Disconnected from FMS MQTT Broker (code: {rc})")
        self.is_connected_mqtt = False
        self.mode = "CONNECTING"

    def _on_mqtt_message(self, client, userdata, msg):
        try:
            self.last_mqtt_packet_time = time.time()
            self.total_packets_received += 1
            if self.mode != "LIVE":
                self.mode = "LIVE"
                logger.info("Real FMS MQTT packets detected! Switched mode to LIVE.")

            topic_parts = msg.topic.split('/')
            if len(topic_parts) < 5:
                return

            serial_number = str(topic_parts[3] if len(topic_parts) == 5 else topic_parts[4])
            msg_type = topic_parts[-1]
            payload = json.loads(msg.payload.decode("utf-8"))

            if serial_number not in self.robot_states:
                # Only dynamically register robots that belong to Map TT
                pos = payload.get("agvPosition", {})
                map_id = pos.get("mapId", "TT")
                if map_id not in self.offset_map_dict and map_id != "TT":
                    return
                if msg_type == "connection" and "x" not in pos:
                    return

                self.robot_states[serial_number] = {
                    "id": str(serial_number),
                    "model": "AMR-L",
                    "floor": 1,
                    "lift_id": None,
                    "lift_stage": None,
                    "position": [0.0, 0.0, 0.0],
                    "heading": 0.0,
                    "velocity": 0.0,
                    "max_speed": 1.8,
                    "battery": 100,
                    "status": "OFFLINE",
                    "fsm": "OFFLINE",
                    "load": {"current": 0, "max": 100},
                    "path": [],
                    "path_index": 0,
                    "destination": None,
                    "map_id": map_id,
                    "last_position_at": 0.0,
                    "last_position_source": "init",
                }

            robot = self.robot_states[serial_number]

            # 0. Connection State
            if msg_type == "connection":
                conn_state = payload.get("connectionState", "OFFLINE")
                if conn_state in ["CONNECTIONBROKEN", "OFFLINE"]:
                    robot["status"] = "OFFLINE"
                    robot["fsm"] = "OFFLINE"
                    robot["velocity"] = 0.0
                elif conn_state == "ONLINE" and robot["status"] == "OFFLINE":
                    robot["status"] = "IDLE"
                    robot["fsm"] = "IDLE"

            # 1. Position & Orientation & Velocity
            pos = payload.get("agvPosition", {})
            if "x" in pos and "y" in pos:
                map_id = pos.get("mapId", "TT")
                theta = pos.get("theta", 0.0)
                pos_3d, heading_3d = self.to_3d_pose(pos["x"], pos["y"], theta, map_id)
                if pos_3d is not None:
                    self._apply_pose_update(
                        robot,
                        pos_3d,
                        heading_3d,
                        map_id,
                        source="mqtt",
                        raw_fms={
                            "x": round(float(pos["x"]), 2),
                            "y": round(float(pos["y"]), 2),
                            "theta": round(float(theta), 2),
                        },
                    )

            vel = payload.get("velocity", {})
            vx = vel.get("vx", 0.0)
            vy = vel.get("vy", 0.0)
            if robot["status"] != "OFFLINE":
                robot["velocity"] = round(math.sqrt(vx**2 + vy**2), 2)
            else:
                robot["velocity"] = 0.0

            # 2. Battery & Operating Mode & Status
            if msg_type == "state":
                battery_state = payload.get("batteryState", {})
                b_charge = battery_state.get("batteryCharge")
                if b_charge is not None:
                    robot["battery"] = int(b_charge)

                is_charging = battery_state.get("charging", False)
                for action in payload.get("actionStates", []):
                    if action.get("actionType") == "startCharging" and action.get("actionStatus") in ["INITIALIZING", "RUNNING", "FINISHED"]:
                        is_charging = True
                        break

                driving = payload.get("driving", False)
                errors = payload.get("errors", [])
                e_stop = payload.get("safetyState", {}).get("eStop", "NONE")

                if robot["status"] == "OFFLINE":
                    pass  # Keep offline if connection is broken
                else:
                    new_status = canonical_fms_status(
                        raw_status=None,
                        driving=driving or robot["velocity"] > 0.02,
                        is_charging=is_charging,
                        errors=errors,
                        e_stop=e_stop
                    )
                    # If FMS Authoritative DataSocket already set RUNNING, don't drop to IDLE on transient zero-velocity pause
                    if new_status == "IDLE" and robot.get("status") == "RUNNING" and not is_charging and not errors and (e_stop == "NONE" or e_stop == "AUTOACK"):
                        pass
                    else:
                        robot["status"] = new_status
                        robot["fsm"] = new_status

                last_node = payload.get("lastNodeId")
                if last_node:
                    robot["destination"] = str(last_node)

        except Exception as e:
            logger.error(f"Error parsing MQTT message: {e}")

    async def _broadcast_loop(self):
        """Broadcasts 10Hz PATCH diffs to all connected WebSocket clients with periodic FMS DB refresh"""
        loop_counter = 0
        while self._is_running:
            await asyncio.sleep(0.1)
            loop_counter += 1
            
            # Periodic DB sync every 5 seconds (50 ticks) to catch any FMS UI updates
            if loop_counter % 50 == 0:
                try:
                    self.load_robots_from_db()
                except Exception as e:
                    logger.debug(f"Periodic FMS DB sync error: {e}")

            if not self.active_websockets:
                continue

            message = self.create_patch_message()
            msg_str = json.dumps(message)

            dead_ws = set()
            for ws in list(self.active_websockets):
                try:
                    await ws.send_text(msg_str)
                except Exception:
                    dead_ws.add(ws)

            for ws in dead_ws:
                self.active_websockets.discard(ws)

    async def _fms_datasocket_loop(self):
        """Connects directly to authoritative FMS Drogon dataSocket WebSocket (ws://192.168.5.105:9009/dataSocket?mapId=110)"""
        if not WEBSOCKETS_AVAILABLE:
            return
            
        uri = f"ws://{self.fms_ip}:9009/dataSocket?mapId=110"
        while self._is_running:
            try:
                logger.info(f"Connecting to FMS authoritative dataSocket: {uri}...")
                async with websockets.connect(uri, ping_interval=10, ping_timeout=5) as ws:
                    logger.info("Connected to FMS authoritative dataSocket!")
                    self.mode = "LIVE"
                    while self._is_running:
                        msg = await ws.recv()
                        self.last_mqtt_packet_time = time.time()
                        self.total_packets_received += 1
                        try:
                            payload = json.loads(msg)
                            robot_list = payload.get("data", [])
                            for r_item in robot_list:
                                loc = r_item.get("location", {})
                                serial_number = str(loc.get("serialNumber") or loc.get("serialInt") or "")
                                if not serial_number:
                                    continue
                                
                                x = float(loc.get("x", 0.0))
                                y = float(loc.get("y", 0.0))
                                theta = float(loc.get("theta", 0.0))
                                pos_3d, heading_3d = self.to_3d_pose(x, y, theta, "TT")
                                if pos_3d is None:
                                    continue
                                
                                status_raw = loc.get("status", "OFFLINE")
                                vx = float(loc.get("vx", 0.0))
                                vy = float(loc.get("vy", 0.0))
                                vel = round(math.sqrt(vx**2 + vy**2), 2)
                                edge_path = r_item.get("edgePath") or []
                                has_active_path = len(edge_path) > 0
                                canonical_status = canonical_fms_status(
                                    raw_status=status_raw,
                                    driving=(has_active_path or vel > 0.02)
                                )
                                bat = r_item.get("battery", {}).get("batteryCharge")
                                battery = int(bat) if bat is not None else 100
                                model = loc.get("model") or r_item.get("model") or "AMR-Vision"

                                robot = self.robot_states.setdefault(serial_number, {
                                    "id": serial_number,
                                    "model": model,
                                    "floor": 1,
                                    "lift_id": None,
                                    "lift_stage": None,
                                    "position": pos_3d,
                                    "heading": heading_3d,
                                    "velocity": 0.0,
                                    "max_speed": 1.8,
                                    "battery": battery,
                                    "status": canonical_status,
                                    "fsm": canonical_status,
                                    "load": {"current": 0, "max": 100},
                                    "path": [],
                                    "path_index": 0,
                                    "destination": None,
                                    "map_id": "TT",
                                    "last_position_at": 0.0,
                                    "last_position_source": "init",
                                })
                                robot["model"] = model
                                robot["battery"] = battery
                                robot["status"] = canonical_status
                                robot["fsm"] = canonical_status
                                self._apply_pose_update(
                                    robot,
                                    pos_3d,
                                    heading_3d,
                                    "TT",
                                    source="datasocket",
                                    raw_fms={
                                        "x": round(x, 2),
                                        "y": round(y, 2),
                                        "theta": round(theta, 2)
                                    },
                                    velocity=vel if status_raw != "OFFLINE" else 0.0,
                                )
                        except Exception as e:
                            logger.debug(f"Error parsing dataSocket payload: {e}")
            except Exception as e:
                logger.debug(f"FMS dataSocket error: {e}. Retrying in 3s...")
                await asyncio.sleep(3.0)

    async def start(self):
        """Initialize and start the FMS bridge"""
        self._is_running = True
        logger.info(f"Starting FMS Bridge (FMS Server: {self.fms_ip}:{self.mqtt_port})...")
        self.load_offset_maps_from_db()
        self.load_robots_from_db()

        if MQTT_AVAILABLE:
            try:
                self._mqtt_client = mqtt.Client()
                if self.mqtt_user and self.mqtt_pass:
                    self._mqtt_client.username_pw_set(self.mqtt_user, self.mqtt_pass)
                self._mqtt_client.on_connect = self._on_mqtt_connect
                self._mqtt_client.on_disconnect = self._on_mqtt_disconnect
                self._mqtt_client.on_message = self._on_mqtt_message

                self._mqtt_client.connect_async(self.fms_ip, self.mqtt_port, 60)
                self._mqtt_client.loop_start()
            except Exception as e:
                logger.error(f"Could not connect to FMS MQTT broker ({self.fms_ip}): {e}")
                self.is_connected_mqtt = False

        # Start broadcast task and direct FMS dataSocket client
        self._broadcast_task = asyncio.create_task(self._broadcast_loop())
        if WEBSOCKETS_AVAILABLE:
            self._datasocket_task = asyncio.create_task(self._fms_datasocket_loop())
        logger.info("FMS Bridge started successfully.")

    async def stop(self):
        """Gracefully stop the FMS bridge"""
        self._is_running = False
        if self._broadcast_task:
            self._broadcast_task.cancel()
        if self._datasocket_task:
            self._datasocket_task.cancel()
        if self._mqtt_client:
            try:
                self._mqtt_client.loop_stop()
                self._mqtt_client.disconnect()
            except Exception:
                pass
        logger.info("FMS Bridge stopped.")

    async def handle_websocket(self, websocket):
        """Handles new WebSocket client connection"""
        await websocket.accept()
        logger.info(f"Frontend connected to FMS 3D WebSocket ({len(self.robot_states)} robots, mode: {self.mode})")

        # Send initial FULL snapshot
        await websocket.send_text(json.dumps(self.create_full_message()))
        self.active_websockets.add(websocket)

        try:
            while True:
                msg = await websocket.receive_text()
                try:
                    data = json.loads(msg)
                    if data.get("type") == "RESYNC":
                        await websocket.send_text(json.dumps(self.create_full_message()))
                except Exception:
                    pass
        except Exception:
            pass
        finally:
            self.active_websockets.discard(websocket)
            logger.info("Client disconnected from FMS 3D WebSocket")

    def get_status(self):
        return {
            "fms_ip": self.fms_ip,
            "mqtt_port": self.mqtt_port,
            "mqtt_connected": self.is_connected_mqtt,
            "mode": self.mode,
            "robot_count": len(self.robot_states),
            "last_packet_time": self.last_mqtt_packet_time,
            "total_packets": self.total_packets_received,
            "offsets": self.offset_map_dict
        }

    def get_robots(self):
        return {
            "status": "success",
            "count": len(self.robot_states),
            "mode": self.mode,
            "robots": self.robot_states,
            "kpi": self.get_fleet_kpi()
        }

    def update_config(self, config: dict):
        if "fms_ip" in config:
            self.fms_ip = config["fms_ip"]
        if "mqtt_port" in config:
            self.mqtt_port = int(config["mqtt_port"])
        if "origin_x" in config:
            self.origin_x = float(config["origin_x"])
        if "origin_y" in config:
            self.origin_y = float(config["origin_y"])
        if "layout_depth" in config:
            self.layout_depth = float(config["layout_depth"])
        return self.get_status()


# Global singleton instance
fms_bridge = FMSBridge()
