'use client';

import React, { useEffect, useRef, useState, useMemo, useCallback } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { useLanguage } from '../LanguageContext';
import { useAppTheme } from '../ThemeContext';
import {
  Bot,
  User,
  Package,
  Boxes,
  Layers,
  Compass,
  RotateCcw,
  BatteryCharging,
  Battery,
  Navigation as NavIcon,
  Crosshair,
  Gauge,
  Clock,
  ShieldAlert,
  Sparkles,
  Activity,
  AlertTriangle,
  Radio,
  CheckCircle2,
  Zap,
  MapPin
} from 'lucide-react';

// ─── Interfaces ─────────────────────────────────────────────────────────────
interface RobotData {
  id: string;
  model: string;
  floor: number;
  position: [number, number, number];
  heading: number;
  velocity: number;
  max_speed?: number;
  battery: number;
  status: 'RUNNING' | 'ACTIVE' | 'CHARGING' | 'IDLE' | 'ERROR' | 'OFFLINE' | 'CARRYING_RACK';
  fsm: string;
  destination: string | null;
  map_id?: string;
  load?: { current: number; max: number };
  has_rack?: boolean;
  carried_rack_id?: string | null;
  raw_fms?: { x: number; y: number; theta: number };
  fms_position?: [number, number, number];
  vision_position?: [number, number, number] | null;
  delta_distance_m?: number | null;
  cross_check_status?: 'SYNCED' | 'CHARGING_VERIFIED' | 'DEVIATED' | 'FMS_ONLY' | 'VISION_ONLY' | 'OFFLINE';
  cross_check_msg?: string;
  cam_id?: string | null;
  source?: string;
  last_seen?: number;
  ui_last_seen?: number;
}

interface PersonData {
  id: string;
  position: [number, number, number];
  heading: number;
  velocity: number;
  status: 'WORKING' | 'WALKING' | 'IDLE' | 'ALERT';
  zone?: string;
  safety_alert?: boolean;
  cam_id?: string;
  track_id?: number;
  near_robot_id?: string | null;
  near_robot_dist?: number | null;
  safety_msg?: string;
  last_seen?: number;
  ui_last_seen?: number;
}

interface RackData {
  id: string;
  position: [number, number, number];
  status: 'STORED' | 'CARRIED';
  carried_by?: string | null;
  slot_id?: string | null;
  cam_id?: string;
  cross_check_msg?: string;
  last_seen?: number;
  ui_last_seen?: number;
}

interface FleetKpi {
  total: number;
  active: number;
  charging: number;
  idle: number;
  warning: number;
  error: number;
  offline?: number;
  persons_count?: number;
  racks_count?: number;
}

interface FmsMeta {
  server_ip: string;
  mqtt_connected: boolean;
  last_packet_time: number;
  total_packets: number;
  mode: 'LIVE' | 'CONNECTING';
}

type Point2 = [number, number];
type Point3 = [number, number, number];

interface FmsLayoutRectItem {
  id: string;
  kind?: string;
  zone?: string;
  rect: [number, number, number, number];
  access_point?: Point2;
}

interface FmsLayoutPointItem {
  id: string;
  kind?: string;
  zone?: string;
  floor?: number;
  position?: Point3;
  access_point?: Point2;
  heading?: number;
  power_kw?: number;
}

interface FmsLayoutPolygonItem {
  id: string;
  name?: string;
  color?: string;
  floor?: number;
  polygon: Point2[];
  robots_allowed?: boolean;
  speed_limit_mps?: number;
}

interface FmsLayoutCamera {
  id: string;
  zone?: string;
  floor?: number;
  position: Point3;
  look_at?: Point3;
  fov_deg?: number;
  range_m?: number;
}

interface FmsLayout {
  id: string;
  name?: string;
  units?: string;
  origin_world?: Point2;
  slam_map?: {
    href?: string;
    source?: string;
    map_name?: string;
    rect?: [number, number, number, number];
    pixel_size?: [number, number];
    flip_y?: boolean;
  };
  slam_walls?: [Point2, Point2][];
  slam_points?: Point2[];
  size?: {
    width?: number;
    depth?: number;
    height?: number;
  };
  grid?: {
    cell_size?: number;
    cols?: number;
    rows?: number;
  };
  zones?: FmsLayoutPolygonItem[];
  walkways?: FmsLayoutPolygonItem[];
  restricted_areas?: FmsLayoutPolygonItem[];
  conveyors?: FmsLayoutRectItem[];
  docks?: FmsLayoutRectItem[];
  racks?: FmsLayoutRectItem[];
  stations?: FmsLayoutRectItem[];
  charging_stations?: FmsLayoutPointItem[];
  parking?: FmsLayoutPolygonItem[];
  locations?: FmsLayoutPointItem[];
  cameras?: FmsLayoutCamera[];
  sensors?: FmsLayoutPointItem[];
  obstacles?: FmsLayoutRectItem[];
}

interface LayoutFrame {
  minX: number;
  minZ: number;
  width: number;
  depth: number;
  height: number;
  centerX: number;
  centerZ: number;
  gridSize: number;
  gridDivisions: number;
}

// ─── Color Palette for Statuses ─────────────────────────────────────────────
const STATUS_COLORS: Record<string, string> = {
  RUNNING: '#16a34a',       // Vibrant Emerald Running (FMS Ground Truth)
  ACTIVE: '#16a34a',        // Vibrant Emerald Active
  CHARGING: '#f59e0b',      // Amber Charging
  IDLE: '#0284c7',          // Sky Standby
  OFFLINE: '#64748b',       // Slate Gray Offline
  CARRYING_RACK: '#8b5cf6', // Purple Carrier
  ERROR: '#ef4444',         // Rose Error
  WORKING: '#38bdf8',       // Cyan Working
  WALKING: '#10b981',       // Green Walking
  STORED: '#64748b',        // Slate Stored
  CARRIED: '#a855f7',       // Violet Carried
};

// ─── Initial FMS Fleet Data Fallback (Empty, populated from FMS Live WebSocket) ──────────
const INITIAL_ROBOTS: Record<string, RobotData> = {};

const INITIAL_PERSONS: Record<string, PersonData> = {};

const INITIAL_RACKS: Record<string, RackData> = {};

const DEFAULT_CAMERAS: FmsLayoutCamera[] = [
  {
    id: 'Cam 1',
    zone: 'MAIN',
    floor: 1,
    position: [13.8, 1.15, 6.85],
    look_at: [10.5, 0.0, 11.8],
    fov_deg: 76,
    range_m: 8.8,
  },
  {
    id: 'Cam 4',
    zone: 'MAIN',
    floor: 1,
    position: [11.2, 1.15, 15.40],
    look_at: [15.2, 0.0, 12.0],
    fov_deg: 78,
    range_m: 8.2,
  },
];

// ─── Motorcycle-Style Analog Speedometer Gauge (Kim gạt từ góc phần 3 bên trái sang đối xứng) ─────────
interface MotorcycleSpeedometerGaugeProps {
  velocity: number;
  maxSpeed?: number;
  status?: string;
  isDark?: boolean;
}

export const MotorcycleSpeedometerGauge: React.FC<MotorcycleSpeedometerGaugeProps> = ({
  velocity,
  maxSpeed = 2.0,
  status = 'RUNNING',
  isDark = true,
}) => {
  const currentSpeed = Math.max(0, velocity);
  const kmh = (currentSpeed * 3.6).toFixed(1);
  const mps = currentSpeed.toFixed(2);
  const ratio = Math.min(1.0, currentSpeed / maxSpeed);

  // 0.0 starts from bottom-left (Quadrant 3, -135 deg) to bottom-right (Quadrant 4, +135 deg)
  // Perfectly symmetrical around top 12 o'clock (0 deg)
  const startAngle = -135;
  const totalSweep = 270;
  const needleAngle = startAngle + ratio * totalSweep;

  const cx = 110;
  const cy = 82;
  const r = 58;

  const ticks = [0.0, 0.5, 1.0, 1.5, 2.0];
  const minorTicks = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4, 1.6, 1.7, 1.8, 1.9];

  const angleToCoord = (centerX: number, centerY: number, radius: number, deg: number) => {
    const rad = (deg * Math.PI) / 180.0;
    return {
      x: centerX + radius * Math.sin(rad),
      y: centerY - radius * Math.cos(rad),
    };
  };

  const makeArcPath = (centerX: number, centerY: number, radius: number, startDeg: number, endDeg: number) => {
    const p1 = angleToCoord(centerX, centerY, radius, startDeg);
    const p2 = angleToCoord(centerX, centerY, radius, endDeg);
    const diff = endDeg - startDeg;
    const largeArc = Math.abs(diff) > 180 ? 1 : 0;
    const sweep = diff > 0 ? 1 : 0;
    return `M ${p1.x} ${p1.y} A ${radius} ${radius} 0 ${largeArc} ${sweep} ${p2.x} ${p2.y}`;
  };

  return (
    <div style={{
      background: isDark ? 'linear-gradient(145deg, #090d16, #131d2e)' : 'linear-gradient(145deg, #f8fafc, #edf2f7)',
      borderRadius: '14px',
      padding: '12px 14px',
      border: isDark ? '1px solid #1e293b' : '1px solid #cbd5e1',
      boxShadow: isDark ? 'inset 0 2px 10px rgba(0,0,0,0.6), 0 4px 16px rgba(0,0,0,0.3)' : 'inset 0 2px 8px rgba(0,0,0,0.05), 0 4px 12px rgba(0,0,0,0.06)',
      position: 'relative',
      overflow: 'hidden',
    }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2px' }}>
        <span style={{ fontSize: '12px', fontWeight: 800, color: 'var(--cyan)', letterSpacing: '0.5px', textTransform: 'uppercase', display: 'flex', alignItems: 'center', gap: '6px' }}>
          <Gauge size={14} /> Tốc độ
        </span>
        <span style={{
          fontSize: '10px',
          fontWeight: 800,
          fontFamily: 'monospace',
          padding: '2px 7px',
          borderRadius: '4px',
          background: status === 'OFFLINE' ? 'rgba(100,116,139,0.25)' : status === 'CHARGING' ? 'rgba(245,158,11,0.2)' : (status === 'RUNNING' || status === 'ACTIVE') ? 'rgba(22,163,74,0.2)' : status === 'ERROR' ? 'rgba(239,68,68,0.2)' : 'rgba(2,132,199,0.2)',
          color: status === 'OFFLINE' ? '#94a3b8' : status === 'CHARGING' ? '#fbbf24' : (status === 'RUNNING' || status === 'ACTIVE') ? '#22c55e' : status === 'ERROR' ? '#ef4444' : '#38bdf8',
        }}>
          {status === 'OFFLINE' ? '⚪ OFFLINE' : status === 'CHARGING' ? '🔌 SẠC' : status === 'ERROR' ? '🚨 LỖI' : (status === 'RUNNING' || status === 'ACTIVE') ? '🟢 RUNNING' : '🔵 IDLE'}
        </span>
      </div>

      {/* SVG Symmetrical Speedometer Dial */}
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
        <svg viewBox="0 0 220 150" style={{ width: '100%', maxHeight: '150px' }}>
          <defs>
            <linearGradient id="speedGradMoto" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#10b981" />
              <stop offset="50%" stopColor="#f59e0b" />
              <stop offset="100%" stopColor="#ef4444" />
            </linearGradient>

            <filter id="needleGlowMoto" x="-50%" y="-50%" width="200%" height="200%">
              <feDropShadow dx="0" dy="0" stdDeviation="3" floodColor="#ef4444" floodOpacity="0.85" />
            </filter>

            <radialGradient id="dialBgMoto" cx="50%" cy="50%" r="50%">
              <stop offset="70%" stopColor={isDark ? "#090d16" : "#ffffff"} />
              <stop offset="100%" stopColor={isDark ? "#1e293b" : "#e2e8f0"} />
            </radialGradient>
          </defs>

          {/* Outer Chrome Bezel Ring */}
          <circle cx={cx} cy={cy} r={r + 14} fill="none" stroke={isDark ? "#1e293b" : "#cbd5e1"} strokeWidth="4" />
          <circle cx={cx} cy={cy} r={r + 11} fill="url(#dialBgMoto)" stroke={isDark ? "#0f172a" : "#e2e8f0"} strokeWidth="1" />

          {/* Background Track Arc (270 deg from -135 to +135) */}
          <path
            d={makeArcPath(cx, cy, r, -135, 135)}
            fill="none"
            stroke={isDark ? "rgba(255,255,255,0.12)" : "rgba(0,0,0,0.12)"}
            strokeWidth="7"
            strokeLinecap="round"
          />

          {/* Speed Zones */}
          <path
            d={makeArcPath(cx, cy, r, -135, -20)}
            fill="none"
            stroke="#10b981"
            strokeWidth="3"
            strokeOpacity="0.45"
          />
          <path
            d={makeArcPath(cx, cy, r, -20, 60)}
            fill="none"
            stroke="#f59e0b"
            strokeWidth="3"
            strokeOpacity="0.45"
          />
          <path
            d={makeArcPath(cx, cy, r, 60, 135)}
            fill="none"
            stroke="#ef4444"
            strokeWidth="3"
            strokeOpacity="0.75"
          />

          {/* Active Speed Arc */}
          {currentSpeed > 0.01 && (
            <path
              d={makeArcPath(cx, cy, r, -135, needleAngle)}
              fill="none"
              stroke="url(#speedGradMoto)"
              strokeWidth="7"
              strokeLinecap="round"
              style={{ filter: 'drop-shadow(0 0 6px rgba(239, 68, 68, 0.65))' }}
            />
          )}

          {/* Minor Ticks */}
          {minorTicks.map((val) => {
            const tickRatio = val / maxSpeed;
            const tickDeg = startAngle + tickRatio * totalSweep;
            const p1 = angleToCoord(cx, cy, r - 5, tickDeg);
            const p2 = angleToCoord(cx, cy, r + 2, tickDeg);
            return (
              <line
                key={`min-tick-${val}`}
                x1={p1.x}
                y1={p1.y}
                x2={p2.x}
                y2={p2.y}
                stroke={isDark ? "rgba(255,255,255,0.3)" : "rgba(0,0,0,0.3)"}
                strokeWidth="1.2"
              />
            );
          })}

          {/* Major Ticks & Speed Numbers */}
          {ticks.map((val) => {
            const tickRatio = val / maxSpeed;
            const tickDeg = startAngle + tickRatio * totalSweep;
            const p1 = angleToCoord(cx, cy, r - 9, tickDeg);
            const p2 = angleToCoord(cx, cy, r + 3, tickDeg);
            const pText = angleToCoord(cx, cy, r - 19, tickDeg);
            const isRedZone = val >= 1.5;

            return (
              <g key={`maj-tick-${val}`}>
                <line
                  x1={p1.x}
                  y1={p1.y}
                  x2={p2.x}
                  y2={p2.y}
                  stroke={isRedZone ? "#ef4444" : isDark ? "#ffffff" : "#0f172a"}
                  strokeWidth="2.5"
                  strokeLinecap="round"
                />
                <text
                  x={pText.x}
                  y={pText.y + 4}
                  textAnchor="middle"
                  fill={isRedZone ? "#ef4444" : isDark ? "#cbd5e1" : "#475569"}
                  fontSize="10"
                  fontWeight="800"
                  fontFamily="monospace"
                >
                  {val.toFixed(1)}
                </text>
              </g>
            );
          })}

          {/* Sweeping Needle (Gạt từ góc phần 3 bên trái sang phải đối xứng) */}
          <g
            style={{
              transformOrigin: `${cx}px ${cy}px`,
              transform: `rotate(${needleAngle}deg)`,
              transition: 'transform 0.25s cubic-bezier(0.34, 1.56, 0.64, 1)',
            }}
          >
            <line
              x1={cx}
              y1={cy}
              x2={cx}
              y2={cy - r + 3}
              stroke="#ef4444"
              strokeWidth="3.5"
              strokeLinecap="round"
              filter="url(#needleGlowMoto)"
            />
            <polygon
              points={`${cx - 3},${cy - r + 22} ${cx + 3},${cy - r + 22} ${cx},${cy - r + 2}`}
              fill="#ffffff"
            />
            <line
              x1={cx}
              y1={cy}
              x2={cx}
              y2={cy + 13}
              stroke="#94a3b8"
              strokeWidth="3"
              strokeLinecap="round"
            />
          </g>

          {/* Center Chrome Cap */}
          <circle cx={cx} cy={cy} r="9" fill={isDark ? "#0f172a" : "#334155"} stroke={isDark ? "#38bdf8" : "#0284c7"} strokeWidth="2.5" />
          <circle cx={cx} cy={cy} r="3.5" fill="#ef4444" />

          {/* Digital LCD Speedometer Readout */}
          <g transform={`translate(${cx}, ${cy + 34})`}>
            <rect
              x="-46"
              y="-14"
              width="92"
              height="28"
              rx="6"
              fill={isDark ? "#050811" : "#0f172a"}
              stroke={isDark ? "rgba(56, 189, 248, 0.4)" : "#334155"}
              strokeWidth="1.5"
            />
            <text x="-5" y="5" textAnchor="end" fill="#38bdf8" fontSize="16" fontWeight="900" fontFamily="monospace" style={{ letterSpacing: '0.5px' }}>
              {mps}
            </text>
            <text x="-2" y="3" textAnchor="start" fill="#94a3b8" fontSize="9" fontWeight="700" fontFamily="sans-serif">
              m/s
            </text>
            <text x="24" y="3" textAnchor="start" fill="#f59e0b" fontSize="9" fontWeight="800" fontFamily="monospace">
              {kmh} <tspan fontSize="7">km/h</tspan>
            </text>
          </g>
        </svg>
      </div>
    </div>
  );
};
const DEFAULT_LAYOUT_FRAME: LayoutFrame = {
  minX: 0.0,
  minZ: 0.0,
  width: 26.0,
  depth: 18.0,
  height: 4.0,
  centerX: 13.0,
  centerZ: 9.0,
  gridSize: 26.0,
  gridDivisions: 52,
};

const getLayoutFrame = (layout: FmsLayout | null): LayoutFrame => {
  const width = layout?.size?.width ?? 26.0;
  const depth = layout?.size?.depth ?? 18.0;
  const height = layout?.size?.height ?? 4.0;
  const centerX = width / 2;
  const centerZ = depth / 2;
  const gridDivisions = Math.round(width / 0.5);

  return {
    minX: 0.0,
    minZ: 0.0,
    width,
    depth,
    height,
    centerX,
    centerZ,
    gridSize: width,
    gridDivisions,
  };
};

const polygonToPointsAttr = (polygon: Point2[]) =>
  polygon.map(([x, z]) => `${x},${z}`).join(' ');

const getItemPoint = (item: FmsLayoutPointItem): Point2 | null => {
  if (item.access_point) return item.access_point;
  if (item.position) return [item.position[0], item.position[2]];
  return null;
};

const getSlamMapRect = (layout: FmsLayout | null, frame: LayoutFrame): [number, number, number, number] =>
  layout?.slam_map?.rect ?? [frame.minX, frame.minZ, frame.minX + frame.width, frame.minZ + frame.depth];

const normalizeListOrMap = <T extends { id: string }>(data: any): Record<string, T> => {
  if (!data) return {};
  if (Array.isArray(data)) {
    return Object.fromEntries(data.map((item: T) => [item.id, item]));
  }
  return data;
};

const UI_ENTITY_PRESERVE_MS = 6500;
const UI_POSE_JUMP_WINDOW_MS = 2500;

const distance3 = (a?: [number, number, number], b?: [number, number, number]) => {
  if (!a || !b) return 0;
  return Math.hypot(a[0] - b[0], a[2] - b[2]);
};

const mergeStableEntities = <T extends {
  id: string;
  position: [number, number, number];
  heading?: number;
  velocity?: number;
  max_speed?: number;
  ui_last_seen?: number;
}>(
  prev: Record<string, T>,
  incoming: Record<string, T>,
  nowMs: number,
  minJumpMeters = 2.5,
): Record<string, T> => {
  const next: Record<string, T> = {};

  Object.entries(prev).forEach(([id, item]) => {
    const lastSeen = item.ui_last_seen ?? 0;
    if (lastSeen && nowMs - lastSeen < UI_ENTITY_PRESERVE_MS) {
      next[id] = item;
    }
  });

  Object.entries(incoming).forEach(([id, item]) => {
    const prevItem = prev[id];
    const lastSeen = prevItem?.ui_last_seen ?? 0;
    const dtSec = lastSeen ? Math.max(0.05, (nowMs - lastSeen) / 1000) : 999;
    const dist = distance3(prevItem?.position, item.position);
    const maxSpeed = Math.max(prevItem?.max_speed ?? item.max_speed ?? 2.0, 0.5);
    const allowedJump = Math.max(minJumpMeters, maxSpeed * dtSec * 4.0 + 0.4);
    const shouldHoldPose = Boolean(prevItem && lastSeen && nowMs - lastSeen < UI_POSE_JUMP_WINDOW_MS && dist > allowedJump);
    
    // Clamp coordinates strictly inside physical SLAM room boundaries
    const rawPos = shouldHoldPose ? prevItem.position : item.position;
    const clampedPos: [number, number, number] = [
      Math.min(Math.max(6.0, rawPos[0]), 22.0),
      rawPos[1],
      Math.min(Math.max(9.8, rawPos[2]), 16.2),
    ];

    next[id] = {
      ...prevItem,
      ...item,
      position: clampedPos,
      heading: shouldHoldPose ? prevItem.heading : item.heading,
      velocity: shouldHoldPose ? 0 : item.velocity,
      ui_last_seen: nowMs,
    } as T;
  });

  return next;
};

// ─── Main Component ─────────────────────────────────────────────────────────
export default function RobotMap3DView() {
  const { t } = useLanguage();
  const { theme } = useAppTheme();
  const isDark = theme !== 'light';

  // Multi-Entity Telemetry States (Populated with full FMS fleet by default)
  const [robots, setRobots] = useState<Record<string, RobotData>>(INITIAL_ROBOTS);
  const [persons, setPersons] = useState<Record<string, PersonData>>(INITIAL_PERSONS);
  const [racks, setRacks] = useState<Record<string, RackData>>(INITIAL_RACKS);

  // Active Selection & Filter States (Default to Robot_2001 so speedometer is immediately alive)
  const [selectedEntity, setSelectedEntity] = useState<{ type: 'robot' | 'person' | 'rack'; id: string } | null>({ type: 'robot', id: 'Robot_2001' });
  const [fleetTab, setFleetTab] = useState<'robots' | 'persons' | 'racks'>('robots');
  const [viewMode, setViewMode] = useState<'3D' | '2D'>('3D');
  const [followTarget, setFollowTarget] = useState<boolean>(false);
  const [showLabels] = useState<boolean>(true);

  const [kpi, setKpi] = useState<FleetKpi>({
    total: 3,
    active: 2,
    charging: 1,
    idle: 0,
    warning: 0,
    error: 0,
    persons_count: 0,
    racks_count: 0,
  });

  const [fmsMeta, setFmsMeta] = useState<FmsMeta>({
    server_ip: '192.168.5.104',
    mqtt_connected: true,
    last_packet_time: 0,
    total_packets: 0,
    mode: 'LIVE',
  });

  const [layout, setLayout] = useState<FmsLayout | null>(null);
  const [packetRate, setPacketRate] = useState<number>(15);

  // References for Three.js
  const mountRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const layoutGroupRef = useRef<THREE.Group | null>(null);
  const entitiesGroupRef = useRef<THREE.Group | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const animFrameIdRef = useRef<number | null>(null);

  // Mesh registries for ultra-fast 60 FPS update with in-place canvas redraw
  const robotMeshesRef = useRef<Map<string, {
    group: THREE.Group;
    targetPos: THREE.Vector3;
    targetHeading: number;
    bodyMesh: THREE.Mesh;
    beaconMesh: THREE.Mesh;
    labelSprite: THREE.Sprite;
    labelCanvas: HTMLCanvasElement;
    labelCtx: CanvasRenderingContext2D;
    labelTexture: THREE.CanvasTexture;
    pulseRing: THREE.Mesh;
    rackMountGroup: THREE.Group;
    lastLabelKey?: string;
  }>>(new Map());

  const personMeshesRef = useRef<Map<string, {
    group: THREE.Group;
    targetPos: THREE.Vector3;
    targetHeading: number;
    labelSprite: THREE.Sprite;
    labelCanvas: HTMLCanvasElement;
    labelCtx: CanvasRenderingContext2D;
    labelTexture: THREE.CanvasTexture;
    haloRing: THREE.Mesh;
    lastLabelKey?: string;
  }>>(new Map());

  const rackMeshesRef = useRef<Map<string, {
    group: THREE.Group;
    targetPos: THREE.Vector3;
    labelSprite: THREE.Sprite;
    labelCanvas: HTMLCanvasElement;
    labelCtx: CanvasRenderingContext2D;
    labelTexture: THREE.CanvasTexture;
    lastLabelKey?: string;
  }>>(new Map());

  const wsRef = useRef<WebSocket | null>(null);
  const packetCountRef = useRef<number>(0);
  const selectedEntityRef = useRef<{ type: 'robot' | 'person' | 'rack'; id: string } | null>(null);
  const followTargetRef = useRef<boolean>(followTarget);

  useEffect(() => {
    selectedEntityRef.current = selectedEntity;
  }, [selectedEntity]);

  useEffect(() => {
    followTargetRef.current = followTarget;
  }, [followTarget]);

  const layoutFrame = useMemo(() => getLayoutFrame(layout), [layout]);
  const slamMapRect = useMemo(() => getSlamMapRect(layout, layoutFrame), [layout, layoutFrame]);
  const slamMapHref = useMemo(() => {
    const href = layout?.slam_map?.href ?? '/maps/fms_map.png';
    const cacheKey = [
      layout?.id ?? 'fms',
      layout?.slam_map?.map_name ?? 'map',
      layout?.slam_map?.pixel_size?.join('x') ?? '',
    ].join(':');
    return `${href}${href.includes('?') ? '&' : '?'}v=${encodeURIComponent(cacheKey)}`;
  }, [layout]);

  const slamWallsSvgPath = useMemo(() => {
    if (!layout?.slam_walls || layout.slam_walls.length === 0) return '';
    return layout.slam_walls.map(([p1, p2]) => `M ${p1[0]} ${p1[1]} L ${p2[0]} ${p2[1]}`).join(' ');
  }, [layout?.slam_walls]);

  const handleResetCamera = useCallback(() => {
    if (!cameraRef.current || !controlsRef.current) return;
    const { centerX, centerZ } = layoutFrame;
    cameraRef.current.position.set(centerX, 15.5, centerZ + 12.5);
    controlsRef.current.target.set(centerX, 0, centerZ);
    controlsRef.current.update();
    setFollowTarget(false);
  }, [layoutFrame]);

  // ─── FMS Layout Loader ────────────────────────────────────────────────────
  useEffect(() => {
    let cancelled = false;
    const loadLayout = async () => {
      const sources = ['/api/backend/fms/layout', '/maps/warehouse_layout.json'];
      for (const source of sources) {
        try {
          const res = await fetch(source, { cache: 'no-store' });
          if (!res.ok) continue;
          const data = await res.json();
          if (!cancelled && !data.error) {
            setLayout(data);
            return;
          }
        } catch (err) {
          console.warn(`[FMS Layout] Failed to load ${source}:`, err);
        }
      }
      if (!cancelled) setLayout(null);
    };
    loadLayout();
    return () => { cancelled = true; };
  }, []);

  // ─── WebSocket Connection for 3D Digital Twin (Robots, Persons, Racks) ────
  useEffect(() => {
    let ws: WebSocket | null = null;
    let reconnectTimeout: NodeJS.Timeout | null = null;
    let rateInterval: NodeJS.Timeout | null = null;

    const connect = () => {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const host = window.location.hostname || 'localhost';
      const wsUrl = `${protocol}//${host}:8000/ws/digital_twin`;

      ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        setFmsMeta(prev => ({ ...prev, mode: 'LIVE', mqtt_connected: true }));
      };

      ws.onmessage = (event) => {
        try {
          packetCountRef.current++;
          const msg = JSON.parse(event.data);
          const nowMs = Date.now();

          if (msg.type === 'DIGITAL_TWIN_SYNC' || msg.type === 'DIGITAL_TWIN_TELEMETRY') {
            if (msg.robots) {
              const incoming = normalizeListOrMap<RobotData>(msg.robots);
              setRobots(prev => mergeStableEntities(prev, incoming, nowMs, 2.5));
            }
            if (msg.persons) {
              const incoming = normalizeListOrMap<PersonData>(msg.persons);
              setPersons(prev => mergeStableEntities(prev, incoming, nowMs, 1.8));
            }
            if (msg.racks) {
              const incoming = normalizeListOrMap<RackData>(msg.racks);
              setRacks(prev => mergeStableEntities(prev, incoming, nowMs, 1.8));
            }
            if (msg.fms_meta) {
              setFmsMeta(msg.fms_meta);
            }
            if (msg.fleet_kpi) {
              setKpi(prev => ({
                ...prev,
                total: msg.fleet_kpi.total_robots ?? (Array.isArray(msg.robots) ? msg.robots.length : Object.keys(msg.robots || {}).length),
                active: msg.fleet_kpi.active_robots ?? 0,
                charging: msg.fleet_kpi.charging_robots ?? 0,
                idle: msg.fleet_kpi.idle_robots ?? 0,
                persons_count: msg.fleet_kpi.total_persons ?? (Array.isArray(msg.persons) ? msg.persons.length : Object.keys(msg.persons || {}).length),
                racks_count: msg.fleet_kpi.total_racks ?? (Array.isArray(msg.racks) ? msg.racks.length : Object.keys(msg.racks || {}).length),
              }));
            }
          } else if (msg.type === 'FULL' || msg.type === 'PATCH') {
            const data = msg.state || msg.patch || {};
            if (data.robots) {
              const incoming = normalizeListOrMap<RobotData>(data.robots);
              setRobots(prev => mergeStableEntities(prev, incoming, nowMs, 2.5));
            }
            if (data.persons) {
              const incoming = normalizeListOrMap<PersonData>(data.persons);
              setPersons(prev => mergeStableEntities(prev, incoming, nowMs, 1.8));
            }
            if (data.racks) {
              const incoming = normalizeListOrMap<RackData>(data.racks);
              setRacks(prev => mergeStableEntities(prev, incoming, nowMs, 1.8));
            }
            if (data.kpi?.fleet) setKpi(prev => ({ ...prev, ...data.kpi.fleet }));
            if (data.fms_meta) setFmsMeta(data.fms_meta);
          }
        } catch (e) {
          console.error('[DigitalTwin WS] Parse error:', e);
        }
      };

      ws.onerror = () => {};
      ws.onclose = () => {
        setFmsMeta(prev => ({ ...prev, mode: 'CONNECTING', mqtt_connected: false }));
        reconnectTimeout = setTimeout(connect, 2500);
      };

      wsRef.current = ws;
    };

    connect();

    rateInterval = setInterval(() => {
      setPacketRate(packetCountRef.current);
      packetCountRef.current = 0;
    }, 1000);

    return () => {
      if (ws) ws.close();
      if (reconnectTimeout) clearTimeout(reconnectTimeout);
      if (rateInterval) clearInterval(rateInterval);
    };
  }, []);

  // ─── Direct In-Place Canvas Redraw Helpers for 3D Labels (Zero Disposal, Zero Flicker) ───
  const updateRobotLabelCanvas = (
    canvas: HTMLCanvasElement,
    ctx: CanvasRenderingContext2D,
    id: string,
    battery: number,
    status: string,
    carriedRack?: string | null,
    crossStatus?: string,
    deltaM?: number | null
  ) => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const isDeviated = crossStatus === 'DEVIATED';
    const isChargingVer = crossStatus === 'CHARGING_VERIFIED';
    const isOffline = status === 'OFFLINE';

    ctx.fillStyle = 'rgba(15, 23, 42, 0.92)';
    ctx.strokeStyle = isOffline
      ? '#64748b'
      : isDeviated
      ? '#ef4444'
      : isChargingVer
      ? '#eab308'
      : status === 'CARRYING_RACK'
      ? '#a855f7'
      : status === 'CHARGING'
      ? '#f59e0b'
      : status === 'ERROR'
      ? '#ef4444'
      : (status === 'RUNNING' || status === 'ACTIVE')
      ? '#22c55e'
      : '#0284c7';
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.roundRect(6, 6, 288, 113, 14);
    ctx.fill();
    ctx.stroke();

    // Line 1: Robot ID + FMS Status
    ctx.fillStyle = isOffline ? '#94a3b8' : '#ffffff';
    ctx.font = 'bold 24px "Space Grotesk", sans-serif';
    ctx.fillText(`🤖 ${id}`, 18, 38);

    // Line 2: Battery + FMS Mode
    const batColor = isOffline ? '#94a3b8' : battery > 50 ? '#22c55e' : battery > 20 ? '#f59e0b' : '#ef4444';
    ctx.fillStyle = batColor;
    ctx.font = 'bold 18px monospace';
    ctx.fillText(`⚡ ${battery}%`, 18, 70);

    ctx.fillStyle = isOffline ? '#94a3b8' : carriedRack ? '#c084fc' : ((status === 'RUNNING' || status === 'ACTIVE') ? '#22c55e' : status === 'CHARGING' ? '#fbbf24' : status === 'ERROR' ? '#ef4444' : '#38bdf8');
    ctx.font = 'bold 15px "Space Grotesk", sans-serif';
    const fmsLabelText = isOffline ? '⚪ OFFLINE' : status === 'CHARGING' ? '🔌 ĐANG SẠC' : status === 'ERROR' ? '🚨 LỖI' : carriedRack ? `📦 ${carriedRack}` : (status === 'RUNNING' || status === 'ACTIVE') ? '🟢 RUNNING' : '🔵 IDLE';
    ctx.fillText(fmsLabelText, 115, 70);

    // Line 3: Vision AI Cross-check status
    ctx.fillStyle = isOffline ? '#94a3b8' : isDeviated ? '#f87171' : isChargingVer ? '#fef08a' : '#67e8f9';
    ctx.font = 'bold 14px "Space Grotesk", sans-serif';
    const crossLabel = isOffline
      ? '⚪ Mất kết nối FMS'
      : isChargingVer
      ? '🎯 Vision: Đúng trạm sạc'
      : isDeviated
      ? `⚠ Vision: Lệch ${deltaM ?? 0}m`
      : deltaM !== null && deltaM !== undefined
      ? `🎯 Vision: Khớp (${deltaM}m)`
      : '📡 Nguồn: FMS Only';
    ctx.fillText(crossLabel, 18, 102);
  };

  const createRobotLabelSprite = (
    id: string,
    battery: number,
    status: string,
    carriedRack?: string | null,
    crossStatus?: string,
    deltaM?: number | null
  ) => {
    const canvas = document.createElement('canvas');
    canvas.width = 280;
    canvas.height = 115;
    const ctx = canvas.getContext('2d')!;
    updateRobotLabelCanvas(canvas, ctx, id, battery, status, carriedRack, crossStatus, deltaM);

    const texture = new THREE.CanvasTexture(canvas);
    texture.minFilter = THREE.LinearFilter;
    texture.magFilter = THREE.LinearFilter;
    texture.needsUpdate = true;
    const mat = new THREE.SpriteMaterial({ map: texture, transparent: true, depthWrite: false });
    const sprite = new THREE.Sprite(mat);
    sprite.scale.set(1.9, 0.78, 1);
    sprite.position.set(0, 1.35, 0);
    return { sprite, canvas, ctx, texture };
  };

  const updatePersonLabelCanvas = (
    canvas: HTMLCanvasElement,
    ctx: CanvasRenderingContext2D,
    id: string,
    status: string,
    isAlert: boolean
  ) => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = isAlert ? 'rgba(239, 68, 68, 0.92)' : 'rgba(15, 23, 42, 0.90)';
    ctx.strokeStyle = isAlert ? '#f87171' : '#f97316';
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.roundRect(6, 6, 248, 78, 14);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 24px "Space Grotesk", sans-serif';
    ctx.fillText(`👤 ${id}`, 20, 42);

    ctx.fillStyle = isAlert ? '#fef08a' : '#fdba74';
    ctx.font = 'bold 18px "Space Grotesk", sans-serif';
    ctx.fillText(isAlert ? '⚠ CẢNH BÁO VA CHẠM' : `👷 ${status}`, 20, 70);
  };

  const createPersonLabelSprite = (id: string, status: string, isAlert: boolean) => {
    const canvas = document.createElement('canvas');
    canvas.width = 260;
    canvas.height = 90;
    const ctx = canvas.getContext('2d')!;
    updatePersonLabelCanvas(canvas, ctx, id, status, isAlert);

    const texture = new THREE.CanvasTexture(canvas);
    texture.minFilter = THREE.LinearFilter;
    texture.magFilter = THREE.LinearFilter;
    texture.needsUpdate = true;
    const mat = new THREE.SpriteMaterial({ map: texture, transparent: true, depthWrite: false });
    const sprite = new THREE.Sprite(mat);
    sprite.scale.set(1.7, 0.65, 1);
    sprite.position.set(0, 1.65, 0);
    return { sprite, canvas, ctx, texture };
  };

  const updateRackLabelCanvas = (
    canvas: HTMLCanvasElement,
    ctx: CanvasRenderingContext2D,
    id: string,
    status: string,
    carriedBy?: string | null
  ) => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = 'rgba(15, 23, 42, 0.90)';
    ctx.strokeStyle = status === 'CARRIED' ? '#c084fc' : '#64748b';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.roundRect(6, 6, 228, 68, 12);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 22px "Space Grotesk", sans-serif';
    ctx.fillText(`📦 ${id}`, 18, 38);

    ctx.fillStyle = status === 'CARRIED' ? '#c084fc' : '#94a3b8';
    ctx.font = 'bold 15px monospace';
    ctx.fillText(status === 'CARRIED' ? `Cõng bởi ${carriedBy || 'Robot'}` : 'Ô Lưu Trữ Tĩnh', 18, 62);
  };

  const createRackLabelSprite = (id: string, status: string, carriedBy?: string | null) => {
    const canvas = document.createElement('canvas');
    canvas.width = 240;
    canvas.height = 80;
    const ctx = canvas.getContext('2d')!;
    updateRackLabelCanvas(canvas, ctx, id, status, carriedBy);

    const texture = new THREE.CanvasTexture(canvas);
    texture.minFilter = THREE.LinearFilter;
    texture.magFilter = THREE.LinearFilter;
    texture.needsUpdate = true;
    const mat = new THREE.SpriteMaterial({ map: texture, transparent: true, depthWrite: false });
    const sprite = new THREE.Sprite(mat);
    sprite.scale.set(1.7, 0.60, 1);
    sprite.position.set(0, 0.72, 0);
    return { sprite, canvas, ctx, texture };
  };

  // ─── Three.js Scene Setup (Stable Lifecycle, Independent of Layout State) ──────────────────────
  useEffect(() => {
    if (!mountRef.current || viewMode !== '3D') return;

    const mountEl = mountRef.current;
    const width = mountEl.clientWidth || 800;
    const height = mountEl.clientHeight || 600;
    const {
      centerX,
      centerZ,
      gridSize,
      gridDivisions,
    } = layoutFrame;

    // 1. Scene & Camera
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(isDark ? '#090a0f' : '#f0f5fa');
    scene.fog = new THREE.Fog(isDark ? '#090a0f' : '#f0f5fa', gridSize * 1.5, gridSize * 4.5);
    sceneRef.current = scene;

    // Dedicated groups for layout and dynamic entities
    const layoutGroup = new THREE.Group();
    scene.add(layoutGroup);
    layoutGroupRef.current = layoutGroup;

    const entitiesGroup = new THREE.Group();
    scene.add(entitiesGroup);
    entitiesGroupRef.current = entitiesGroup;

    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    camera.position.set(centerX, 15.5, centerZ + 12.5);
    cameraRef.current = camera;

    // 2. Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    rendererRef.current = renderer;
    mountEl.appendChild(renderer.domElement);

    // 3. Controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.target.set(centerX, 0, centerZ);
    controls.maxPolarAngle = Math.PI / 2 - 0.04;
    controls.minDistance = 3;
    controls.maxDistance = gridSize * 3;
    controlsRef.current = controls;

    // 4. Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, isDark ? 1.6 : 2.0);
    scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, isDark ? 2.2 : 2.6);
    dirLight.position.set(centerX + 12, 24, centerZ + 12);
    dirLight.castShadow = true;
    dirLight.shadow.mapSize.width = 2048;
    dirLight.shadow.mapSize.height = 2048;
    scene.add(dirLight);

    // 5. Grid Helper (Lưới ô vuông chuẩn xác 0.5m / ô)
    const gridDiv = Math.max(52, gridDivisions);
    const gridHelper = new THREE.GridHelper(
      gridSize,
      gridDiv,
      isDark ? 0x0284c7 : 0x38bdf8,
      isDark ? 0x1e293b : 0xdbeafe
    );
    gridHelper.position.set(centerX, 0.001, centerZ);
    scene.add(gridHelper);

    // 5B. Central Coordinate Axes
    const axisMat = new THREE.LineBasicMaterial({
      color: isDark ? 0x06b6d4 : 0x0284c7,
      transparent: true,
      opacity: 0.6,
    });
    const axisXGeo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(centerX - gridSize * 0.75, 0.003, centerZ),
      new THREE.Vector3(centerX + gridSize * 0.75, 0.003, centerZ),
    ]);
    const axisZGeo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(centerX, 0.003, centerZ - gridSize * 0.4),
      new THREE.Vector3(centerX, 0.003, centerZ + gridSize * 0.4),
    ]);
    scene.add(new THREE.Line(axisXGeo, axisMat));
    scene.add(new THREE.Line(axisZGeo, axisMat));

    // 6. Raycaster for Interactive Entity Selection
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();

    const onPointerDown = (event: MouseEvent) => {
      const rect = renderer.domElement.getBoundingClientRect();
      mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

      raycaster.setFromCamera(mouse, camera);

      const interactiveObjects: THREE.Object3D[] = [];
      robotMeshesRef.current.forEach((m) => interactiveObjects.push(m.group));
      personMeshesRef.current.forEach((m) => interactiveObjects.push(m.group));
      rackMeshesRef.current.forEach((m) => interactiveObjects.push(m.group));

      const intersects = raycaster.intersectObjects(interactiveObjects, true);
      if (intersects.length > 0) {
        let current: THREE.Object3D | null = intersects[0].object;
        while (current && !current.userData.robotId && !current.userData.personId && !current.userData.rackId) {
          current = current.parent;
        }
        if (current) {
          if (current.userData.robotId) {
            setSelectedEntity({ type: 'robot', id: current.userData.robotId });
            setFleetTab('robots');
          } else if (current.userData.personId) {
            setSelectedEntity({ type: 'person', id: current.userData.personId });
            setFleetTab('persons');
          } else if (current.userData.rackId) {
            setSelectedEntity({ type: 'rack', id: current.userData.rackId });
            setFleetTab('racks');
          }
        }
      }
    };

    renderer.domElement.addEventListener('pointerdown', onPointerDown);

    // 7. Resize Observer
    const handleResize = () => {
      if (!rendererRef.current || !cameraRef.current) return;
      const w = mountEl.clientWidth;
      const h = mountEl.clientHeight;
      if (w > 10 && h > 10) {
        cameraRef.current.aspect = w / h;
        cameraRef.current.updateProjectionMatrix();
        rendererRef.current.setSize(w, h);
      }
    };
    const resizeObserver = new ResizeObserver(handleResize);
    resizeObserver.observe(mountEl);

    // 8. 60 FPS LERP Smoothing Animation Loop
    const animate = () => {
      animFrameIdRef.current = requestAnimationFrame(animate);
      controls.update();

      // A. Robots Animation
      robotMeshesRef.current.forEach((meshData, rid) => {
        const { group, targetPos, targetHeading, pulseRing } = meshData;
        group.position.x += (targetPos.x - group.position.x) * 0.18;
        group.position.z += (targetPos.z - group.position.z) * 0.18;

        let deltaHeading = targetHeading - group.rotation.y;
        while (deltaHeading > Math.PI) deltaHeading -= 2 * Math.PI;
        while (deltaHeading < -Math.PI) deltaHeading += 2 * Math.PI;
        group.rotation.y += deltaHeading * 0.18;

        const isSelected = selectedEntityRef.current?.type === 'robot' && selectedEntityRef.current?.id === rid;
        pulseRing.visible = isSelected;
        if (isSelected) {
          const s = 1 + Math.sin(Date.now() * 0.006) * 0.18;
          pulseRing.scale.set(s, s, s);

          if (followTargetRef.current && controlsRef.current) {
            controlsRef.current.target.x += (group.position.x - controlsRef.current.target.x) * 0.06;
            controlsRef.current.target.z += (group.position.z - controlsRef.current.target.z) * 0.06;
          }
        }
      });

      // B. Persons Animation
      personMeshesRef.current.forEach((meshData, pid) => {
        const { group, targetPos, targetHeading } = meshData;
        group.position.x += (targetPos.x - group.position.x) * 0.20;
        group.position.z += (targetPos.z - group.position.z) * 0.20;

        let deltaHeading = targetHeading - group.rotation.y;
        while (deltaHeading > Math.PI) deltaHeading -= 2 * Math.PI;
        while (deltaHeading < -Math.PI) deltaHeading += 2 * Math.PI;
        group.rotation.y += deltaHeading * 0.20;

        const isMoving = Math.hypot(targetPos.x - group.position.x, targetPos.z - group.position.z) > 0.05;
        if (isMoving) {
          group.position.y = Math.abs(Math.sin(Date.now() * 0.008)) * 0.06;
        } else {
          group.position.y = 0.0;
        }

        const isSelected = selectedEntityRef.current?.type === 'person' && selectedEntityRef.current?.id === pid;
        if (isSelected && followTargetRef.current && controlsRef.current) {
          controlsRef.current.target.x += (group.position.x - controlsRef.current.target.x) * 0.06;
          controlsRef.current.target.z += (group.position.z - controlsRef.current.target.z) * 0.06;
        }
      });

      // C. Racks Animation
      rackMeshesRef.current.forEach((meshData) => {
        const { group, targetPos } = meshData;
        group.position.x += (targetPos.x - group.position.x) * 0.18;
        group.position.y += (targetPos.y - group.position.y) * 0.18;
        group.position.z += (targetPos.z - group.position.z) * 0.18;
      });

      renderer.render(scene, camera);
    };

    animate();

    return () => {
      if (animFrameIdRef.current) cancelAnimationFrame(animFrameIdRef.current);
      resizeObserver.disconnect();
      renderer.domElement.removeEventListener('pointerdown', onPointerDown);
      if (renderer.domElement.parentElement === mountEl) {
        mountEl.removeChild(renderer.domElement);
      }
      renderer.dispose();
      robotMeshesRef.current.clear();
      personMeshesRef.current.clear();
      rackMeshesRef.current.clear();
      layoutGroupRef.current = null;
      entitiesGroupRef.current = null;
    };
  }, [viewMode, isDark]);

  // ─── Build / Update Layout Layer (SLAM Floor, Walkways, Waypoints, Stations, Charging Stations, SLAM Walls) ───
  useEffect(() => {
    if (viewMode !== '3D' || !layoutGroupRef.current) return;
    const layoutGroup = layoutGroupRef.current;

    // Clear previous layout children cleanly
    while (layoutGroup.children.length > 0) {
      const child = layoutGroup.children[0];
      layoutGroup.remove(child);
      if ((child as THREE.Mesh).geometry) {
        (child as THREE.Mesh).geometry.dispose();
      }
    }

    const [slamX1, slamZ1, slamX2, slamZ2] = slamMapRect;

    // A. SLAM LiDAR Floor Texture Overlay (Khớp chuẩn 1:1 với tọa độ SLAM)
    const createHorizontalPlaneGeometry = (x1: number, z1: number, x2: number, z2: number, y = 0.004) => {
      const geometry = new THREE.BufferGeometry();
      const vertices = new Float32Array([
        x1, y, z1,
        x2, y, z1,
        x1, y, z2,
        x2, y, z2,
      ]);
      const uvs = new Float32Array([
        0, 1,
        1, 1,
        0, 0,
        1, 0,
      ]);
      geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
      geometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
      geometry.setIndex([0, 2, 1, 2, 3, 1]);
      geometry.computeVertexNormals();
      return geometry;
    };

    const textureLoader = new THREE.TextureLoader();
    textureLoader.load(slamMapHref, (tex) => {
      tex.colorSpace = THREE.SRGBColorSpace;
      tex.flipY = true;
      tex.anisotropy = 8;
      tex.needsUpdate = true;
      const overlayGeo = createHorizontalPlaneGeometry(slamX1, slamZ1, slamX2, slamZ2, 0.004);
      const overlayMat = new THREE.MeshBasicMaterial({
        map: tex,
        transparent: true,
        opacity: isDark ? 0.35 : 0.25,
        depthWrite: false,
        side: THREE.DoubleSide,
      });
      const overlayMesh = new THREE.Mesh(overlayGeo, overlayMat);
      layoutGroup.add(overlayMesh);
    });

    // B. Smooth Continuous 3D Walkways (Mạng lưới đường chạy FMS mượt mà)
    const trackColor = isDark ? 0x1e293b : 0xe2e8f0;
    const trackMat = new THREE.MeshStandardMaterial({
      color: trackColor,
      roughness: 0.4,
      metalness: 0.15,
      side: THREE.DoubleSide,
    });

    (layout?.walkways ?? []).forEach((w) => {
      if (!w.polygon || w.polygon.length < 3) return;
      const shape = new THREE.Shape();
      shape.moveTo(w.polygon[0][0], w.polygon[0][1]);
      w.polygon.slice(1).forEach(([x, z]) => shape.lineTo(x, z));
      shape.closePath();
      const geom = new THREE.ShapeGeometry(shape);
      const mesh = new THREE.Mesh(geom, trackMat);
      mesh.rotation.x = Math.PI / 2;
      mesh.position.y = 0.006;
      mesh.receiveShadow = true;
      layoutGroup.add(mesh);
    });

    // C. FMS Waypoints 3D (Các điểm nút tròn nhỏ gọn, cân đối đúng chuẩn FMS)
    (layout?.locations ?? []).forEach((loc) => {
      const pt = getItemPoint(loc);
      if (!pt) return;
      const isChargingPt = loc.kind === 'CHARGING' || loc.id.toLowerCase().includes('charge') || loc.id === 'AutoXing';
      const isStationEnd = loc.id.startsWith('19') && (pt[1] > 14.5 || loc.id.includes('AXP'));
      
      const dotGeo = new THREE.CylinderGeometry(0.07, 0.07, 0.012, 24);
      const dotMat = new THREE.MeshStandardMaterial({
        color: isChargingPt ? 0xf59e0b : isStationEnd ? 0x0284c7 : 0x16a34a,
        emissive: isChargingPt ? 0xd97706 : isStationEnd ? 0x0369a1 : 0x15803d,
        emissiveIntensity: 0.6,
        roughness: 0.3
      });
      const dot = new THREE.Mesh(dotGeo, dotMat);
      dot.position.set(pt[0], 0.010, pt[1]);
      layoutGroup.add(dot);

      // White center inner dot
      const centerDotGeo = new THREE.CylinderGeometry(0.03, 0.03, 0.014, 16);
      const centerDot = new THREE.Mesh(centerDotGeo, new THREE.MeshBasicMaterial({ color: 0xffffff }));
      centerDot.position.set(pt[0], 0.011, pt[1]);
      layoutGroup.add(centerDot);
    });

    // D. Work Stations 3D (Trạm làm việc xanh lá FMS)
    (layout?.stations ?? []).forEach((st) => {
      const w = st.rect[2] - st.rect[0];
      const d = st.rect[3] - st.rect[1];
      const cx = (st.rect[0] + st.rect[2]) / 2;
      const cz = (st.rect[1] + st.rect[3]) / 2;
      const stMesh = new THREE.Mesh(
        new THREE.BoxGeometry(w, 0.02, d),
        new THREE.MeshStandardMaterial({
          color: 0x10b981,
          emissive: 0x059669,
          emissiveIntensity: 0.2,
          transparent: true,
          opacity: 0.55,
          roughness: 0.4
        })
      );
      stMesh.position.set(cx, 0.014, cz);
      layoutGroup.add(stMesh);
    });

    // E. Dynamic Real Charging Stations from FMS layout
    (layout?.charging_stations ?? []).forEach((cs) => {
      const csGroup = new THREE.Group();
      const posX = cs.position?.[0] ?? cs.access_point?.[0] ?? 0;
      const posZ = cs.position?.[2] ?? cs.access_point?.[1] ?? 0;
      csGroup.position.set(posX, 0, posZ);

      // Floor metallic charging base plate
      const basePad = new THREE.Mesh(
        new THREE.CylinderGeometry(0.38, 0.40, 0.02, 32),
        new THREE.MeshStandardMaterial({
          color: 0x0f172a,
          roughness: 0.35,
          metalness: 0.8,
        })
      );
      basePad.position.y = 0.01;
      csGroup.add(basePad);

      // Amber glowing charging contact ring
      const chargingRing = new THREE.Mesh(
        new THREE.RingGeometry(0.14, 0.26, 32),
        new THREE.MeshStandardMaterial({
          color: 0xf59e0b,
          emissive: 0xd97706,
          emissiveIntensity: 0.9,
          side: THREE.DoubleSide
        })
      );
      chargingRing.rotation.x = -Math.PI / 2;
      chargingRing.position.y = 0.021;
      csGroup.add(chargingRing);

      // Industrial Charging Pillar
      const pillar = new THREE.Mesh(
        new THREE.BoxGeometry(0.20, 0.65, 0.14),
        new THREE.MeshStandardMaterial({ color: 0x1e293b, roughness: 0.3, metalness: 0.7 })
      );
      pillar.position.set(0, 0.33, -0.22);
      csGroup.add(pillar);

      // Pillar Top Screen / Status Indicator
      const screen = new THREE.Mesh(
        new THREE.BoxGeometry(0.16, 0.12, 0.02),
        new THREE.MeshStandardMaterial({
          color: 0x38bdf8,
          emissive: 0x0284c7,
          emissiveIntensity: 1.2
        })
      );
      screen.position.set(0, 0.52, -0.14);
      csGroup.add(screen);

      // Floating 3D Text Badge with Station Name (e.g. "⚡ AutoXing" / "⚡ 199953TT202049")
      const canvas = document.createElement('canvas');
      canvas.width = 320;
      canvas.height = 96;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.fillStyle = 'rgba(15, 23, 42, 0.90)';
        ctx.beginPath();
        ctx.roundRect(4, 4, 312, 88, 16);
        ctx.fill();
        ctx.lineWidth = 4;
        ctx.strokeStyle = '#f59e0b';
        ctx.stroke();

        ctx.font = 'bold 28px "Space Grotesk", sans-serif';
        ctx.fillStyle = '#fbbf24';
        ctx.textAlign = 'center';
        ctx.fillText(`⚡ ${cs.id}`, 160, 56);
      }
      const badgeTex = new THREE.CanvasTexture(canvas);
      badgeTex.needsUpdate = true;
      const badgeMat = new THREE.SpriteMaterial({ map: badgeTex, transparent: true, depthWrite: false });
      const badgeSprite = new THREE.Sprite(badgeMat);
      badgeSprite.position.set(0, 0.95, -0.1);
      badgeSprite.scale.set(0.95, 0.28, 1);
      csGroup.add(badgeSprite);

      layoutGroup.add(csGroup);
    });

    // F. Solid 3D Walls Extruded on ALL SLAM LiDAR Scanned Walls
    const slamWallSegs = (layout?.slam_walls ?? []).filter(([p1, p2]) => {
      const dx = p2[0] - p1[0];
      const dz = p2[1] - p1[1];
      return Math.hypot(dx, dz) > 0.002;
    });

    if (slamWallSegs.length > 0) {
      const wallHeight = 1.25;
      const wallThickness = 0.14; // Perfectly covers all underlying SLAM LiDAR map marks
      const halfThick = wallThickness / 2;

      const segCount = slamWallSegs.length;
      const vertexCount = segCount * 8;
      const faceCount = segCount * 12;

      const positions = new Float32Array(vertexCount * 3);
      const indices = new Uint32Array(faceCount * 3);
      const topEdgePositions = new Float32Array(segCount * 12);
      const baseEdgePositions = new Float32Array(segCount * 12);

      let vIdx = 0;
      let iIdx = 0;
      let topEdgeIdx = 0;
      let baseEdgeIdx = 0;

      slamWallSegs.forEach((seg) => {
        const [p1, p2] = seg;
        const x1 = p1[0], z1 = p1[1];
        const x2 = p2[0], z2 = p2[1];
        const dx = x2 - x1;
        const dz = z2 - z1;
        const len = Math.hypot(dx, dz);
        if (len < 0.002) return;
        const nx = (-dz / len) * halfThick;
        const nz = (dx / len) * halfThick;

        const baseV = vIdx / 3;

        // Base 4 vertices (y = 0)
        positions[vIdx++] = x1 + nx; positions[vIdx++] = 0.0; positions[vIdx++] = z1 + nz; // 0
        positions[vIdx++] = x1 - nx; positions[vIdx++] = 0.0; positions[vIdx++] = z1 - nz; // 1
        positions[vIdx++] = x2 - nx; positions[vIdx++] = 0.0; positions[vIdx++] = z2 - nz; // 2
        positions[vIdx++] = x2 + nx; positions[vIdx++] = 0.0; positions[vIdx++] = z2 + nz; // 3

        // Top 4 vertices (y = H)
        positions[vIdx++] = x1 + nx; positions[vIdx++] = wallHeight; positions[vIdx++] = z1 + nz; // 4
        positions[vIdx++] = x1 - nx; positions[vIdx++] = wallHeight; positions[vIdx++] = z1 - nz; // 5
        positions[vIdx++] = x2 - nx; positions[vIdx++] = wallHeight; positions[vIdx++] = z2 - nz; // 6
        positions[vIdx++] = x2 + nx; positions[vIdx++] = wallHeight; positions[vIdx++] = z2 + nz; // 7

        // 12 Triangles (6 Faces)
        indices[iIdx++] = baseV + 0; indices[iIdx++] = baseV + 3; indices[iIdx++] = baseV + 7;
        indices[iIdx++] = baseV + 0; indices[iIdx++] = baseV + 7; indices[iIdx++] = baseV + 4;

        indices[iIdx++] = baseV + 1; indices[iIdx++] = baseV + 5; indices[iIdx++] = baseV + 6;
        indices[iIdx++] = baseV + 1; indices[iIdx++] = baseV + 6; indices[iIdx++] = baseV + 2;

        indices[iIdx++] = baseV + 4; indices[iIdx++] = baseV + 7; indices[iIdx++] = baseV + 6;
        indices[iIdx++] = baseV + 4; indices[iIdx++] = baseV + 6; indices[iIdx++] = baseV + 5;

        indices[iIdx++] = baseV + 0; indices[iIdx++] = baseV + 1; indices[iIdx++] = baseV + 2;
        indices[iIdx++] = baseV + 0; indices[iIdx++] = baseV + 2; indices[iIdx++] = baseV + 3;

        indices[iIdx++] = baseV + 0; indices[iIdx++] = baseV + 4; indices[iIdx++] = baseV + 5;
        indices[iIdx++] = baseV + 0; indices[iIdx++] = baseV + 5; indices[iIdx++] = baseV + 1;

        indices[iIdx++] = baseV + 3; indices[iIdx++] = baseV + 2; indices[iIdx++] = baseV + 6;
        indices[iIdx++] = baseV + 3; indices[iIdx++] = baseV + 6; indices[iIdx++] = baseV + 7;

        // Top edge lines
        topEdgePositions[topEdgeIdx++] = x1 + nx; topEdgePositions[topEdgeIdx++] = wallHeight; topEdgePositions[topEdgeIdx++] = z1 + nz;
        topEdgePositions[topEdgeIdx++] = x2 + nx; topEdgePositions[topEdgeIdx++] = wallHeight; topEdgePositions[topEdgeIdx++] = z2 + nz;
        topEdgePositions[topEdgeIdx++] = x1 - nx; topEdgePositions[topEdgeIdx++] = wallHeight; topEdgePositions[topEdgeIdx++] = z1 - nz;
        topEdgePositions[topEdgeIdx++] = x2 - nx; topEdgePositions[topEdgeIdx++] = wallHeight; topEdgePositions[topEdgeIdx++] = z2 - nz;

        // Base edge lines
        baseEdgePositions[baseEdgeIdx++] = x1 + nx; baseEdgePositions[baseEdgeIdx++] = 0.01; baseEdgePositions[baseEdgeIdx++] = z1 + nz;
        baseEdgePositions[baseEdgeIdx++] = x2 + nx; baseEdgePositions[baseEdgeIdx++] = 0.01; baseEdgePositions[baseEdgeIdx++] = z2 + nz;
        baseEdgePositions[baseEdgeIdx++] = x1 - nx; baseEdgePositions[baseEdgeIdx++] = 0.01; baseEdgePositions[baseEdgeIdx++] = z1 - nz;
        baseEdgePositions[baseEdgeIdx++] = x2 - nx; baseEdgePositions[baseEdgeIdx++] = 0.01; baseEdgePositions[baseEdgeIdx++] = z2 - nz;
      });

      const wallGeo = new THREE.BufferGeometry();
      wallGeo.setAttribute('position', new THREE.BufferAttribute(positions.slice(0, vIdx), 3));
      wallGeo.setIndex(new THREE.BufferAttribute(indices.slice(0, iIdx), 1));
      wallGeo.computeVertexNormals();

      const wallMat = new THREE.MeshStandardMaterial({
        color: isDark ? 0x1e3a8a : 0x2563eb,
        emissive: isDark ? 0x172554 : 0x1d4ed8,
        emissiveIntensity: 0.4,
        roughness: 0.3,
        metalness: 0.4,
        side: THREE.DoubleSide,
      });
      const wallMesh = new THREE.Mesh(wallGeo, wallMat);
      wallMesh.castShadow = true;
      wallMesh.receiveShadow = true;
      layoutGroup.add(wallMesh);

      // Top glowing rim line
      const topGeo = new THREE.BufferGeometry();
      topGeo.setAttribute('position', new THREE.BufferAttribute(topEdgePositions.slice(0, topEdgeIdx), 3));
      const topMat = new THREE.LineBasicMaterial({
        color: isDark ? 0x60a5fa : 0x3b82f6,
        transparent: true,
        opacity: 0.9,
      });
      layoutGroup.add(new THREE.LineSegments(topGeo, topMat));

      // Base glowing line
      const baseGeo = new THREE.BufferGeometry();
      baseGeo.setAttribute('position', new THREE.BufferAttribute(baseEdgePositions.slice(0, baseEdgeIdx), 3));
      const baseMat = new THREE.LineBasicMaterial({
        color: isDark ? 0x38bdf8 : 0x1d4ed8,
        transparent: true,
        opacity: 0.5,
      });
      layoutGroup.add(new THREE.LineSegments(baseGeo, baseMat));
    }

    // G. Ultra-Realistic 3D Wall-Mounted CCTV Cameras with Strict Blue-Wall Interior FOV Frustums
    const rawCameras = (layout?.cameras && layout.cameras.length > 0) ? layout.cameras : DEFAULT_CAMERAS;

    // Strict physical interior boundaries of the blue SLAM room
    const ROOM_X_MIN = 6.30;
    const ROOM_X_MAX = 21.80;
    const ROOM_Z_MIN = 6.92;
    const ROOM_Z_MAX = 15.38;
    const FLOOR_Y = 0.02;

    rawCameras.forEach((cam, camIdx) => {
      const isCam1 = cam.id.toLowerCase().includes('1') || camIdx === 0;
      const camX = cam.position[0];
      const camY = cam.position[1] ?? 1.15;
      const camZ = cam.position[2];
      const lookAtX = cam.look_at ? cam.look_at[0] : (isCam1 ? 10.5 : 15.2);
      const lookAtY = cam.look_at ? cam.look_at[1] : 0.0;
      const lookAtZ = cam.look_at ? cam.look_at[2] : (isCam1 ? 11.8 : 12.0);
      const fovDeg = cam.fov_deg ?? (isCam1 ? 76 : 78);
      const rangeM = cam.range_m ?? (isCam1 ? 8.8 : 8.2);
      const themeColor = isCam1 ? 0x22d3ee : 0x818cf8; // Cyan for Cam 1, Indigo for Cam 4
      const hexColorStr = isCam1 ? '#22d3ee' : '#818cf8';

      const camGroup = new THREE.Group();
      camGroup.position.set(camX, camY, camZ);

      // 1. Industrial Wall Mounting Backplate (Gắn trực tiếp trên bề mặt tường xanh)
      const wallPlateGeo = new THREE.BoxGeometry(0.18, 0.18, 0.04);
      const wallPlateMat = new THREE.MeshStandardMaterial({
        color: 0x1e293b,
        metalness: 0.85,
        roughness: 0.25,
      });
      const wallPlate = new THREE.Mesh(wallPlateGeo, wallPlateMat);
      wallPlate.position.set(0, 0, isCam1 ? -0.02 : 0.02);
      camGroup.add(wallPlate);

      // 2. Heavy-Duty Cantilever Wall Extension Arm (Vươn ngang từ tường)
      const armLength = 0.18;
      const armGeo = new THREE.CylinderGeometry(0.016, 0.016, armLength, 12);
      const armMat = new THREE.MeshStandardMaterial({
        color: 0x475569,
        metalness: 0.9,
        roughness: 0.2,
      });
      const arm = new THREE.Mesh(armGeo, armMat);
      arm.rotation.x = Math.PI / 2;
      arm.position.set(0, 0, isCam1 ? armLength / 2 : -armLength / 2);
      camGroup.add(arm);

      // 3. Articulated Swivel Joint & Aiming Head
      const swivelGeo = new THREE.SphereGeometry(0.032, 12, 12);
      const swivelMat = new THREE.MeshStandardMaterial({ color: 0x334155, metalness: 0.8, roughness: 0.3 });
      const swivel = new THREE.Mesh(swivelGeo, swivelMat);
      const swivelZ = isCam1 ? armLength : -armLength;
      swivel.position.set(0, 0, swivelZ);
      camGroup.add(swivel);

      // 4. Camera Aiming Group
      const aimGroup = new THREE.Group();
      aimGroup.position.set(0, 0, swivelZ);
      const lookTarget = new THREE.Vector3(lookAtX, lookAtY, lookAtZ);
      const camWorldPos = new THREE.Vector3(camX, camY, camZ + swivelZ);
      const aimDir = lookTarget.clone().sub(camWorldPos).normalize();
      aimGroup.lookAt(aimDir);

      // 5. Industrial Bullet Camera Housing
      const bodyGeo = new THREE.CylinderGeometry(0.052, 0.065, 0.22, 16);
      const bodyMat = new THREE.MeshStandardMaterial({
        color: isDark ? 0xe2e8f0 : 0xf8fafc,
        metalness: 0.5,
        roughness: 0.2,
      });
      const body = new THREE.Mesh(bodyGeo, bodyMat);
      body.rotation.x = Math.PI / 2;
      body.position.set(0, 0, 0.10);
      aimGroup.add(body);

      // 6. Protective Sunshield Canopy
      const shieldGeo = new THREE.CylinderGeometry(0.072, 0.072, 0.16, 16, 1, true, 0, Math.PI);
      const shieldMat = new THREE.MeshStandardMaterial({
        color: isDark ? 0x0f172a : 0x334155,
        metalness: 0.7,
        roughness: 0.3,
        side: THREE.DoubleSide,
      });
      const shield = new THREE.Mesh(shieldGeo, shieldMat);
      shield.rotation.x = Math.PI / 2;
      shield.rotation.z = Math.PI;
      shield.position.set(0, 0.02, 0.12);
      aimGroup.add(shield);

      // 7. Lens Rim & Optical Tinted Glass
      const lensRimGeo = new THREE.CylinderGeometry(0.050, 0.050, 0.02, 16);
      const lensRimMat = new THREE.MeshStandardMaterial({ color: 0x0f172a, metalness: 0.9, roughness: 0.1 });
      const lensRim = new THREE.Mesh(lensRimGeo, lensRimMat);
      lensRim.rotation.x = Math.PI / 2;
      lensRim.position.set(0, 0, 0.21);
      aimGroup.add(lensRim);

      const lensGlassGeo = new THREE.CircleGeometry(0.038, 16);
      const lensGlassMat = new THREE.MeshBasicMaterial({ color: 0x0284c7, side: THREE.DoubleSide });
      const lensGlass = new THREE.Mesh(lensGlassGeo, lensGlassMat);
      lensGlass.position.set(0, 0, 0.221);
      aimGroup.add(lensGlass);

      // 8. Active Status LED
      const ledGeo = new THREE.SphereGeometry(0.012, 8, 8);
      const ledMat = new THREE.MeshBasicMaterial({ color: 0x10b981 });
      const led = new THREE.Mesh(ledGeo, ledMat);
      led.position.set(0.032, 0.032, 0.21);
      aimGroup.add(led);

      camGroup.add(aimGroup);
      layoutGroup.add(camGroup);

      // 9. Strict Blue-Wall Interior Ray & Volumetric FOV Clipping
      const forwardDir = aimDir.clone();
      const worldUp = new THREE.Vector3(0, 1, 0);
      const rightDir = new THREE.Vector3().crossVectors(forwardDir, worldUp).normalize();
      const upDir = new THREE.Vector3().crossVectors(rightDir, forwardDir).normalize();

      const halfHRad = (fovDeg * Math.PI) / 360.0;
      const halfW = rangeM * Math.tan(halfHRad);
      const halfH = halfW * (9 / 16);

      const farCenter = camWorldPos.clone().addScaledVector(forwardDir, rangeM);
      const rawC1 = farCenter.clone().addScaledVector(rightDir, -halfW).addScaledVector(upDir, halfH);
      const rawC2 = farCenter.clone().addScaledVector(rightDir, halfW).addScaledVector(upDir, halfH);
      const rawC3 = farCenter.clone().addScaledVector(rightDir, halfW).addScaledVector(upDir, -halfH);
      const rawC4 = farCenter.clone().addScaledVector(rightDir, -halfW).addScaledVector(upDir, -halfH);

      // Ray-Box Intersect: Clips any ray strictly against the blue SLAM wall boundary
      const clipRayInsideWalls = (origin: THREE.Vector3, target: THREE.Vector3): THREE.Vector3 => {
        const dx = target.x - origin.x;
        const dy = target.y - origin.y;
        const dz = target.z - origin.z;
        let tMax = 1.0;

        if (dx < -1e-5 && origin.x + dx < ROOM_X_MIN) tMax = Math.min(tMax, (ROOM_X_MIN - origin.x) / dx);
        if (dx > 1e-5 && origin.x + dx > ROOM_X_MAX) tMax = Math.min(tMax, (ROOM_X_MAX - origin.x) / dx);
        if (dz < -1e-5 && origin.z + dz < ROOM_Z_MIN) tMax = Math.min(tMax, (ROOM_Z_MIN - origin.z) / dz);
        if (dz > 1e-5 && origin.z + dz > ROOM_Z_MAX) tMax = Math.min(tMax, (ROOM_Z_MAX - origin.z) / dz);
        if (dy < -1e-5 && origin.y + dy < FLOOR_Y) tMax = Math.min(tMax, (FLOOR_Y - origin.y) / dy);

        tMax = Math.max(0.05, Math.min(1.0, tMax));
        return new THREE.Vector3(
          Math.max(ROOM_X_MIN, Math.min(ROOM_X_MAX, origin.x + tMax * dx)),
          Math.max(FLOOR_Y, origin.y + tMax * dy),
          Math.max(ROOM_Z_MIN, Math.min(ROOM_Z_MAX, origin.z + tMax * dz))
        );
      };

      const c1 = clipRayInsideWalls(camWorldPos, rawC1);
      const c2 = clipRayInsideWalls(camWorldPos, rawC2);
      const c3 = clipRayInsideWalls(camWorldPos, rawC3);
      const c4 = clipRayInsideWalls(camWorldPos, rawC4);

      // Floor Footprint Points (strictly inside blue walls)
      const g1 = new THREE.Vector3(c1.x, FLOOR_Y, c1.z);
      const g2 = new THREE.Vector3(c2.x, FLOOR_Y, c2.z);
      const g3 = new THREE.Vector3(c3.x, FLOOR_Y, c3.z);
      const g4 = new THREE.Vector3(c4.x, FLOOR_Y, c4.z);

      // 3D Volumetric Mesh (Semi-transparent Viewing Pyramid inside room)
      const fGeo = new THREE.BufferGeometry();
      const fVertices = new Float32Array([
        // Top triangle (cam -> c1 -> c2)
        camWorldPos.x, camWorldPos.y, camWorldPos.z,
        c1.x, c1.y, c1.z,
        c2.x, c2.y, c2.z,

        // Right triangle (cam -> c2 -> c3)
        camWorldPos.x, camWorldPos.y, camWorldPos.z,
        c2.x, c2.y, c2.z,
        c3.x, c3.y, c3.z,

        // Bottom triangle (cam -> c3 -> c4)
        camWorldPos.x, camWorldPos.y, camWorldPos.z,
        c3.x, c3.y, c3.z,
        c4.x, c4.y, c4.z,

        // Left triangle (cam -> c4 -> c1)
        camWorldPos.x, camWorldPos.y, camWorldPos.z,
        c4.x, c4.y, c4.z,
        c1.x, c1.y, c1.z,

        // Far quad (c1 -> c2 -> c3 & c1 -> c3 -> c4)
        c1.x, c1.y, c1.z,
        c2.x, c2.y, c2.z,
        c3.x, c3.y, c3.z,

        c1.x, c1.y, c1.z,
        c3.x, c3.y, c3.z,
        c4.x, c4.y, c4.z,
      ]);
      fGeo.setAttribute('position', new THREE.BufferAttribute(fVertices, 3));
      fGeo.computeVertexNormals();

      const fMat = new THREE.MeshBasicMaterial({
        color: themeColor,
        transparent: true,
        opacity: isDark ? 0.09 : 0.06,
        side: THREE.DoubleSide,
        depthWrite: false,
      });
      layoutGroup.add(new THREE.Mesh(fGeo, fMat));

      // Glowing Wireframe Rays
      const rayGeo = new THREE.BufferGeometry();
      const rayPositions = new Float32Array([
        camWorldPos.x, camWorldPos.y, camWorldPos.z, c1.x, c1.y, c1.z,
        camWorldPos.x, camWorldPos.y, camWorldPos.z, c2.x, c2.y, c2.z,
        camWorldPos.x, camWorldPos.y, camWorldPos.z, c3.x, c3.y, c3.z,
        camWorldPos.x, camWorldPos.y, camWorldPos.z, c4.x, c4.y, c4.z,
        c1.x, c1.y, c1.z, c2.x, c2.y, c2.z,
        c2.x, c2.y, c2.z, c3.x, c3.y, c3.z,
        c3.x, c3.y, c3.z, c4.x, c4.y, c4.z,
        c4.x, c4.y, c4.z, c1.x, c1.y, c1.z,
      ]);
      rayGeo.setAttribute('position', new THREE.BufferAttribute(rayPositions, 3));
      const rayMat = new THREE.LineBasicMaterial({
        color: themeColor,
        transparent: true,
        opacity: 0.60,
      });
      layoutGroup.add(new THREE.LineSegments(rayGeo, rayMat));

      // Floor Footprint Polygon (Strictly inside blue room)
      const floorGeo = new THREE.BufferGeometry();
      const floorVertices = new Float32Array([
        g1.x, FLOOR_Y, g1.z,
        g2.x, FLOOR_Y, g2.z,
        g3.x, FLOOR_Y, g3.z,

        g1.x, FLOOR_Y, g1.z,
        g3.x, FLOOR_Y, g3.z,
        g4.x, FLOOR_Y, g4.z,
      ]);
      floorGeo.setAttribute('position', new THREE.BufferAttribute(floorVertices, 3));
      const floorMat = new THREE.MeshBasicMaterial({
        color: themeColor,
        transparent: true,
        opacity: isDark ? 0.13 : 0.09,
        side: THREE.DoubleSide,
        depthWrite: false,
      });
      layoutGroup.add(new THREE.Mesh(floorGeo, floorMat));

      // Floor Footprint Border Line
      const floorBorderGeo = new THREE.BufferGeometry();
      const floorBorderPos = new Float32Array([
        g1.x, FLOOR_Y + 0.005, g1.z, g2.x, FLOOR_Y + 0.005, g2.z,
        g2.x, FLOOR_Y + 0.005, g2.z, g3.x, FLOOR_Y + 0.005, g3.z,
        g3.x, FLOOR_Y + 0.005, g3.z, g4.x, FLOOR_Y + 0.005, g4.z,
        g4.x, FLOOR_Y + 0.005, g4.z, g1.x, FLOOR_Y + 0.005, g1.z,
      ]);
      floorBorderGeo.setAttribute('position', new THREE.BufferAttribute(floorBorderPos, 3));
      const floorBorderMat = new THREE.LineBasicMaterial({
        color: themeColor,
        transparent: true,
        opacity: 0.90,
      });
      layoutGroup.add(new THREE.LineSegments(floorBorderGeo, floorBorderMat));

      // 10. Floating 3D Camera Name Badge
      const camCanvas = document.createElement('canvas');
      camCanvas.width = 280;
      camCanvas.height = 72;
      const camCtx = camCanvas.getContext('2d');
      if (camCtx) {
        camCtx.fillStyle = 'rgba(15, 23, 42, 0.92)';
        camCtx.beginPath();
        camCtx.roundRect(4, 4, 272, 64, 14);
        camCtx.fill();
        camCtx.lineWidth = 3;
        camCtx.strokeStyle = hexColorStr;
        camCtx.stroke();

        camCtx.font = 'bold 24px "Space Grotesk", sans-serif';
        camCtx.fillStyle = hexColorStr;
        camCtx.textAlign = 'center';
        camCtx.fillText(`📹 ${cam.id}`, 140, 44);
      }
      const camTex = new THREE.CanvasTexture(camCanvas);
      camTex.needsUpdate = true;
      const camBadgeMat = new THREE.SpriteMaterial({ map: camTex, transparent: true, depthWrite: false });
      const camBadgeSprite = new THREE.Sprite(camBadgeMat);
      camBadgeSprite.position.set(camX, camY + 0.35, camZ);
      camBadgeSprite.scale.set(0.95, 0.25, 1);
      layoutGroup.add(camBadgeSprite);
    });
  }, [layout, isDark, viewMode, slamMapHref, slamMapRect]);

  // ─── 3D Mesh Sync: High-Fidelity Ultra-Realistic Robot, Person, and Rack Meshes ───────────
  useEffect(() => {
    if (viewMode !== '3D' || !entitiesGroupRef.current) return;
    const entitiesGroup = entitiesGroupRef.current;

    // 1. Synchronize Robots (Thiết kế Robot AGV chuẩn đẹp thực tế giống FMS)
    Object.entries(robots).forEach(([rid, r]) => {
      let meshData = robotMeshesRef.current.get(rid);

      if (!meshData) {
        const group = new THREE.Group();
        group.userData.robotId = rid;

        // 0. Realistic Contact Shadow on floor
        const shadowGeo = new THREE.PlaneGeometry(1.2, 0.9);
        const shadowMat = new THREE.MeshBasicMaterial({
          color: 0x000000,
          transparent: true,
          opacity: 0.35,
          depthWrite: false
        });
        const shadowMesh = new THREE.Mesh(shadowGeo, shadowMat);
        shadowMesh.rotation.x = -Math.PI / 2;
        shadowMesh.position.y = 0.003;
        group.add(shadowMesh);

        // A. Lower Chassis (Dark Anthracite)
        const lowerGeo = new THREE.BoxGeometry(0.96, 0.12, 0.68);
        const lowerMat = new THREE.MeshStandardMaterial({
          color: 0x0f172a,
          metalness: 0.8,
          roughness: 0.25,
        });
        const lower = new THREE.Mesh(lowerGeo, lowerMat);
        lower.position.y = 0.10;
        lower.castShadow = true;
        group.add(lower);

        // B. Main Body (Vibrant Emerald Green Industrial Finish matching FMS Image 2)
        const isOffline = r.status === 'OFFLINE';
        const isCharging = r.status === 'CHARGING';
        const bodyColor = isOffline ? 0x475569 : isCharging ? 0xf59e0b : r.status === 'ERROR' ? 0xef4444 : r.status === 'IDLE' ? 0x0284c7 : 0x16a34a;
        const beaconColor = isOffline ? 0x64748b : isCharging ? 0xf59e0b : 0x10b981;
        const beaconEmissive = isOffline ? 0x000000 : isCharging ? 0xf59e0b : 0x10b981;
        const bodyGeo = new THREE.BoxGeometry(0.92, 0.16, 0.64);
        const bodyMat = new THREE.MeshStandardMaterial({
          color: bodyColor,
          metalness: 0.35,
          roughness: 0.25,
        });
        const bodyMesh = new THREE.Mesh(bodyGeo, bodyMat);
        bodyMesh.position.y = 0.22;
        bodyMesh.castShadow = true;
        group.add(bodyMesh);

        // C. Top Shell (Beveled cover)
        const topGeo = new THREE.BoxGeometry(0.78, 0.05, 0.54);
        const topMat = new THREE.MeshStandardMaterial({
          color: 0x1e293b,
          metalness: 0.6,
          roughness: 0.35,
        });
        const topShell = new THREE.Mesh(topGeo, topMat);
        topShell.position.y = 0.32;
        group.add(topShell);

        // D. Front Black Sensor Visor (Sleek dark glass visor matching FMS Image 2)
        const visorGeo = new THREE.BoxGeometry(0.10, 0.12, 0.56);
        const visorMat = new THREE.MeshStandardMaterial({
          color: 0x020617,
          metalness: 0.95,
          roughness: 0.08,
        });
        const visor = new THREE.Mesh(visorGeo, visorMat);
        visor.position.set(0.44, 0.22, 0);
        group.add(visor);

        // Internal Glowing LiDAR diode in Visor
        const diodeGeo = new THREE.CylinderGeometry(0.02, 0.02, 0.03, 16);
        const diodeMat = new THREE.MeshStandardMaterial({ color: 0x38bdf8, emissive: 0x38bdf8, emissiveIntensity: isOffline ? 0.2 : 2.0 });
        const diode = new THREE.Mesh(diodeGeo, diodeMat);
        diode.position.set(0.48, 0.22, 0);
        group.add(diode);

        // E. Dual Front LED Headlights (Bright cyan illumination)
        const headlightMat = new THREE.MeshStandardMaterial({
          color: isOffline ? 0x64748b : 0x38bdf8,
          emissive: isOffline ? 0x000000 : 0x38bdf8,
          emissiveIntensity: isOffline ? 0.0 : 1.5,
        });
        [-0.20, 0.20].forEach((zOffset) => {
          const lightMesh = new THREE.Mesh(new THREE.BoxGeometry(0.04, 0.04, 0.08), headlightMat);
          lightMesh.position.set(0.48, 0.22, zOffset);
          group.add(lightMesh);
        });

        // F. Dual Rear Taillights (Ruby Red)
        const tailLightMat = new THREE.MeshStandardMaterial({
          color: isOffline ? 0x64748b : 0xf43f5e,
          emissive: isOffline ? 0x000000 : 0xf43f5e,
          emissiveIntensity: isOffline ? 0.0 : 1.2,
        });
        [-0.20, 0.20].forEach((zOffset) => {
          const tailMesh = new THREE.Mesh(new THREE.BoxGeometry(0.03, 0.04, 0.08), tailLightMat);
          tailMesh.position.set(-0.47, 0.22, zOffset);
          group.add(tailMesh);
        });

        // G. Top Turntable Lifting Disc (Docking platform for Racks)
        const turntableGeo = new THREE.CylinderGeometry(0.24, 0.24, 0.04, 32);
        const turntableMat = new THREE.MeshStandardMaterial({
          color: 0x334155,
          metalness: 0.8,
          roughness: 0.2,
        });
        const turntable = new THREE.Mesh(turntableGeo, turntableMat);
        turntable.position.set(0, 0.36, 0);
        group.add(turntable);

        // H. Glowing Center Beacon / Status LED
        const beaconGeo = new THREE.CylinderGeometry(0.06, 0.06, 0.03, 16);
        const beaconMat = new THREE.MeshStandardMaterial({
          color: beaconColor,
          emissive: beaconEmissive,
          emissiveIntensity: isOffline ? 0.0 : 1.5,
        });
        const beaconMesh = new THREE.Mesh(beaconGeo, beaconMat);
        beaconMesh.position.set(0, 0.39, 0);
        group.add(beaconMesh);

        // I. 4 Rugged Rubber Tread Wheels with Metallic Rims
        const tireGeo = new THREE.CylinderGeometry(0.10, 0.10, 0.07, 18);
        const tireMat = new THREE.MeshStandardMaterial({ color: 0x090d16, roughness: 0.95 });
        const rimGeo = new THREE.CylinderGeometry(0.06, 0.06, 0.072, 18);
        const rimMat = new THREE.MeshStandardMaterial({ color: 0x64748b, metalness: 0.85, roughness: 0.2 });

        [
          [-0.30, 0.10, -0.32],
          [0.30, 0.10, -0.32],
          [-0.30, 0.10, 0.32],
          [0.30, 0.10, 0.32],
        ].forEach(([wx, wy, wz]) => {
          const wheelGroup = new THREE.Group();
          wheelGroup.position.set(wx, wy, wz);
          wheelGroup.rotation.x = Math.PI / 2;

          const tire = new THREE.Mesh(tireGeo, tireMat);
          const rim = new THREE.Mesh(rimGeo, rimMat);
          wheelGroup.add(tire);
          wheelGroup.add(rim);
          group.add(wheelGroup);
        });

        // J. Dynamic Rack Mount on Robot Back (1-Tier Single Deck Platform & Cargo)
        const rackMountGroup = new THREE.Group();
        rackMountGroup.name = 'rackMount';
        const postGeo = new THREE.CylinderGeometry(0.018, 0.018, 0.32, 8);
        const postMat = new THREE.MeshStandardMaterial({ color: 0x475569, metalness: 0.8, roughness: 0.2 });
        [[-0.28, -0.20], [0.28, -0.20], [-0.28, 0.20], [0.28, 0.20]].forEach(p => {
          const post = new THREE.Mesh(postGeo, postMat);
          post.position.set(p[0], 0.38 + 0.16, p[1]);
          rackMountGroup.add(post);
        });
        const shelfGeo = new THREE.BoxGeometry(0.68, 0.03, 0.48);
        const shelfMat = new THREE.MeshStandardMaterial({ color: 0x7c3aed, metalness: 0.5, roughness: 0.3 });
        const singleDeck = new THREE.Mesh(shelfGeo, shelfMat);
        singleDeck.position.set(0, 0.38 + 0.12, 0);
        rackMountGroup.add(singleDeck);

        const boxGeo = new THREE.BoxGeometry(0.50, 0.22, 0.36);
        const cargoBox = new THREE.Mesh(boxGeo, new THREE.MeshStandardMaterial({ color: 0x0284c7, roughness: 0.35, metalness: 0.2 }));
        cargoBox.position.set(0, 0.38 + 0.24, 0);
        rackMountGroup.add(cargoBox);

        const lidGeo = new THREE.BoxGeometry(0.52, 0.03, 0.38);
        const cargoLid = new THREE.Mesh(lidGeo, new THREE.MeshStandardMaterial({ color: 0x0369a1, roughness: 0.3, metalness: 0.4 }));
        cargoLid.position.set(0, 0.38 + 0.36, 0);
        rackMountGroup.add(cargoLid);

        rackMountGroup.visible = Boolean(r.has_rack || r.carried_rack_id);
        group.add(rackMountGroup);

        // K. Pulsing Selection Ring on floor
        const ringGeo = new THREE.RingGeometry(0.85, 1.05, 32);
        const ringMat = new THREE.MeshBasicMaterial({
          color: 0x22d3ee,
          side: THREE.DoubleSide,
          transparent: true,
          opacity: 0.85,
        });
        const pulseRing = new THREE.Mesh(ringGeo, ringMat);
        pulseRing.rotation.x = -Math.PI / 2;
        pulseRing.position.y = 0.02;
        pulseRing.visible = false;
        group.add(pulseRing);

        // L. Floating 3D Holographic Badge with Direct In-Place Canvas (Zero Texture Allocation Stalls)
        const labelKey = `${r.id}:${r.battery}:${r.status}:${r.carried_rack_id}:${r.cross_check_status}:${r.delta_distance_m}`;
        const { sprite: labelSprite, canvas: labelCanvas, ctx: labelCtx, texture: labelTexture } = createRobotLabelSprite(
          r.id, r.battery, r.status, r.carried_rack_id, r.cross_check_status, r.delta_distance_m
        );
        labelSprite.visible = showLabels;
        group.add(labelSprite);

        group.position.set(r.position[0], 0, r.position[2]);
        group.rotation.y = -r.heading;
        entitiesGroup.add(group);

        meshData = {
          group,
          targetPos: new THREE.Vector3(r.position[0], 0, r.position[2]),
          targetHeading: -r.heading,
          bodyMesh,
          beaconMesh,
          labelSprite,
          labelCanvas,
          labelCtx,
          labelTexture,
          pulseRing,
          rackMountGroup,
          lastLabelKey: labelKey,
        };
        robotMeshesRef.current.set(rid, meshData);
      }

      // Live updates
      meshData.targetPos.set(r.position[0], 0, r.position[2]);
      meshData.targetHeading = -r.heading;
      
      const isOffline = r.status === 'OFFLINE';
      const isCharging = r.status === 'CHARGING';
      const bodyColor = isOffline ? 0x475569 : isCharging ? 0xf59e0b : r.status === 'ERROR' ? 0xef4444 : r.status === 'IDLE' ? 0x0284c7 : 0x16a34a;
      const beaconColor = isOffline ? 0x64748b : isCharging ? 0xf59e0b : 0x10b981;
      const beaconEmissive = isOffline ? 0x000000 : isCharging ? 0xf59e0b : 0x10b981;
      (meshData.bodyMesh.material as THREE.MeshStandardMaterial).color.set(bodyColor);
      (meshData.beaconMesh.material as THREE.MeshStandardMaterial).color.set(beaconColor);
      (meshData.beaconMesh.material as THREE.MeshStandardMaterial).emissive.set(beaconEmissive);
      (meshData.beaconMesh.material as THREE.MeshStandardMaterial).emissiveIntensity = isOffline ? 0.0 : 1.5;
      
      meshData.rackMountGroup.visible = Boolean(r.has_rack || r.carried_rack_id);
      meshData.labelSprite.visible = showLabels;
      
      const labelKey = `${r.id}:${r.battery}:${r.status}:${r.carried_rack_id}:${r.cross_check_status}:${r.delta_distance_m}`;
      if (meshData.lastLabelKey !== labelKey) {
        meshData.lastLabelKey = labelKey;
        updateRobotLabelCanvas(meshData.labelCanvas, meshData.labelCtx, r.id, r.battery, r.status, r.carried_rack_id, r.cross_check_status, r.delta_distance_m);
        meshData.labelTexture.needsUpdate = true;
      }
    });

    robotMeshesRef.current.forEach((meshData, rid) => {
      if (!robots[rid]) {
        meshData.labelTexture.dispose();
        entitiesGroup.remove(meshData.group);
        robotMeshesRef.current.delete(rid);
      }
    });

    // 2. Synchronize Persons (Workers)
    Object.entries(persons).forEach(([pid, p]) => {
      let meshData = personMeshesRef.current.get(pid);

      if (!meshData) {
        const group = new THREE.Group();
        group.userData.personId = pid;

        const torsoGeo = new THREE.CylinderGeometry(0.20, 0.16, 0.65, 16);
        const vestMat = new THREE.MeshStandardMaterial({ color: 0xf97316, roughness: 0.4 });
        const torso = new THREE.Mesh(torsoGeo, vestMat);
        torso.position.y = 0.95;
        torso.castShadow = true;
        group.add(torso);

        const bandGeo = new THREE.CylinderGeometry(0.205, 0.205, 0.05, 16);
        const bandMat = new THREE.MeshStandardMaterial({ color: 0xf1f5f9, metalness: 0.8, roughness: 0.2 });
        const b1 = new THREE.Mesh(bandGeo, bandMat);
        b1.position.y = 1.05;
        const b2 = new THREE.Mesh(bandGeo, bandMat);
        b2.position.y = 0.85;
        group.add(b1);
        group.add(b2);

        const headGeo = new THREE.SphereGeometry(0.14, 16, 16);
        const headMat = new THREE.MeshStandardMaterial({ color: 0xfde047, roughness: 0.3 });
        const head = new THREE.Mesh(headGeo, headMat);
        head.position.y = 1.40;
        group.add(head);

        const hatGeo = new THREE.CylinderGeometry(0.18, 0.16, 0.09, 16);
        const hatMat = new THREE.MeshStandardMaterial({ color: 0xeab308, roughness: 0.2 });
        const hat = new THREE.Mesh(hatGeo, hatMat);
        hat.position.y = 1.48;
        group.add(hat);

        const haloGeo = new THREE.RingGeometry(0.60, 0.85, 32);
        const haloMat = new THREE.MeshBasicMaterial({
          color: p.safety_alert ? 0xef4444 : 0x10b981,
          side: THREE.DoubleSide,
          transparent: true,
          opacity: 0.65
        });
        const haloRing = new THREE.Mesh(haloGeo, haloMat);
        haloRing.rotation.x = -Math.PI / 2;
        haloRing.position.y = 0.02;
        group.add(haloRing);

        const labelKey = `${p.id}:${p.status}:${p.safety_alert}`;
        const { sprite: labelSprite, canvas: labelCanvas, ctx: labelCtx, texture: labelTexture } = createPersonLabelSprite(
          p.id, p.status, Boolean(p.safety_alert)
        );
        labelSprite.visible = showLabels;
        group.add(labelSprite);

        group.position.set(p.position[0], 0, p.position[2]);
        entitiesGroup.add(group);

        meshData = {
          group,
          targetPos: new THREE.Vector3(p.position[0], 0, p.position[2]),
          targetHeading: -p.heading,
          labelSprite,
          labelCanvas,
          labelCtx,
          labelTexture,
          haloRing,
          lastLabelKey: labelKey,
        };
        personMeshesRef.current.set(pid, meshData);
      }

      meshData.targetPos.set(p.position[0], 0, p.position[2]);
      meshData.targetHeading = -p.heading;
      (meshData.haloRing.material as THREE.MeshBasicMaterial).color.set(p.safety_alert ? 0xef4444 : 0x10b981);
      meshData.labelSprite.visible = showLabels;
      
      const labelKey = `${p.id}:${p.status}:${p.safety_alert}`;
      if (meshData.lastLabelKey !== labelKey) {
        meshData.lastLabelKey = labelKey;
        updatePersonLabelCanvas(meshData.labelCanvas, meshData.labelCtx, p.id, p.status, Boolean(p.safety_alert));
        meshData.labelTexture.needsUpdate = true;
      }
    });

    personMeshesRef.current.forEach((meshData, pid) => {
      if (!persons[pid]) {
        meshData.labelTexture.dispose();
        entitiesGroup.remove(meshData.group);
        personMeshesRef.current.delete(pid);
      }
    });

    // 3. Synchronize Racks (Storage Shelves & Cargo)
    Object.entries(racks).forEach(([rkid, rk]) => {
      if (rk.status === 'CARRIED') {
        const meshData = rackMeshesRef.current.get(rkid);
        if (meshData) meshData.group.visible = false;
        return;
      }

      let meshData = rackMeshesRef.current.get(rkid);

      if (!meshData) {
        const group = new THREE.Group();
        group.userData.rackId = rkid;

        // 1-Tier Industrial Warehouse Rack (Kệ hàng 1 tầng minh họa)
        const postGeo = new THREE.CylinderGeometry(0.022, 0.022, 0.46, 8);
        const postMat = new THREE.MeshStandardMaterial({ color: 0x334155, metalness: 0.8, roughness: 0.3 });
        [[-0.35, -0.28], [0.35, -0.28], [-0.35, 0.28], [0.35, 0.28]].forEach(p => {
          const post = new THREE.Mesh(postGeo, postMat);
          post.position.set(p[0], 0.23, p[1]);
          post.castShadow = true;
          group.add(post);
        });

        // Single Shelf Deck Platform
        const shelfGeo = new THREE.BoxGeometry(0.88, 0.04, 0.68);
        const shelfMat = new THREE.MeshStandardMaterial({ color: 0x6366f1, metalness: 0.6, roughness: 0.3 });
        const shelf = new THREE.Mesh(shelfGeo, shelfMat);
        shelf.position.set(0, 0.16, 0);
        shelf.castShadow = true;
        group.add(shelf);

        // Single Tier Industrial Cargo Container Box
        const boxGeo = new THREE.BoxGeometry(0.65, 0.28, 0.52);
        const cargoBox = new THREE.Mesh(boxGeo, new THREE.MeshStandardMaterial({ color: 0x0284c7, roughness: 0.35, metalness: 0.2 }));
        cargoBox.position.set(0, 0.32, 0);
        cargoBox.castShadow = true;
        group.add(cargoBox);

        // Container Lid
        const lidGeo = new THREE.BoxGeometry(0.67, 0.03, 0.54);
        const cargoLid = new THREE.Mesh(lidGeo, new THREE.MeshStandardMaterial({ color: 0x0369a1, roughness: 0.3, metalness: 0.4 }));
        cargoLid.position.set(0, 0.47, 0);
        group.add(cargoLid);

        const labelKey = `${rk.id}:${rk.status}:${rk.carried_by}`;
        const { sprite: labelSprite, canvas: labelCanvas, ctx: labelCtx, texture: labelTexture } = createRackLabelSprite(
          rk.id, rk.status, rk.carried_by
        );
        labelSprite.visible = showLabels;
        group.add(labelSprite);

        group.position.set(rk.position[0], 0, rk.position[2]);
        entitiesGroup.add(group);

        meshData = {
          group,
          targetPos: new THREE.Vector3(rk.position[0], 0, rk.position[2]),
          labelSprite,
          labelCanvas,
          labelCtx,
          labelTexture,
          lastLabelKey: labelKey,
        };
        rackMeshesRef.current.set(rkid, meshData);
      }

      meshData.group.visible = true;
      meshData.targetPos.set(rk.position[0], 0, rk.position[2]);
      meshData.labelSprite.visible = showLabels;
      
      const labelKey = `${rk.id}:${rk.status}:${rk.carried_by}`;
      if (meshData.lastLabelKey !== labelKey) {
        meshData.lastLabelKey = labelKey;
        updateRackLabelCanvas(meshData.labelCanvas, meshData.labelCtx, rk.id, rk.status, rk.carried_by);
        meshData.labelTexture.needsUpdate = true;
      }
    });

    rackMeshesRef.current.forEach((meshData, rkid) => {
      if (!racks[rkid]) {
        meshData.labelTexture.dispose();
        entitiesGroup.remove(meshData.group);
        rackMeshesRef.current.delete(rkid);
      }
    });

    // 4. Dynamic Anti-Collision & Vertical Staggering for 3D Label Cards
    // Prevents floating badges from overlapping when robots, persons, or racks are close together
    const allActiveLabels: {
      id: string;
      x: number;
      z: number;
      sprite: THREE.Sprite;
      baseY: number;
    }[] = [];

    robotMeshesRef.current.forEach((meshData, rid) => {
      const r = robots[rid];
      if (r) {
        allActiveLabels.push({
          id: rid,
          x: r.position[0],
          z: r.position[2],
          sprite: meshData.labelSprite,
          baseY: 1.35,
        });
      }
    });

    personMeshesRef.current.forEach((meshData, pid) => {
      const p = persons[pid];
      if (p) {
        allActiveLabels.push({
          id: pid,
          x: p.position[0],
          z: p.position[2],
          sprite: meshData.labelSprite,
          baseY: 1.65,
        });
      }
    });

    rackMeshesRef.current.forEach((meshData, rkid) => {
      const rk = racks[rkid];
      if (rk && rk.status !== 'CARRIED') {
        allActiveLabels.push({
          id: rkid,
          x: rk.position[0],
          z: rk.position[2],
          sprite: meshData.labelSprite,
          baseY: 1.40,
        });
      }
    });

    // Sort deterministically by ID to avoid jitter across render cycles
    allActiveLabels.sort((a, b) => a.id.localeCompare(b.id));

    // Form proximity clusters within 2.2m radius and stack their heights neatly
    const processedEntityIds = new Set<string>();
    allActiveLabels.forEach((item) => {
      if (processedEntityIds.has(item.id)) return;

      const cluster = allActiveLabels.filter(other => {
        const dx = other.x - item.x;
        const dz = other.z - item.z;
        return Math.hypot(dx, dz) < 2.2;
      });

      cluster.forEach((member, index) => {
        processedEntityIds.add(member.id);
        // Stagger each card vertically by 0.55m so every badge is completely legible and unobscured
        member.sprite.position.y = member.baseY + index * 0.55;
      });
    });

  }, [robots, persons, racks, viewMode, showLabels]);

  const selectedRobot = selectedEntity?.type === 'robot' ? robots[selectedEntity.id] : null;
  const selectedPerson = selectedEntity?.type === 'person' ? persons[selectedEntity.id] : null;
  const selectedRack = selectedEntity?.type === 'rack' ? racks[selectedEntity.id] : null;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: 'calc(100vh - 84px)', width: '100%', background: 'var(--bg-main)', overflow: 'hidden' }}>
      {/* ── TOP KPI & CONTROL BAR ── */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '12px 20px',
        background: 'var(--bg-card)',
        borderBottom: '1px solid var(--border)',
        zIndex: 10,
      }}>
        {/* Left: Title & Realtime Broadcaster Status */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Sparkles size={20} style={{ color: 'var(--cyan)' }} />
            <span style={{ fontWeight: 700, fontSize: '15px', color: 'var(--text-primary)' }}>
              3D Digital Twin & Real-Time Tracking
            </span>
          </div>

          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            padding: '5px 12px',
            borderRadius: '999px',
            background: 'rgba(22, 163, 74, 0.15)',
            border: '1px solid rgba(22, 163, 74, 0.35)',
          }}>
            <span style={{
              width: '8px', height: '8px', borderRadius: '50%',
              background: '#16a34a', boxShadow: '0 0 8px #16a34a',
              animation: 'pulse 2s infinite'
            }} />
            <span style={{ fontSize: '12px', fontWeight: 700, color: '#16a34a', fontFamily: 'monospace' }}>
              VISION + FMS FUSION (15Hz)
            </span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', fontSize: '12px', color: 'var(--text-sub)' }}>
            <span>Tần số: <strong style={{ color: 'var(--cyan)' }}>{packetRate} Hz</strong></span>
          </div>
        </div>

        {/* Center: Multi-Entity KPI Badges */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div style={{
            display: 'flex', alignItems: 'center', gap: '6px',
            background: 'rgba(22, 163, 74, 0.12)', padding: '5px 12px',
            borderRadius: '8px', border: '1px solid rgba(22, 163, 74, 0.32)',
          }}>
            <Bot size={15} color="#22c55e" />
            <span style={{ fontSize: '11px', color: '#22c55e', fontWeight: 600 }}>Robots:</span>
            <strong style={{ fontSize: '13px', color: '#22c55e', fontFamily: 'monospace' }}>{Object.keys(robots).length}</strong>
          </div>

          <div style={{
            display: 'flex', alignItems: 'center', gap: '6px',
            background: 'rgba(249, 115, 22, 0.12)', padding: '5px 12px',
            borderRadius: '8px', border: '1px solid rgba(249, 115, 22, 0.35)',
          }}>
            <User size={15} color="#fb923c" />
            <span style={{ fontSize: '11px', color: '#fb923c', fontWeight: 600 }}>Công Nhân:</span>
            <strong style={{ fontSize: '13px', color: '#fb923c', fontFamily: 'monospace' }}>{Object.keys(persons).length}</strong>
          </div>

          <div style={{
            display: 'flex', alignItems: 'center', gap: '6px',
            background: 'rgba(168, 85, 247, 0.12)', padding: '5px 12px',
            borderRadius: '8px', border: '1px solid rgba(168, 85, 247, 0.35)',
          }}>
            <Package size={15} color="#c084fc" />
            <span style={{ fontSize: '11px', color: '#c084fc', fontWeight: 600 }}>Kệ Hàng:</span>
            <strong style={{ fontSize: '13px', color: '#c084fc', fontFamily: 'monospace' }}>{Object.keys(racks).length}</strong>
          </div>
        </div>

        {/* Right: View & Camera Controls */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div style={{
            display: 'flex', background: 'var(--bg-elevated)',
            borderRadius: '10px', padding: '3px', border: '1px solid var(--border)',
          }}>
            <button
              onClick={() => setViewMode('3D')}
              style={{
                padding: '6px 14px', borderRadius: '8px', fontSize: '12px', fontWeight: 600,
                cursor: 'pointer',
                background: viewMode === '3D' ? 'linear-gradient(135deg, #16a34a, #15803d)' : 'transparent',
                color: viewMode === '3D' ? '#ffffff' : 'var(--text-sub)',
                border: 'none', transition: 'all 0.2s',
              }}
            >
              3D Digital Twin
            </button>
            <button
              onClick={() => setViewMode('2D')}
              style={{
                padding: '6px 14px', borderRadius: '8px', fontSize: '12px', fontWeight: 600,
                cursor: 'pointer',
                background: viewMode === '2D' ? 'linear-gradient(135deg, #16a34a, #15803d)' : 'transparent',
                color: viewMode === '2D' ? '#ffffff' : 'var(--text-sub)',
                border: 'none', transition: 'all 0.2s',
              }}
            >
              2D Bản Đồ FMS
            </button>
          </div>

          {viewMode === '3D' && (
            <button
              onClick={handleResetCamera}
              style={{
                display: 'flex', alignItems: 'center', gap: '6px',
                padding: '7px 12px', borderRadius: '8px', fontSize: '12px', fontWeight: 600,
                cursor: 'pointer', background: 'var(--bg-elevated)', color: 'var(--text-label)',
                border: '1px solid var(--border)',
              }}
            >
              <RotateCcw size={13} />
              Reset Cam
            </button>
          )}

          {viewMode === '3D' && selectedEntity && (
            <button
              onClick={() => setFollowTarget(!followTarget)}
              style={{
                display: 'flex', alignItems: 'center', gap: '6px',
                padding: '7px 14px', borderRadius: '8px', fontSize: '12px', fontWeight: 700,
                cursor: 'pointer',
                background: followTarget ? 'rgba(22, 163, 74, 0.15)' : 'var(--bg-elevated)',
                color: followTarget ? '#22c55e' : 'var(--text-label)',
                border: `1px solid ${followTarget ? 'rgba(22, 163, 74, 0.4)' : 'var(--border)'}`,
              }}
            >
              <Crosshair size={14} />
              Follow {selectedEntity.type === 'robot' ? 'Robot' : selectedEntity.type === 'person' ? 'Person' : 'Rack'}
            </button>
          )}
        </div>
      </div>

      {/* ── MAIN WORKSPACE ── */}
      <div style={{ display: 'flex', flex: 1, position: 'relative', overflow: 'hidden' }}>
        {/* ── LEFT PANEL: MULTI-ENTITY FLEET SELECTOR ── */}
        <div style={{
          width: '320px',
          background: 'var(--bg-card)',
          borderRight: '1px solid var(--border)',
          display: 'flex',
          flexDirection: 'column',
          zIndex: 5,
        }}>
          {/* Tabs */}
          <div style={{ display: 'flex', borderBottom: '1px solid var(--border)' }}>
            <button
              onClick={() => setFleetTab('robots')}
              style={{
                flex: 1, padding: '10px 4px', fontSize: '11px', fontWeight: 700,
                cursor: 'pointer', background: fleetTab === 'robots' ? 'var(--bg-elevated)' : 'transparent',
                color: fleetTab === 'robots' ? '#22c55e' : 'var(--text-muted)',
                borderBottom: fleetTab === 'robots' ? '2px solid #22c55e' : '2px solid transparent',
                borderTop: 'none', borderLeft: 'none', borderRight: 'none',
                display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '4px'
              }}
            >
              <Bot size={14} /> Robots ({Object.keys(robots).length})
            </button>
            <button
              onClick={() => setFleetTab('persons')}
              style={{
                flex: 1, padding: '10px 4px', fontSize: '11px', fontWeight: 700,
                cursor: 'pointer', background: fleetTab === 'persons' ? 'var(--bg-elevated)' : 'transparent',
                color: fleetTab === 'persons' ? '#fb923c' : 'var(--text-muted)',
                borderBottom: fleetTab === 'persons' ? '2px solid #fb923c' : '2px solid transparent',
                borderTop: 'none', borderLeft: 'none', borderRight: 'none',
                display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '4px'
              }}
            >
              <User size={14} /> Người ({Object.keys(persons).length})
            </button>
            <button
              onClick={() => setFleetTab('racks')}
              style={{
                flex: 1, padding: '10px 4px', fontSize: '11px', fontWeight: 700,
                cursor: 'pointer', background: fleetTab === 'racks' ? 'var(--bg-elevated)' : 'transparent',
                color: fleetTab === 'racks' ? '#c084fc' : 'var(--text-muted)',
                borderBottom: fleetTab === 'racks' ? '2px solid #c084fc' : '2px solid transparent',
                borderTop: 'none', borderLeft: 'none', borderRight: 'none',
                display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '4px'
              }}
            >
              <Package size={14} /> Kệ ({Object.keys(racks).length})
            </button>
          </div>

          {/* List Content */}
          <div style={{ flex: 1, overflowY: 'auto', padding: '10px' }}>
            {fleetTab === 'robots' && (
              Object.keys(robots).length === 0 ? (
                <div style={{ padding: '30px 10px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '12px' }}>
                  Chưa phát hiện Robot nào trên sàn...
                </div>
              ) : (
                Object.values(robots).map((r) => {
                  const isSelected = selectedEntity?.type === 'robot' && selectedEntity?.id === r.id;
                  const statusColor = STATUS_COLORS[r.status] || '#16a34a';
                  return (
                    <div
                      key={r.id}
                      onClick={() => setSelectedEntity({ type: 'robot', id: r.id })}
                      style={{
                        padding: '12px', borderRadius: '8px', marginBottom: '8px', cursor: 'pointer',
                        background: isSelected ? 'rgba(22,163,74,0.15)' : 'var(--bg-elevated)',
                        border: `1px solid ${isSelected ? '#16a34a' : 'var(--border)'}`,
                        boxShadow: isSelected ? '0 0 12px rgba(22,163,74,0.25)' : 'none',
                        transition: 'all 0.2s',
                      }}
                    >
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontWeight: 700, fontSize: '13px', color: 'var(--text-primary)' }}>
                          <Bot size={15} color="#22c55e" />
                          <span>{r.id}</span>
                          {r.carried_rack_id && (
                            <span style={{ fontSize: '10px', background: 'rgba(168,85,247,0.2)', color: '#c084fc', padding: '1px 5px', borderRadius: '4px' }}>
                              📦 {r.carried_rack_id}
                            </span>
                          )}
                        </div>
                        <span style={{ fontSize: '10px', fontWeight: 700, padding: '2px 6px', borderRadius: '4px', background: `${statusColor}20`, color: statusColor }}>
                          {r.status}
                        </span>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: 'var(--text-sub)', fontFamily: 'monospace' }}>
                        <span>⚡ {r.battery}% · {r.velocity.toFixed(2)} m/s</span>
                        <span>({r.position[0].toFixed(1)}, {r.position[2].toFixed(1)})</span>
                      </div>
                    </div>
                  );
                })
              )
            )}

            {fleetTab === 'persons' && (
              Object.keys(persons).length === 0 ? (
                <div style={{ padding: '30px 10px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '12px' }}>
                  Chưa phát hiện người/công nhân nào...
                </div>
              ) : (
                Object.values(persons).map((p) => {
                  const isSelected = selectedEntity?.type === 'person' && selectedEntity?.id === p.id;
                  return (
                    <div
                      key={p.id}
                      onClick={() => setSelectedEntity({ type: 'person', id: p.id })}
                      style={{
                        padding: '12px', borderRadius: '8px', marginBottom: '8px', cursor: 'pointer',
                        background: isSelected ? 'rgba(249,115,22,0.15)' : 'var(--bg-elevated)',
                        border: `1px solid ${isSelected ? '#f97316' : 'var(--border)'}`,
                      }}
                    >
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontWeight: 700, fontSize: '13px', color: 'var(--text-primary)' }}>
                          <User size={15} color="#fb923c" />
                          <span>{p.id}</span>
                        </div>
                        <span style={{ fontSize: '10px', fontWeight: 700, padding: '2px 6px', borderRadius: '4px', background: p.safety_alert ? 'rgba(239,68,68,0.2)' : 'rgba(249,115,22,0.2)', color: p.safety_alert ? '#ef4444' : '#fb923c' }}>
                          {p.safety_alert ? '⚠ CẢNH BÁO' : p.status}
                        </span>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: 'var(--text-sub)', fontFamily: 'monospace' }}>
                        <span>Vận tốc: {p.velocity.toFixed(2)} m/s</span>
                        <span>({p.position[0].toFixed(1)}, {p.position[2].toFixed(1)})</span>
                      </div>
                    </div>
                  );
                })
              )
            )}

            {fleetTab === 'racks' && (
              Object.keys(racks).length === 0 ? (
                <div style={{ padding: '30px 10px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '12px' }}>
                  Chưa phát hiện Kệ Hàng nào...
                </div>
              ) : (
                Object.values(racks).map((rk) => {
                  const isSelected = selectedEntity?.type === 'rack' && selectedEntity?.id === rk.id;
                  const isCarried = rk.status === 'CARRIED';
                  return (
                    <div
                      key={rk.id}
                      onClick={() => setSelectedEntity({ type: 'rack', id: rk.id })}
                      style={{
                        padding: '12px', borderRadius: '8px', marginBottom: '8px', cursor: 'pointer',
                        background: isSelected ? 'rgba(168,85,247,0.15)' : 'var(--bg-elevated)',
                        border: `1px solid ${isSelected ? '#a855f7' : 'var(--border)'}`,
                      }}
                    >
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontWeight: 700, fontSize: '13px', color: 'var(--text-primary)' }}>
                          <Package size={15} color="#c084fc" />
                          <span>{rk.id}</span>
                        </div>
                        <span style={{ fontSize: '10px', fontWeight: 700, padding: '2px 6px', borderRadius: '4px', background: isCarried ? 'rgba(168,85,247,0.2)' : 'rgba(100,116,139,0.2)', color: isCarried ? '#c084fc' : '#94a3b8' }}>
                          {isCarried ? 'TRÊN LƯNG XE' : 'Ô LƯU TRỮ'}
                        </span>
                      </div>
                      <div style={{ fontSize: '11px', color: 'var(--text-sub)', fontFamily: 'monospace' }}>
                        {isCarried ? `Cõng bởi: ${rk.carried_by || 'Robot'}` : `Tọa độ: (${rk.position[0].toFixed(1)}, ${rk.position[2].toFixed(1)})`}
                      </div>
                    </div>
                  );
                })
              )
            )}
          </div>
        </div>

        {/* ── CENTER: 3D DIGITAL TWIN VIEWPORT OR 2D MAP ── */}
        <div style={{ flex: 1, position: 'relative', height: '100%', overflow: 'hidden' }}>
          {viewMode === '3D' ? (
            <div ref={mountRef} style={{ width: '100%', height: '100%' }} />
          ) : (
            /* 2D FMS Map View with Robots, Persons, and Racks (Đúng chuẩn FMS) */
            <div style={{
              width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center',
              padding: '16px', background: isDark ? '#06070a' : '#f8fafc',
            }}>
              <svg
                viewBox={`${layoutFrame.minX} ${layoutFrame.minZ} ${layoutFrame.width} ${layoutFrame.depth}`}
                style={{
                  width: '100%', maxHeight: '100%', background: isDark ? '#090a0f' : '#ffffff',
                  borderRadius: '16px', border: isDark ? '1px solid #1e293b' : '1px solid #e2e8f0',
                  boxShadow: '0 4px 24px rgba(0,0,0,0.12)'
                }}
              >
                {/* Subtle Grid Lines */}
                <defs>
                  <pattern id="fmsGrid" width="0.5" height="0.5" patternUnits="userSpaceOnUse">
                    <path d="M 0.5 0 L 0 0 0 0.5" fill="none" stroke={isDark ? "rgba(255,255,255,0.04)" : "rgba(0,0,0,0.04)"} strokeWidth="0.02" />
                  </pattern>
                </defs>
                <rect x={layoutFrame.minX} y={layoutFrame.minZ} width={layoutFrame.width} height={layoutFrame.depth} fill="url(#fmsGrid)" />

                {/* SLAM Floor Map */}
                <image
                  href={slamMapHref}
                  x={slamMapRect[0]}
                  y={slamMapRect[1]}
                  width={slamMapRect[2] - slamMapRect[0]}
                  height={slamMapRect[3] - slamMapRect[1]}
                  preserveAspectRatio="none"
                  opacity={isDark ? '0.85' : '0.65'}
                />

                {/* Walkways (Đường chạy robot xám/xanh nhạt chuẩn FMS) */}
                {(layout?.walkways ?? []).map((walkway) => (
                  <polygon
                    key={`walkway-${walkway.id}`}
                    points={polygonToPointsAttr(walkway.polygon)}
                    fill={isDark ? "#1e293b" : "#e2e8f0"}
                    fillOpacity={isDark ? 0.75 : 0.85}
                    stroke="none"
                  />
                ))}

                {/* Directional Chevrons on Walkway Tracks (Giống FMS gốc) */}
                <g stroke={isDark ? "rgba(255,255,255,0.25)" : "rgba(100,116,139,0.4)"} strokeWidth="0.06" fill="none" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M 9.9 12.1 L 10.1 11.9 L 10.3 12.1" />
                  <path d="M 9.9 12.35 L 10.1 12.15 L 10.3 12.35" />
                  <path d="M 9.9 13.3 L 10.1 13.1 L 10.3 13.3" />
                  <path d="M 9.9 13.55 L 10.1 13.35 L 10.3 13.55" />
                </g>

                {/* SLAM Physical Walls */}
                {slamWallsSvgPath && (
                  <path d={slamWallsSvgPath} stroke="#2563eb" strokeWidth="0.10" fill="none" strokeLinecap="round" strokeLinejoin="round" />
                )}

                {/* FMS Waypoints (Các điểm tròn xanh lá & bánh răng trên đường) */}
                {(layout?.locations ?? []).map((loc) => {
                  const pt = getItemPoint(loc);
                  if (!pt) return null;
                  const isChargingPt = loc.kind === 'CHARGING' || loc.id.toLowerCase().includes('charge') || loc.id === 'AutoXing';
                  const isStationEnd = loc.id.startsWith('19') && (pt[1] > 14.5 || loc.id.includes('AXP'));

                  return (
                    <g key={`2d-wpt-${loc.id}`} transform={`translate(${pt[0]}, ${pt[1]})`}>
                      {isStationEnd ? (
                        /* Blue Cog/Gear icon for Station Endpoints */
                        <g>
                          <circle cx="0" cy="0" r="0.10" fill="#0284c7" stroke="#ffffff" strokeWidth="0.02" />
                          <circle cx="0" cy="0" r="0.04" fill="#ffffff" />
                        </g>
                      ) : (
                        /* Green Circle Waypoints */
                        <g>
                          <circle cx="0" cy="0" r="0.09" fill={isChargingPt ? "#f59e0b" : "#16a34a"} stroke="#ffffff" strokeWidth="0.02" />
                          <circle cx="0" cy="0" r="0.04" fill="#ffffff" opacity="0.8" />
                        </g>
                      )}
                    </g>
                  );
                })}

                {/* Dynamic Real Charging Stations from layout */}
                {(layout?.charging_stations ?? []).map((cs) => {
                  const posX = cs.position?.[0] ?? cs.access_point?.[0] ?? 0;
                  const posZ = cs.position?.[2] ?? cs.access_point?.[1] ?? 0;
                  return (
                    <g key={`2d-cs-${cs.id}`} transform={`translate(${posX}, ${posZ})`}>
                      <rect x="-0.28" y="-0.28" width="0.56" height="0.56" rx="0.10" fill="#f59e0b" stroke="#d97706" strokeWidth="0.03" />
                      <circle cx="0" cy="0" r="0.16" fill="#0f172a" />
                      <path d="M -0.05,-0.12 L 0.05,-0.01 L -0.01,0.00 L 0.06,0.12 L -0.05,0.01 L 0.01,-0.00 Z" fill="#fbbf24" />
                      <text x="0" y="0.48" textAnchor="middle" fill="#fbbf24" fontSize="0.22" fontWeight="bold">
                        ⚡ {cs.id}
                      </text>
                    </g>
                  );
                })}

                {/* Station checkmark and purple barcode box (Khớp chuẩn FMS) */}
                <g transform="translate(15.5, 12.8)">
                  <rect x="-0.30" y="-0.16" width="0.32" height="0.32" rx="0.04" fill="#10b981" />
                  <path d="M -0.22,0.0 L -0.16,0.08 L -0.04,-0.06" fill="none" stroke="#ffffff" strokeWidth="0.04" strokeLinecap="round" strokeLinejoin="round" />
                  <rect x="0.08" y="-0.16" width="0.32" height="0.32" rx="0.04" fill="#9333ea" />
                  <line x1="0.16" y1="-0.10" x2="0.16" y2="0.10" stroke="#ffffff" strokeWidth="0.03" />
                  <line x1="0.24" y1="-0.10" x2="0.24" y2="0.10" stroke="#ffffff" strokeWidth="0.03" />
                  <line x1="0.32" y1="-0.10" x2="0.32" y2="0.10" stroke="#ffffff" strokeWidth="0.03" />
                </g>

                {/* Stations (Trạm bốc dỡ) */}
                {(layout?.stations ?? []).map((st) => (
                  <g key={`2d-st-${st.id}`}>
                    <rect
                      x={st.rect[0]}
                      y={st.rect[1]}
                      width={st.rect[2] - st.rect[0]}
                      height={st.rect[3] - st.rect[1]}
                      rx="0.1"
                      fill="rgba(22, 163, 74, 0.20)"
                      stroke="#16a34a"
                      strokeWidth="0.03"
                    />
                  </g>
                ))}

                {/* 2D Industrial Cameras with FOV Frustums (Strictly clipped inside blue walls) */}
                {((layout?.cameras && layout.cameras.length > 0) ? layout.cameras : DEFAULT_CAMERAS).map((cam, camIdx) => {
                  const isCam1 = cam.id.toLowerCase().includes('1') || camIdx === 0;
                  const col = isCam1 ? '#06b6d4' : '#6366f1';
                  const posX = cam.position[0];
                  const posZ = cam.position[2];
                  const lookX = cam.look_at ? cam.look_at[0] : (isCam1 ? 10.5 : 15.2);
                  const lookZ = cam.look_at ? cam.look_at[2] : (isCam1 ? 11.8 : 12.0);
                  const angleRad = Math.atan2(lookZ - posZ, lookX - posX);
                  const angleDeg = (angleRad * 180) / Math.PI;
                  const fov = cam.fov_deg ?? (isCam1 ? 76 : 78);
                  const range = cam.range_m ?? (isCam1 ? 8.8 : 8.2);
                  const halfFovRad = ((fov / 2) * Math.PI) / 180;

                  const rawP1x = posX + range * Math.cos(angleRad - halfFovRad);
                  const rawP1z = posZ + range * Math.sin(angleRad - halfFovRad);
                  const rawP2x = posX + range * Math.cos(angleRad + halfFovRad);
                  const rawP2z = posZ + range * Math.sin(angleRad + halfFovRad);

                  const clampPt = (x: number, z: number) => {
                    const dx = x - posX;
                    const dz = z - posZ;
                    let t = 1.0;
                    if (dx < -1e-5 && posX + dx < 6.30) t = Math.min(t, (6.30 - posX) / dx);
                    if (dx > 1e-5 && posX + dx > 21.80) t = Math.min(t, (21.80 - posX) / dx);
                    if (dz < -1e-5 && posZ + dz < 6.92) t = Math.min(t, (6.92 - posZ) / dz);
                    if (dz > 1e-5 && posZ + dz > 15.38) t = Math.min(t, (15.38 - posZ) / dz);
                    t = Math.max(0.05, Math.min(1.0, t));
                    return [
                      Math.max(6.30, Math.min(21.80, posX + t * dx)),
                      Math.max(6.92, Math.min(15.38, posZ + t * dz)),
                    ];
                  };

                  const [p1x, p1z] = clampPt(rawP1x, rawP1z);
                  const [p2x, p2z] = clampPt(rawP2x, rawP2z);

                  return (
                    <g key={`2d-cam-${cam.id}`}>
                      {/* FOV Frustum Cone */}
                      <polygon
                        points={`${posX},${posZ} ${p1x},${p1z} ${p2x},${p2z}`}
                        fill={col}
                        fillOpacity={isDark ? 0.16 : 0.12}
                        stroke={col}
                        strokeWidth="0.03"
                        strokeDasharray="0.1 0.05"
                      />
                      {/* Camera Body Icon */}
                      <g transform={`translate(${posX}, ${posZ}) rotate(${angleDeg})`}>
                        <rect x="-0.22" y="-0.14" width="0.44" height="0.28" rx="0.06" fill="#0f172a" stroke={col} strokeWidth="0.04" />
                        <polygon points="0.22,-0.10 0.36,-0.18 0.36,0.18 0.22,0.10" fill={col} />
                        <circle cx="-0.06" cy="0" r="0.05" fill={col} />
                      </g>
                      {/* Camera Floating Label */}
                      <rect x={posX - 0.55} y={posZ - 0.48} width="1.1" height="0.32" rx="0.06" fill="rgba(15,23,42,0.85)" stroke={col} strokeWidth="0.02" />
                      <text x={posX} y={posZ - 0.27} textAnchor="middle" fill={col} fontSize="0.18" fontWeight="bold">
                        📹 {cam.id}
                      </text>
                    </g>
                  );
                })}

                {/* 2D Racks */}
                {Object.values(racks).map((rk) => (
                  <g key={`2d-rack-${rk.id}`} transform={`translate(${rk.position[0]}, ${rk.position[2]})`}>
                    <rect x="-0.35" y="-0.25" width="0.70" height="0.50" rx="0.08" fill="#8b5cf6" stroke="#7c3aed" strokeWidth="0.03" />
                    <text x="0" y="0.07" textAnchor="middle" fill="#ffffff" fontSize="0.18" fontWeight="bold">
                      📦 {rk.id}
                    </text>
                  </g>
                ))}

                {/* 2D Persons */}
                {Object.values(persons).map((p) => (
                  <g key={`2d-person-${p.id}`} transform={`translate(${p.position[0]}, ${p.position[2]})`}>
                    <circle cx="0" cy="0" r="0.30" fill="#f97316" stroke="#ea580c" strokeWidth="0.04" />
                    <circle cx="0" cy="0" r="0.13" fill="#eab308" />
                    <text x="0" y="-0.40" textAnchor="middle" fill="#f97316" fontSize="0.20" fontWeight="bold">
                      👤 {p.id}
                    </text>
                  </g>
                ))}

                {/* 2D Robots */}
                {Object.values(robots).map((r) => {
                  const headingDeg = (r.heading * 180) / Math.PI;
                  const isOffline = r.status === 'OFFLINE';
                  const isCharging = r.status === 'CHARGING';
                  const robotColor = isOffline ? '#64748b' : isCharging ? '#f59e0b' : r.status === 'ERROR' ? '#ef4444' : r.status === 'IDLE' ? '#0284c7' : '#16a34a';
                  const isSelected = selectedEntity?.type === 'robot' && selectedEntity?.id === r.id;

                  return (
                    <g key={`2d-robot-${r.id}`} transform={`translate(${r.position[0]}, ${r.position[2]})`}>
                      <g transform={`rotate(${headingDeg})`}>
                        {/* Cyan Glowing Active / Selected Halo Box */}
                        <rect
                          x="-0.50"
                          y="-0.62"
                          width="1.00"
                          height="1.24"
                          rx="0.20"
                          fill={isSelected ? "rgba(6, 182, 212, 0.25)" : "transparent"}
                          stroke={isSelected ? "#22d3ee" : "transparent"}
                          strokeWidth="0.06"
                          style={{ filter: isSelected ? 'drop-shadow(0 0 6px #22d3ee)' : 'none' }}
                        />

                        {/* 4 Wheels */}
                        <rect x="-0.38" y="-0.48" width="0.08" height="0.22" rx="0.03" fill="#0f172a" />
                        <rect x="0.30" y="-0.48" width="0.08" height="0.22" rx="0.03" fill="#0f172a" />
                        <rect x="-0.38" y="0.26" width="0.08" height="0.22" rx="0.03" fill="#0f172a" />
                        <rect x="0.30" y="0.26" width="0.08" height="0.22" rx="0.03" fill="#0f172a" />

                        {/* Main Body */}
                        <rect x="-0.34" y="-0.48" width="0.68" height="0.96" rx="0.14" fill={robotColor} stroke="#0f172a" strokeWidth="0.03" />

                        {/* Front Dark Sensor Visor */}
                        <rect x="-0.26" y="-0.48" width="0.52" height="0.16" rx="0.04" fill="#020617" />

                        {/* Center Turntable Disc */}
                        <circle cx="0" cy="0.05" r="0.18" fill="#1e293b" />
                        <circle cx="0" cy="0.05" r="0.08" fill={isOffline ? '#64748b' : isCharging ? '#fbbf24' : '#22c55e'} />
                      </g>

                      {/* Floating ID & Battery Badge */}
                      <rect x="-0.75" y="-0.95" width="1.50" height="0.36" rx="0.08" fill="rgba(15, 23, 42, 0.88)" stroke={robotColor} strokeWidth="0.02" />
                      <text x="0" y="-0.71" textAnchor="middle" fill={isOffline ? '#94a3b8' : '#ffffff'} fontSize="0.18" fontWeight="bold" fontFamily="monospace">
                        🤖 {r.id} {isOffline ? '(OFFLINE)' : `(${r.battery}%)`}
                      </text>
                    </g>
                  );
                })}
              </svg>
            </div>
          )}
        </div>

        {/* ── RIGHT PANEL: SELECTED ENTITY TELEMETRY & INSPECTOR ── */}
        <div style={{
          width: '360px',
          background: 'var(--bg-card)',
          borderLeft: '1px solid var(--border)',
          display: 'flex',
          flexDirection: 'column',
          zIndex: 5,
        }}>
          <div style={{ flex: 1, overflowY: 'auto', padding: '16px' }}>
            {selectedRobot ? (
              <div>
                {/* 1. Header Card with ID & FMS Status */}
                <div style={{ padding: '14px', borderRadius: '10px', background: 'var(--bg-elevated)', border: '1px solid var(--border)', marginBottom: '14px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <h3 style={{ margin: 0, fontSize: '17px', fontWeight: 700, color: 'var(--text-primary)' }}>🤖 {selectedRobot.id}</h3>
                    <span style={{
                      fontSize: '11px', fontWeight: 700, padding: '3px 10px', borderRadius: '6px',
                      background: selectedRobot.status === 'OFFLINE' ? 'rgba(100,116,139,0.25)' : selectedRobot.status === 'CHARGING' ? 'rgba(245,158,11,0.2)' : (selectedRobot.status === 'RUNNING' || selectedRobot.status === 'ACTIVE') ? 'rgba(22,163,74,0.2)' : selectedRobot.status === 'ERROR' ? 'rgba(239,68,68,0.2)' : 'rgba(2,132,199,0.2)',
                      color: selectedRobot.status === 'OFFLINE' ? '#94a3b8' : selectedRobot.status === 'CHARGING' ? '#fbbf24' : (selectedRobot.status === 'RUNNING' || selectedRobot.status === 'ACTIVE') ? '#22c55e' : selectedRobot.status === 'ERROR' ? '#ef4444' : '#38bdf8',
                      border: `1px solid ${selectedRobot.status === 'OFFLINE' ? 'rgba(100,116,139,0.4)' : selectedRobot.status === 'CHARGING' ? 'rgba(245,158,11,0.4)' : (selectedRobot.status === 'RUNNING' || selectedRobot.status === 'ACTIVE') ? 'rgba(22,163,74,0.4)' : 'rgba(2,132,199,0.4)'}`
                    }}>
                      {selectedRobot.status === 'OFFLINE' ? '⚪ OFFLINE (FMS)' : selectedRobot.status === 'CHARGING' ? '🔌 ĐANG SẠC (FMS)' : selectedRobot.status === 'ERROR' ? '🚨 BÁO LỖI (FMS)' : (selectedRobot.status === 'RUNNING' || selectedRobot.status === 'ACTIVE') ? '🟢 RUNNING (FMS)' : '🔵 IDLE (FMS)'}
                    </span>
                  </div>
                </div>

                {/* 1B. SPEEDOMETER GAUGE (Tốc độ - kim gạt từ góc phần 3 bên trái đối xứng) */}
                <div style={{ marginBottom: '14px' }}>
                  <MotorcycleSpeedometerGauge
                    velocity={selectedRobot.velocity}
                    maxSpeed={selectedRobot.max_speed || 2.0}
                    status={selectedRobot.status}
                    isDark={isDark}
                  />
                </div>

                {/* 2. Robot Telemetry (State & Battery) */}
                <div style={{ padding: '14px', background: 'var(--bg-elevated)', borderRadius: '10px', border: '1px solid var(--border)' }}>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', marginBottom: '10px' }}>
                    <div style={{ padding: '8px', background: 'var(--bg-card)', borderRadius: '6px' }}>
                      <div style={{ fontSize: '10px', color: 'var(--text-muted)' }}>Dung lượng Pin</div>
                      <div style={{ fontSize: '16px', fontWeight: 700, color: selectedRobot.battery > 50 ? '#22c55e' : selectedRobot.battery > 20 ? '#f59e0b' : '#ef4444' }}>
                        ⚡ {selectedRobot.battery}%
                      </div>
                    </div>
                    <div style={{ padding: '8px', background: 'var(--bg-card)', borderRadius: '6px' }}>
                      <div style={{ fontSize: '10px', color: 'var(--text-muted)' }}>Vận tốc FMS</div>
                      <div style={{ fontSize: '16px', fontWeight: 700, color: 'var(--text-primary)' }}>
                        {selectedRobot.velocity.toFixed(2)} m/s
                      </div>
                    </div>
                  </div>

                  {selectedRobot.destination && (
                    <div style={{ fontSize: '11px', color: 'var(--text-sub)', marginBottom: '6px' }}>
                      Điểm đích FMS: <b style={{ color: '#22c55e' }}>{selectedRobot.destination}</b>
                    </div>
                  )}

                  <div style={{ fontSize: '11px', color: 'var(--text-sub)', fontFamily: 'monospace', lineHeight: '1.6', marginTop: '4px' }}>
                    {selectedRobot.raw_fms ? (
                      <div>
                        Tọa độ FMS: <b style={{ color: '#38bdf8' }}>X: {selectedRobot.raw_fms.x.toFixed(2)} Y: {selectedRobot.raw_fms.y.toFixed(2)} θ: {selectedRobot.raw_fms.theta.toFixed(2)}</b>
                      </div>
                    ) : selectedRobot.fms_position ? (
                      <div>
                        Tọa độ FMS: <b style={{ color: '#38bdf8' }}>X: {(selectedRobot.fms_position[0] + 185.0).toFixed(2)} Y: {(194.5 + (18.0 - selectedRobot.fms_position[2])).toFixed(2)} θ: {(-selectedRobot.heading).toFixed(2)}</b>
                      </div>
                    ) : null}
                    <div>
                      Vị trí 3D Scene: <b>[{selectedRobot.position[0].toFixed(2)}, {selectedRobot.position[2].toFixed(2)}]</b>
                    </div>
                  </div>
                </div>
              </div>
            ) : selectedPerson ? (
              <div>
                <div style={{ padding: '14px', borderRadius: '10px', background: 'var(--bg-elevated)', border: '1px solid var(--border)', marginBottom: '14px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                    <h3 style={{ margin: 0, fontSize: '16px', fontWeight: 700, color: 'var(--text-primary)' }}>👤 {selectedPerson.id}</h3>
                    <span style={{ fontSize: '11px', fontWeight: 700, padding: '2px 8px', borderRadius: '5px', background: 'rgba(249,115,22,0.2)', color: '#fb923c' }}>
                      {selectedPerson.status}
                    </span>
                  </div>
                  <div style={{ fontSize: '12px', color: 'var(--text-sub)' }}>
                    Vùng: {selectedPerson.zone || 'Khu vực Chung'} · Camera: {selectedPerson.cam_id || 'Cam_01'}
                  </div>
                </div>

                <div style={{
                  padding: '14px',
                  borderRadius: '10px',
                  background: selectedPerson.safety_alert ? 'rgba(239,68,68,0.12)' : 'rgba(22,163,74,0.08)',
                  border: `1px solid ${selectedPerson.safety_alert ? 'rgba(239,68,68,0.4)' : 'rgba(22,163,74,0.3)'}`,
                  marginBottom: '14px',
                }}>
                  <div style={{ fontSize: '11px', fontWeight: 700, color: selectedPerson.safety_alert ? '#f87171' : '#22c55e', textTransform: 'uppercase', marginBottom: '6px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <ShieldAlert size={14} /> An Toàn Cự Ly (Vision AI)
                  </div>
                  <div style={{ fontSize: '13px', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '4px' }}>
                    {selectedPerson.safety_msg || (selectedPerson.safety_alert ? '⚠ Nguy cơ va chạm với Robot' : '✔ Khoảng cách an toàn')}
                  </div>
                  {selectedPerson.near_robot_id && (
                    <div style={{ fontSize: '12px', color: 'var(--text-sub)' }}>
                      Robot gần nhất: <b>{selectedPerson.near_robot_id}</b> ({selectedPerson.near_robot_dist}m)
                    </div>
                  )}
                </div>

                <div style={{ padding: '14px', background: 'var(--bg-elevated)', borderRadius: '10px', border: '1px solid var(--border)', marginBottom: '14px' }}>
                  <div style={{ fontSize: '12px', color: 'var(--text-sub)', marginBottom: '6px' }}>Vận tốc di chuyển</div>
                  <div style={{ fontSize: '22px', fontWeight: 700, color: '#fb923c', fontFamily: 'monospace' }}>
                    {selectedPerson.velocity.toFixed(2)} m/s
                  </div>
                </div>

                <div style={{ fontSize: '12px', color: 'var(--text-sub)', display: 'flex', flexDirection: 'column', gap: '6px' }}>
                  <div>Tọa độ mặt sàn: <b>X={selectedPerson.position[0].toFixed(2)}m, Z={selectedPerson.position[2].toFixed(2)}m</b></div>
                </div>
              </div>
            ) : selectedRack ? (
              <div>
                <div style={{ padding: '14px', borderRadius: '10px', background: 'var(--bg-elevated)', border: '1px solid var(--border)', marginBottom: '14px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                    <h3 style={{ margin: 0, fontSize: '16px', fontWeight: 700, color: 'var(--text-primary)' }}>📦 {selectedRack.id}</h3>
                    <span style={{ fontSize: '11px', fontWeight: 700, padding: '2px 8px', borderRadius: '5px', background: 'rgba(168,85,247,0.2)', color: '#c084fc' }}>
                      {selectedRack.status}
                    </span>
                  </div>
                </div>

                <div style={{ padding: '14px', background: 'var(--bg-elevated)', borderRadius: '10px', border: '1px solid var(--border)', marginBottom: '14px' }}>
                  <div style={{ fontSize: '11px', fontWeight: 700, color: '#c084fc', textTransform: 'uppercase', marginBottom: '8px' }}>
                    Đối Chiếu Vị Trí Kệ Hàng
                  </div>
                  <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '6px' }}>
                    {selectedRack.cross_check_msg || (selectedRack.status === 'CARRIED' ? `Đang di chuyển trên ${selectedRack.carried_by}` : 'Đặt tại ô lưu trữ')}
                  </div>
                  <div style={{ fontSize: '12px', color: 'var(--text-sub)' }}>
                    Tọa độ: <b>X={selectedRack.position[0].toFixed(2)}m, Z={selectedRack.position[2].toFixed(2)}m</b>
                  </div>
                </div>
              </div>
            ) : (
              <div style={{ padding: '30px 10px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '13px' }}>
                <Crosshair size={28} style={{ margin: '0 auto 10px', opacity: 0.5 }} />
                <p style={{ margin: 0 }}>Click vào Robot, Người hoặc Kệ trên bản đồ 3D để xem đối chiếu FMS & Vision AI.</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
