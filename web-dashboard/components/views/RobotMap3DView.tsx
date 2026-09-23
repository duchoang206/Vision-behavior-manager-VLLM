'use client';

import React, { useEffect, useRef, useState, useMemo, useCallback } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { RoundedBoxGeometry } from 'three/examples/jsm/geometries/RoundedBoxGeometry.js';
import { createFactoryFloor, createRobotTrail, disposeTwinObject, getFactoryView, updateRobotTrail } from '../../lib/factory-twin-scene';
import { useTab } from '../TabContext';
import styles from './RobotMap3DView.module.css';
import CameraProjectionPanel from './CameraProjectionPanel';
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

interface CalibratedCamera {
  camera_id: string;
  name: string;
  footprint: Point2[];
  anchor: Point3;
  anchor_kind: 'camera_pose' | 'coverage_center';
  physical_pose_known: boolean;
  method?: string;
  save_id?: string;
  yaw?: number | null;
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
  fms_position?: [number, number, number];
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
    
    const authoritative = item.fms_position?.every(Number.isFinite) ? item.fms_position : null;

    next[id] = {
      ...prevItem,
      ...item,
      position: authoritative ?? (shouldHoldPose ? prevItem.position : item.position),
      heading: shouldHoldPose && !authoritative ? prevItem.heading : item.heading,
      velocity: shouldHoldPose && !authoritative ? 0 : item.velocity,
      ui_last_seen: nowMs,
    } as T;
  });

  return next;
};

// ─── Main Component ─────────────────────────────────────────────────────────
export default function RobotMap3DView() {
  const { t } = useLanguage();
  const { activeTab } = useTab();
  const { isDark } = useAppTheme();
  const activeTabRef = useRef(activeTab);
  activeTabRef.current = activeTab;
  const [showFleet, setShowFleet] = useState(true);
  const [showInspector, setShowInspector] = useState(false);
  const [telemetryConnected, setTelemetryConnected] = useState(false);
  const lastTelemetryAt = useRef(0);

  // Multi-Entity Telemetry States (Populated with full FMS fleet by default)
  const [robots, setRobots] = useState<Record<string, RobotData>>(INITIAL_ROBOTS);
  const [persons, setPersons] = useState<Record<string, PersonData>>(INITIAL_PERSONS);
  const [racks, setRacks] = useState<Record<string, RackData>>(INITIAL_RACKS);
  const telemetryRef = useRef({ robots: INITIAL_ROBOTS, persons: INITIAL_PERSONS, racks: INITIAL_RACKS });

  // Active Selection & Filter States (Default to Robot_2001 so speedometer is immediately alive)
  const [selectedEntity, setSelectedEntity] = useState<{ type: 'robot' | 'person' | 'rack'; id: string } | null>({ type: 'robot', id: 'Robot_2001' });
  const [fleetTab, setFleetTab] = useState<'robots' | 'persons' | 'racks'>('robots');
  const [viewMode, setViewMode] = useState<'3D' | '2D'>('3D');
  const viewModeRef = useRef(viewMode);
  viewModeRef.current = viewMode;
  const [showCameraProjection, setShowCameraProjection] = useState(false);
  const cameraProjectionRef = useRef(false);
  cameraProjectionRef.current = showCameraProjection;
  const [projectionPoint, setProjectionPoint] = useState<Point2 | null>(null);
  const [followTarget, setFollowTarget] = useState<boolean>(false);
  const [showLabels] = useState<boolean>(true);

  const [kpi, setKpi] = useState<FleetKpi>({
    total: 0,
    active: 0,
    charging: 0,
    idle: 0,
    warning: 0,
    error: 0,
    persons_count: 0,
    racks_count: 0,
  });

  const [fmsMeta, setFmsMeta] = useState<FmsMeta>({
    server_ip: '',
    mqtt_connected: false,
    last_packet_time: 0,
    total_packets: 0,
    mode: 'CONNECTING',
  });

  const [layout, setLayout] = useState<FmsLayout | null>(null);
  const [calibratedCameras, setCalibratedCameras] = useState<CalibratedCamera[]>([]);
  const [packetRate, setPacketRate] = useState<number>(0);

  // References for Three.js
  const mountRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const layoutGroupRef = useRef<THREE.Group | null>(null);
  const entitiesGroupRef = useRef<THREE.Group | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const animFrameIdRef = useRef<number | null>(null);
  const trailsRef = useRef(new Map<string, ReturnType<typeof createRobotTrail>>());

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
    const { centerX, centerZ, span } = getFactoryView(layout);
    const distance = span * Math.max(1, 1.3 / cameraRef.current.aspect);
    cameraRef.current.position.set(centerX + distance * 0.53, distance * 0.72, centerZ + distance * 0.66);
    controlsRef.current.target.set(centerX, 0, centerZ);
    controlsRef.current.update();
    setFollowTarget(false);
  }, [layout]);

  useEffect(() => { handleResetCamera(); }, [handleResetCamera, viewMode]);

  useEffect(() => {
    let cancelled = false;
    let controller: AbortController | null = null;
    const pollStatus = async () => {
      controller = new AbortController();
      const timeout = setTimeout(() => controller?.abort(), 2500);
      try {
        const response = await fetch('/api/backend/fms/status', { cache: 'no-store', signal: controller.signal });
        if (!response.ok) throw new Error('FMS status unavailable');
        const status = await response.json();
        if (!cancelled) setFmsMeta({ ...status, server_ip: status.fms_ip });
      } catch {
        if (!cancelled) setFmsMeta(previous => ({ ...previous, mqtt_connected: false, mode: 'CONNECTING' }));
      } finally {
        clearTimeout(timeout);
      }
    };
    void pollStatus();
    const interval = setInterval(pollStatus, 3000);
    return () => { cancelled = true; controller?.abort(); clearInterval(interval); };
  }, []);

  useEffect(() => {
    if (!activeTab || activeTab !== 'robot_map') return;
    const controller = new AbortController();
    fetch('/api/backend/calibration/cameras', { signal: controller.signal, cache: 'no-store' })
      .then(response => response.ok ? response.json() : null)
      .then(data => { if (data?.cameras) setCalibratedCameras(data.cameras); })
      .catch(() => undefined);
    return () => controller.abort();
  }, [activeTab]);

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
    let disposed = false;
    let ws: WebSocket | null = null;
    let reconnectTimeout: NodeJS.Timeout | null = null;
    let rateInterval: NodeJS.Timeout | null = null;

    const connect = () => {
      if (disposed) return;
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const host = window.location.hostname || 'localhost';
      const wsUrl = `${protocol}//${host}:8000/ws/digital_twin`;

      ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        setTelemetryConnected(false);
      };

      const socket = ws;
      let lastMessageTimestamp = 0;
      let lastUiAt = 0;
      lastTelemetryAt.current = Date.now();
      ws.onmessage = (event) => {
        if (disposed || socket !== ws) return;
        try {
          const msg = JSON.parse(event.data);
          const twin = msg.type === 'DIGITAL_TWIN_SYNC' || msg.type === 'DIGITAL_TWIN_TELEMETRY';
          if (!twin && msg.type !== 'FULL' && msg.type !== 'PATCH') return;
          const nowMs = Date.now();
          const timestamp = Number(msg.timestamp || 0) * (Number(msg.timestamp || 0) < 1e11 ? 1000 : 1);
          if (timestamp && (timestamp < lastMessageTimestamp || nowMs - timestamp > 2000)) return;
          lastMessageTimestamp = timestamp;
          packetCountRef.current++;
          lastTelemetryAt.current = nowMs;
          const data = twin ? msg : msg.state || msg.patch || {};
          const current = telemetryRef.current;
          if (data.robots) current.robots = mergeStableEntities(current.robots, normalizeListOrMap<RobotData>(data.robots), nowMs, 2.5);
          if (data.persons) current.persons = mergeStableEntities(current.persons, normalizeListOrMap<PersonData>(data.persons), nowMs, 1.8);
          if (data.racks) current.racks = mergeStableEntities(current.racks, normalizeListOrMap<RackData>(data.racks), nowMs, 1.8);
          if (document.hidden || activeTabRef.current !== 'robot_map' || nowMs - lastUiAt < (viewModeRef.current === '3D' ? 200 : 60)) return;
          lastUiAt = nowMs;
          setTelemetryConnected(true);
          setRobots(current.robots);
          setPersons(current.persons);
          setRacks(current.racks);
          if (data.fms_meta) setFmsMeta(data.fms_meta);
          if (data.fleet_kpi) setKpi(previous => ({ ...previous,
            total: data.fleet_kpi.total_robots ?? Object.keys(current.robots).length,
            active: data.fleet_kpi.active_robots ?? 0,
            charging: data.fleet_kpi.charging_robots ?? 0,
            idle: data.fleet_kpi.idle_robots ?? 0,
            persons_count: data.fleet_kpi.total_persons ?? Object.keys(current.persons).length,
            racks_count: data.fleet_kpi.total_racks ?? Object.keys(current.racks).length,
          }));
          if (data.kpi?.fleet) setKpi(previous => ({ ...previous, ...data.kpi.fleet }));
        } catch {
          socket.close(1003, 'Invalid telemetry');
        }
      };
      ws.onerror = () => socket.close();
      ws.onclose = () => {
        if (socket !== ws) return;
        setTelemetryConnected(false);
        if (!disposed) reconnectTimeout = setTimeout(connect, 1500);
      };

      wsRef.current = ws;
    };

    connect();

    rateInterval = setInterval(() => {
      setPacketRate(packetCountRef.current);
      packetCountRef.current = 0;
      if (Date.now() - lastTelemetryAt.current > 3000) {
        setTelemetryConnected(false);
        if (ws?.readyState === WebSocket.OPEN || ws?.readyState === WebSocket.CONNECTING) ws.close(4000, 'Telemetry stalled');
      }
    }, 1000);

    return () => {
      disposed = true;
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
    deltaM?: number | null,
    velocity = 0
  ) => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const accent = status === 'OFFLINE' ? '#78939a' : status === 'CHARGING' ? '#f6c76a' : status === 'ERROR' ? '#fb8d7b' : '#65eadd';
    ctx.fillStyle = 'rgba(5, 23, 29, 0.94)';
    ctx.strokeStyle = accent;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.roundRect(3, 3, canvas.width - 6, canvas.height - 6, 8);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = accent;
    ctx.fillRect(3, 15, 3, 26);
    ctx.fillStyle = '#e3f5f3';
    ctx.font = '600 22px sans-serif';
    ctx.fillText(id, 18, 32);
    ctx.fillStyle = '#8fb8bb';
    ctx.font = '16px monospace';
    ctx.fillText(`BAT ${battery}%  |  ${Math.max(0, velocity).toFixed(2)} m/s`, 18, 57);
    ctx.fillStyle = accent;
    ctx.font = '12px monospace';
    ctx.fillText(`FMS / ${status}`, 18, 77);
  };

  const createRobotLabelSprite = (
    id: string,
    battery: number,
    status: string,
    carriedRack?: string | null,
    crossStatus?: string,
    deltaM?: number | null,
    velocity = 0
  ) => {
    const canvas = document.createElement('canvas');
    canvas.width = 280;
    canvas.height = 88;
    const ctx = canvas.getContext('2d')!;
    updateRobotLabelCanvas(canvas, ctx, id, battery, status, carriedRack, crossStatus, deltaM, velocity);

    const texture = new THREE.CanvasTexture(canvas);
    texture.minFilter = THREE.LinearFilter;
    texture.magFilter = THREE.LinearFilter;
    texture.needsUpdate = true;
    const mat = new THREE.SpriteMaterial({ map: texture, transparent: true, depthWrite: false });
    const sprite = new THREE.Sprite(mat);
    sprite.scale.set(2.1, 0.66, 1);
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
    const { centerX, centerZ, span: gridSize } = getFactoryView(layout);

    // 1. Scene & Camera
    const scene = new THREE.Scene();
    scene.background = new THREE.Color('#061117');
    scene.fog = new THREE.Fog('#061117', gridSize * 2, gridSize * 5);
    sceneRef.current = scene;

    // Dedicated groups for layout and dynamic entities
    const layoutGroup = new THREE.Group();
    scene.add(layoutGroup);
    layoutGroupRef.current = layoutGroup;

    const entitiesGroup = new THREE.Group();
    scene.add(entitiesGroup);
    entitiesGroupRef.current = entitiesGroup;

    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    const distance = gridSize * Math.max(1, 1.3 / camera.aspect);
    camera.position.set(centerX + distance * 0.53, distance * 0.72, centerZ + distance * 0.66);
    cameraRef.current = camera;

    // 2. Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
    renderer.shadowMap.enabled = false;
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
    const ambientLight = new THREE.HemisphereLight(0x91cfd6, 0x07151d, 1.25);
    scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0xb7e8eb, 1.55);
    dirLight.position.set(centerX + 12, 24, centerZ + 12);
    dirLight.castShadow = false;
    scene.add(dirLight);


    // 6. Raycaster for Interactive Entity Selection
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();

    const onPointerDown = (event: MouseEvent) => {
      const rect = renderer.domElement.getBoundingClientRect();
      mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

      raycaster.setFromCamera(mouse, camera);
      if (event.shiftKey && cameraProjectionRef.current) {
        const point = raycaster.ray.intersectPlane(new THREE.Plane(new THREE.Vector3(0, 1, 0), 0), new THREE.Vector3());
        if (point) setProjectionPoint([point.x, point.z]);
        return;
      }

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
          setShowInspector(true);
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
    let lastRenderAt = 0;
    const animate = (timestamp = 0) => {
      animFrameIdRef.current = requestAnimationFrame(animate);
      if (document.hidden || activeTabRef.current !== 'robot_map' || mountEl.clientWidth < 10) return;
      const elapsed = Math.min(0.1, (timestamp - lastRenderAt) / 1000);
      lastRenderAt = timestamp;
      const smoothing = 1 - Math.exp(-18 * elapsed);
      controls.update();

      // A. Robots Animation
      robotMeshesRef.current.forEach((meshData, rid) => {
        const latest = telemetryRef.current.robots[rid];
        if (latest) {
          meshData.targetPos.set(latest.position[0], 0, latest.position[2]);
          meshData.targetHeading = -latest.heading;
        }
        const { group, targetPos, targetHeading, pulseRing } = meshData;
        group.position.x += (targetPos.x - group.position.x) * smoothing;
        group.position.z += (targetPos.z - group.position.z) * smoothing;

        let deltaHeading = targetHeading - group.rotation.y;
        while (deltaHeading > Math.PI) deltaHeading -= 2 * Math.PI;
        while (deltaHeading < -Math.PI) deltaHeading += 2 * Math.PI;
        group.rotation.y += deltaHeading * smoothing;

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
        const latest = telemetryRef.current.persons[pid];
        if (latest) {
          meshData.targetPos.set(latest.position[0], 0, latest.position[2]);
          meshData.targetHeading = -latest.heading;
        }
        const { group, targetPos, targetHeading } = meshData;
        group.position.x += (targetPos.x - group.position.x) * smoothing;
        group.position.z += (targetPos.z - group.position.z) * smoothing;

        let deltaHeading = targetHeading - group.rotation.y;
        while (deltaHeading > Math.PI) deltaHeading -= 2 * Math.PI;
        while (deltaHeading < -Math.PI) deltaHeading += 2 * Math.PI;
        group.rotation.y += deltaHeading * smoothing;

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
      rackMeshesRef.current.forEach((meshData, identifier) => {
        const latest = telemetryRef.current.racks[identifier];
        if (latest) meshData.targetPos.set(latest.position[0], 0, latest.position[2]);
        const { group, targetPos } = meshData;
        group.position.x += (targetPos.x - group.position.x) * smoothing;
        group.position.y += (targetPos.y - group.position.y) * smoothing;
        group.position.z += (targetPos.z - group.position.z) * smoothing;
      });

      renderer.render(scene, camera);
    };

    animate();

    return () => {
      if (animFrameIdRef.current) cancelAnimationFrame(animFrameIdRef.current);
      resizeObserver.disconnect();
      controls.dispose();
      disposeTwinObject(entitiesGroup);
      trailsRef.current.forEach(trail => { scene.remove(trail.line); disposeTwinObject(trail.line); });
      trailsRef.current.clear();
      renderer.domElement.removeEventListener('pointerdown', onPointerDown);
      if (renderer.domElement.parentElement === mountEl) {
        mountEl.removeChild(renderer.domElement);
      }
      renderer.dispose();
      rendererRef.current = null;
      sceneRef.current = null;
      cameraRef.current = null;
      controlsRef.current = null;
      robotMeshesRef.current.clear();
      personMeshesRef.current.clear();
      rackMeshesRef.current.clear();
      layoutGroupRef.current = null;
      entitiesGroupRef.current = null;
    };
  }, [viewMode]);

  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene) return;
    const background = isDark ? '#061117' : '#eaf1f4';
    scene.background = new THREE.Color(background);
    if (scene.fog) scene.fog.color.set(background);
    scene.children.forEach(object => {
      if (object instanceof THREE.HemisphereLight) {
        object.color.set(isDark ? 0x91cfd6 : 0xffffff);
        object.groundColor.set(isDark ? 0x07151d : 0xb3c4c8);
      }
    });
  }, [isDark, viewMode]);

  useEffect(() => {
    const parent = layoutGroupRef.current;
    if (viewMode !== '3D' || !parent) return;
    const floor = createFactoryFloor(layout, isDark);
    parent.add(floor);
    return () => {
      parent.remove(floor);
      disposeTwinObject(floor);
    };
  }, [layout, viewMode, isDark]);

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
        const lowProfileRobot = /28100|6969/i.test(r.id);
        const forkliftRobot = /2001/i.test(r.id);

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
        const lowerGeo = new RoundedBoxGeometry(0.96, 0.12, 0.68, 3, 0.07);
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
        const bodyColor = isOffline ? 0x475569 : isCharging ? 0xcaa56c : r.status === 'ERROR' ? 0xdd6c62 : lowProfileRobot ? 0xe97828 : forkliftRobot ? 0xe2e8f0 : r.status === 'IDLE' ? 0x338b9c : 0x4ab9b4;
        const beaconColor = isOffline ? 0x64748b : isCharging ? 0xf59e0b : 0x4cebdd;
        const beaconEmissive = isOffline ? 0x000000 : isCharging ? 0xf59e0b : 0x4cebdd;
        const bodyGeo = new RoundedBoxGeometry(0.92, 0.16, 0.64, 3, 0.08);
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
        const topGeo = new RoundedBoxGeometry(0.78, 0.05, 0.54, 3, 0.035);
        const topMat = new THREE.MeshStandardMaterial({
          color: 0x1e293b,
          metalness: 0.6,
          roughness: 0.35,
        });
        const topShell = new THREE.Mesh(topGeo, topMat);
        topShell.position.y = 0.32;
        group.add(topShell);

        // D. Front Black Sensor Visor (Sleek dark glass visor matching FMS Image 2)
        const visorGeo = new RoundedBoxGeometry(0.10, 0.12, 0.56, 3, 0.025);
        const visorMat = new THREE.MeshStandardMaterial({
          color: 0x020617,
          metalness: 0.95,
          roughness: 0.08,
        });
        const visor = new THREE.Mesh(visorGeo, visorMat);
        visor.position.set(0.44, 0.22, 0);
        group.add(visor);

        if (forkliftRobot) {
          const forkMaterial = new THREE.MeshStandardMaterial({ color: 0xb9c4cf, metalness: 0.82, roughness: 0.22 });
          const forkRail = new THREE.Mesh(new RoundedBoxGeometry(0.56, 0.045, 0.055, 2, 0.018), forkMaterial);
          forkRail.position.set(0.67, 0.16, -0.17);
          const forkRailTwo = forkRail.clone();
          forkRailTwo.position.z = 0.17;
          group.add(forkRail, forkRailTwo);
        }
        if (lowProfileRobot) {
          const bumperMaterial = new THREE.MeshStandardMaterial({ color: 0xff8a34, emissive: 0x321304, emissiveIntensity: 0.25, metalness: 0.35, roughness: 0.3 });
          const bumper = new THREE.Mesh(new RoundedBoxGeometry(0.90, 0.045, 0.60, 3, 0.02), bumperMaterial);
          bumper.position.set(0, 0.31, 0);
          group.add(bumper);
        }

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
          const lightMesh = new THREE.Mesh(new RoundedBoxGeometry(0.04, 0.04, 0.08, 2, 0.012), headlightMat);
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
          const tailMesh = new THREE.Mesh(new RoundedBoxGeometry(0.03, 0.04, 0.08, 2, 0.009), tailLightMat);
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
        const labelKey = `${r.id}:${r.battery}:${r.status}:${r.carried_rack_id}:${r.cross_check_status}:${r.delta_distance_m}:${r.velocity}`;
        const { sprite: labelSprite, canvas: labelCanvas, ctx: labelCtx, texture: labelTexture } = createRobotLabelSprite(
          r.id, r.battery, r.status, r.carried_rack_id, r.cross_check_status, r.delta_distance_m, r.velocity
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

      let trail = trailsRef.current.get(rid);
      if (!trail && sceneRef.current) {
        trail = createRobotTrail();
        sceneRef.current.add(trail.line);
        trailsRef.current.set(rid, trail);
      }
      if (trail) updateRobotTrail(trail, r.position, Date.now(), r.status === 'CHARGING' ? 0xefc06e : 0x48d9ce);
      
      const isOffline = r.status === 'OFFLINE';
      const isCharging = r.status === 'CHARGING';
      const lowProfileRobot = /28100|6969/i.test(r.id);
      const forkliftRobot = /2001/i.test(r.id);
      const bodyColor = isOffline ? 0x475569 : isCharging ? 0xcaa56c : r.status === 'ERROR' ? 0xdd6c62 : lowProfileRobot ? 0xe97828 : forkliftRobot ? 0xe2e8f0 : r.status === 'IDLE' ? 0x338b9c : 0x4ab9b4;
      const beaconColor = isOffline ? 0x64748b : isCharging ? 0xf59e0b : 0x4cebdd;
      const beaconEmissive = isOffline ? 0x000000 : isCharging ? 0xf59e0b : 0x4cebdd;
      (meshData.bodyMesh.material as THREE.MeshStandardMaterial).color.set(bodyColor);
      (meshData.beaconMesh.material as THREE.MeshStandardMaterial).color.set(beaconColor);
      (meshData.beaconMesh.material as THREE.MeshStandardMaterial).emissive.set(beaconEmissive);
      (meshData.beaconMesh.material as THREE.MeshStandardMaterial).emissiveIntensity = isOffline ? 0.0 : 1.5;
      
      meshData.rackMountGroup.visible = Boolean(r.has_rack || r.carried_rack_id);
      meshData.labelSprite.visible = showLabels;
      
      const labelKey = `${r.id}:${r.battery}:${r.status}:${r.carried_rack_id}:${r.cross_check_status}:${r.delta_distance_m}:${r.velocity}`;
      if (meshData.lastLabelKey !== labelKey) {
        meshData.lastLabelKey = labelKey;
        updateRobotLabelCanvas(meshData.labelCanvas, meshData.labelCtx, r.id, r.battery, r.status, r.carried_rack_id, r.cross_check_status, r.delta_distance_m, r.velocity);
        meshData.labelTexture.needsUpdate = true;
      }
    });

    robotMeshesRef.current.forEach((meshData, rid) => {
      if (!robots[rid]) {
        disposeTwinObject(meshData.group);
        const trail = trailsRef.current.get(rid);
        if (trail) { sceneRef.current?.remove(trail.line); disposeTwinObject(trail.line); trailsRef.current.delete(rid); }
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
        disposeTwinObject(meshData.group);
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
        disposeTwinObject(meshData.group);
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
  const fmsLive = telemetryConnected && fmsMeta.mqtt_connected && fmsMeta.mode === 'LIVE';
  const alertCount = Object.values(robots).filter(robot => robot.status === 'ERROR' || robot.cross_check_status === 'DEVIATED').length;

  return (
    <div className={styles.root} data-theme={isDark ? 'dark' : 'light'}>
      <header className={styles.header}>
        <div className={styles.brand}>
          <div className={styles.brandIcon}><Boxes size={22} /></div>
          <div>
            <div className={styles.eyebrow}>R-SKYVIEW / RTC TECHNOLOGY</div>
            <h2 className={styles.title}>REAL-TIME FACTORY FLOOR TWIN</h2>
          </div>
        </div>
        <div className={styles.statusRow}>
          <span className={styles.connection + (fmsLive ? '' : ' ' + styles.disconnected)}>
            <i />{fmsLive ? 'FMS LIVE' : 'FMS · CHỜ KẾT NỐI'}
          </span>
          <span>{Object.keys(robots).length} robots · {packetRate} Hz</span>
          {alertCount > 0 && <span className={styles.alert}><AlertTriangle size={14} />{alertCount} cảnh báo</span>}
          <span>{layout?.slam_map?.map_name || 'FMS'} / MÉT</span>
        </div>
      </header>

      {/* ── MAIN WORKSPACE ── */}
      <div className={styles.workspace}>
        <div className={styles.toolbar}>
          <div className={styles.toolbarGroup}>
            <button aria-pressed={viewMode === '3D'} onClick={() => setViewMode('3D')}><Layers size={14} />3D Twin</button>
            <button aria-pressed={viewMode === '2D'} onClick={() => setViewMode('2D')}><MapPin size={14} />2D FMS</button>
            <button title="Góc nhìn tổng thể" aria-label="Góc nhìn tổng thể" disabled={viewMode !== '3D'} onClick={handleResetCamera}><RotateCcw size={14} /></button>
          </div>
          <div className={styles.toolbarGroup}>
            <button aria-pressed={showFleet && !showCameraProjection} onClick={() => { setShowFleet(showCameraProjection || !showFleet); setShowCameraProjection(false); }}><Bot size={14} />Đội xe</button>
            <button aria-pressed={showInspector} onClick={() => setShowInspector(previous => !previous)}><Activity size={14} />Chi tiết</button>
            <button aria-pressed={showCameraProjection} onClick={() => setShowCameraProjection(previous => !previous)}><Crosshair size={14} />Camera ↔ FMS ({calibratedCameras.length})</button>
            <button aria-pressed={followTarget} disabled={viewMode !== '3D' || !selectedEntity} title="Theo đối tượng đã chọn" onClick={() => setFollowTarget(previous => !previous)}><Crosshair size={14} /></button>
          </div>
        </div>
        {/* ── LEFT PANEL: MULTI-ENTITY FLEET SELECTOR ── */}
        <div className={styles.fleet} style={{
          width: '320px',
          background: 'var(--bg-card)',
          borderRight: '1px solid var(--border)',
          display: showFleet && !showCameraProjection ? 'flex' : 'none',
          flexDirection: 'column',
          zIndex: 5,
        }}>
          <div className={styles.fleetHeading}><span>FLEET OVERVIEW</span><strong>{Object.keys(robots).length.toString().padStart(2, '0')} UNITS</strong></div>
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
        <div className={styles.viewport} style={{ flex: 1, position: 'relative', height: '100%', overflow: 'hidden' }}>
          {showCameraProjection && <CameraProjectionPanel cameras={calibratedCameras} floorPoint={projectionPoint} onClose={() => setShowCameraProjection(false)} />}
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
                    fill={isDark ? "#15504d" : "#bfe4df"}
                    fillOpacity={isDark ? 0.75 : 0.85}
                    stroke={isDark ? '#4cebdd' : '#008e8b'}
                    strokeWidth="0.025"
                    style={{ filter: 'drop-shadow(0 0 0.08px #20c4b5)' }}
                  />
                ))}

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
          <div className={styles.legend}>
            <span><i />Đường FMS</span>
            <span><i className={styles.amber} />Trạm sạc</span>
            <span><i className={styles.trail} />Vệt di chuyển thực</span>
            <span>Lưới {layout?.grid?.cell_size ?? 0.5} m</span>
          </div>
          <div className={styles.mapStamp}><strong>{layout?.name || 'FMS FLOOR'}</strong>FMS COORDINATES · LIVE TELEMETRY</div>
          {!telemetryConnected && <div className={styles.empty}>Đang kết nối dữ liệu realtime… Không mô phỏng vị trí robot.</div>}
        </div>

        {/* ── RIGHT PANEL: SELECTED ENTITY TELEMETRY & INSPECTOR ── */}
        <div className={styles.inspector} style={{
          width: '360px',
          background: 'var(--bg-card)',
          borderLeft: '1px solid var(--border)',
          display: showInspector ? 'flex' : 'none',
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
                        Tọa độ scene FMS: <b style={{ color: '#38bdf8' }}>X: {selectedRobot.fms_position[0].toFixed(2)} Z: {selectedRobot.fms_position[2].toFixed(2)}</b>
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
