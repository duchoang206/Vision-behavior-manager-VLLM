'use client';

import React, { useEffect, useState, useRef, useCallback } from 'react';
import { useLanguage } from '../LanguageContext';
import { useCameras, Camera } from '../CameraContext';
import { useAppTheme } from '../ThemeContext';
import { RegisteredMask, validRegisteredMask } from '../../lib/registered-mask';
import { Activity, ShieldAlert, Users, Layers, AlertCircle, ArrowRightLeft, Radio, Boxes, Package, CheckCircle2, Grid3X3, Bot, Tag, Sparkles, Trash2, X, Check, Search, Sliders } from 'lucide-react';


type TrackedObject = {
  id: number;
  local_id?: number;
  class: string;
  label?: string;
  mask?: RegisteredMask | null;
  x: number; // 0..1
  y: number; // 0..1
  w: number; // 0..1
  h: number; // 0..1
  floor_x?: number;
  floor_y?: number;
  confidence?: number;
  fms_status?: string;
  fms_battery?: number;
  fms_speed?: number;
  carried_rack?: string | null;
  cross_check?: string;
  delta_distance_m?: number;
  fms_pos?: [number, number, number];
  keypoints?: number[][];
  posture?: string;
  fall_detected?: boolean;
  fall_event?: boolean;
  world_position?: [number | null, number, number | null];
  spatial_valid?: boolean;
};

type LiveAlert = {
  cam_id: string;
  global_id: number;
  rule_type: string;
  severity: string;
  description: string;
  timestamp: number;
};

type InterpolatedTrack = {
  id: number;
  class?: string;
  label?: string;
  mask?: RegisteredMask | null;
  curX: number;
  curY: number;
  curW: number;
  curH: number;
  targetX: number;
  targetY: number;
  targetW: number;
  targetH: number;
  floorX: number;
  floorY: number;
  lastUpdated: number;
  fms_status?: string;
  fms_battery?: number;
  fms_speed?: number;
  carried_rack?: string | null;
  cross_check?: string;
  delta_distance_m?: number;
  fms_pos?: [number, number, number];
  keypoints?: number[][];
  posture?: string;
  fall_detected?: boolean;
  world_position?: [number | null, number, number | null];
  spatial_valid?: boolean;
};

type InterpolatedFloorTrack = {
  id: number;
  class?: string;
  curFx: number;
  curFy: number;
  targetFx: number;
  targetFy: number;
  cam: string;
  lastUpdated: number;
};

type ROIState = {
  roi_id: string;
  name: string;
  status: 'OCCUPIED' | 'CARFULL' | 'EMPTY';
  overlap_ratio: number;
  occupant_ids: (number | string)[];
  occupant_labels?: string[];
  polygon: number[][];
  fms_polygon?: number[][];
  rule_type: string;
  coordinate_space?: string;
  threshold?: number;
  last_updated?: number;
};

type CameraRuleResponse = {
  id: string;
  name?: string;
  type?: string;
  rule_type?: string;
  points?: number[][];
  camera_points?: number[][];
  fms_points?: number[][];
  coordinate_space?: string;
  threshold?: number;
};

type RegisteredTargetItem = {
  label: string;
  cam_id?: string;
  class_name?: string;
  category?: string;
  last_cam?: string;
  samples_count?: number;
};

type MapOverviewItem = Record<string, unknown>;

type RegistryTargetsResponse = {
  targets?: RegisteredTargetItem[];
};

const resolveStreamCamId = (rawCamId: string, index: number, cameras: Camera[]) => {
  if (cameras.some(c => c.id === rawCamId)) return rawCamId;
  const indexedAliases = new Set([String(index), `cam_${index}`, `source_${index}`, `camera_${index}`]);
  if (indexedAliases.has(String(rawCamId)) && cameras[index]) return cameras[index].id;
  if (cameras.length === 1) return cameras[0].id;
  return rawCamId;
};

const isPersonClass = (className?: string) => (className || '').toLowerCase().includes('person');
const isRobotClass = (className?: string) => {
  const c = (className || '').toLowerCase();
  return c.includes('robot') || c.includes('agv') || c.includes('amr') || c.includes('rack');
};
const hasRegisteredLabel = (label?: string) => Boolean(label && label.trim() !== '');
const shouldShowTrackIdentity = (className?: string, label?: string) =>
  isPersonClass(className) || (hasRegisteredLabel(label) && isRobotClass(className));
const isValidPointPolygon = (points?: number[][]) =>
  Array.isArray(points) && points.length >= 3 && points.every(p => Array.isArray(p) && p.length >= 2 && Number.isFinite(p[0]) && Number.isFinite(p[1]));
const isValidNormPolygon = (points?: number[][]) =>
  isValidPointPolygon(points) && points!.every(p => p[0] >= 0 && p[0] <= 1 && p[1] >= 0 && p[1] <= 1);
const isStorageSlotRoi = (roi: ROIState) =>
  (roi.rule_type || '').toLowerCase() === 'occupancy'
  && isValidNormPolygon(roi.polygon)
  && isValidPointPolygon(roi.fms_polygon);
const isCameraDrawableRoi = (roi: ROIState) =>
  (roi.coordinate_space || 'camera') !== 'fms'
  && isValidNormPolygon(roi.polygon)
  && ((roi.rule_type || '').toLowerCase() !== 'occupancy' || isStorageSlotRoi(roi));
const filterRealConfiguredRois = (rois: ROIState[]) =>
  rois.filter(roi => (roi.rule_type || '').toLowerCase() !== 'occupancy' || isStorageSlotRoi(roi));
const TRACK_HOLD_MS = 750;
const TRACK_FADE_START_MS = 250;

const mergeRois = (existing: ROIState[] = [], incoming: ROIState[] = []) => {
  if (incoming.length === 0) return [];
  const byId = new Map<string, ROIState>();
  existing.forEach(roi => byId.set(roi.roi_id, roi));
  incoming.forEach(roi => {
    const prev = byId.get(roi.roi_id);
    byId.set(roi.roi_id, {
      ...prev,
      ...roi,
      polygon: roi.polygon && roi.polygon.length > 0 ? roi.polygon : (prev?.polygon || []),
      fms_polygon: roi.fms_polygon && roi.fms_polygon.length > 0 ? roi.fms_polygon : (prev?.fms_polygon || []),
      coordinate_space: roi.coordinate_space || prev?.coordinate_space || 'camera',
      occupant_labels: roi.occupant_labels || prev?.occupant_labels || [],
      name: roi.name || prev?.name || roi.roi_id,
      rule_type: roi.rule_type || prev?.rule_type || 'roi',
    });
  });
  return filterRealConfiguredRois(Array.from(byId.values()));
};

function CameraStreamCard({
  cam,
  hostName,
  isVisible,
  activeTab,
  metadataMap,
  roisMap,
  metadataCount,
  metadataConnected
}: {
  cam: Camera;
  hostName: string;
  isVisible: boolean;
  activeTab: string;
  metadataMap: React.MutableRefObject<Map<string, Map<number, InterpolatedTrack>>>;
  roisMap: React.MutableRefObject<Map<string, ROIState[]>>;
  metadataCount: number;
  metadataConnected: boolean;
}) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [useIframeFallback, setUseIframeFallback] = useState(false);
  const pcRef = useRef<RTCPeerConnection | null>(null);

  // 1. Ultra-Low-Latency Direct WHEP WebRTC Connection
	  useEffect(() => {
	    let isCancelled = false;
	    let reconnectTimeout: ReturnType<typeof setTimeout> | null = null;

    const connectWHEP = async () => {
      if (pcRef.current) {
        try { pcRef.current.close(); } catch (e) {}
        pcRef.current = null;
      }

      try {
        const pc = new RTCPeerConnection({
          iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
        });
        pcRef.current = pc;

        pc.addTransceiver('video', { direction: 'recvonly' });
        pc.addTransceiver('audio', { direction: 'recvonly' });

        pc.ontrack = (event) => {
          if (videoRef.current && event.streams[0]) {
            videoRef.current.srcObject = event.streams[0];
            videoRef.current.play().catch(() => {});
            setIsPlaying(true);
          }
        };

        pc.oniceconnectionstatechange = () => {
          if (pc.iceConnectionState === 'disconnected' || pc.iceConnectionState === 'failed') {
            setIsPlaying(false);
	            if (!isCancelled) {
	              if (reconnectTimeout) clearTimeout(reconnectTimeout);
	              reconnectTimeout = setTimeout(connectWHEP, 1500);
	            }
          }
        };

        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);

        // Fetch WHEP Answer from MediaMTX
        const whepUrl = `http://${hostName}:8081/${cam.id}/whep`;
        const res = await fetch(whepUrl, {
          method: 'POST',
          headers: { 'Content-Type': 'application/sdp' },
          body: offer.sdp
        });

        if (!res.ok) {
          throw new Error(`WHEP HTTP ${res.status}`);
        }

        const answerSdp = await res.text();
        if (!isCancelled && pc.signalingState !== 'closed') {
          await pc.setRemoteDescription(new RTCSessionDescription({ type: 'answer', sdp: answerSdp }));
        }
      } catch (err) {
        if (!isCancelled) {
          // Fallback to clean iframe if direct WHEP endpoint fails
	          setUseIframeFallback(true);
	          if (reconnectTimeout) clearTimeout(reconnectTimeout);
	          reconnectTimeout = setTimeout(connectWHEP, 4000);
	        }
      }
    };

    connectWHEP();

	    return () => {
	      isCancelled = true;
	      if (reconnectTimeout) clearTimeout(reconnectTimeout);
      if (pcRef.current) {
        try { pcRef.current.close(); } catch (e) {}
        pcRef.current = null;
      }
    };
  }, [cam.id, hostName]);

  // 2. 60 FPS Real-time Tracking & ROI Canvas Render
  useEffect(() => {
    let animId: number;

    const render = () => {
      const canvas = canvasRef.current;
      const container = containerRef.current;
      if (canvas && container) {
        const rect = container.getBoundingClientRect();
        const rw = Math.round(rect.width);
        const rh = Math.round(rect.height);
        if (rw > 10 && rh > 10 && (canvas.width !== rw || canvas.height !== rh)) {
          canvas.width = rw;
          canvas.height = rh;
        }

        const ctx = canvas.getContext('2d');
        if (ctx && canvas.width > 0 && canvas.height > 0) {
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          const video = videoRef.current;
          const sourceW = video?.videoWidth || 16;
          const sourceH = video?.videoHeight || 9;
          const sourceAspect = sourceW / Math.max(1, sourceH);
          const canvasAspect = canvas.width / Math.max(1, canvas.height);
          let videoDrawW = canvas.width;
          let videoDrawH = canvas.height;
          let videoOffsetX = 0;
          let videoOffsetY = 0;
          if (sourceAspect > canvasAspect) {
            videoDrawW = canvas.width;
            videoDrawH = canvas.width / sourceAspect;
            videoOffsetY = (canvas.height - videoDrawH) / 2;
          } else {
            videoDrawH = canvas.height;
            videoDrawW = canvas.height * sourceAspect;
            videoOffsetX = (canvas.width - videoDrawW) / 2;
          }
          const mapVideoX = (x: number) => videoOffsetX + x * videoDrawW;
          const mapVideoY = (y: number) => videoOffsetY + y * videoDrawH;

          // --- A. Tracked Objects (BBox + Tags only, no ROI overlays on stream) ---
          const camTracks = metadataMap.current.get(cam.id);
          const now = Date.now();

          if (camTracks) {
            camTracks.forEach((track, id) => {
              const age = now - track.lastUpdated;
              if (age > TRACK_HOLD_MS) {
                camTracks.delete(id);
                return;
              }
              const alpha = age <= TRACK_FADE_START_MS
                ? 1
                : Math.max(0.35, 1 - ((age - TRACK_FADE_START_MS) / (TRACK_HOLD_MS - TRACK_FADE_START_MS)));

              track.curX = track.targetX;
              track.curY = track.targetY;
              track.curW = track.targetW;
              track.curH = track.targetH;

              const px = mapVideoX(track.curX);
              const py = mapVideoY(track.curY);
              const pw = track.curW * videoDrawW;
              const ph = track.curH * videoDrawH;

              // Class & label formatting
              const rawClass = (track.label || track.class || 'Object').toLowerCase();
              const isRobot = rawClass.includes('robot');
              const isRack = rawClass.includes('rack');
              const isPerson = rawClass.includes('person') || rawClass.includes('human') || rawClass.includes('worker');
              const isFallen = Boolean(track.fall_detected || track.posture === 'fallen');
              const hasCustomLabel = hasRegisteredLabel(track.label);
              const drawMask = !isPerson && hasCustomLabel && validRegisteredMask(track.mask)
                && (!track.mask.observed_at || now - track.mask.observed_at < 500);
              if (!shouldShowTrackIdentity(track.class, track.label)) {
                return;
              }
              const displayClass = isRobot ? 'robot' : (isRack ? 'rack' : rawClass);
              let label = hasCustomLabel ? `🎯 ${track.label}` : `${displayClass} #${track.id}`;
              if (hasCustomLabel && (track.fms_battery !== undefined || track.fms_status)) {
                const batStr = track.fms_battery !== undefined ? `${track.fms_battery}%` : '';
                const statStr = track.fms_status ? ` · ${track.fms_status}` : '';
                const spdStr = (track.fms_speed !== undefined && track.fms_speed > 0.05) ? ` · ${track.fms_speed.toFixed(1)}m/s` : '';
                label = `🎯 ${track.label} (${batStr}${statStr}${spdStr})`;
              }

              // Dynamic Accent Colors (Cyan for verified registered targets, Rose for robots, Indigo for racks)
              const strokeColor = isFallen ? '#ef4444' : hasCustomLabel ? '#22d3ee' : (isRobot ? '#f43f5e' : isPerson ? '#22c55e' : '#6366f1');
              const fillColor = isFallen ? 'rgba(239, 68, 68, 0.28)' : hasCustomLabel ? 'rgba(34, 211, 238, 0.22)' : (isRobot ? 'rgba(244, 63, 94, 0.20)' : isPerson ? 'rgba(34, 197, 94, 0.15)' : 'rgba(99, 102, 241, 0.15)');

              ctx.save();
              ctx.globalAlpha = alpha;
              ctx.strokeStyle = strokeColor;
              ctx.lineWidth = hasCustomLabel ? 2.5 : 1.5;
              ctx.shadowColor = strokeColor;
              ctx.shadowBlur = hasCustomLabel ? 14 : 8;
              if (drawMask && track.mask) {
                ctx.beginPath();
                track.mask.polygons.forEach(ring => {
                  ctx.moveTo(mapVideoX(ring[0][0]), mapVideoY(ring[0][1]));
                  ring.slice(1).forEach(point => ctx.lineTo(mapVideoX(point[0]), mapVideoY(point[1])));
                  ctx.closePath();
                });
                ctx.fillStyle = fillColor;
                ctx.fill('evenodd');
                ctx.stroke();
              } else {
                ctx.strokeRect(px, py, pw, ph);
              }
              ctx.shadowBlur = 0;

              if (isPerson && Array.isArray(track.keypoints) && track.keypoints.length > 0) {
                const skeleton: [number, number][] = [[5, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16]];
                ctx.strokeStyle = strokeColor;
                ctx.lineWidth = 1.5;
                skeleton.forEach(([first, second]) => {
                  const firstPoint = track.keypoints?.[first];
                  const secondPoint = track.keypoints?.[second];
                  if (!firstPoint || !secondPoint || (firstPoint[2] ?? 0) < 0.35 || (secondPoint[2] ?? 0) < 0.35) return;
                  ctx.beginPath();
                  ctx.moveTo(mapVideoX(firstPoint[0]), mapVideoY(firstPoint[1]));
                  ctx.lineTo(mapVideoX(secondPoint[0]), mapVideoY(secondPoint[1]));
                  ctx.stroke();
                });
                ctx.fillStyle = strokeColor;
                track.keypoints.forEach((point) => {
                  if (!point || (point[2] ?? 0) < 0.35) return;
                  ctx.beginPath();
                  ctx.arc(mapVideoX(point[0]), mapVideoY(point[1]), 3, 0, Math.PI * 2);
                  ctx.fill();
                });
              }

              ctx.fillStyle = fillColor;
              if (!drawMask) ctx.fillRect(px, py, pw, ph);

              // Label Tag Badge
              ctx.font = 'bold 12px "JetBrains Mono", "Space Grotesk", monospace';
              const textMetrics = ctx.measureText(label);
              const tagW = textMetrics.width + 12;
              const tagH = 20;
              const tagY = Math.max(0, py - tagH);

              ctx.fillStyle = hasCustomLabel ? 'rgba(6, 182, 212, 0.95)' : (isRobot ? 'rgba(244, 63, 94, 0.90)' : 'rgba(99, 102, 241, 0.90)');
              ctx.beginPath();
              ctx.roundRect(px, tagY, tagW, tagH, [4, 4, 0, 0]);
              ctx.fill();

              ctx.fillStyle = '#ffffff';
              ctx.fillText(label, px + 6, tagY + 14);
              ctx.restore();
            });
          }
        }
      }
      animId = requestAnimationFrame(render);
    };

    animId = requestAnimationFrame(render);
    return () => cancelAnimationFrame(animId);
  }, [cam.id, metadataMap, roisMap]);

  return (
    <div
      className="video-card-fms"
      style={{
        display: isVisible ? 'flex' : 'none',
        flexDirection: 'column',
        borderRadius: '10px',
        overflow: 'hidden',
        border: '1px solid #2a2a38',
        background: '#18181d',
        boxShadow: '0 4px 24px rgba(0,0,0,0.4)',
        ...(activeTab !== 'all' ? { width: '100%', maxWidth: '1200px', margin: '0 auto' } : {})
      }}
    >
      {/* Card Header */}
      <div style={{
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        padding: '10px 14px',
        background: '#1f1f27',
        borderBottom: '1px solid #2a2a38',
        color: '#f4f4f5'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{
            display: 'inline-block', width: '7px', height: '7px', borderRadius: '50%',
            background: '#22d3ee', boxShadow: '0 0 8px #22d3ee'
          }}></span>
          <span style={{ fontWeight: 600, fontSize: '13px', letterSpacing: '0.01em' }}>{cam.name}</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{
            fontSize: '10px', background: 'rgba(99,102,241,0.15)', color: '#818cf8',
            padding: '2px 8px', borderRadius: '4px', fontWeight: 600,
            border: '1px solid rgba(99,102,241,0.3)', fontFamily: 'monospace'
          }}>
            WHEP · WebRTC
          </span>
          <span style={{ fontSize: '10px', color: '#52525b', fontFamily: 'monospace' }}>&lt; 25ms</span>
          <span style={{
            fontSize: '10px',
            background: metadataCount > 0 ? 'rgba(34,211,238,0.14)' : 'rgba(244,63,94,0.12)',
            color: metadataCount > 0 ? '#22d3ee' : metadataConnected ? '#fb7185' : '#a1a1aa',
            padding: '2px 8px',
            borderRadius: '4px',
            fontWeight: 700,
            border: `1px solid ${metadataCount > 0 ? 'rgba(34,211,238,0.32)' : 'rgba(244,63,94,0.28)'}`,
            fontFamily: 'monospace'
          }}>
            {metadataCount > 0 ? `TRACKING ${metadataCount} ID` : metadataConnected ? 'NO METADATA' : 'WS OFFLINE'}
          </span>
        </div>
      </div>

      <div
        className="video-frame"
        ref={containerRef}
        style={{
          position: 'relative',
          width: '100%',
          aspectRatio: '16/9',
          background: '#09090b',
          overflow: 'hidden',
          cursor: 'crosshair'
        }}
      >
        {useIframeFallback ? (
          <iframe
            src={`http://${hostName}:8081/${cam.id}/?controls=0&autoplay=1&muted=1&playsinline=1`}
            style={{ width: '100%', height: '100%', border: 'none', position: 'absolute', top: 0, left: 0 }}
            title={cam.name}
            scrolling="no"
          />
        ) : (
          <video
            ref={videoRef}
            autoPlay
            muted
            playsInline
            style={{
              width: '100%',
              height: '100%',
              objectFit: 'contain',
              position: 'absolute',
              top: 0,
              left: 0,
              background: '#09090b'
            }}
          />
        )}

        {/* Loading Spinner */}
        {!isPlaying && (
          <div style={{
            position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column',
            alignItems: 'center', justifyContent: 'center',
            background: 'rgba(9, 9, 11, 0.85)', zIndex: 5, gap: '10px'
          }}>
            <div style={{
              width: '26px', height: '26px',
              border: '2px solid #2a2a38', borderTopColor: '#6366f1',
              borderRadius: '50%', animation: 'spin 1s linear infinite'
            }} />
            <span style={{ fontSize: '11px', color: '#52525b', fontFamily: 'monospace' }}>Đang kết nối luồng camera...</span>
          </div>
        )}

        {/* Tracking status */}
        <div style={{
          position: 'absolute',
          bottom: '8px',
          left: '8px',
          zIndex: 12,
          background: 'rgba(15, 23, 42, 0.75)',
          backdropFilter: 'blur(6px)',
          border: '1px solid rgba(255, 255, 255, 0.12)',
          borderRadius: '6px',
          padding: '4px 8px',
          fontSize: '11px',
          color: '#cbd5e1',
          pointerEvents: 'none',
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
          boxShadow: '0 2px 8px rgba(0,0,0,0.5)'
        }}>
          <span style={{ color: '#22d3ee', fontSize: '12px' }}>🎯</span>
          <span>{metadataCount > 0 ? `Đang bám vết ${metadataCount} object` : 'Đang chờ metadata tracking từ backend'}</span>
        </div>

        <canvas
          ref={canvasRef}
          style={{
            position: 'absolute', top: 0, left: 0,
            width: '100%', height: '100%',
            pointerEvents: 'none', zIndex: 10
          }}
        />
      </div>
    </div>
  );
}

export default function MonitorView() {
	  const { cameras } = useCameras();
	  const [activeTab, setActiveTab] = useState('all');
	  const [hostName] = useState(() => (typeof window !== 'undefined' ? (window.location.hostname || '192.168.5.104') : '192.168.5.104'));
	  const [totalDetections, setTotalDetections] = useState(0);
	  const [liveAlerts, setLiveAlerts] = useState<LiveAlert[]>([]);
	  const [globalTrackList, setGlobalTrackList] = useState<{ id: number; fx: number; fy: number; cam: string; class?: string }[]>([]);
	  const [floorTracks, setFloorTracks] = useState<InterpolatedFloorTrack[]>([]);
	  const [mapOverview, setMapOverview] = useState<MapOverviewItem[]>([]);
  const [cameraRois, setCameraRois] = useState<Record<string, ROIState[]>>({});
  const [metadataConnected, setMetadataConnected] = useState(false);
  const [metadataCounts, setMetadataCounts] = useState<Record<string, number>>({});
  const [slotFilter, setSlotFilter] = useState<'ALL' | 'OCCUPIED' | 'EMPTY'>('ALL');
  const { t } = useLanguage();

	  const [showFleetRegistryModal, setShowFleetRegistryModal] = useState(false);
	  const [registeredTargets, setRegisteredTargets] = useState<RegisteredTargetItem[]>([]);

  const fetchRegisteredTargets = useCallback(() => {
    fetch('/api/backend/registry/targets')
      .then(res => res.ok ? res.json() : null)
	      .then((data: RegistryTargetsResponse | null) => {
	        if (data?.targets && Array.isArray(data.targets)) {
	          const targets = data.targets;
	          setRegisteredTargets(prev => {
	            if (prev.length === targets.length) {
	              const unchanged = prev.every((t, i) => {
	                const n = targets[i];
	                return n && t.label === n.label && t.class_name === n.class_name;
	              });
	              if (unchanged) return prev;
	            }
	            return targets;
	          });
	        }
	      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    fetchRegisteredTargets();
    const interval = setInterval(fetchRegisteredTargets, 4000);
    return () => clearInterval(interval);
  }, [fetchRegisteredTargets]);

  const handleDeleteRegisteredTarget = async (label: string, camId?: string) => {
    if (!confirm(`Bạn có chắc muốn xóa nhận diện '${label}' khỏi Registry?`)) return;
    try {
      const url = camId
        ? `/api/backend/camera/${encodeURIComponent(camId)}/registry/target/${encodeURIComponent(label)}`
        : `/api/backend/registry/target/${encodeURIComponent(label)}`;
      const res = await fetch(url, {
        method: 'DELETE',
      });
      if (res.ok) {
        fetchRegisteredTargets();
      }
    } catch (e) {
      console.error(e);
    }
  };

  // Storage Slot Occupancy Aggregate across all cameras (ONLY real configured ROIs)
  const allStorageSlots = React.useMemo(() => {
    const list: {
      id: string;
      name: string;
      camId: string;
      camName: string;
      status: 'CARFULL' | 'EMPTY';
      occupants: string[];
      lastUpdated?: number;
    }[] = [];

	    // Gather ONLY real configured ROIs from cameras (NO dummy/mock data)
	    Object.entries(cameraRois).forEach(([cId, rois]) => {
	      const camName = cameras.find(c => c.id === cId)?.name || `Camera ${cId}`;
	      rois.filter(isStorageSlotRoi).forEach((r, idx) => {
	        const isOccupied = (r.status as string) === 'CARFULL' || r.status === 'OCCUPIED';
	        const occupants = r.occupant_labels && r.occupant_labels.length > 0
	          ? r.occupant_labels
	          : (r.occupant_ids ? r.occupant_ids.map(id => `#${id}`) : []);
	        list.push({
	          id: r.roi_id || `${cId}-roi-${idx}`,
	          name: r.name || `Ô Chứa #${idx + 1}`,
	          camId: cId,
	          camName,
	          status: isOccupied ? 'CARFULL' : 'EMPTY',
	          occupants,
	          lastUpdated: r.last_updated,
	        });
	      });
    });

    return list;
  }, [cameraRois, cameras]);

  const filteredSlots = React.useMemo(() => {
    if (slotFilter === 'OCCUPIED') return allStorageSlots.filter(s => s.status === 'CARFULL');
    if (slotFilter === 'EMPTY') return allStorageSlots.filter(s => s.status === 'EMPTY');
    return allStorageSlots;
  }, [allStorageSlots, slotFilter]);

  const occupiedSlotsCount = allStorageSlots.filter(s => s.status === 'CARFULL').length;
  const emptySlotsCount = allStorageSlots.length - occupiedSlotsCount;
  const occupancyPercent = allStorageSlots.length > 0 ? Math.round((occupiedSlotsCount / allStorageSlots.length) * 100) : 0;

  const metadataMap = useRef<Map<string, Map<number, InterpolatedTrack>>>(new Map());
  const roisMap = useRef<Map<string, ROIState[]>>(new Map());
  const floorTrackMap = useRef<Map<number, InterpolatedFloorTrack>>(new Map());
  const camerasRef = useRef<Camera[]>([]);
  const lastPrimaryObjectsByCamRef = useRef<Map<string, number>>(new Map());

  useEffect(() => {
    camerasRef.current = cameras;
  }, [cameras]);

  // Fetch map overview calibration polygons
  useEffect(() => {
    fetch('/api/backend/calibration/map-overview')
      .then(res => (res.ok ? res.json() : null))
      .then(data => {
        if (data?.calibrations) setMapOverview(data.calibrations);
      })
      .catch(() => {});
  }, [cameras]);

  // Fetch initial saved camera rules/ROIs from database
  useEffect(() => {
    cameras.forEach(cam => {
      fetch(`/api/backend/camera/${cam.id}/rules`)
        .then(res => (res.ok ? res.json() : null))
        .then(data => {
	          if (data?.rules && Array.isArray(data.rules)) {
	            const initialRois: ROIState[] = filterRealConfiguredRois(data.rules.map((r: CameraRuleResponse) => {
	              const coordinateSpace = r.coordinate_space || 'camera';
	              const cameraPoints = Array.isArray(r.camera_points) && r.camera_points.length > 0
	                ? r.camera_points
	                : (coordinateSpace === 'camera' ? (r.points || []) : []);
	              const fmsPoints = Array.isArray(r.fms_points) && r.fms_points.length > 0
	                ? r.fms_points
	                : (coordinateSpace === 'fms' ? (r.points || []) : []);
	              return {
	                roi_id: r.id,
	                name: r.name,
	                status: 'EMPTY',
	                overlap_ratio: 0,
	                occupant_ids: [],
	                occupant_labels: [],
	                polygon: cameraPoints,
	                fms_polygon: fmsPoints,
	                rule_type: r.type || r.rule_type || 'roi',
	                coordinate_space: coordinateSpace,
	                threshold: r.threshold
	              };
	            }));
	            roisMap.current.set(cam.id, initialRois);
            setCameraRois(Object.fromEntries(roisMap.current));
          }
        })
        .catch(() => {});
    });
  }, [cameras]);

	  // WebSockets for Metadata & Events
	  useEffect(() => {
	    if (typeof window !== 'undefined') {
	      const host = hostName;

      let wsMeta: WebSocket | null = null;
      let wsEvents: WebSocket | null = null;

      const connectMeta = () => {
        wsMeta = new WebSocket(`ws://${host}:8000/ws/metadata`);
        wsMeta.onopen = () => setMetadataConnected(true);
        wsMeta.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            if (data.streams && Array.isArray(data.streams)) {
              let count = 0;
              const now = Date.now();
              const source = data.source || 'deepstream';
              const activeGlobals: { id: number; fx: number; fy: number; cam: string; class?: string }[] = [];
              const countsByCam: Record<string, number> = {};
              let processedAnyStream = false;

              data.streams.forEach((stream: { cam_id: string; objects: TrackedObject[]; rois?: ROIState[] }, index: number) => {
                const camId = resolveStreamCamId(stream.cam_id, index, camerasRef.current);
                const identities = new Map<string, TrackedObject>();
                (Array.isArray(stream.objects) ? stream.objects : []).forEach(obj => {
                  if (!shouldShowTrackIdentity(obj.class, obj.label)) return;
                  const key = hasRegisteredLabel(obj.label) ? `label:${obj.label!.trim().toLowerCase()}` : `person:${obj.id}`;
                  const existing = identities.get(key);
                  if (!existing || (obj.confidence ?? 0) > (existing.confidence ?? 0)) identities.set(key, obj);
                });
                const streamObjects = Array.from(identities.values());
                const lastPrimaryAt = lastPrimaryObjectsByCamRef.current.get(camId) || 0;
                // The CPU fallback is person-only in production and complements,
                // rather than replaces, the DeepStream robot/rack stream.
                processedAnyStream = true;
                if (source !== 'cpu_fallback' && streamObjects.length > 0) {
                  lastPrimaryObjectsByCamRef.current.set(camId, now);
                }

                if (Array.isArray(stream.rois)) {
                  roisMap.current.set(camId, mergeRois(roisMap.current.get(camId), stream.rois));
                }

                if (!metadataMap.current.has(camId)) {
                  metadataMap.current.set(camId, new Map());
                }
                const tracks = metadataMap.current.get(camId)!;
                const incomingIds = new Set(streamObjects.map(obj => obj.id));
                tracks.forEach((track, trackId) => {
                  if (!incomingIds.has(trackId)) tracks.delete(trackId);
                });

                const visibleObjects = streamObjects.filter(obj => shouldShowTrackIdentity(obj.class, obj.label));
                countsByCam[camId] = visibleObjects.length;

                streamObjects.forEach(obj => {
                  const showIdentity = shouldShowTrackIdentity(obj.class, obj.label);
                  if (showIdentity) count++;
                  const fx = obj.floor_x ?? (obj.x + obj.w / 2);
                  const fy = obj.floor_y ?? (obj.y + obj.h);
                  if (showIdentity) {
                    activeGlobals.push({ id: obj.id, fx, fy, cam: camId, class: obj.label || obj.class });
                  }

                  if (showIdentity) {
                    if (!floorTrackMap.current.has(obj.id)) {
                      floorTrackMap.current.set(obj.id, {
                        id: obj.id, class: obj.label || obj.class,
                        curFx: fx, curFy: fy,
                        targetFx: fx, targetFy: fy,
                        cam: camId, lastUpdated: now
                      });
                    } else {
                      const ft = floorTrackMap.current.get(obj.id)!;
                      if (obj.label || obj.class) ft.class = obj.label || obj.class;
                      ft.targetFx = fx; ft.targetFy = fy;
                      ft.cam = camId; ft.lastUpdated = now;
                    }
                  }

                  if (!tracks.has(obj.id)) {
                    tracks.set(obj.id, {
                      id: obj.id, class: obj.class, label: obj.label,
                      mask: !isPersonClass(obj.class) && hasRegisteredLabel(obj.label) && validRegisteredMask(obj.mask) ? obj.mask : null,
                      curX: obj.x, curY: obj.y, curW: obj.w, curH: obj.h,
                      targetX: obj.x, targetY: obj.y, targetW: obj.w, targetH: obj.h,
                      floorX: fx, floorY: fy, lastUpdated: now,
                      fms_status: obj.fms_status,
                      fms_battery: obj.fms_battery,
                      fms_speed: obj.fms_speed,
                      carried_rack: obj.carried_rack,
                      cross_check: obj.cross_check,
                      delta_distance_m: obj.delta_distance_m,
                      fms_pos: obj.fms_pos,
                      keypoints: obj.keypoints,
                      posture: obj.posture,
                      fall_detected: obj.fall_detected,
                      world_position: obj.world_position,
                      spatial_valid: obj.spatial_valid,
                    });
                  } else {
                    const track = tracks.get(obj.id)!;
                    track.class = obj.class;
                    track.label = obj.label;
                    track.mask = !isPersonClass(obj.class) && hasRegisteredLabel(obj.label) && validRegisteredMask(obj.mask) ? obj.mask : null;
                    track.targetX = obj.x; track.targetY = obj.y;
                    track.targetW = obj.w; track.targetH = obj.h;
                    track.floorX = fx; track.floorY = fy;
                    track.lastUpdated = now;
                    if (obj.fms_status !== undefined) track.fms_status = obj.fms_status;
                    if (obj.fms_battery !== undefined) track.fms_battery = obj.fms_battery;
                    if (obj.fms_speed !== undefined) track.fms_speed = obj.fms_speed;
                    if (obj.carried_rack !== undefined) track.carried_rack = obj.carried_rack;
                    if (obj.cross_check !== undefined) track.cross_check = obj.cross_check;
                    if (obj.delta_distance_m !== undefined) track.delta_distance_m = obj.delta_distance_m;
                    if (obj.fms_pos !== undefined) track.fms_pos = obj.fms_pos;
                    if (obj.keypoints !== undefined) track.keypoints = obj.keypoints;
                    if (obj.posture !== undefined) track.posture = obj.posture;
                    if (obj.fall_detected !== undefined) track.fall_detected = obj.fall_detected;
                    if (obj.world_position !== undefined) track.world_position = obj.world_position;
                    if (obj.spatial_valid !== undefined) track.spatial_valid = obj.spatial_valid;
                  }
                });
              });

              if (!processedAnyStream) return;
              setTotalDetections(count);
              setMetadataCounts(prev => ({ ...prev, ...countsByCam }));
              setGlobalTrackList(activeGlobals);
              setCameraRois(Object.fromEntries(roisMap.current));
            }
          } catch (e) {}
        };
        wsMeta.onerror = () => setMetadataConnected(false);
        wsMeta.onclose = () => {
          setMetadataConnected(false);
          setTimeout(connectMeta, 2500);
        };
      };

      const connectEvents = () => {
        wsEvents = new WebSocket(`ws://${host}:8000/ws/events`);
        wsEvents.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            if (data.event) {
              setLiveAlerts(prev => [data.event, ...prev].slice(0, 30));
            }
          } catch (e) {}
        };
        wsEvents.onclose = () => setTimeout(connectEvents, 2500);
      };

      connectMeta();
      connectEvents();

      return () => {
        if (wsMeta) wsMeta.close();
        if (wsEvents) wsEvents.close();
      };
    }
	  }, [hostName]);

	  const { colors: C, isDark } = useAppTheme();
	  const activeStorageRois = activeTab !== 'all'
	    ? (cameraRois[activeTab] || []).filter(isStorageSlotRoi)
	    : [];

	  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', background: C.bg, transition: 'background-color 0.2s ease' }}>

      {/* Top Monitor Navigation */}
      <div style={{
        background: C.surface,
        borderBottom: `1px solid ${C.border}`,
        padding: '12px 20px',
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div style={{ display: 'flex', gap: '6px' }}>
            <button
              onClick={() => setActiveTab('all')}
              style={{
                padding: '7px 14px', borderRadius: '7px', cursor: 'pointer',
                fontSize: '12px', fontWeight: 600,
                border: activeTab === 'all' ? `1px solid ${C.accent}` : `1px solid ${C.border}`,
                background: activeTab === 'all' ? C.accentDim : 'transparent',
                color: activeTab === 'all' ? C.accentGlow : C.textSecondary,
                boxShadow: activeTab === 'all' ? `0 0 12px ${C.accentGlow}` : 'none',
                transition: 'all 0.2s'
              }}
            >
              {t.monitor.allCamera} ({cameras.length})
            </button>
            {cameras.map(c => (
              <button
                key={c.id}
                onClick={() => setActiveTab(c.id)}
                style={{
                  padding: '7px 14px', borderRadius: '7px', cursor: 'pointer',
                  fontSize: '12px', fontWeight: 600,
                  border: activeTab === c.id ? `1px solid ${C.accent}` : `1px solid ${C.border}`,
                  background: activeTab === c.id ? C.accentDim : 'transparent',
                  color: activeTab === c.id ? C.accentGlow : C.textSecondary,
                  boxShadow: activeTab === c.id ? `0 0 12px ${C.accentDim}` : 'none',
                  transition: 'all 0.2s cubic-bezier(0.16, 1, 0.3, 1)'
                }}
              >
                {c.name}
              </button>
            ))}
          </div>

          {/* Right Action & Status Badges */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <button
              onClick={() => setShowFleetRegistryModal(true)}
              style={{
                display: 'flex', alignItems: 'center', gap: '6px',
                background: 'rgba(99,102,241,0.12)',
                border: '1px solid rgba(99,102,241,0.3)',
                padding: '6px 12px', borderRadius: '7px',
                cursor: 'pointer', color: C.accentGlow,
                fontSize: '12px', fontWeight: 600,
                boxShadow: '0 2px 8px rgba(0,0,0,0.2)',
                transition: 'all 0.2s'
              }}
            >
              <Bot size={14} />
              <span>Robot Registry ({registeredTargets.length})</span>
            </button>

            {/* MTMC Status Badge */}
            <div style={{
              display: 'flex', alignItems: 'center', gap: '8px',
              background: 'rgba(34,211,238,0.08)',
              border: '1px solid rgba(34,211,238,0.25)',
              padding: '5px 12px', borderRadius: '20px'
            }}>
              <span style={{
                width: '6px', height: '6px', borderRadius: '50%',
                background: C.cyan, animation: 'pulse 2s infinite',
                boxShadow: `0 0 8px ${C.cyan}`
              }}></span>
              <span style={{ fontSize: '12px', fontWeight: 600, color: C.cyan, fontFamily: 'monospace' }}>
                MTMC FUSION ACTIVE
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div style={{ flex: 1, padding: '20px', display: 'flex', gap: '20px', overflow: 'hidden' }}>
        {cameras.length === 0 ? (
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: C.textMuted }}>
            <div style={{
              width: '64px', height: '64px', borderRadius: '16px',
              background: 'rgba(99,102,241,0.1)', border: `1px solid ${C.border}`,
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              marginBottom: '16px'
            }}>
              <Activity size={28} color={C.textMuted} />
            </div>
            <p style={{ fontSize: '18px', fontWeight: 600, color: C.textSecondary }}>{t.monitor.noCamera}</p>
            <p style={{ fontSize: '13px', color: C.textMuted, marginTop: '8px', textAlign: 'center' }}>
              Chuyển sang tab <b style={{ color: C.accentGlow }}>Building</b> để thêm Camera RTSP và thiết lập Calibration
            </p>
          </div>
        ) : (
          <>
            {/* Left: Video Grid */}
            <div style={{ flex: 3, overflowY: 'auto', paddingRight: '6px' }}>
              <div style={{
                display: 'grid',
                gridTemplateColumns: activeTab === 'all' ? 'repeat(auto-fit, minmax(400px, 1fr))' : '1fr',
                gap: '16px'
              }}>
                {cameras.map(cam => (
                  <CameraStreamCard
                    key={cam.id}
                    cam={cam}
                    hostName={hostName}
                    isVisible={activeTab === 'all' || activeTab === cam.id}
                    activeTab={activeTab}
                    metadataMap={metadataMap}
                    roisMap={roisMap}
                    metadataCount={metadataCounts[cam.id] || 0}
                    metadataConnected={metadataConnected}
                  />
                ))}
              </div>
            </div>


            {/* Right: Side Panel */}
            <div style={{ flex: 1.2, display: 'flex', flexDirection: 'column', gap: '16px', minWidth: '320px' }}>

              {/* SINGLE CAMERA: ROI Bay Status */}
              {activeTab !== 'all' ? (
                <div style={{
                  background: C.surface,
                  borderRadius: '12px', padding: '20px',
                  border: `1px solid ${C.border}`,
                  display: 'flex', flexDirection: 'column', gap: '16px'
                }}>
                  <div style={{
                    display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                    borderBottom: `1px solid ${C.border}`, paddingBottom: '12px'
                  }}>
                    <div>
                      <h3 style={{ margin: 0, fontSize: '15px', fontWeight: 700, color: C.textPrimary }}>
                        Trạng Thái Ô Chứa Hàng
                      </h3>
                      <p style={{ margin: '4px 0 0 0', fontSize: '12px', color: C.textMuted }}>
                        Camera: <b style={{ color: C.accentGlow }}>{cameras.find(c => c.id === activeTab)?.name || activeTab}</b>
                      </p>
                    </div>
                    <span style={{
                      fontSize: '10px', background: 'rgba(99,102,241,0.12)',
                      color: C.accentGlow, padding: '3px 8px', borderRadius: '5px',
                      fontWeight: 600, fontFamily: 'monospace',
                      border: '1px solid rgba(99,102,241,0.25)'
                    }}>
                      REAL-TIME
                    </span>
                  </div>

	                  {activeStorageRois.length === 0 ? (
                    <div style={{
                      padding: '32px 20px', textAlign: 'center',
                      color: C.textMuted, fontSize: '13px',
                      background: C.card, borderRadius: '10px',
                      border: `1px dashed ${C.border}`
                    }}>
                      <p style={{ margin: '0 0 6px 0', fontSize: '14px', fontWeight: 600, color: C.textSecondary }}>Chưa có vùng ROI nào</p>
                      Chuyển sang tab <b style={{ color: C.accentGlow }}>Building</b> &rarr; <b>Cấu hình Phân tích Hành vi</b> để vẽ vùng ô chứa hàng.
                    </div>
                  ) : (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
	                      {activeStorageRois.map(r => {
                        const isCarFull = (r.status as string) === 'CARFULL' || r.status === 'OCCUPIED';
                        return (
                          <div
                            key={r.roi_id}
                            style={{
                              padding: '16px',
                              borderRadius: '10px',
                              background: isCarFull
                                ? 'rgba(244, 63, 94, 0.08)'
                                : 'rgba(34, 211, 238, 0.06)',
                              border: `1px solid ${isCarFull ? 'rgba(244,63,94,0.35)' : 'rgba(34,211,238,0.25)'}`,
                              boxShadow: isCarFull
                                ? '0 0 20px rgba(244,63,94,0.12)'
                                : '0 0 20px rgba(34,211,238,0.08)',
                              display: 'flex', flexDirection: 'column', gap: '10px',
                              transition: 'all 0.3s ease'
                            }}
                          >
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                              <span style={{ fontWeight: 700, fontSize: '13px', color: C.textPrimary, fontFamily: 'monospace', letterSpacing: '0.05em' }}>
                                VỊ TRÍ · {r.name}
                              </span>
                            </div>

                            {/* Status Badge */}
                            <div style={{
                              display: 'flex', alignItems: 'center', justifyContent: 'center',
                              padding: '10px 16px', borderRadius: '8px',
                              background: isCarFull ? 'rgba(244,63,94,0.18)' : 'rgba(34,211,238,0.12)',
                              color: isCarFull ? '#fb7185' : '#67e8f9',
                              fontWeight: 700, fontSize: '14px',
                              letterSpacing: '0.08em', fontFamily: 'monospace',
                              border: `1px solid ${isCarFull ? 'rgba(244,63,94,0.4)' : 'rgba(34,211,238,0.3)'}`,
                              textShadow: isCarFull ? '0 0 12px rgba(244,63,94,0.5)' : '0 0 12px rgba(34,211,238,0.5)'
                            }}>
                              {isCarFull ? '⚠ CÓ HÀNG — CARFULL' : '◎ TRỐNG — EMPTY'}
                            </div>

                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '12px', color: C.textMuted }}>
                              <span>{isCarFull ? 'Chiếm dụng bởi:' : 'Tình trạng:'}</span>
                              <span style={{ fontWeight: 600, color: isCarFull ? '#fb7185' : '#67e8f9', fontFamily: 'monospace' }}>
	                                {isCarFull
	                                  ? (r.occupant_labels && r.occupant_labels.length > 0 ? r.occupant_labels.join(', ') : (r.occupant_ids && r.occupant_ids.length > 0 ? r.occupant_ids.map(id => `#${id}`).join(', ') : 'Có vật thể / xe'))
	                                  : 'Sẵn sàng tiếp nhận'}
                              </span>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              ) : (
                <>
                  {/* ── Trạng thái Các Ô Chứa Hàng (Storage Slot Occupancy Overview) ── */}
                  <div style={{
                    background: C.surface, borderRadius: '12px', padding: '14px',
                    border: `1px solid ${C.border}`, display: 'flex', flexDirection: 'column', gap: '12px'
                  }}>
                    {/* Header */}
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontWeight: 700, color: C.textPrimary, fontSize: '13px' }}>
                        <Boxes size={17} color={C.cyan} />
                        Trạng Thái Ô Chứa Hàng
                      </div>
                      <span style={{
                        fontSize: '10px', background: 'rgba(6,182,212,0.12)', color: C.cyan,
                        padding: '3px 8px', borderRadius: '5px', fontWeight: 700, fontFamily: 'monospace',
                        border: '1px solid rgba(6,182,212,0.25)'
                      }}>
	                        CARFULL · FMS/CAMERA
                      </span>
                    </div>

                    {allStorageSlots.length === 0 ? (
                      <div style={{
                        padding: '28px 16px', textAlign: 'center',
                        color: C.textMuted, fontSize: '12px',
                        background: C.card, borderRadius: '10px',
                        border: `1px dashed ${C.border}`,
                        display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '8px'
                      }}>
                        <Boxes size={28} color={C.textMuted} style={{ opacity: 0.6 }} />
                        <p style={{ margin: 0, fontSize: '13px', fontWeight: 600, color: C.textSecondary }}>
                          Chưa có ô chứa hàng nào được thiết lập
                        </p>
                        <p style={{ margin: 0, fontSize: '11px', color: C.textMuted, lineHeight: 1.5 }}>
                          Chuyển sang tab <b style={{ color: C.accentGlow }}>Building</b> &rarr; <b>Cấu hình Phân tích & ROI</b> để vẽ các vùng ô chứa hàng thực tế.
                        </p>
                      </div>
                    ) : (
                      <>
                        {/* KPI Metric Summary Bar */}
                        <div style={{
                          display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '8px',
                          background: C.card, padding: '10px', borderRadius: '8px', border: `1px solid ${C.border}`
                        }}>
                          <div>
                            <div style={{ fontSize: '11px', color: C.textMuted }}>Tổng số ô</div>
                            <div style={{ fontSize: '16px', fontWeight: 700, color: C.textPrimary, fontFamily: 'monospace' }}>{allStorageSlots.length}</div>
                          </div>
                          <div>
                            <div style={{ fontSize: '11px', color: '#fb7185' }}>Có hàng</div>
                            <div style={{ fontSize: '16px', fontWeight: 700, color: '#fb7185', fontFamily: 'monospace' }}>{occupiedSlotsCount}</div>
                          </div>
                          <div>
                            <div style={{ fontSize: '11px', color: '#34d399' }}>Ô trống</div>
                            <div style={{ fontSize: '16px', fontWeight: 700, color: '#34d399', fontFamily: 'monospace' }}>{emptySlotsCount}</div>
                          </div>
                        </div>

                        {/* Occupancy Progress Bar */}
                        <div>
                          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', marginBottom: '4px', color: C.textMuted }}>
                            <span>Tỷ lệ chiếm dụng</span>
                            <span style={{ fontWeight: 600, color: C.textPrimary, fontFamily: 'monospace' }}>{occupancyPercent}% ({occupiedSlotsCount}/{allStorageSlots.length})</span>
                          </div>
                          <div style={{ width: '100%', height: '6px', background: isDark ? 'rgba(255,255,255,0.06)' : 'rgba(15,23,42,0.08)', borderRadius: '3px', overflow: 'hidden' }}>
                            <div style={{
                              width: `${occupancyPercent}%`,
                              height: '100%',
                              background: occupancyPercent > 70 ? 'linear-gradient(90deg, #fb7185, #e11d48)' : 'linear-gradient(90deg, #06b6d4, #10b981)',
                              borderRadius: '3px',
                              transition: 'width 0.4s ease'
                            }} />
                          </div>
                        </div>

                        {/* Filter Segmented Controls */}
                        <div style={{ display: 'flex', gap: '6px' }}>
                          {(['ALL', 'OCCUPIED', 'EMPTY'] as const).map(f => {
                            const isActive = slotFilter === f;
                            const label = f === 'ALL' ? `Tất cả (${allStorageSlots.length})` : f === 'OCCUPIED' ? `Có hàng (${occupiedSlotsCount})` : `Trống (${emptySlotsCount})`;
                            return (
                              <button
                                key={f}
                                onClick={() => setSlotFilter(f)}
                                style={{
                                  flex: 1, padding: '5px 8px', borderRadius: '6px', fontSize: '11px', fontWeight: 600,
                                  cursor: 'pointer', border: '1px solid transparent',
                                  background: isActive ? (isDark ? 'rgba(255,255,255,0.12)' : 'rgba(15,23,42,0.10)') : 'transparent',
                                  color: isActive ? C.textPrimary : C.textMuted,
                                  borderColor: isActive ? C.border : 'transparent',
                                  transition: 'all 0.15s ease'
                                }}
                              >
                                {label}
                              </button>
                            );
                          })}
                        </div>

                        {/* Slot Cards List */}
                        <div style={{
                          display: 'flex', flexDirection: 'column', gap: '8px',
                          maxHeight: '260px', overflowY: 'auto', paddingRight: '2px'
                        }}>
                          {filteredSlots.length === 0 ? (
                            <div style={{ padding: '20px', textAlign: 'center', color: C.textMuted, fontSize: '12px' }}>
                              Không có ô chứa hàng nào phù hợp bộ lọc
                            </div>
                          ) : (
                            filteredSlots.map(slot => {
                              const isOccupied = slot.status === 'CARFULL';
                              return (
                                <div
                                  key={slot.id}
                                  style={{
                                    padding: '10px 12px',
                                    borderRadius: '8px',
                                    background: isOccupied
                                      ? (isDark ? 'rgba(244, 63, 94, 0.08)' : 'rgba(244, 63, 94, 0.05)')
                                      : (isDark ? 'rgba(16, 185, 129, 0.06)' : 'rgba(16, 185, 129, 0.04)'),
                                    border: `1px solid ${isOccupied ? 'rgba(244,63,94,0.3)' : 'rgba(16,185,129,0.25)'}`,
                                    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                                    transition: 'all 0.2s ease'
                                  }}
                                >
                                  {/* Left: Icon & Slot Name */}
                                  <div style={{ display: 'flex', alignItems: 'center', gap: '9px' }}>
                                    <div style={{
                                      width: '30px', height: '30px', borderRadius: '6px',
                                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                                      background: isOccupied ? 'rgba(244,63,94,0.18)' : 'rgba(16,185,129,0.15)',
                                      color: isOccupied ? '#fb7185' : '#34d399',
                                    }}>
                                      <Package size={16} />
                                    </div>
                                    <div>
                                      <div style={{ fontSize: '12px', fontWeight: 700, color: C.textPrimary }}>
                                        {slot.name}
                                      </div>
                                      <div style={{ fontSize: '10px', color: C.textMuted, fontFamily: 'monospace' }}>
                                        {slot.camName}
                                      </div>
                                    </div>
                                  </div>

                                  {/* Right: Status Pill */}
                                  <div style={{ textAlign: 'right' }}>
                                    <span style={{
                                      fontSize: '11px',
                                      fontWeight: 700,
                                      padding: '3px 8px',
                                      borderRadius: '5px',
                                      background: isOccupied ? 'rgba(244,63,94,0.2)' : 'rgba(16,185,129,0.15)',
                                      color: isOccupied ? '#fb7185' : '#34d399',
                                      border: `1px solid ${isOccupied ? 'rgba(244,63,94,0.4)' : 'rgba(16,185,129,0.3)'}`,
                                      fontFamily: 'monospace',
                                      letterSpacing: '0.04em'
                                    }}>
                                      {isOccupied ? '⚠ CÓ HÀNG' : '◎ TRỐNG'}
                                    </span>
                                    {isOccupied && slot.occupants.length > 0 && (
                                      <div style={{ fontSize: '10px', color: C.textMuted, marginTop: '3px', fontFamily: 'monospace' }}>
                                        {slot.occupants.join(', ')}
                                      </div>
                                    )}
                                  </div>
                                </div>
                              );
                            })
                          )}
                        </div>
                      </>
                    )}
                  </div>

                  {/* Live Alerts Feed */}
                  <div style={{
                    background: C.surface, borderRadius: '12px', padding: '14px',
                    border: `1px solid ${C.border}`,
                    flex: 1, display: 'flex', flexDirection: 'column'
                  }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '7px', fontWeight: 600, color: C.textPrimary, fontSize: '13px' }}>
                        <ShieldAlert size={16} color={C.rose} /> Live Behavior Alarms
                      </div>
                      <span style={{ fontSize: '10px', color: C.textMuted, fontFamily: 'monospace' }}>PostgreSQL Sync</span>
                    </div>

                    <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '6px', maxHeight: '260px' }}>
                      {liveAlerts.length === 0 ? (
                        <div style={{ padding: '24px', textAlign: 'center', color: C.textMuted, fontSize: '12px' }}>
                          Chưa phát hiện sự kiện bất thường
                        </div>
                      ) : (
                        liveAlerts.map((ev, i) => {
                          const isIntrusion = ev.rule_type === 'intrusion';
                          const isTripwire = ev.rule_type === 'tripwire';
                          const accentColor = isIntrusion ? C.rose : isTripwire ? C.cyan : C.orange;
                          return (
                            <div
                              key={i}
                              style={{
                                padding: '9px 12px 9px 14px',
                                borderRadius: '7px',
                                background: C.card,
                                border: `1px solid ${C.border}`,
                                borderLeft: `3px solid ${accentColor}`,
                                fontSize: '12px'
                              }}
                            >
                              <div style={{ display: 'flex', justifyContent: 'space-between', fontWeight: 600, marginBottom: '3px' }}>
                                <span style={{ textTransform: 'uppercase', fontSize: '10px', color: accentColor, fontFamily: 'monospace', letterSpacing: '0.08em' }}>
                                  {ev.rule_type}
                                </span>
                                <span style={{ color: C.textMuted, fontSize: '10px', fontFamily: 'monospace' }}>
                                  {new Date(ev.timestamp).toLocaleTimeString()}
                                </span>
                              </div>
                              <p style={{ margin: 0, color: C.textSecondary, lineHeight: 1.4 }}>{ev.description}</p>
                            </div>
                          );
                        })
                      )}
                    </div>
                  </div>
                </>
              )}

            </div>
          </>
        )}
      </div>

      {/* ── MODAL 2: REGISTERED FLEET MANAGER DRAWER ── */}
      {showFleetRegistryModal && (
        <div style={{
          position: 'fixed', inset: 0, zIndex: 9999,
          background: 'rgba(0, 0, 0, 0.75)', backdropFilter: 'blur(8px)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          padding: '20px'
        }}>
          <div style={{
            width: '100%', maxWidth: '640px',
            background: '#18181f',
            border: '1px solid #3b3b4f',
            borderRadius: '16px',
            boxShadow: '0 20px 60px rgba(0,0,0,0.8), 0 0 30px rgba(99,102,241,0.2)',
            overflow: 'hidden', maxHeight: '80vh', display: 'flex', flexDirection: 'column'
          }}>
            {/* Header */}
            <div style={{
              padding: '16px 20px',
              background: 'linear-gradient(90deg, rgba(99,102,241,0.15), rgba(34,211,238,0.1))',
              borderBottom: '1px solid #2e2e3f',
              display: 'flex', justifyContent: 'space-between', alignItems: 'center'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <div style={{
                  width: '34px', height: '34px', borderRadius: '8px',
                  background: 'rgba(99,102,241,0.25)', border: '1px solid #6366f1',
                  display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#818cf8'
                }}>
                  <Bot size={18} />
                </div>
                <div>
                  <h3 style={{ margin: 0, fontSize: '15px', fontWeight: 700, color: '#f4f4f5' }}>
                    Danh Sách Robot & Kệ Hàng Đã Đăng Ký
                  </h3>
                  <p style={{ margin: 0, fontSize: '11px', color: '#94a3b8' }}>
                    Tổng số: {registeredTargets.length} mục tiêu đang theo dõi Re-ID
                  </p>
                </div>
              </div>
              <button
                onClick={() => setShowFleetRegistryModal(false)}
                style={{
                  background: 'transparent', border: 'none', color: '#71717a',
                  cursor: 'pointer', padding: '6px', borderRadius: '6px',
                  display: 'flex', alignItems: 'center', justifyContent: 'center'
                }}
              >
                <X size={18} />
              </button>
            </div>

            {/* Targets Table */}
            <div style={{ padding: '16px 20px', overflowY: 'auto', flex: 1, display: 'flex', flexDirection: 'column', gap: '8px' }}>
              {registeredTargets.length === 0 ? (
                <div style={{ padding: '32px 20px', textAlign: 'center', color: '#71717a', fontSize: '13px' }}>
                  <Bot size={32} color="#52525b" style={{ margin: '0 auto 10px auto' }} />
                  <p style={{ margin: 0, fontWeight: 600, color: '#a1a1aa' }}>Chưa có robot/kệ hàng nào được đăng ký</p>
                  <p style={{ margin: '4px 0 0 0', fontSize: '11px', color: '#71717a' }}>
                    Vào Building &gt; Label để crop frame camera và gán nhãn tracking.
                  </p>
                </div>
              ) : (
                registeredTargets.map((tgt) => (
                  <div
                    key={tgt.label}
                    style={{
                      padding: '12px 14px',
                      borderRadius: '8px',
                      background: 'rgba(255,255,255,0.03)',
                      border: '1px solid #2e2e3f',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      gap: '12px'
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                      <div style={{
                        width: '32px', height: '32px', borderRadius: '6px',
                        background: tgt.label.startsWith('Rack_') ? 'rgba(99,102,241,0.18)' : 'rgba(244,63,94,0.18)',
                        color: tgt.label.startsWith('Rack_') ? '#a5b4fc' : '#fb7185',
                        display: 'flex', alignItems: 'center', justifyContent: 'center'
                      }}>
                        {tgt.label.startsWith('Rack_') ? <Package size={16} /> : <Bot size={16} />}
                      </div>
                      <div>
                        <div style={{ fontSize: '13px', fontWeight: 700, color: '#f4f4f5', fontFamily: 'monospace' }}>
                          {tgt.label}
                        </div>
                        <div style={{ fontSize: '11px', color: '#71717a', marginTop: '2px' }}>
                          Camera: <b style={{ color: '#22d3ee' }}>{tgt.last_cam || 'N/A'}</b> · Mẫu: {tgt.samples_count || 1}
                        </div>
                      </div>
                    </div>

                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                      <span style={{
                        fontSize: '10px',
                        padding: '3px 8px',
                        borderRadius: '4px',
                        background: 'rgba(52,211,153,0.15)',
                        color: '#34d399',
                        fontWeight: 600,
                        border: '1px solid rgba(52,211,153,0.3)',
                        fontFamily: 'monospace'
                      }}>
                        ACTIVE 512D
                      </span>
                      <button
                        onClick={() => handleDeleteRegisteredTarget(tgt.label, tgt.cam_id || tgt.last_cam)}
                        style={{
                          background: 'rgba(244,63,94,0.12)',
                          border: '1px solid rgba(244,63,94,0.3)',
                          color: '#fb7185',
                          cursor: 'pointer',
                          padding: '6px 8px',
                          borderRadius: '6px',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '4px',
                          fontSize: '11px',
                          fontWeight: 600
                        }}
                      >
                        <Trash2 size={13} />
                        <span>Xóa</span>
                      </button>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
