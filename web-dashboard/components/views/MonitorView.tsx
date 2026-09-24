'use client';

import React, { useEffect, useState, useRef, useCallback } from 'react';
import dynamic from 'next/dynamic';
import { useLanguage } from '../LanguageContext';
import { useCameras, Camera } from '../CameraContext';
import { useAppTheme } from '../ThemeContext';
import { RegisteredMask, validRegisteredMask } from '../../lib/registered-mask';
import { LiveRegisteredMask, isLiveRegisteredMask, metadataReceivedAt, updateLiveRegisteredMask } from '../../lib/live-registered-mask';
import { segmentationColors, segmentationPath } from '../../lib/segmentation-overlay';
import { connectRealtimeSocket, createMetadataClock } from '../../lib/realtime-socket';
import { connectRealtimeVideo } from '../../lib/realtime-video';
import { Activity, ShieldAlert, Users, Layers, AlertCircle, ArrowRightLeft, Radio, Boxes, Package, Grid3X3, Bot, Sparkles, Trash2, X, Check, Search, Sliders } from 'lucide-react';

const MonitorLabelDialog = dynamic(() => import('./MonitorLabelDialog'), { ssr: false });
const ActiveLearningDialog = dynamic(() => import('./ActiveLearningDialog'), { ssr: false });
import MonitorChatAssistant from './MonitorChatAssistant';

type TrackedObject = {
  id: number;
  model_id?: string;
  generation?: string;
  identity_verified?: boolean;
  label_prompt_id?: string;
  category?: string;
  local_id?: string | number;
  class: string;
  label?: string;
  mask?: RegisteredMask | null;
  observed_at?: number;
  tracking_state?: string;
  mask_stale?: boolean;
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
  obb?: { points: number[][]; angle?: number };
  attributes?: Record<string, string | boolean | number>;
  alert?: boolean;
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
  local_id?: string | number;
  model_id?: string;
  generation?: string;
  labelPrompt?: boolean;
  class?: string;
  label?: string;
  maskFrame?: LiveRegisteredMask | null;
  observedAt?: number;
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
  obb?: { points: number[][]; angle?: number };
  attributes?: Record<string, string | boolean | number>;
  alert?: boolean;
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

type MapOverviewItem = Record<string, unknown>;

const resolveStreamCamId = (rawCamId: string, index: number, cameras: Camera[]) => {
  if (cameras.some(c => c.id === rawCamId)) return rawCamId;
  const indexedAliases = new Set([String(index), `cam_${index}`, `source_${index}`, `camera_${index}`]);
  if (indexedAliases.has(String(rawCamId)) && cameras[index]) return cameras[index].id;
  if (cameras.length === 1) return cameras[0].id;
  return rawCamId;
};

const isPersonClass = (className?: string) => /person|human|worker|pedestrian/i.test(className || '');
const isRobotClass = (className?: string) => {
  const c = (className || '').toLowerCase();
  return /robot|agv|amr|forklift|rack|shelf|pallet|kệ/.test(c);
};
const hasRegisteredLabel = (label?: string) => Boolean(label && label.trim() !== '');
const shouldShowTrackIdentity = (className?: string, modelId?: string) =>
  Boolean(modelId) || isPersonClass(className);
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
const TRACK_HOLD_MS = 1200;
const TRACK_FADE_START_MS = 350;

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

const CameraStreamCard = React.memo(function CameraStreamCard({
  cam,
  hostName,
  isVisible,
  isActive,
  activeTab,
  metadataMap,
  roisMap,
  metadataCount,
  metadataConnected,
  onLabel,
  onLearn
}: {
  cam: Camera;
  hostName: string;
  isVisible: boolean;
  isActive: boolean;
  activeTab: string;
  metadataMap: React.MutableRefObject<Map<string, Map<number, InterpolatedTrack>>>;
  roisMap: React.MutableRefObject<Map<string, ROIState[]>>;
  metadataCount: number;
  metadataConnected: boolean;
  onLabel: (cameraId: string) => void;
  onLearn: (cameraId: string) => void;
}) {
  const { isDark } = useAppTheme();
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const visibleRef = useRef(isActive && isVisible);
  const [maskAgeMs, setMaskAgeMs] = useState<number | null>(null);

  useEffect(() => { visibleRef.current = isActive && isVisible; }, [isActive, isVisible]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    return connectRealtimeVideo({
      video,
      // The browser consumes a lightweight transcoded preview. DeepStream keeps
      // reading the full-resolution relay for model inference.
      url: `http://${hostName}:8081/${encodeURIComponent(`${cam.id}_preview`)}/whep`,
      isVisible: () => visibleRef.current,
      onPlayingChange: setIsPlaying,
      onReset: () => {
        const canvas = canvasRef.current;
        const context = canvas?.getContext('2d');
        if (context && canvas) {
          context.setTransform(1, 0, 0, 1, 0, 0);
          context.clearRect(0, 0, canvas.width, canvas.height);
        }
      },
    });
  }, [cam.id, hostName]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    const video = videoRef.current;
    if (!canvas || !container) return;
    const context = canvas.getContext('2d');
    if (!context) return;
    const clear = () => {
      context.setTransform(1, 0, 0, 1, 0, 0);
      context.clearRect(0, 0, canvas.width, canvas.height);
    };
    if (!isActive || !isVisible) {
      clear();
      return;
    }
    let animId = 0;
    let videoFrameId = 0;
    let disposed = false;
    let lastVideoAt = performance.now();
    let lastStatusAt = 0;
    let width = container.clientWidth;
    let height = container.clientHeight;
    let paths = new WeakMap<RegisteredMask, Path2D>();
    let sourceSize = '';
    const resize = new ResizeObserver(entries => {
      const bounds = entries[0]?.contentRect;
      if (!bounds) return;
      width = bounds.width;
      height = bounds.height;
      paths = new WeakMap();
    });
    resize.observe(container);

    const render = () => {
      if (document.hidden || width < 10 || height < 10 || !video?.srcObject || video.readyState < 2) {
        clear();
        return;
      }
      if (canvas && container) {
        const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
        const rw = Math.round(width * pixelRatio);
        const rh = Math.round(height * pixelRatio);
        if (rw > 10 && rh > 10 && (canvas.width !== rw || canvas.height !== rh)) {
          canvas.width = rw;
          canvas.height = rh;
          paths = new WeakMap();
        }

        const ctx = context;
        if (ctx && canvas.width > 0 && canvas.height > 0) {
          clear();
          ctx.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
          const sourceW = video?.videoWidth || 16;
          const sourceH = video?.videoHeight || 9;
          const currentSourceSize = `${sourceW}:${sourceH}`;
          if (currentSourceSize !== sourceSize) {
            sourceSize = currentSourceSize;
            paths = new WeakMap();
          }
          const sourceAspect = sourceW / Math.max(1, sourceH);
          const canvasAspect = width / Math.max(1, height);
          let videoDrawW = width;
          let videoDrawH = height;
          let videoOffsetX = 0;
          let videoOffsetY = 0;
          if (sourceAspect > canvasAspect) {
            videoDrawW = width;
            videoDrawH = width / sourceAspect;
            videoOffsetY = (height - videoDrawH) / 2;
          } else {
            videoDrawH = height;
            videoDrawW = height * sourceAspect;
            videoOffsetX = (width - videoDrawW) / 2;
          }
          const mapVideoX = (x: number) => videoOffsetX + x * videoDrawW;
          const mapVideoY = (y: number) => videoOffsetY + y * videoDrawH;

          // --- A. Tracked Objects (BBox + Tags only, no ROI overlays on stream) ---
          const camTracks = metadataMap.current.get(cam.id);
          const now = Date.now();
          let oldestMaskAge: number | null = null;

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

              // Class & label formatting
              const rawClass = (track.class || 'Object').toLowerCase();
              const isRobot = rawClass.includes('robot');

              if (isRobot) {
                track.curX += (track.targetX - track.curX) * 0.70;
                track.curY += (track.targetY - track.curY) * 0.70;
                track.curW += (track.targetW - track.curW) * 0.35;
                track.curH += (track.targetH - track.curH) * 0.35;
              } else {
                track.curX = track.targetX;
                track.curY = track.targetY;
                track.curW = track.targetW;
                track.curH = track.targetH;
              }

              const px = mapVideoX(track.curX);
              const py = mapVideoY(track.curY);
              const pw = track.curW * videoDrawW;
              const ph = track.curH * videoDrawH;

              const isRack = rawClass.includes('rack');
              const isPerson = rawClass.includes('person') || rawClass.includes('human') || rawClass.includes('worker');
              const isFallen = Boolean(track.fall_detected || track.posture === 'fallen' || track.alert);
              const hasCustomLabel = hasRegisteredLabel(track.label);
              const mask = isLiveRegisteredMask(track.maskFrame, now)
                ? track.maskFrame.mask : null;
              const drawMask = Boolean(mask);
              if (!shouldShowTrackIdentity(track.class, track.model_id)) {
                return;
              }
              const displayClass = isRobot ? 'robot' : (isRack ? 'rack' : rawClass);
              const label = hasCustomLabel ? track.label!
                : `${displayClass} #${track.local_id ?? track.id}`;
              const displayLabel = label;

              const colors = segmentationColors(track.label || `${track.model_id}:${track.local_id ?? track.id}`, isFallen);
              const strokeColor = isPerson && !isFallen ? '#4ade80' : colors.stroke;
              const fillColor = colors.fill;

              ctx.save();
              ctx.globalAlpha = alpha;
              ctx.strokeStyle = strokeColor;
              ctx.lineWidth = hasCustomLabel ? 1.6 : 1.3;
              ctx.lineJoin = 'round';
              ctx.lineCap = 'round';
              ctx.shadowColor = strokeColor;
              ctx.shadowBlur = 0;
              if (drawMask && mask) {
                oldestMaskAge = Math.max(oldestMaskAge ?? 0, now - track.maskFrame!.receivedAt);
                let path = paths.get(mask);
                if (!path) {
                  path = segmentationPath(mask, videoDrawW, videoDrawH, videoOffsetX, videoOffsetY);
                  paths.set(mask, path);
                }
                ctx.fillStyle = fillColor;
                ctx.fill(path, 'evenodd');
                ctx.lineWidth = 3.25;
                ctx.strokeStyle = 'rgba(5, 12, 24, 0.5)';
                ctx.stroke(path);
                ctx.lineWidth = 1.4;
                ctx.strokeStyle = strokeColor;
                ctx.stroke(path);
              } else if (!isPerson) {
                ctx.strokeRect(px, py, pw, ph);
              }
              if (track.obb?.points?.length === 4) {
                const points = track.obb.points;
                ctx.beginPath();
                points.forEach((point, index) => {
                  const x = mapVideoX(point[0]); const y = mapVideoY(point[1]);
                  if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
                });
                ctx.closePath(); ctx.stroke();
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
              if (!drawMask && !isPerson) ctx.fillRect(px, py, pw, ph);

              ctx.font = `${isPerson ? '500 10px' : '600 11px'} "Space Grotesk", sans-serif`;
              const textMetrics = ctx.measureText(displayLabel);
              const tagPadding = 8;
              const tagW = Math.min(width, textMetrics.width + tagPadding * 2 + 10);
              const tagH = isPerson ? 19 : 24;
              const tagX = Math.max(0, Math.min(px, width - tagW));
              const tagY = Math.max(0, Math.min(height - tagH, py - tagH - 5));

              ctx.fillStyle = 'rgba(9, 16, 30, 0.88)';
              ctx.strokeStyle = strokeColor;
              ctx.lineWidth = .8;
              ctx.beginPath();
              ctx.roundRect(tagX, tagY, tagW, tagH, 6);
              ctx.fill();
              ctx.stroke();
              ctx.fillStyle = strokeColor;
              ctx.beginPath();
              ctx.arc(tagX + tagPadding, tagY + tagH / 2, 2.5, 0, Math.PI * 2);
              ctx.fill();

              ctx.fillStyle = '#f1f5f9';
              ctx.textBaseline = 'middle';
              ctx.fillText(displayLabel, tagX + tagPadding + 9, tagY + tagH / 2, Math.max(1, tagW - tagPadding * 2 - 9));
              if (track.attributes && Object.keys(track.attributes).length) {
                const badges = Object.entries(track.attributes).map(([key, value]) => `${key}: ${value === true ? 'OK' : value === false ? 'THIẾU' : value}`).join(' · ');
                ctx.font = '500 9px "Space Grotesk", sans-serif';
                const badgeW = Math.min(width, ctx.measureText(badges).width + 12);
                const badgeY = Math.min(height - 16, tagY + tagH + 3);
                ctx.fillStyle = 'rgba(9, 16, 30, .88)'; ctx.fillRect(tagX, badgeY, badgeW, 16);
                ctx.fillStyle = Object.values(track.attributes).some(value => value === false) ? '#ef4444' : '#e2e8f0';
                ctx.fillText(badges, tagX + 6, badgeY + 8, badgeW - 10);
              }
              ctx.restore();
            });
          }
          if (now - lastStatusAt >= 500) {
            lastStatusAt = now;
            setMaskAgeMs(oldestMaskAge === null ? null : Math.round(oldestMaskAge));
          }
        }
      }
    };

    const onVideoFrame = () => {
      if (disposed || !video) return;
      lastVideoAt = performance.now();
      render();
      videoFrameId = video.requestVideoFrameCallback(onVideoFrame);
    };
    const onAnimationFrame = () => {
      if (disposed) return;
      render();
      animId = requestAnimationFrame(onAnimationFrame);
    };
    const followsVideo = video && typeof video.requestVideoFrameCallback === 'function';
    if (followsVideo) videoFrameId = video.requestVideoFrameCallback(onVideoFrame);
    else animId = requestAnimationFrame(onAnimationFrame);
    const watchdog = setInterval(() => {
      if (document.hidden || (followsVideo && performance.now() - lastVideoAt > 350)) {
        clear();
        setMaskAgeMs(null);
      }
    }, 100);
    document.addEventListener('visibilitychange', clear);
    video?.addEventListener('emptied', clear);
    return () => {
      disposed = true;
      cancelAnimationFrame(animId);
      if (videoFrameId) video?.cancelVideoFrameCallback(videoFrameId);
      clearInterval(watchdog);
      resize.disconnect();
      document.removeEventListener('visibilitychange', clear);
      video?.removeEventListener('emptied', clear);
      clear();
    };
  }, [cam.id, metadataMap, isActive, isVisible]);

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
        {(metadataCount > 0 || !metadataConnected) && (
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
            <span style={{ color: metadataCount > 0 ? '#22d3ee' : '#eab308', fontSize: '12px' }}>
              {metadataCount > 0 ? '🎯' : '⚠️'}
            </span>
            <span>
              {metadataCount > 0 ? `Đang bám vết ${metadataCount} đối tượng` : 'Đang nối lại metadata tracking'}
            </span>
          </div>
        )}

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
});

export default function MonitorView({ isActive = true }: { isActive?: boolean } = {}) {
	  const { cameras } = useCameras();
	  const [activeTab, setActiveTab] = useState('all');
	  const [hostName] = useState(() => (typeof window !== 'undefined' ? (window.location.hostname || '192.168.0.84') : '192.168.0.84'));
	  const [totalDetections, setTotalDetections] = useState(0);
	  const [liveAlerts, setLiveAlerts] = useState<LiveAlert[]>([]);
	  const [globalTrackList, setGlobalTrackList] = useState<{ id: number; fx: number; fy: number; cam: string; class?: string }[]>([]);
	  const [floorTracks, setFloorTracks] = useState<InterpolatedFloorTrack[]>([]);
	  const [mapOverview, setMapOverview] = useState<MapOverviewItem[]>([]);
  const [cameraRois, setCameraRois] = useState<Record<string, ROIState[]>>({});
  const [metadataConnected, setMetadataConnected] = useState(false);
  const [metadataCounts, setMetadataCounts] = useState<Record<string, number>>({});
  const [labelCameraId, setLabelCameraId] = useState<string | null>(null);
  const [learningCameraId, setLearningCameraId] = useState<string | null>(null);
  const [slotFilter, setSlotFilter] = useState<'ALL' | 'OCCUPIED' | 'EMPTY'>('ALL');
  const { t } = useLanguage();

  const lastUiUpdateRef = useRef<number>(0);

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

      const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
      const metadataClock = createMetadataClock();
      const latestPacketByCam = new Map<string, number>();
      const disconnectMeta = connectRealtimeSocket({
        url: `${protocol}://${host}:8000/ws/metadata`,
        idleTimeoutMs: 3000,
        onStatus: setMetadataConnected,
        onMessage: (event) => {
          try {
            const data = JSON.parse(event.data);
            if (!data || (!Array.isArray(data.streams) && data.type !== 'metadata_heartbeat')) return false;
            const transportAgeMs = metadataClock(data.timestamp);
            if (transportAgeMs === null) return false;
            if (data.type === 'metadata_heartbeat') return true;
            if (data.streams && Array.isArray(data.streams)) {
              let count = 0;
              const now = Date.now();
              const source = data.source || 'deepstream';
              if (source === 'identity_template' || source === 'identity_people') return true;
              const activeGlobals: { id: number; fx: number; fy: number; cam: string; class?: string }[] = [];
              let processedAnyStream = false;

              data.streams.forEach((stream: { cam_id: string; model_id?: string; generation?: string; monitor_hidden?: boolean;
                objects: TrackedObject[]; rois?: ROIState[]; identity_rejected_labels?: string[];
                segmentation?: { revoked_track_ids?: number[]; labels?: { active_labels?: string[] } } }, index: number) => {
                const camId = resolveStreamCamId(stream.cam_id, index, camerasRef.current);
                if (Number.isFinite(data.timestamp)) {
                  if (data.timestamp < (latestPacketByCam.get(camId) ?? -Infinity)) return;
                  latestPacketByCam.set(camId, data.timestamp);
                }
                const identities = new Map<string, TrackedObject>();
                (Array.isArray(stream.objects) ? stream.objects : []).forEach(obj => {
                  if (!shouldShowTrackIdentity(obj.class, obj.model_id)
                    || /lost|removed|deleted/i.test(obj.tracking_state || '')) return;
                  const key = obj.model_id ? `model:${obj.model_id}:${obj.id}` : hasRegisteredLabel(obj.label) ? `label:${obj.label!.trim().toLowerCase()}` : `person:${obj.id}`;
                  const existing = identities.get(key);
                  const newer = (obj.observed_at ?? 0) > (existing?.observed_at ?? 0);
                  const sameTime = obj.observed_at === existing?.observed_at;
                  const betterMask = validRegisteredMask(obj.mask) && !validRegisteredMask(existing?.mask);
                  if (!existing || newer || (sameTime && (betterMask
                    || (validRegisteredMask(obj.mask) === validRegisteredMask(existing.mask)
                      && (obj.confidence ?? 0) > (existing.confidence ?? 0))))) identities.set(key, obj);
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
                const rejectedLabels = new Set((stream.identity_rejected_labels || []).map(label => label.trim().toLowerCase()));
                tracks.forEach((track, trackId) => {
                  if (!track.label || !rejectedLabels.has(track.label.trim().toLowerCase())) return;
                  tracks.delete(trackId);
                  if (floorTrackMap.current.get(trackId)?.cam === camId) floorTrackMap.current.delete(trackId);
                });
                streamObjects.forEach(obj => {
                  if (obj.model_id || tracks.has(obj.id) || !hasRegisteredLabel(obj.label) || !isRobotClass(obj.class)) return;
                  const previous = Array.from(tracks.entries()).find(([, track]) =>
                    track.label?.trim().toLowerCase() === obj.label!.trim().toLowerCase());
                  if (previous && now - previous[1].lastUpdated <= TRACK_HOLD_MS) {
                    tracks.delete(previous[0]);
                    previous[1].id = obj.id;
                    tracks.set(obj.id, previous[1]);
                  }
                });
                const incomingIds = new Set(streamObjects.map(obj => obj.id));
                const revokedIds = new Set(stream.segmentation?.revoked_track_ids || []);
                const activeLabels = stream.segmentation?.labels?.active_labels;
                const activeLabelNames = activeLabels ? new Set(activeLabels.map(label => label.trim().toLowerCase())) : null;
                tracks.forEach((track, trackId) => {
                  const removed = stream.objects?.some(obj => obj.id === trackId
                    && /lost|removed|deleted/i.test(obj.tracking_state || ''));
                  const streamApplies = !stream.model_id || track.model_id === stream.model_id;
                  const matchingObject = streamObjects.find(obj => obj.id === trackId && obj.model_id === track.model_id);
                  const invalidated = (streamApplies && stream.monitor_hidden) || (streamApplies && revokedIds.has(trackId))
                    || (streamApplies && track.generation && matchingObject?.generation && track.generation !== matchingObject.generation)
                    || (track.labelPrompt && activeLabelNames && !activeLabelNames.has(track.label?.trim().toLowerCase() || ''));
                  const replaced = track.model_id && track.label && streamObjects.some(obj => obj.id !== trackId
                    && obj.model_id === track.model_id && obj.identity_verified
                    && obj.label?.trim().toLowerCase() === track.label?.trim().toLowerCase());
                  const expired = track.model_id ? !isLiveRegisteredMask(track.maskFrame, now)
                    : !isRobotClass(track.class) || now - track.lastUpdated > TRACK_HOLD_MS;
                  if ((streamApplies && removed) || invalidated || (streamApplies && replaced) || (streamApplies && !incomingIds.has(trackId) && expired)) {
                    tracks.delete(trackId);
                    if (floorTrackMap.current.get(trackId)?.cam === camId) floorTrackMap.current.delete(trackId);
                  }
                });

                streamObjects.forEach(obj => {
                  if (stream.monitor_hidden || revokedIds.has(obj.id)) return;
                  const previous = tracks.get(obj.id);
                  if (previous?.observedAt !== undefined && obj.observed_at !== undefined
                    && obj.observed_at < previous.observedAt) return;
                  const repeated = previous?.observedAt !== undefined && previous.observedAt === obj.observed_at;
                  const receivedAt = repeated ? previous!.lastUpdated
                    : metadataReceivedAt(now, obj.observed_at, data.timestamp, transportAgeMs);
                  const maskFrame = obj.model_id
                    ? updateLiveRegisteredMask(previous?.maskFrame, obj, now, data.timestamp, transportAgeMs)
                    : null;
                  const showIdentity = shouldShowTrackIdentity(obj.class, obj.model_id);
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
                        cam: camId, lastUpdated: receivedAt
                      });
                    } else {
                      const ft = floorTrackMap.current.get(obj.id)!;
                      if (obj.label || obj.class) ft.class = obj.label || obj.class;
                      ft.targetFx = fx; ft.targetFy = fy;
                      ft.cam = camId; ft.lastUpdated = receivedAt;
                    }
                  }

                  if (!tracks.has(obj.id)) {
                    tracks.set(obj.id, {
                      id: obj.id, local_id: obj.local_id, class: obj.class, label: obj.label, model_id: obj.model_id,
                      generation: obj.generation || stream.generation, labelPrompt: Boolean(obj.label_prompt_id),
                      maskFrame, observedAt: obj.observed_at,
                      curX: obj.x, curY: obj.y, curW: obj.w, curH: obj.h,
                      targetX: obj.x, targetY: obj.y, targetW: obj.w, targetH: obj.h,
                      floorX: fx, floorY: fy, lastUpdated: receivedAt,
                      fms_status: obj.fms_status,
                      fms_battery: obj.fms_battery,
                      fms_speed: obj.fms_speed,
                      carried_rack: obj.carried_rack,
                      cross_check: obj.cross_check,
                      delta_distance_m: obj.delta_distance_m,
                      fms_pos: obj.fms_pos,
                      keypoints: obj.keypoints,
                      obb: obj.obb,
                      attributes: obj.attributes,
                      alert: obj.alert,
                      posture: obj.posture,
                      fall_detected: obj.fall_detected,
                      world_position: obj.world_position,
                      spatial_valid: obj.spatial_valid,
                    });
                  } else {
                    const track = tracks.get(obj.id)!;
                    track.class = obj.class;
                    track.local_id = obj.local_id;
                    track.label = obj.label;
                    track.model_id = obj.model_id;
                    track.generation = obj.generation || stream.generation;
                    track.labelPrompt = Boolean(obj.label_prompt_id);
                    track.maskFrame = maskFrame;
                    track.observedAt = obj.observed_at;
                    track.targetX = obj.x; track.targetY = obj.y;
                    track.targetW = obj.w; track.targetH = obj.h;
                    track.floorX = fx; track.floorY = fy;
                    track.lastUpdated = receivedAt;
                    if (obj.fms_status !== undefined) track.fms_status = obj.fms_status;
                    if (obj.fms_battery !== undefined) track.fms_battery = obj.fms_battery;
                    if (obj.fms_speed !== undefined) track.fms_speed = obj.fms_speed;
                    if (obj.carried_rack !== undefined) track.carried_rack = obj.carried_rack;
                    if (obj.cross_check !== undefined) track.cross_check = obj.cross_check;
                    if (obj.delta_distance_m !== undefined) track.delta_distance_m = obj.delta_distance_m;
                    if (obj.fms_pos !== undefined) track.fms_pos = obj.fms_pos;
                    if (obj.keypoints !== undefined) track.keypoints = obj.keypoints;
                    if (obj.obb !== undefined) track.obb = obj.obb;
                    if (obj.attributes !== undefined) track.attributes = obj.attributes;
                    if (obj.alert !== undefined) track.alert = obj.alert;
                    if (obj.posture !== undefined) track.posture = obj.posture;
                    if (obj.fall_detected !== undefined) track.fall_detected = obj.fall_detected;
                    if (obj.world_position !== undefined) track.world_position = obj.world_position;
                    if (obj.spatial_valid !== undefined) track.spatial_valid = obj.spatial_valid;
                  }
                });
              });

              if (!processedAnyStream) return true;
              if (now - lastUiUpdateRef.current >= 250) {
                lastUiUpdateRef.current = now;
                setTotalDetections(count);
                setMetadataCounts(Object.fromEntries(Array.from(metadataMap.current, ([cameraId, tracks]) => [
                  cameraId, Array.from(tracks.values()).filter(track => now - track.lastUpdated <= TRACK_HOLD_MS).length,
                ])));
                setGlobalTrackList(activeGlobals);
                setCameraRois(Object.fromEntries(roisMap.current));
              }
            }
            return true;
          } catch { return false; }
        },
      });

      const disconnectEvents = connectRealtimeSocket({
        url: `${protocol}://${host}:8000/ws/events`,
        onMessage: (event) => {
          try {
            const data = JSON.parse(event.data);
            if (data.event) {
              setLiveAlerts(prev => [data.event, ...prev].slice(0, 30));
            }
            return true;
          } catch { return false; }
        },
      });

      return () => {
        disconnectMeta();
        disconnectEvents();
      };
    }
	  }, [hostName]);

	  const { colors: C, isDark } = useAppTheme();
	  const activeStorageRois = activeTab !== 'all'
	    ? (cameraRois[activeTab] || []).filter(isStorageSlotRoi)
	    : [];

	  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', background: C.bg, transition: 'background-color 0.2s ease' }}>
      {isActive && labelCameraId && <MonitorLabelDialog cameraId={labelCameraId} onClose={() => setLabelCameraId(null)} />}
      {isActive && learningCameraId && <ActiveLearningDialog cameraId={learningCameraId} onClose={() => setLearningCameraId(null)} />}

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
                    isActive={isActive}
                    activeTab={activeTab}
                    metadataMap={metadataMap}
                    roisMap={roisMap}
                    metadataCount={metadataCounts[cam.id] || 0}
                    metadataConnected={metadataConnected}
                    onLabel={setLabelCameraId}
                    onLearn={setLearningCameraId}
                  />
                ))}
              </div>
            </div>


            {/* Right: Side Panel (AI Chatbot & Real-Time Environment Hub) */}
            <div style={{ flex: 1.2, display: 'flex', flexDirection: 'column', minWidth: '340px' }}>
              <MonitorChatAssistant
                cameras={cameras}
                liveAlerts={liveAlerts}
                storageSlots={allStorageSlots}
                hostName={hostName}
                onSelectCameraTab={setActiveTab}
              />
            </div>
          </>
        )}
      </div>

    </div>
  );
}
