'use client';

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Camera, Crosshair, Map, RefreshCw, Ruler, Save, Sparkles, Trash2, Undo2, ZoomIn, ZoomOut, Maximize2 } from 'lucide-react';
import { useCameras } from '../CameraContext';
import { connectRealtimeSocket } from '../../lib/realtime-socket';
import { CalibrationPair, FmsFrame, Point2, floorToFms, isDistinctPair, svgPoint } from '../../lib/calibration-points';
import styles from './CalibrationView.module.css';
import CalibrationRuler from './CalibrationRuler';
import { RulerCalibration, ImageViewport, clampImageViewport, zoomImageViewport } from '../../lib/calibration-ruler';
import CalibrationInputs, { LengthConstraint } from './CalibrationInputs';
import DeepCalibPanel from './DeepCalibPanel';
import { DeepCalibPreview, DeepCalibProfile, DeepCalibStatus, snapshotDataUrl, useCalibrationDraft, visibleRectifiedPoint } from '../../lib/deepcalib';

type Layout = { name?: string; size: { width: number; depth: number }; slam_map: { href: string; rect: [number, number, number, number]; map_name: string; flip_y?: boolean } };
type Config = RulerCalibration & { src_points: Point2[]; dst_points: Point2[]; reprojection_error?: number; point_errors_m?: number[]; length_constraints?: LengthConstraint[]; save_id?: string; persisted_at?: number; intrinsic_profile?: DeepCalibProfile; deepcalib_points_rectified?: boolean; rectified_src_points?: Point2[]; rectified_length_constraints?: LengthConstraint[] };
type Robot = { id: string; fms_position?: number[]; position?: number[]; status?: string; source?: string };
type Viewport = [number, number, number, number];
type CalibrationSubtab = 'calibration' | 'deepcalib' | 'ruler';

async function readJson(response: Response) {
  const data = await response.json();
  if (!response.ok || data.error) throw new Error(typeof data.detail === 'string' ? data.detail : data.error || `HTTP ${response.status}`);
  return data;
}

export default function CalibrationView({ active }: { active: boolean }) {
  const { cameras, fetchCameras } = useCameras();
  const [camId, setCamId] = useState('');
  const [subtab, setSubtab] = useState<CalibrationSubtab>('calibration');
  const [layout, setLayout] = useState<Layout | null>(null);
  const [frame, setFrame] = useState<FmsFrame | null>(null);
  const rawDraft = useCalibrationDraft();
  const deepDraft = useCalibrationDraft();
  const isDeep = subtab === 'deepcalib';
  const { pairs, setPairs, pending, setPending, lengthConstraints, setLengthConstraints,
    lengthPoints, setLengthPoints, lengthMode, setLengthMode, dirty, setDirty } = isDeep ? deepDraft : rawDraft;
  const [cameraViewport, setCameraViewport] = useState<ImageViewport>([0, 0, 1, 1]);
  const cameraSvg = useRef<SVGSVGElement>(null);
  const cameraDrag = useRef<Point2 | null>(null);
  const cameraDragged = useRef(false);
  const [config, setConfig] = useState<Config | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const [snapshot, setSnapshot] = useState('');
  const [snapshotKey, setSnapshotKey] = useState(0);
  const [rawImageSize, setRawImageSize] = useState<Point2 | null>(null);
  const [snapshotAt, setSnapshotAt] = useState('');
  const [robots, setRobots] = useState<Robot[]>([]);
  const [connected, setConnected] = useState(false);
  const [mapReady, setMapReady] = useState(false);
  const [deepcalibStatus, setDeepcalibStatus] = useState<DeepCalibStatus | null>(null);
  const [deepPreview, setDeepPreview] = useState<DeepCalibPreview | null>(null);
  const deepRequest = useRef<AbortController | null>(null);
  const [deepcalibRunning, setDeepcalibRunning] = useState(false);
  const [viewport, setViewport] = useState<Viewport>([0, 0, 26, 18]);
  const panning = useRef<{ point: Point2; view: Viewport } | null>(null);
  const dragged = useRef(false);
  const mapSvg = useRef<SVGSVGElement>(null);
  const currentCamera = useRef(camId);
  currentCamera.current = camId;
  const preview = deepPreview?.camera_id === camId && deepPreview.snapshot === snapshot ? deepPreview : null;
  const imageSize = useMemo<Point2 | null>(() => isDeep ? preview ? [
    preview.profile.rectified_width ?? preview.profile.image_width,
    preview.profile.rectified_height ?? preview.profile.image_height,
  ] : null : rawImageSize, [isDeep, preview, rawImageSize]);
  const editorImage = isDeep ? preview?.rectified_image || '' : snapshot;

  useEffect(() => { if (!camId && cameras[0]) setCamId(cameras[0].id); }, [camId, cameras]);

  useEffect(() => {
    const controller = new AbortController();
    fetch('/api/backend/fms/layout?compact=true', { signal: controller.signal }).then(readJson).then(data => {
      if (!data.size || !data.slam_map?.href || !data.slam_map?.rect) throw new Error('Chưa có bản đồ SLAM FMS để hiệu chuẩn.');
      setLayout(data);
      setViewport([0, 0, data.size.width, data.size.depth]);
    }).catch(reason => { if (!controller.signal.aborted) setError(String(reason.message)); });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    fetch('/api/backend/deepcalib/status', { signal: controller.signal, cache: 'no-store' })
      .then(readJson).then(data => { if (!controller.signal.aborted) setDeepcalibStatus(data.deepcalib); })
      .catch(() => { if (!controller.signal.aborted) setDeepcalibStatus(null); });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (!camId) return;
    const controller = new AbortController();
    setLoading(true); setConfig(null); setFrame(null); setMessage(''); setError('');
    setDeepPreview(null);
    for (const draft of [rawDraft, deepDraft]) {
      draft.setPending(null); draft.setPairs([]); draft.setDirty(false);
      draft.setLengthConstraints([]); draft.setLengthPoints([]); draft.setLengthMode(false);
    }
    fetch(`/api/backend/camera/${encodeURIComponent(camId)}/calibration`, { signal: controller.signal, cache: 'no-store' })
      .then(readJson).then(data => {
        if (controller.signal.aborted) return;
        setFrame(data.fms_frame);
        const saved: Config | null = data.calibration;
        setConfig(saved);
        rawDraft.setLengthConstraints(saved?.length_constraints || []);
        rawDraft.setPairs(saved?.src_points?.map((camera, index) => ({ camera, floor: saved.dst_points[index] })) || []);
        deepDraft.setLengthConstraints(saved?.rectified_length_constraints || []);
        deepDraft.setPairs(saved?.rectified_src_points?.map((camera, index) => ({ camera, floor: saved.dst_points[index] })) || []);
      }).catch(reason => { if (!controller.signal.aborted) setError(String(reason.message)); })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [camId]);

  useEffect(() => {
    if (!camId) return;
    const controller = new AbortController();
    let objectUrl = '';
    deepRequest.current?.abort(); setDeepcalibRunning(false); setDeepPreview(null);
    setSnapshot(''); setRawImageSize(null); setSnapshotAt('');
    fetch(`/api/backend/camera/${encodeURIComponent(camId)}/snapshot?t=${Date.now()}`, { signal: controller.signal, cache: 'no-store' })
      .then(async response => { if (!response.ok) throw new Error('Không lấy được ảnh camera. Bấm Lấy frame mới để thử lại.'); return response.blob(); })
      .then(blob => { if (!controller.signal.aborted) { objectUrl = URL.createObjectURL(blob); setSnapshot(objectUrl); setSnapshotAt(new Date().toLocaleTimeString()); } })
      .catch(reason => { if (!controller.signal.aborted) setError(String(reason.message)); });
    return () => { controller.abort(); deepRequest.current?.abort(); if (objectUrl) URL.revokeObjectURL(objectUrl); };
  }, [camId, snapshotKey]);

  useEffect(() => { if (imageSize) setCameraViewport([0, 0, imageSize[0], imageSize[1]]); }, [imageSize]);

  useEffect(() => {
    if (!active || subtab === 'ruler') { setConnected(false); return; }
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    return connectRealtimeSocket({ url: `${protocol}//${window.location.hostname}:8000/ws/digital_twin`, idleTimeoutMs: 3000,
      onStatus: value => { setConnected(value); if (!value) setRobots([]); },
      onMessage: event => {
        const data = JSON.parse(event.data);
        if (!['DIGITAL_TWIN_TELEMETRY', 'DIGITAL_TWIN_SYNC'].includes(data.type) || !data.robots) return false;
        const incoming: Robot[] = Array.isArray(data.robots) ? data.robots : Object.values(data.robots);
        setRobots(incoming.filter(robot => robot.status !== 'OFFLINE' && Boolean(robot.fms_position)));
        return true;
      },
    });
  }, [active, subtab]);

  const canDraw = !loading && !saving && !deepcalibRunning && Boolean(frame && imageSize && editorImage);
  const savedDate = config?.calibration_updated_at ? new Date(config.calibration_updated_at * 1000).toLocaleString() : null;
  const changePairs = (next: CalibrationPair[]) => { setPairs(next); setDirty(true); setMessage(''); setError(''); };

  const clickCamera = (event: React.MouseEvent<SVGSVGElement>) => {
    if (cameraDragged.current) { cameraDragged.current = false; return; }
    if (!canDraw || !imageSize) return;
    const point = svgPoint(event.currentTarget, event.clientX, event.clientY);
    if (!point || point[0] < 0 || point[1] < 0 || point[0] > imageSize[0] || point[1] > imageSize[1]) return;
    const normalized: Point2 = [point[0] / imageSize[0], point[1] / imageSize[1]];
    if (isDeep && preview && !visibleRectifiedPoint(normalized, preview.profile)) {
      setError('Vị trí này nằm trong viền trống của ảnh hiệu chỉnh. Hãy chọn điểm trên mặt sàn nhìn thấy được.'); return;
    }
    if (lengthMode) { setLengthPoints(previous => previous.length === 2 ? [normalized] : [...previous, normalized]); return; }
    setPending(normalized);
    setError(''); setMessage('');
  };

  const clickMap = (event: React.MouseEvent<SVGSVGElement>) => {
    if (dragged.current) { dragged.current = false; return; }
    if (!canDraw || !mapReady || event.shiftKey || !pending || !layout) return;
    const point = svgPoint(event.currentTarget, event.clientX, event.clientY);
    if (!point || point[0] < 0 || point[1] < 0 || point[0] > layout.size.width || point[1] > layout.size.depth) return;
    if (!isDistinctPair(pairs, pending, point)) { setError('Điểm bị trùng. Hãy chọn vị trí khác cho cặp điểm mới.'); return; }
    changePairs([...pairs, { camera: pending, floor: point }]);
    setPending(null);
  };

  const zoom = useCallback((factor: number, center?: Point2) => {
    if (!layout) return;
    setViewport(previous => {
      const width = Math.min(layout.size.width, Math.max(layout.size.width / 8, previous[2] * factor));
      const height = width * layout.size.depth / layout.size.width;
      const anchor = center || [previous[0] + previous[2] / 2, previous[1] + previous[3] / 2];
      return [Math.max(0, Math.min(layout.size.width - width, anchor[0] - (anchor[0] - previous[0]) * width / previous[2])),
        Math.max(0, Math.min(layout.size.depth - height, anchor[1] - (anchor[1] - previous[1]) * height / previous[3])), width, height];
    });
  }, [layout]);

  useEffect(() => {
    const svg = mapSvg.current;
    if (!svg || !active || subtab === 'ruler') return;
    const wheel = (event: WheelEvent) => { event.preventDefault(); const point = svgPoint(svg, event.clientX, event.clientY); zoom(event.deltaY > 0 ? 1.15 : 1 / 1.15, point || undefined); };
    svg.addEventListener('wheel', wheel, { passive: false });
    return () => svg.removeEventListener('wheel', wheel);
  }, [active, subtab, layout, zoom]);

  useEffect(() => {
    const svg = cameraSvg.current;
    if (!svg || !imageSize || !active || subtab === 'ruler') return;
    const wheel = (event: WheelEvent) => { event.preventDefault(); const point = svgPoint(svg, event.clientX, event.clientY);
      setCameraViewport(previous => zoomImageViewport(previous, imageSize, event.deltaY > 0 ? 1.15 : 1 / 1.15, point || undefined)); };
    svg.addEventListener('wheel', wheel, { passive: false });
    return () => svg.removeEventListener('wheel', wheel);
  }, [active, subtab, imageSize, snapshot]);

  const save = async () => {
    if (!frame || !layout || pairs.length < 4 || pending || lengthPoints.length || !canDraw) return;
    const method = isDeep ? 'deepcalib_camera_fms' : 'manual_camera_fms_click';
    const requestCamera = camId;
    setSaving(true); setError(''); setMessage('');
    try {
      const data = await fetch(`/api/backend/camera/${encodeURIComponent(requestCamera)}/calibration`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ src_points: pairs.map(pair => pair.camera), dst_points: pairs.map(pair => pair.floor),
          method, map_id: layout.slam_map.map_name, fms_frame: frame, length_constraints: lengthConstraints,
          points_space: isDeep ? 'rectified' : 'raw', deepcalib_preview_id: isDeep ? preview?.preview_id : undefined }),
      }).then(readJson);
      if (currentCamera.current !== requestCamera) return;
      if (!data.persisted || !data.save_id) throw new Error('Backend chưa xác nhận đã lưu PostgreSQL. Hiệu chuẩn chưa được xác nhận.');
      setConfig(data.config); setDirty(false);
      if (isDeep && !rawDraft.dirty) {
        rawDraft.setPairs(data.config.src_points.map((camera: Point2, index: number) => ({ camera, floor: data.config.dst_points[index] })));
        rawDraft.setLengthConstraints(data.config.length_constraints || []);
      }
      setMessage(`${method === 'deepcalib_camera_fms' ? 'DeepCalib' : 'Calibration'} đã lưu ${pairs.length} cặp điểm vào PostgreSQL${data.save_id ? ` · mã ${data.save_id}` : ''} và kích hoạt cho camera này.`);
      await fetchCameras();
    } catch (reason) { if (currentCamera.current === requestCamera) setError(reason instanceof Error ? reason.message : String(reason)); }
    finally { setSaving(false); }
  };

  const runDeepcalib = async (reuse = false) => {
    if (!camId || !snapshot || deepcalibRunning || loading || saving) return;
    if (deepDraft.dirty && !confirm('Chạy lại ảnh hiệu chỉnh sẽ bỏ bản nháp DeepCalib chưa lưu. Tiếp tục?')) return;
    const requestCamera = camId;
    const requestSnapshot = snapshot;
    const controller = new AbortController();
    deepRequest.current?.abort(); deepRequest.current = controller;
    setDeepcalibRunning(true); setError(''); setMessage('');
    try {
      const image = await snapshotDataUrl(requestSnapshot, controller.signal);
      const data = await fetch(`/api/backend/camera/${encodeURIComponent(requestCamera)}/deepcalib/estimate`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, cache: 'no-store', signal: controller.signal,
        body: JSON.stringify({ image, use_saved_profile: reuse, calibration_save_id: reuse ? config?.save_id : undefined }),
      }).then(readJson);
      if (controller.signal.aborted || currentCamera.current !== requestCamera) return;
      setDeepPreview({ ...data, snapshot: requestSnapshot });
      if (data.deepcalib) setDeepcalibStatus(data.deepcalib);
      deepDraft.setPairs(reuse ? config?.rectified_src_points?.map((camera, index) => ({ camera, floor: config.dst_points[index] })) || [] : []);
      deepDraft.setLengthConstraints(reuse ? config?.rectified_length_constraints || [] : []);
      deepDraft.setPending(null); deepDraft.setLengthPoints([]); deepDraft.setLengthMode(false); deepDraft.setDirty(!reuse);
      setMessage(reuse ? 'Đã mở ảnh bằng profile đã lưu. Có thể thêm điểm hoặc đoạn đo rồi Save.' : 'Đã hiệu chỉnh ảnh bằng DeepCalib. Chấm các cặp ảnh ↔ FMS bên dưới, thêm đoạn đo mét rồi Save.');
    } catch (reason) { if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : String(reason)); }
    finally { if (deepRequest.current === controller) setDeepcalibRunning(false); }
  };

  const refreshSnapshot = () => {
    if (saving || deepcalibRunning) return;
    if ((rawDraft.dirty || deepDraft.dirty || pending || lengthPoints.length) && !confirm('Ảnh mới sẽ bỏ các điểm chưa lưu. Tiếp tục?')) return;
    rawDraft.setPairs(config?.src_points?.map((camera, index) => ({ camera, floor: config.dst_points[index] })) || []);
    rawDraft.setLengthConstraints(config?.length_constraints || []);
    deepDraft.setPairs(config?.rectified_src_points?.map((camera, index) => ({ camera, floor: config.dst_points[index] })) || []);
    deepDraft.setLengthConstraints(config?.rectified_length_constraints || []);
    for (const draft of [rawDraft, deepDraft]) { draft.setDirty(false); draft.setPending(null); draft.setLengthPoints([]); }
    setError(''); setMessage(''); setSnapshotKey(value => value + 1);
  };

  const marker = (point: Point2, index: number, radius: number, waiting = false) => <g key={index} pointerEvents="none">
    <circle cx={point[0]} cy={point[1]} r={radius} fill={waiting ? '#f59e0b' : '#ffd77d'} stroke="#171d25" strokeWidth={radius * .13} />
    <text x={point[0]} y={point[1]} dy=".34em" textAnchor="middle" fill="#17202a" fontWeight="800" fontSize={radius * 1.2}>{index + 1}</text>
    <circle cx={point[0]} cy={point[1]} r={radius * 1.5} fill="none" stroke="#ffd77d" strokeWidth={radius * .07} />
  </g>;

  const rect = layout?.slam_map.rect;
  return <section className={styles.view}>
    <div className={styles.subtabs} role="tablist" aria-label="Chức năng Calibration" onKeyDown={event => {
      if (saving || deepcalibRunning) return;
      if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
      event.preventDefault();
      const tabs: CalibrationSubtab[] = ['calibration', 'deepcalib', 'ruler'];
      const current = tabs.indexOf(subtab);
      const next = event.key === 'Home' ? tabs[0] : event.key === 'End' ? tabs[tabs.length - 1] : tabs[(current + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length];
      setSubtab(next); document.getElementById(`calibration-tab-${next}`)?.focus();
    }}>
      <button disabled={saving || deepcalibRunning} role="tab" id="calibration-tab-calibration" aria-controls="calibration-panel-calibration" aria-selected={subtab === 'calibration'} tabIndex={subtab === 'calibration' ? 0 : -1} onClick={() => setSubtab('calibration')}><Crosshair size={16} />Hiệu chuẩn Robot / FMS</button>
      <button disabled={saving || deepcalibRunning} role="tab" id="calibration-tab-deepcalib" aria-controls="calibration-panel-deepcalib" aria-selected={subtab === 'deepcalib'} tabIndex={subtab === 'deepcalib' ? 0 : -1} onClick={() => setSubtab('deepcalib')}><Sparkles size={16} />DeepCalib pipeline</button>
      <button disabled={saving || deepcalibRunning} role="tab" id="calibration-tab-ruler" aria-controls="calibration-panel-ruler" aria-selected={subtab === 'ruler'} tabIndex={subtab === 'ruler' ? 0 : -1} onClick={() => setSubtab('ruler')}><Ruler size={16} />Thước đo</button>
    </div>
    <div className={styles.toolbar}>
      <div className={styles.actions}><Camera size={18} /><select aria-label="Camera hiệu chuẩn" value={camId} disabled={saving || deepcalibRunning} onChange={event => {
        if ((!rawDraft.dirty && !deepDraft.dirty && !rawDraft.pending && !deepDraft.pending) || confirm('Chuyển camera sẽ bỏ các điểm chưa lưu. Tiếp tục?')) setCamId(event.target.value);
      }}>{!cameras.length && <option value="">Chưa có camera</option>}{cameras.map(camera => <option value={camera.id} key={camera.id}>{camera.name}</option>)}</select></div>
      {subtab !== 'ruler' && <div className={styles.actions}>
        <button disabled={!pairs.length && !pending || !canDraw} onClick={() => { if (pending) setPending(null); else changePairs(pairs.slice(0, -1)); }}><Undo2 size={15} />Hoàn tác</button>
        <button disabled={!pairs.length || !canDraw} onClick={() => { if (confirm('Xóa bản nháp điểm? Hiệu chuẩn đang chạy chỉ thay đổi khi bạn bấm Save.')) { changePairs([]); setLengthConstraints([]); setLengthPoints([]); setPending(null); } }}><Trash2 size={15} />Chấm lại</button>
        <button className={styles.save} onClick={() => save()} disabled={!dirty || pairs.length < 4 || Boolean(pending) || lengthPoints.length > 0 || !canDraw}><Save size={15} />{saving ? 'Đang lưu…' : isDeep ? 'Save DeepCalib · Áp dụng' : 'Save · Áp dụng'}</button>
      </div>}
    </div>
    <div role="tabpanel" id="calibration-panel-deepcalib" aria-labelledby="calibration-tab-deepcalib" hidden={subtab !== 'deepcalib'}>
      <DeepCalibPanel snapshot={snapshot} preview={preview} status={deepcalibStatus}
        pairCount={deepDraft.pairs.length} lengthCount={deepDraft.lengthConstraints.length}
        saved={Boolean(config?.deepcalib_points_rectified && !deepDraft.dirty)} canReuse={Boolean(config?.deepcalib_points_rectified && (config.intrinsic_profile?.geometry_version ?? 1) >= 2)}
        onRun={runDeepcalib} onRefresh={refreshSnapshot} running={deepcalibRunning} disabled={loading || saving || !camId} />
    </div>
    {snapshot && <img src={snapshot} alt="" style={{ display: 'none' }} onLoad={event => setRawImageSize([event.currentTarget.naturalWidth, event.currentTarget.naturalHeight])} onError={() => setError('Không giải mã được ảnh camera.')} />}
    <div role="tabpanel" id="calibration-panel-calibration" aria-labelledby={isDeep ? 'calibration-tab-deepcalib' : 'calibration-tab-calibration'} hidden={subtab === 'ruler'}><div className={styles.subpanel}>
    <div className={styles.guide}><strong>{pending ? `Cặp #${pairs.length + 1}: chấm điểm tương ứng trên bản đồ FMS bên phải.` : `Chấm điểm #${pairs.length + 1} trên ảnh camera, rồi chấm đúng vị trí đó trên FMS.`}</strong>
      <p>Ít nhất 4 cặp điểm, không giới hạn số cặp. Chọn các điểm cố định trên cùng mặt sàn, trải rộng khắp vùng cần tracking. Không chấm trên nóc robot/kệ. Bấm Save khi hoàn tất; hiệu chuẩn cũ vẫn hoạt động trong lúc bạn chấm.</p>
    </div>
    <CalibrationInputs key={`${camId}:${isDeep}:${preview?.preview_id || ''}`} pairs={pairs} pending={pending} imageSize={imageSize} frame={frame} disabled={!canDraw}
      onPair={(pair, index) => { changePairs(index === null ? [...pairs, pair] : pairs.map((previous, number) => number === index ? pair : previous)); setPending(null); }}
      constraints={lengthConstraints} lengthPoints={lengthPoints} lengthMode={lengthMode}
      onLengthMode={value => { setLengthMode(value); setLengthPoints([]); setPending(null); }}
      onLengths={value => { setLengthConstraints(value); setDirty(true); setMessage(''); }} />
    <div className={styles.panels}>
      <article className={styles.panel}><div className={styles.heading}><h2><Camera size={16} />{isDeep ? 'Ảnh hiệu chỉnh · chấm điểm tại đây' : 'Ảnh camera gốc'}</h2><button disabled={!camId || saving || deepcalibRunning} onClick={refreshSnapshot}><RefreshCw size={13} />Lấy frame mới</button></div>
        <div className={styles.stage}>
          {editorImage && imageSize ? <svg ref={cameraSvg} aria-label={isDeep ? 'Chấm điểm trên ảnh hiệu chỉnh' : 'Chấm điểm trên camera'} viewBox={cameraViewport.join(' ')} onClick={clickCamera}
            onPointerDown={event => { cameraDragged.current = false; if (event.shiftKey || event.button === 1) { cameraDragged.current = true; cameraDrag.current = svgPoint(event.currentTarget, event.clientX, event.clientY); event.currentTarget.setPointerCapture(event.pointerId); } }}
            onPointerMove={event => { const start = cameraDrag.current; if (!start) return; const point = svgPoint(event.currentTarget, event.clientX, event.clientY); if (point) setCameraViewport(previous => clampImageViewport([previous[0] + start[0] - point[0], previous[1] + start[1] - point[1], previous[2], previous[3]], imageSize)); }}
            onPointerUp={() => { cameraDrag.current = null; }} onPointerCancel={() => { cameraDrag.current = null; }}>
            <image href={editorImage} width={imageSize[0]} height={imageSize[1]} pointerEvents="none" />
            {[...lengthConstraints.map(item => item.points), lengthPoints].map((points, index) => <g key={`length-${index}`} pointerEvents="none"><polyline points={points.map(point => `${point[0] * imageSize[0]},${point[1] * imageSize[1]}`).join(' ')} fill="none" stroke="#2de8c4" strokeWidth={cameraViewport[2] / 450} />{points.map((point, pointIndex) => <circle key={pointIndex} cx={point[0] * imageSize[0]} cy={point[1] * imageSize[1]} r={cameraViewport[2] / 110} fill="#2de8c4" />)}</g>)}
            {pairs.map((pair, index) => marker([pair.camera[0] * imageSize[0], pair.camera[1] * imageSize[1]], index, cameraViewport[2] / 65))}
            {pending && marker([pending[0] * imageSize[0], pending[1] * imageSize[1]], pairs.length, cameraViewport[2] / 65, true)}
          </svg> : <div className={styles.placeholder}>{isDeep ? 'Chạy DeepCalib hoặc mở profile đã lưu để bắt đầu chấm trên ảnh hiệu chỉnh.' : 'Đang lấy một frame mới nhất…'}</div>}
        </div><div className={styles.caption}>Ảnh tĩnh {snapshotAt && `lúc ${snapshotAt}`} · Cuộn để zoom, Shift + kéo · {lengthMode ? `Chọn đoạn đo: ${lengthPoints.length}/2 điểm` : 'Chấm tiếp điểm trên mặt sàn'} <button onClick={() => imageSize && setCameraViewport([0, 0, ...imageSize])}>Toàn ảnh</button></div>
      </article>
      <article className={styles.panel}><div className={styles.heading}><h2><Map size={16} />FMS 2D · {layout?.slam_map.map_name || '…'}</h2><div className={styles.actions}>
        <button aria-label="Thu nhỏ bản đồ" onClick={() => zoom(1.3)}><ZoomOut size={14} /></button><button aria-label="Phóng to bản đồ" onClick={() => zoom(1 / 1.3)}><ZoomIn size={14} /></button>
        <button aria-label="Toàn bộ bản đồ" onClick={() => layout && setViewport([0, 0, layout.size.width, layout.size.depth])}><Maximize2 size={14} /></button>
      </div></div><div className={styles.stage}>
        {layout && rect ? <svg ref={mapSvg} aria-label="Chấm điểm trên bản đồ FMS" viewBox={viewport.join(' ')} onClick={clickMap}
          onPointerDown={event => { dragged.current = false; if (event.shiftKey || event.button === 1) { const point = svgPoint(event.currentTarget, event.clientX, event.clientY); if (point) { dragged.current = true; panning.current = { point, view: viewport }; event.currentTarget.setPointerCapture(event.pointerId); } } }}
          onPointerMove={event => { if (!panning.current || !layout) return; const point = svgPoint(event.currentTarget, event.clientX, event.clientY); if (!point) return; const start = panning.current;
            setViewport(previous => [Math.max(0, Math.min(layout.size.width - previous[2], previous[0] + start.point[0] - point[0])), Math.max(0, Math.min(layout.size.depth - previous[3], previous[1] + start.point[1] - point[1])), previous[2], previous[3]]); }}
          onPointerUp={() => { panning.current = null; }} onPointerCancel={() => { panning.current = null; }}>
          <rect width={layout.size.width} height={layout.size.depth} fill="#e4e7e8" pointerEvents="none" />
          <image href={layout.slam_map.href} x={rect[0]} y={rect[1]} width={rect[2] - rect[0]} height={rect[3] - rect[1]} preserveAspectRatio="none" pointerEvents="none"
            transform={layout.slam_map.flip_y ? `translate(0 ${rect[1] + rect[3]}) scale(1 -1)` : undefined} onLoad={() => setMapReady(true)} onError={() => { setMapReady(false); setError('Không tải được ảnh bản đồ FMS.'); }} />
          {connected && robots.map(robot => { const position = robot.fms_position; return position && position.every(Number.isFinite) ? <g key={robot.id} pointerEvents="none">
            <circle cx={position[0]} cy={position[2]} r={viewport[2] / 105} fill="#087e67" stroke="#fff" strokeWidth={viewport[2] / 900} />
            <text x={position[0]} y={position[2] - viewport[2] / 70} textAnchor="middle" fontSize={viewport[2] / 85} fill="#075946" stroke="#fff" strokeWidth={viewport[2] / 1500} paintOrder="stroke" fontWeight="700">{robot.id}</text>
          </g> : null; })}
          {pairs.map((pair, index) => marker(pair.floor, index, viewport[2] / 100))}
        </svg> : <div className={styles.placeholder}>Đang tải bản đồ SLAM từ FMS…</div>}
      </div><div className={styles.caption}>Cuộn để zoom · Shift + kéo để di chuyển · Không tự đổi tỷ lệ theo robot. <span className={connected ? styles.live : styles.offline}>{connected ? 'FMS đang cập nhật' : 'Đang nối lại FMS…'}</span></div></article>
    </div>
    {message && <div role="status" className={styles.message}>{message}</div>}
    <div className={styles.status}><Crosshair size={15} /><strong>{pairs.length} cặp điểm</strong><span>· {dirty ? 'Bản nháp chưa áp dụng' : config ? `Đã lưu PostgreSQL${config.save_id ? ` · ${config.save_id}` : ''} và đang áp dụng` : 'Camera chưa có hiệu chuẩn'}</span>
      {savedDate && <span>· {savedDate}</span>}{config?.reprojection_error !== undefined && <span>· Sai số khớp: {config.reprojection_error.toFixed(3)} m</span>}
    </div>
    <article className={styles.panel}><div className={styles.tableWrap}>{pairs.length ? <table className={styles.table}><thead><tr><th>Cặp</th><th>Camera (u, v)</th><th>FMS (x, y) · mét</th><th /></tr></thead><tbody>
      {pairs.map((pair, index) => { const coordinate = frame ? floorToFms(pair.floor, frame) : pair.floor; return <tr key={index}><td>#{index + 1}</td><td>{pair.camera.map(value => value.toFixed(4)).join(', ')}</td><td>{coordinate.map(value => value.toFixed(3)).join(', ')}</td>
        <td><button disabled={saving || loading} aria-label={`Xóa cặp điểm ${index + 1}`} onClick={() => changePairs(pairs.filter((_, pairIndex) => pairIndex !== index))}><Trash2 size={12} /></button></td></tr>; })}
    </tbody></table> : <div className={styles.empty}>Chưa có cặp điểm. Chấm camera → chấm FMS, lặp lại đến khi phủ đủ mặt sàn.</div>}</div></article>
    </div></div>
    <div role="tabpanel" id="calibration-panel-ruler" aria-labelledby="calibration-tab-ruler" hidden={subtab !== 'ruler'}>
      <CalibrationRuler key={`${camId}:${snapshotKey}`} cameraId={camId} active={active && subtab === 'ruler'} snapshot={snapshot} imageSize={rawImageSize} snapshotAt={snapshotAt} savedCalibration={config}
        onCalibrate={() => setSubtab(config?.deepcalib_points_rectified ? 'deepcalib' : 'calibration')} onRefresh={refreshSnapshot} />
    </div>
    {error && <div role="alert" className={`${styles.message} ${styles.error}`}>{error}</div>}
  </section>;
}
