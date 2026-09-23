'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { Crosshair, X } from 'lucide-react';
import styles from './CameraProjectionPanel.module.css';

type Camera = { camera_id: string; name: string; save_id?: string; physical_pose_known: boolean; footprint: number[][]; error?: string };
type Point = [number, number];
type Projection = { image: Point; floor: Point; fms: Point | null; inside_image: boolean; inside_calibrated_area: boolean };
type Space = 'image' | 'floor' | 'fms';

export default function CameraProjectionPanel({ cameras, floorPoint, onClose }: { cameras: Camera[]; floorPoint: Point | null; onClose: () => void }) {
  const [cameraId, setCameraId] = useState('');
  const selected = cameras.find(camera => camera.camera_id === cameraId) || cameras[0];
  const [source, setSource] = useState<Space>('fms');
  const [first, setFirst] = useState('');
  const [second, setSecond] = useState('');
  const [projection, setProjection] = useState<{ cameraId: string; saveId?: string; point: Projection }>();
  const result = projection?.cameraId === selected?.camera_id && projection?.saveId === selected?.save_id ? projection?.point : undefined;
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const requestRef = useRef<AbortController | null>(null);
  const project = useCallback(async (points: Point, space: Space) => {
    requestRef.current?.abort();
    if (!selected || !points.every(Number.isFinite)) return;
    const controller = new AbortController();
    requestRef.current = controller;
    setBusy(true); setError(''); setProjection(undefined);
    try {
      const response = await fetch(`/api/backend/camera/${encodeURIComponent(selected.camera_id)}/calibration/project`, {
        method: 'POST', signal: controller.signal, headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ points: [points], source: space, save_id: selected.save_id }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : `HTTP ${response.status}`);
      if (!controller.signal.aborted) setProjection({ cameraId: selected.camera_id, saveId: selected.save_id, point: data.points[0] });
    } catch (reason) {
      if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      if (!controller.signal.aborted) setBusy(false);
    }
  }, [selected]);
  useEffect(() => {
    if (floorPoint) void project(floorPoint, 'floor');
    return () => requestRef.current?.abort();
  }, [floorPoint, project]);
  const resetProjection = () => {
    requestRef.current?.abort();
    setProjection(undefined);
    setError('');
    setBusy(false);
  };
  const format = (point?: Point | null) => point ? point.map(value => value.toFixed(4)).join(', ') : '—';
  return <aside className={styles.panel}>
    <header><Crosshair size={16} /><strong>CAMERA ↔ FMS</strong><button onClick={onClose} aria-label="Đóng quy chiếu"><X size={16} /></button></header>
    <p>Giữ <b>Shift + click</b> trên mặt sàn map3D để tìm điểm tương ứng trên ảnh gốc.</p>
    <select aria-label="Camera hiệu chuẩn" value={selected?.camera_id || ''} onChange={event => { resetProjection(); setCameraId(event.target.value); }}>{!cameras.length && <option>Chưa có camera đã calib</option>}{cameras.map(camera => <option key={camera.camera_id} value={camera.camera_id}>{camera.name}</option>)}</select>
    <small>Quy chiếu theo hiệu chuẩn đã lưu; không vẽ vị trí hoặc vùng phủ camera trên map.</small>
    <form onSubmit={event => { event.preventDefault(); void project([Number(first), Number(second)], source); }}>
      <label>Hoặc nhập tọa độ<select value={source} onChange={event => { resetProjection(); setSource(event.target.value as Space); setFirst(''); setSecond(''); }}><option value="fms">FMS (X, Y) · mét</option><option value="image">Ảnh gốc (u, v) · 0..1</option><option value="floor">Mặt sàn map (X, Z) · mét</option></select></label>
      <div className={styles.pair}><input aria-label="Tọa độ thứ nhất" required type="number" step="any" min={source === 'image' ? 0 : undefined} max={source === 'image' ? 1 : undefined} value={first} onChange={event => setFirst(event.target.value)} /><input aria-label="Tọa độ thứ hai" required type="number" step="any" min={source === 'image' ? 0 : undefined} max={source === 'image' ? 1 : undefined} value={second} onChange={event => setSecond(event.target.value)} /></div>
      <button disabled={!selected || busy || !selected.footprint?.length}>{busy ? 'Đang quy chiếu…' : 'Quy đổi tọa độ'}</button>
    </form>
    {(error || selected?.error) && <p role="alert" className={styles.warning}>{error || selected?.error}</p>}
    {result && <dl><dt>Ảnh gốc u, v</dt><dd>{format(result.image)}</dd><dt>Mặt sàn X, Z (m)</dt><dd>{format(result.floor)}</dd><dt>FMS X, Y (m)</dt><dd>{format(result.fms)}</dd><dt>Độ tin cậy vùng</dt><dd className={!result.inside_calibrated_area || !result.inside_image ? styles.warning : ''}>{!result.inside_image ? 'Ngoài ảnh camera' : result.inside_calibrated_area ? 'Trong vùng đã hiệu chuẩn' : 'Ngoài vùng neo: chỉ ngoại suy'}</dd></dl>}
    <small>Phép chiếu chỉ áp dụng điểm trên mặt sàn; không suy ra độ cao vật thể.</small>
  </aside>;
}
