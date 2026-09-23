'use client';

import React, { useEffect, useRef, useState } from 'react';
import { RefreshCw, Trash2, Undo2 } from 'lucide-react';
import styles from './WorkflowView.module.css';

export default function WorkflowGeometry({ cameraId, type, points, onChange }: {
  cameraId?: string; type: string; points: number[][]; onChange: (points: number[][]) => void;
}) {
  const [snapshot, setSnapshot] = useState('');
  const [size, setSize] = useState([1280, 720]);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const pending = useRef<AbortController | null>(null);
  useEffect(() => () => pending.current?.abort(), []);
  useEffect(() => () => { if (snapshot) URL.revokeObjectURL(snapshot); }, [snapshot]);

  async function capture() {
    if (!cameraId) return;
    pending.current?.abort();
    const controller = new AbortController();
    pending.current = controller;
    setLoading(true); setError('');
    try {
      const response = await fetch(`/api/backend/camera/${encodeURIComponent(cameraId)}/snapshot`, { signal: controller.signal, cache: 'no-store' });
      if (!response.ok) throw new Error('Camera chưa có ảnh; kiểm tra kết nối trong Building.');
      const blob = await response.blob();
      if (controller.signal.aborted) return;
      const url = URL.createObjectURL(blob);
      const image = new Image();
      image.src = url;
      try { await image.decode(); }
      catch (reason) { URL.revokeObjectURL(url); throw reason; }
      if (controller.signal.aborted) { URL.revokeObjectURL(url); return; }
      setSize([image.naturalWidth, image.naturalHeight]); setSnapshot(url);
    } catch (reason) { if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : String(reason)); }
    finally { if (!controller.signal.aborted) setLoading(false); }
  }

  function addPoint(event: React.MouseEvent<SVGSVGElement>) {
    if (!snapshot || points.length >= (type === 'line' ? 2 : 128)) return;
    const svg = event.currentTarget;
    const matrix = svg.getScreenCTM();
    if (!matrix) return;
    const cursor = svg.createSVGPoint();
    cursor.x = event.clientX; cursor.y = event.clientY;
    const local = cursor.matrixTransform(matrix.inverse());
    const point = [local.x / size[0], local.y / size[1]];
    if (point.some(value => value < 0 || value > 1)) return;
    onChange([...points, point.map(value => Math.round(value * 100000) / 100000)]);
  }

  return <div className={styles.configContent}>
    <strong className={styles.fieldLabel}>{type === 'line' ? 'Chấm 2 điểm tạo vạch' : 'Chấm vùng ROI trên ảnh'}</strong>
    <div className={styles.toolbarActions}>
      <button type="button" className={styles.secondaryButton} onClick={capture} disabled={!cameraId || loading}><RefreshCw size={13} />{loading ? 'Đang chụp' : 'Lấy 1 frame mới'}</button>
      <button type="button" className={styles.iconButton} title="Bỏ điểm cuối" onClick={() => onChange(points.slice(0, -1))}><Undo2 size={14} /></button>
      <button type="button" className={styles.iconButton} title="Xóa vùng để chấm lại" onClick={() => onChange([])}><Trash2 size={14} /></button>
    </div>
    {!cameraId && <p className={styles.fieldHelp}>Chọn đúng một camera nguồn trước khi vẽ vùng.</p>}
    {error && <p role="alert" className={styles.badText}>{error}</p>}
    <svg className={styles.geometry} viewBox={`0 0 ${size[0]} ${size[1]}`} style={{ aspectRatio: `${size[0]} / ${size[1]}` }} onClick={addPoint} aria-label="Chấm vùng camera">
      {snapshot && <image href={snapshot} width={size[0]} height={size[1]} />}
      {type === 'roi' ? <polygon points={points.map(point => `${point[0] * size[0]},${point[1] * size[1]}`).join(' ')} fill="#12c5dc35" stroke="#49ebfa" strokeWidth={size[0] / 300} /> : <polyline points={points.map(point => `${point[0] * size[0]},${point[1] * size[1]}`).join(' ')} fill="none" stroke="#f5be55" strokeWidth={size[0] / 300} />}
      {points.map((point, index) => <g key={index}><circle cx={point[0] * size[0]} cy={point[1] * size[1]} r={size[0] / 120} fill="#fbf4da" /><text x={point[0] * size[0] + size[0] / 80} y={point[1] * size[1]} fill="#fff" fontSize={size[0] / 35}>{index + 1}</text></g>)}
    </svg>
    <p className={styles.fieldHelp}>{points.length} điểm. Lấy ảnh tĩnh, xóa vùng mẫu rồi chấm lại. Vùng xét điểm chân đối tượng.</p>
  </div>;
}
