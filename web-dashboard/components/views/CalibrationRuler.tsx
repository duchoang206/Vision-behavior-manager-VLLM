'use client';

import React, { useCallback, useEffect, useRef, useState } from 'react';
import { AlertTriangle, Camera, CheckCircle2, Hand, Maximize2, RefreshCw, Ruler, Trash2, Undo2, ZoomIn, ZoomOut } from 'lucide-react';
import { Point2, svgPoint } from '../../lib/calibration-points';
import { ImageViewport, RulerCalibration, RulerMeasurement, calibrationMethodLabel, clampImageViewport, zoomImageViewport } from '../../lib/calibration-ruler';
import styles from './CalibrationView.module.css';

type Props = {
  cameraId: string;
  active: boolean;
  snapshot: string;
  imageSize: Point2 | null;
  snapshotAt: string;
  savedCalibration: RulerCalibration | null;
  onRefresh: () => void;
  onCalibrate: () => void;
};
type Drag = { origin: Point2; inverse: DOMMatrix; view: ImageViewport; pointerId: number };
type CompletedMeasurement = { data: RulerMeasurement; points: Point2[]; calibration: RulerCalibration };

async function readResponse(response: Response) {
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : `Không xử lý được phép đo (HTTP ${response.status}).`);
  return data;
}

export default function CalibrationRuler({ cameraId, active, snapshot, imageSize, snapshotAt, savedCalibration, onRefresh, onCalibrate }: Props) {
  const [points, setPoints] = useState<Point2[]>([]);
  const [calibration, setCalibration] = useState<RulerCalibration | null>(null);
  const [checking, setChecking] = useState(true);
  const [calibrationError, setCalibrationError] = useState('');
  const [error, setError] = useState('');
  const [completed, setCompleted] = useState<CompletedMeasurement | null>(null);
  const [measuring, setMeasuring] = useState(false);
  const [refreshKey, setRefreshKey] = useState(0);
  const [unit, setUnit] = useState<'m' | 'cm' | 'mm'>('m');
  const [viewport, setViewport] = useState<ImageViewport>([0, 0, 1, 1]);
  const [panMode, setPanMode] = useState(false);
  const [dragging, setDragging] = useState(false);
  const imageSvg = useRef<SVGSVGElement>(null);
  const drag = useRef<Drag | null>(null);
  const suppressClick = useRef(false);
  const calibrated = Boolean(calibration?.matrix);
  const result = completed?.points === points && completed.calibration === calibration ? completed.data : null;

  useEffect(() => {
    if (imageSize) setViewport([0, 0, imageSize[0], imageSize[1]]);
  }, [imageSize]);

  useEffect(() => {
    if (!active || !cameraId) return;
    const controller = new AbortController();
    setChecking(true); setCalibrationError(''); setCompleted(null);
    fetch(`/api/backend/camera/${encodeURIComponent(cameraId)}/calibration`, { signal: controller.signal, cache: 'no-store' })
      .then(readResponse).then(data => { if (!controller.signal.aborted) setCalibration(data.calibration); })
      .catch(reason => { if (!controller.signal.aborted) { setCalibration(null); setCalibrationError(reason.message); } })
      .finally(() => { if (!controller.signal.aborted) setChecking(false); });
    return () => controller.abort();
  }, [active, cameraId, savedCalibration, refreshKey]);

  useEffect(() => {
    setCompleted(null); setError('');
    if (!active || checking || !calibrated || !calibration || points.length < 2) { setMeasuring(false); return; }
    const controller = new AbortController();
    setMeasuring(true);
    const timer = window.setTimeout(() => {
      fetch(`/api/backend/camera/${encodeURIComponent(cameraId)}/calibration/measure`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, signal: controller.signal, cache: 'no-store',
        body: JSON.stringify({ points }),
      }).then(readResponse).then((data: RulerMeasurement) => {
        if (!controller.signal.aborted && data.camera_id === cameraId) setCompleted({ data, points, calibration });
      }).catch(reason => { if (!controller.signal.aborted) setError(reason.message); })
        .finally(() => { if (!controller.signal.aborted) setMeasuring(false); });
    }, 120);
    return () => { window.clearTimeout(timer); controller.abort(); };
  }, [active, cameraId, points, checking, calibrated, calibration]);

  const zoom = useCallback((factor: number, center?: Point2) => {
    if (imageSize && !drag.current) setViewport(previous => zoomImageViewport(previous, imageSize, factor, center));
  }, [imageSize]);

  useEffect(() => {
    const svg = imageSvg.current;
    if (!active || !svg) return;
    const wheel = (event: WheelEvent) => {
      event.preventDefault();
      zoom(event.deltaY > 0 ? 1.15 : 1 / 1.15, svgPoint(svg, event.clientX, event.clientY) || undefined);
    };
    svg.addEventListener('wheel', wheel, { passive: false });
    return () => svg.removeEventListener('wheel', wheel);
  }, [active, snapshot, imageSize, zoom]);

  const addPoint = (event: React.MouseEvent<SVGSVGElement>) => {
    if (suppressClick.current) { suppressClick.current = false; return; }
    if (!active || !calibrated || checking || panMode || event.shiftKey || event.button !== 0 || event.detail > 1 || !imageSize) return;
    const point = svgPoint(event.currentTarget, event.clientX, event.clientY);
    if (!point || point[0] < 0 || point[1] < 0 || point[0] > imageSize[0] || point[1] > imageSize[1]) return;
    if (points.length >= 256) { setError('Đường đo đã có 256 điểm. Bấm Đo mới để bắt đầu đường khác.'); return; }
    const normalized: Point2 = [point[0] / imageSize[0], point[1] / imageSize[1]];
    const previous = points[points.length - 1];
    if (previous && Math.hypot(normalized[0] - previous[0], normalized[1] - previous[1]) < 0.00005) return;
    setPoints([...points, normalized]);
  };

  const startPan = (event: React.PointerEvent<SVGSVGElement>) => {
    suppressClick.current = false;
    if (!(panMode || event.shiftKey || event.button === 1)) return;
    const matrix = event.currentTarget.getScreenCTM();
    if (!matrix) return;
    event.preventDefault();
    const inverse = matrix.inverse();
    const origin = new DOMPoint(event.clientX, event.clientY).matrixTransform(inverse);
    drag.current = { origin: [origin.x, origin.y], inverse, view: viewport, pointerId: event.pointerId };
    suppressClick.current = true; setDragging(true);
    event.currentTarget.setPointerCapture(event.pointerId);
  };

  const movePan = (event: React.PointerEvent<SVGSVGElement>) => {
    const start = drag.current;
    if (!start || start.pointerId !== event.pointerId || !imageSize) return;
    const point = new DOMPoint(event.clientX, event.clientY).matrixTransform(start.inverse);
    setViewport(clampImageViewport([start.view[0] + start.origin[0] - point.x, start.view[1] + start.origin[1] - point.y, start.view[2], start.view[3]], imageSize));
  };

  const stopPan = (event: React.PointerEvent<SVGSVGElement>) => {
    if (drag.current?.pointerId !== event.pointerId) return;
    drag.current = null; setDragging(false);
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
  };

  const lengthText = (meters: number) => `${(meters * (unit === 'cm' ? 100 : unit === 'mm' ? 1000 : 1)).toLocaleString('vi-VN', { minimumFractionDigits: unit === 'm' ? 3 : 1, maximumFractionDigits: unit === 'm' ? 3 : 1 })} ${unit}`;
  const pixelPoints = imageSize ? points.map(point => [point[0] * imageSize[0], point[1] * imageSize[1]] as Point2) : [];
  const markerSize = viewport[2] / 130;
  const updatedAt = result?.calibration.updated_at || calibration?.calibration_updated_at || calibration?.calibrated_at;

  return <div className={styles.subpanel}>
    <div className={styles.guide}><strong>Chấm ít nhất 2 điểm trên ảnh. Thước tự nối từng đoạn và tính chiều dài thực theo hiệu chuẩn đang lưu.</strong>
      <p>Cuộn để zoom tới 16× · Shift + kéo hoặc bật bàn tay để di chuyển ảnh. Chỉ đo các điểm trên cùng mặt sàn đã hiệu chuẩn; không dùng để đo chiều cao hay nóc robot/kệ. Số đo phụ thuộc độ chính xác của calib.</p>
    </div>
    <div className={styles.toolbar}><div className={styles.actions}>
      <button disabled={!points.length} onClick={() => setPoints(points.slice(0, -1))}><Undo2 size={14} />Hoàn tác điểm</button>
      <button disabled={!points.length} onClick={() => { setPoints([]); setError(''); }}><Trash2 size={14} />Đo mới</button>
      <button disabled={checking || !cameraId} onClick={() => setRefreshKey(value => value + 1)}><RefreshCw size={14} />Đo lại</button>
    </div><label className={styles.actions}>Đơn vị <select aria-label="Đơn vị thước đo" value={unit} onChange={event => setUnit(event.target.value as typeof unit)}><option value="m">Mét (m)</option><option value="cm">Centimét (cm)</option><option value="mm">Milimét (mm)</option></select></label></div>
    {checking ? <div className={styles.status}>Đang đọc hiệu chuẩn đã lưu của camera…</div> : calibrationError ? <div role="alert" className={`${styles.message} ${styles.error}`}>{calibrationError}</div>
      : !calibrated ? <div className={`${styles.message} ${styles.warning}`} role="status">Camera chưa có hiệu chuẩn hợp lệ, chưa thể đo kích thước thực. <button onClick={onCalibrate}>Mở tab hiệu chuẩn</button></div>
      : <div className={styles.status}><CheckCircle2 size={15} className={styles.live} /><strong>{calibrationMethodLabel(result?.calibration.method || calibration?.method)}</strong>
        <span>· Dùng bản đã lưu, không dùng điểm calib đang chấm dở</span>{updatedAt && <span>· {new Date(updatedAt * 1000).toLocaleString('vi-VN')}</span>}</div>}
    <div className={styles.rulerPanels}>
      <article className={styles.panel}>
        <div className={styles.heading}><h2><Camera size={16} />Ảnh camera · Thước đo</h2><div className={styles.actions}>
          <button aria-label="Di chuyển ảnh thước đo" aria-pressed={panMode} disabled={!imageSize} className={panMode ? styles.selectedControl : ''} onClick={() => setPanMode(value => !value)}><Hand size={14} /></button>
          <button aria-label="Thu nhỏ ảnh thước đo" disabled={!imageSize || viewport[2] >= imageSize[0]} onClick={() => zoom(1.3)}><ZoomOut size={14} /></button>
          <span className={styles.zoomValue}>{imageSize ? `${(imageSize[0] / viewport[2]).toFixed(1)}×` : '1.0×'}</span>
          <button aria-label="Phóng to ảnh thước đo" disabled={!imageSize || viewport[2] <= imageSize[0] / 16} onClick={() => zoom(1 / 1.3)}><ZoomIn size={14} /></button>
          <button aria-label="Toàn bộ ảnh thước đo" disabled={!imageSize} onClick={() => imageSize && setViewport([0, 0, imageSize[0], imageSize[1]])}><Maximize2 size={14} /></button>
        </div></div>
        <div className={`${styles.stage} ${styles.rulerStage}`}>
          {snapshot && imageSize ? <svg ref={imageSvg} aria-label="Chấm điểm thước đo trên camera" viewBox={viewport.join(' ')} onClick={addPoint} onPointerDown={startPan} onPointerMove={movePan} onPointerUp={stopPan} onPointerCancel={stopPan} onLostPointerCapture={stopPan} style={{ cursor: dragging ? 'grabbing' : panMode ? 'grab' : calibrated ? 'crosshair' : 'not-allowed' }}>
            <image href={snapshot} width={imageSize[0]} height={imageSize[1]} pointerEvents="none" />
            {calibration?.coverage_polygon && <polygon points={calibration.coverage_polygon.map(point => `${point[0] * imageSize[0]},${point[1] * imageSize[1]}`).join(' ')} fill="none" stroke="#f3ce78" strokeWidth="1" strokeDasharray="5 5" vectorEffect="non-scaling-stroke" opacity="0.65" pointerEvents="none" />}
            {pixelPoints.slice(1).map((point, index) => { const start = pixelPoints[index]; return <g key={`segment-${index}`} pointerEvents="none">
              <line x1={start[0]} y1={start[1]} x2={point[0]} y2={point[1]} stroke="#121b26" strokeWidth="5" vectorEffect="non-scaling-stroke" />
              <line x1={start[0]} y1={start[1]} x2={point[0]} y2={point[1]} stroke="#64f4da" strokeWidth="2" vectorEffect="non-scaling-stroke" />
              {result?.segments[index] && <text x={(start[0] + point[0]) / 2} y={(start[1] + point[1]) / 2 - markerSize} textAnchor="middle" fontSize={markerSize * 1.6} fill="#b8ffed" stroke="#101b26" strokeWidth={markerSize * 0.4} paintOrder="stroke" fontWeight="700">{lengthText(result.segments[index].distance_m)}</text>}
            </g>; })}
            {pixelPoints.map((point, index) => <g key={`point-${index}`} pointerEvents="none">
              <circle cx={point[0]} cy={point[1]} r={markerSize * 0.65} fill="#13232b" stroke={result?.points[index]?.inside_calibrated_area === false ? '#ffc46b' : '#64f4da'} strokeWidth="2" vectorEffect="non-scaling-stroke" />
              <circle cx={point[0]} cy={point[1]} r={markerSize * 0.18} fill="#fff" />
              <text x={point[0] + markerSize} y={point[1] - markerSize} fontSize={markerSize * 1.65} fontWeight="700" fill="#fff" stroke="#101b26" strokeWidth={markerSize * 0.4} paintOrder="stroke">{index + 1}</text>
            </g>)}
          </svg> : <div className={styles.placeholder}>{cameraId ? 'Đang lấy một frame mới nhất…' : 'Chọn camera để bắt đầu.'}</div>}
        </div>
        <div className={`${styles.caption} ${styles.toolbar}`}><span>Ảnh tĩnh {snapshotAt && `lúc ${snapshotAt}`} · Nét đứt: vùng calib · Không mở thêm luồng video.</span>
          <button disabled={!cameraId} onClick={() => { if (!points.length || confirm('Lấy ảnh mới sẽ xóa đường đo trên ảnh cũ. Tiếp tục?')) onRefresh(); }}><RefreshCw size={13} />Lấy frame mới</button></div>
      </article>
      <aside className={styles.panel}>
        <div className={styles.heading}><h2><Ruler size={16} />Kết quả đo</h2><small>{points.length} điểm · {Math.max(0, points.length - 1)} đoạn</small></div>
        <div className={styles.measureSummary} aria-live="polite"><span>TỔNG CHIỀU DÀI</span><strong>{result ? lengthText(result.total_distance_m) : '—'}</strong>
          <small>{measuring ? 'Đang quy đổi ở backend…' : result ? 'Tổng các đoạn trên mặt sàn' : points.length === 1 ? 'Chấm điểm tiếp theo để đo.' : 'Chấm lần lượt các điểm cần đo.'}</small>
          {result && points.length > 2 && <div className={styles.directDistance}>Đầu → cuối (đường thẳng)<b>{lengthText(result.direct_distance_m)}</b></div>}
        </div>
        {result && <div className={styles.tableWrap}><table className={styles.table}><thead><tr><th>Đoạn</th><th>Chiều dài</th></tr></thead><tbody>{result.segments.map(segment => <tr key={segment.from}><td>{segment.from} → {segment.to}</td><td>{lengthText(segment.distance_m)}</td></tr>)}</tbody></table></div>}
        <div className={styles.caption}>Phép đo chỉ đọc ma trận của camera, không thay đổi hiệu chuẩn hay pipeline tracking. Bấm Đo lại để lấy bản hiệu chuẩn mới nhất.
          {result?.calibration.reprojection_error_m != null && <p>Sai số khớp calib: {lengthText(result.calibration.reprojection_error_m)}. Đây không phải bảo đảm sai số thực tế của thước.</p>}</div>
      </aside>
    </div>
    {error && <div role="alert" className={`${styles.message} ${styles.error}`}>{error}</div>}
    {result?.warnings.map(warning => <div role="status" key={warning} className={`${styles.message} ${styles.warning}`}><AlertTriangle size={16} />{warning}</div>)}
  </div>;
}
