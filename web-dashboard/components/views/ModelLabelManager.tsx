'use client';

import { PointerEvent, useCallback, useEffect, useRef, useState } from 'react';
import { Camera, Check, Crosshair, Images, Plus, RefreshCw, Save, Scan, Tag, Trash2 } from 'lucide-react';
import { useCameras } from '../CameraContext';
import type { VisionModel } from './ModelManager';
import { RegisteredMask, maskPath } from '../../lib/registered-mask';
import styles from './ModelLabelManager.module.css';

type Point = [number, number];
type Box = [number, number, number, number];
type Snapshot = { snapshot_id: string; image: string; width: number; height: number; frame_id: number; expires_in: number };
type Preview = { preview_id: string; mask: RegisteredMask };
type Label = { label: string; class_name: string; category: string; negative: boolean; samples: number; cameras: string[] };
type Metric = { state?: string; samples?: number; training?: boolean; last_loss?: number | null; error?: string; revision?: number; incompatible_samples?: number };
type IdentityRejection = { candidate_label?: string; class_name?: string; reason?: string; score?: number; raw_score?: number };
type Runtime = { model_id?: string; state?: string; cameras?: Record<string, { age_ms?: number; segmentation?: {
  ready?: boolean; error?: string; labels?: Metric; observed_at?: number; visible_labels?: string[];
  identity_rejected?: IdentityRejection[]; detector_candidates?: { class_name: string; confidence: number }[];
  verification_candidates?: number; targets?: number; masks?: number;
} }> };
type Gallery = { labels: Label[]; revision: number; require_labels: boolean; runtime?: Runtime;
  runtime_sync?: { mode: string; error?: string; activation?: { mode: string } } };
type SavedView = { id: string; label: string; camera_id: string; created_at: string; mask: RegisteredMask; frame_id: number };
type SavedViews = { samples: SavedView[]; total: number; limit: number; offset: number };

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api/backend/models${path}`, { ...init, cache: 'no-store' });
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : `Yêu cầu không hợp lệ (${response.status}).`);
  return data;
}

export default function ModelLabelManager({ active, initialCameraId = '' }: { active: boolean; initialCameraId?: string }) {
  const { cameras } = useCameras();
  const [models, setModels] = useState<VisionModel[]>([]);
  const [modelId, setModelId] = useState('');
  const [cameraId, setCameraId] = useState(initialCameraId);
  const [gallery, setGallery] = useState<Gallery | null>(null);
  const [viewsLabel, setViewsLabel] = useState('');
  const [viewsOffset, setViewsOffset] = useState(0);
  const [savedViews, setSavedViews] = useState<SavedViews | null>(null);
  const [viewsError, setViewsError] = useState('');
  const [runtime, setRuntime] = useState<Runtime | null>(null);
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [preview, setPreview] = useState<Preview | null>(null);
  const [box, setBox] = useState<Box | null>(null);
  const [points, setPoints] = useState<{ point: Point; label: 0 | 1 }[]>([]);
  const [mode, setMode] = useState<'box' | 'positive' | 'negative'>('box');
  const [zoom, setZoom] = useState(1);
  const [label, setLabel] = useState('');
  const [className, setClassName] = useState('');
  const [negative, setNegative] = useState(false);
  const [viewMode, setViewMode] = useState<'new' | 'additional'>('new');
  const [selectedLabel, setSelectedLabel] = useState('');
  const [busy, setBusy] = useState('');
  const [error, setError] = useState('');
  const [message, setMessage] = useState('');
  const [imageReady, setImageReady] = useState(false);
  const [expiresAt, setExpiresAt] = useState(0);
  const [now, setNow] = useState(Date.now());
  const origin = useRef<Point | null>(null);
  const generation = useRef(0);
  const pending = useRef<AbortController | null>(null);
  const model = models.find(item => item.id === modelId);
  const liveRuntime = gallery?.runtime ?? runtime;
  const cameraRuntime = liveRuntime?.model_id === modelId ? liveRuntime.cameras?.[cameraId] : undefined;
  const metric = cameraRuntime?.segmentation?.labels;
  const expired = !!snapshot && now >= expiresAt;

  useEffect(() => {
    if (!active) return;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      try {
        const result = await request<{ models: VisionModel[]; runtime: Runtime; deployment?: { model_id: string } }>('', { signal: controller.signal });
        if (controller.signal.aborted) return;
        setModels(result.models);
        setRuntime(result.runtime);
        setModelId(current => current || result.runtime?.model_id || result.deployment?.model_id || result.models.find(item => item.state === 'ready')?.id || '');
      } catch (reason) {
        if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : String(reason));
      } finally {
        if (!controller.signal.aborted) timer = setTimeout(poll, 4000);
      }
    };
    void poll();
    return () => { controller.abort(); clearTimeout(timer); };
  }, [active]);

  useEffect(() => {
    if (!cameras.some(camera => camera.id === cameraId)) setCameraId(cameras[0]?.id || '');
  }, [cameras, cameraId]);

  useEffect(() => {
    generation.current += 1;
    pending.current?.abort();
    pending.current = null;
    origin.current = null;
    setBusy(''); setSnapshot(null); setPreview(null); setBox(null); setPoints([]);
    setImageReady(false); setError(''); setMessage(''); setZoom(1);
    return () => { generation.current += 1; pending.current?.abort(); };
  }, [modelId, cameraId, active]);

  useEffect(() => {
    setLabel(''); setNegative(false); setClassName(model?.labels[0] || '');
    setViewMode('new'); setSelectedLabel('');
    setViewsLabel(''); setViewsOffset(0);
  }, [modelId, model?.labels.join('|')]);

  useEffect(() => {
    setGallery(null);
    if (!active || !modelId) return;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      try {
        const result = await request<Gallery>(`/${modelId}/labels`, { signal: controller.signal });
        if (!controller.signal.aborted) setGallery(result);
      } catch (reason) {
        if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : String(reason));
      } finally {
        if (!controller.signal.aborted) timer = setTimeout(poll, 2500);
      }
    };
    void poll();
    return () => { controller.abort(); clearTimeout(timer); };
  }, [active, modelId]);

  useEffect(() => {
    setSavedViews(null); setViewsError('');
    if (!active || !modelId || !viewsLabel) return;
    const controller = new AbortController();
    void request<SavedViews>(`/${modelId}/labels/samples?label=${encodeURIComponent(viewsLabel)}&limit=12&offset=${viewsOffset}`, { signal: controller.signal })
      .then(result => { if (!controller.signal.aborted) setSavedViews(result); })
      .catch(reason => { if (!controller.signal.aborted) setViewsError(reason instanceof Error ? reason.message : String(reason)); });
    return () => controller.abort();
  }, [active, modelId, viewsLabel, viewsOffset, gallery?.revision]);

  useEffect(() => {
    if (gallery && selectedLabel && !gallery.labels.some(item => item.label === selectedLabel)) {
      setSelectedLabel(''); setViewMode('new'); setLabel(''); setPreview(null);
    }
  }, [gallery, selectedLabel]);

  useEffect(() => {
    if (!active || !snapshot) return;
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [active, snapshot]);

  const perform = useCallback(async <Result,>(name: string, path: string, body: unknown, apply: (result: Result) => void, method = 'POST') => {
    if (pending.current) return;
    const controller = new AbortController();
    const current = generation.current;
    pending.current = controller;
    setBusy(name); setError(''); setMessage('');
    try {
      const result = await request<Result>(`/${modelId}/labels${path}`, { method, signal: controller.signal,
        headers: { 'Content-Type': 'application/json' }, ...(body === undefined ? {} : { body: JSON.stringify(body) }) });
      if (!controller.signal.aborted && current === generation.current) apply(result);
    } catch (reason) {
      if (!controller.signal.aborted && current === generation.current) setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      if (pending.current === controller) pending.current = null;
      if (current === generation.current) setBusy('');
    }
  }, [modelId]);

  const capture = () => void perform<Snapshot>('Lấy frame GPU…', '/snapshot', { camera_id: cameraId }, result => {
    setSnapshot(result); setPreview(null); setBox(null); setPoints([]); setMode('box'); setImageReady(false);
    setExpiresAt(Date.now() + result.expires_in * 1000); setNow(Date.now());
  });
  const drawPoint = (event: PointerEvent<SVGSVGElement>): Point => {
    const bounds = event.currentTarget.getBoundingClientRect();
    return [Math.min(1, Math.max(0, (event.clientX - bounds.left) / bounds.width)), Math.min(1, Math.max(0, (event.clientY - bounds.top) / bounds.height))];
  };
  const pointerDown = (event: PointerEvent<SVGSVGElement>) => {
    if (busy || !imageReady || expired || event.button !== 0) return;
    const point = drawPoint(event);
    setPreview(null);
    if (mode === 'box') {
      event.currentTarget.setPointerCapture(event.pointerId);
      origin.current = point; setBox([point[0], point[1], 0, 0]); setPoints([]);
    } else if (points.length < 32) {
      setPoints(current => [...current, { point, label: mode === 'positive' ? 1 : 0 }]);
    }
  };
  const pointerMove = (event: PointerEvent<SVGSVGElement>) => {
    if (!origin.current) return;
    const point = drawPoint(event);
    setBox([Math.min(origin.current[0], point[0]), Math.min(origin.current[1], point[1]), Math.abs(origin.current[0] - point[0]), Math.abs(origin.current[1] - point[1])]);
  };
  const previewMask = () => snapshot && box && void perform<Preview>('SAM2 GPU đang tách vật…', '/preview', {
    camera_id: cameraId, snapshot_id: snapshot.snapshot_id, bbox: box,
    points: points.map(item => item.point), point_labels: points.map(item => item.label),
  }, setPreview);
  const save = () => preview && void perform<Gallery>(viewMode === 'additional' ? 'Lưu thêm góc nhìn…' : 'Lưu mẫu & cập nhật triplet…', '/samples', {
    camera_id: cameraId, preview_id: preview.preview_id, label: label.trim(), class_name: className, negative,
  }, result => {
    setGallery(result);
    const saved = result.labels.find(item => item.label.toLowerCase() === label.trim().toLowerCase());
    if (saved) {
      setLabel(saved.label); setClassName(saved.class_name); setNegative(saved.negative);
      setSelectedLabel(saved.label); setViewMode('additional');
    }
    setSnapshot(null); setPreview(null); setBox(null); setPoints([]); setImageReady(false); setExpiresAt(0);
    const sync = result.runtime_sync;
    const tracking = negative ? 'Đã cập nhật mẫu loại.' : sync?.mode === 'applied'
      ? sync.activation?.mode === 'mask_initialized' ? 'Mask đã khởi tạo trên Monitor.'
        : 'Đã nạp mask vào luồng bám live, không cần chờ YOLO.'
      : 'Mẫu đã lưu; worker chưa xác nhận, đang chờ đồng bộ để bám live.';
    setMessage(`Đã lưu ${viewMode === 'additional' ? 'góc nhìn mới của' : 'Label'} ${label.trim()} vào PostgreSQL + SSD. ${tracking} Hãy lấy frame mới để đăng ký góc tiếp theo.`);
  });
  const selectLabel = (item: Label) => {
    setLabel(item.label); setClassName(item.class_name); setNegative(item.negative);
    setSelectedLabel(item.label); setViewMode('additional');
    setSnapshot(null); setPreview(null); setBox(null); setPoints([]); setImageReady(false); setExpiresAt(0);
    setMessage(`Đang thêm góc nhìn cho ${item.label}. Nhấn “Lấy frame mới từ GPU”, chọn vật và chạy SAM2 trước khi lưu.`);
  };
  const newLabel = () => {
    setViewMode('new'); setSelectedLabel(''); setLabel(''); setNegative(false);
    setSnapshot(null); setPreview(null); setBox(null); setPoints([]); setImageReady(false);
    setMessage('Tạo Label riêng cho vật khác. Không dùng lại tên của vật đã đăng ký.');
  };
  const labelStatus = (item: Label) => {
    if (item.negative) return 'Mẫu loại: không hiển thị như một danh tính trên Monitor.';
    if (!cameraRuntime || (cameraRuntime.age_ms ?? Infinity) > 1000) return 'Camera đang chọn chưa có metadata model mới. Kiểm tra deployment/kết nối.';
    const segmentation = cameraRuntime.segmentation;
    if (segmentation?.error || metric?.error) return segmentation?.error || metric?.error;
    if (!segmentation?.ready) return 'SAM2 GPU đang nạp.';
    if (metric?.revision !== gallery?.revision) return 'Đang nạp bộ mẫu mới vào GPU.';
    if (segmentation?.visible_labels?.some(name => name.toLowerCase() === item.label.toLowerCase())) return 'Đã xác thực nhãn và có mask trong đầu ra AI của camera này.';
    const rejected = segmentation?.identity_rejected?.find(result => result.candidate_label?.toLowerCase() === item.label.toLowerCase());
    if (rejected) {
      const reason = rejected.reason === 'ambiguous_identity' ? 'còn giống nhiều vật'
        : rejected.reason === 'category_conflict' ? 'khác loại đối tượng'
          : rejected.reason === 'negative_sample' ? 'khớp mẫu loại' : 'ngoại hình chưa đủ khớp';
      return `Chưa xác nhận: ${reason}${rejected.score != null ? ` (độ tương đồng ${rejected.score.toFixed(3)})` : ''}. Thêm góc nhìn đúng vật nếu nó đang trong ảnh; không gán theo vị trí cũ.`;
    }
    if (!item.cameras.includes(cameraId)) return 'Chưa có góc nhìn ở camera này. Chụp frame và thêm góc nhìn để khởi tạo mask tại đây.';
    return 'Mask đã lưu là mốc khởi tạo SAM2; đang chuyển từ frame đã gán nhãn sang frame live. Không cần chờ YOLO phát hiện.';
  };

  return <section className={styles.root}>
    <header className={styles.header}><div><span className={styles.eyebrow}>MODEL + MULTI-VIEW IDENTITY</span><h2><Tag size={20} /> Label · Xác thực ngoại hình</h2>
      <p>Lưu mask người dùng → khởi tạo SAM2 → bám frame live → Monitor. DeepStream và mẫu đa góc nhìn hỗ trợ giữ đúng vật.</p></div>
      <span className={styles.badge}>Ưu tiên Label người dùng · GPU</span></header>
    <div className={styles.notice}>Lưu nhiều góc nhìn của <strong>cùng một vật dưới cùng tên Label</strong>. Vật khác dùng tên khác; thêm mẫu dễ nhầm để loại nền/robot sai.
      Mẫu gốc khớp rõ và khác biệt được ưu tiên hơn điểm triplet. Triplet hỗ trợ góc nhìn mới, <strong>không huấn luyện lại weights YOLO ONNX/TensorRT</strong>.</div>
    <div className={styles.viewSelector}>
      <label>Thêm góc nhìn cho vật đã đăng ký<select aria-label="Vật cần thêm góc nhìn" value={selectedLabel} disabled={!!busy} onChange={event => {
        const selected = gallery?.labels.find(item => item.label === event.target.value);
        if (selected) selectLabel(selected); else newLabel();
      }}><option value="">Tạo Label mới</option>{gallery?.labels.map(item => <option key={item.label} value={item.label}>{item.label} · {item.samples} góc nhìn{item.negative ? ' · mẫu loại' : ''}</option>)}</select></label>
      <button disabled={!!busy} onClick={newLabel}><Plus size={15} /> Vật khác / Label mới</button>
    </div>
    {viewMode === 'additional' && <div className={styles.viewMode}><strong>Đang thêm góc nhìn:</strong> {selectedLabel || label} · mỗi lần lưu bắt buộc lấy frame mới và chạy SAM2 lại.</div>}
    <div className={styles.selectors}>
      <label>Model đã upload<select value={modelId} disabled={!!busy} onChange={event => setModelId(event.target.value)}><option value="">Chọn model</option>{models.map(item => <option key={item.id} value={item.id}>{item.name} · {item.state}</option>)}</select></label>
      <label>Camera<select value={cameraId} disabled={!!busy} onChange={event => setCameraId(event.target.value)}><option value="">Chọn camera</option>{cameras.map(camera => <option key={camera.id} value={camera.id}>{camera.name}</option>)}</select></label>
      <button className={styles.primary} disabled={!!busy || !cameraId || !modelId || !cameraRuntime?.segmentation?.ready || (cameraRuntime.age_ms ?? Infinity) > 2000} onClick={capture}><Camera size={16} /> {viewMode === 'additional' ? 'Lấy góc nhìn mới từ GPU' : 'Lấy frame mới từ GPU'}</button>
    </div>
    {(!cameraRuntime || !cameraRuntime.segmentation?.ready) && <p className={styles.hint}>Deploy model cho camera ở Model / TensorRT trước. {cameraRuntime?.segmentation?.error || 'Chờ worker GPU sẵn sàng.'}</p>}
    <div className={styles.layout}>
      <div className={styles.canvasCard}>
        <div className={styles.toolbar}><strong>01 · Khoanh đúng vật</strong><label>Zoom <input aria-label="Phóng to ảnh" type="range" min={1} max={3} step={0.25} value={zoom} onChange={event => setZoom(Number(event.target.value))} /> {zoom}×</label></div>
        <div className={styles.tools}>
          <button aria-pressed={mode === 'box'} disabled={!!busy} onClick={() => setMode('box')}><Scan size={15} /> Khoanh vật</button>
          <button aria-pressed={mode === 'positive'} disabled={!box || !!busy} onClick={() => setMode('positive')}><Plus size={15} /> Giữ vùng</button>
          <button aria-pressed={mode === 'negative'} disabled={!box || !!busy} onClick={() => setMode('negative')}><Crosshair size={15} /> Loại vùng</button>
          <button disabled={!points.length || !!busy} onClick={() => { setPoints(current => current.slice(0, -1)); setPreview(null); }}>Bỏ điểm cuối</button>
        </div>
        <div className={styles.viewport}>{snapshot ? <div className={styles.imageFrame} style={{ width: `${zoom * 100}%`, aspectRatio: `${snapshot.width} / ${snapshot.height}` }}>
          <img key={snapshot.snapshot_id} src={`data:image/jpeg;base64,${snapshot.image}`} alt="Frame GPU cố định để đăng ký mẫu" draggable={false} onLoad={() => setImageReady(true)} onError={() => { setImageReady(false); setError('Không hiển thị được ảnh. Lấy frame mới.'); }} />
          <svg viewBox="0 0 1 1" preserveAspectRatio="none" onPointerDown={pointerDown} onPointerMove={pointerMove} onPointerUp={event => { pointerMove(event); origin.current = null; }} onPointerCancel={() => { origin.current = null; }}>
            {preview && <path d={maskPath(preview.mask)} fill="rgba(45,212,191,.23)" stroke="#5eead4" strokeWidth={1.5} vectorEffect="non-scaling-stroke" fillRule="evenodd" />}
            {box && <rect x={box[0]} y={box[1]} width={box[2]} height={box[3]} fill="none" stroke="#fbbf24" strokeWidth={1.5} vectorEffect="non-scaling-stroke" strokeDasharray="6 4" />}
            {points.map((item, index) => <ellipse key={index} cx={item.point[0]} cy={item.point[1]} rx={0.004 / zoom} ry={0.004 * snapshot.width / snapshot.height / zoom} fill={item.label ? '#34d399' : '#fb7185'} stroke="white" strokeWidth={1} vectorEffect="non-scaling-stroke" />)}
          </svg>
        </div> : <div className={styles.empty}><Scan size={40} /><h3>Chọn một frame, không phát thêm video</h3><p>Kéo khung sát Robot/Kệ. Thêm điểm giữ/loại vùng rồi xem trước mask trước khi lưu.</p></div>}</div>
        <div className={styles.toolbar}><small>{snapshot ? `Frame #${snapshot.frame_id} · ${expired ? 'Hết hạn — lấy frame mới' : `Còn ${Math.max(0, Math.ceil((expiresAt - now) / 1000))}s để lưu`}` : 'Ảnh và mask được xử lý tại worker GPU.'}</small>
          <button disabled={!box || box[2] < .005 || box[3] < .005 || !imageReady || expired || !!busy} onClick={previewMask}><RefreshCw size={15} /> Xem trước SAM2</button></div>
      </div>
      <aside className={styles.form}>
        <h3>02 · Xác nhận & lưu mẫu</h3>
        <label>Lớp tương ứng trong model<select disabled={!!busy || viewMode === 'additional'} value={className} onChange={event => setClassName(event.target.value)}>{model?.labels.map(name => <option key={name}>{name}</option>)}</select></label>
        <label>Danh tính vật thể<input value={label} disabled={!!busy || viewMode === 'additional'} maxLength={120} onChange={event => setLabel(event.target.value)} placeholder="Robot_2001 hoặc Rack_A01" /></label>
        <label className={styles.check}><input type="checkbox" disabled={!!busy || viewMode === 'additional'} checked={negative} onChange={event => setNegative(event.target.checked)} /> Đây là mẫu sai / vật dễ nhận nhầm (hard negative)</label>
        <p>Robot_2001 tương ứng robot 2001 trên FMS khi camera đã calib. Không đặt cùng tên cho hai vật khác nhau.</p>
        <button className={styles.primary} disabled={!preview || !label.trim() || !className || expired || !!busy} onClick={save}><Save size={16} /> {viewMode === 'additional' ? 'Lưu thêm góc nhìn' : 'Lưu góc nhìn này'}</button>
        <div className={styles.status}><strong>Triplet · {metric?.training ? 'Đang học GPU' : metric?.state === 'trained' ? 'Đã học' : 'Chờ đủ mẫu'}</strong>
          <p>Cần ít nhất 2 góc của một vật và mẫu của vật khác / mẫu sai để tạo bộ A–P–N.</p>
          <small>{metric?.samples ?? 0} mẫu trong gallery GPU · revision {metric?.revision ?? '—'}{metric?.last_loss != null ? ` · loss ${metric.last_loss.toFixed(4)}` : ''}</small>
          {gallery && metric?.revision !== undefined && gallery.revision !== metric.revision && <p>Worker đang cập nhật gallery mới…</p>}
          {(metric?.error || !!metric?.incompatible_samples) && <p role="alert">{metric.error || 'Có mẫu khác phiên bản SAM2. Đăng ký lại để xác thực an toàn.'}</p>}
        </div>
        <div className={styles.status}><strong>Đầu ra AI · {cameras.find(camera => camera.id === cameraId)?.name || 'Camera đang chọn'}</strong>
          <p>{cameraRuntime && (cameraRuntime.age_ms ?? Infinity) <= 1000
            ? `${cameraRuntime.segmentation?.visible_labels?.length ?? 0} nhãn đã xác thực · ${cameraRuntime.segmentation?.verification_candidates ?? 0} detection yếu đang qua xác thực Label`
            : 'Chưa có metadata model mới từ camera. Kiểm tra camera đã được deploy.'}</p>
          <small>Lưu Label khởi tạo mask trên Monitor từ đúng hình dạng bạn đã chọn. AI hỗ trợ di chuyển mask theo vật, không cần chờ detection YOLO. Khi AI cập nhật chậm, hệ thống nối các kết quả mới; không ghim mask cũ vô hạn khi vật đã mất dấu.</small>
        </div>
      </aside>
    </div>
    {busy && <p role="status" className={styles.message}>{busy}</p>}{error && <p role="alert" className={styles.error}>{error}</p>}{message && <p role="status" className={styles.message}><Check size={15} /> {message}</p>}
    <section className={styles.gallery}>
      <div className={styles.toolbar}><div><h3>03 · Bộ mẫu đã xác nhận ({gallery?.labels.length ?? 0} Label)</h3><p>Mọi góc nhìn đã lưu đều tham gia đối chiếu trên GPU; không loại góc cũ khi thêm mẫu mới. Ảnh và dữ liệu được lưu trên SSD + PostgreSQL, có thể xóa từng góc.</p></div>
        <label className={styles.check}><input type="checkbox" checked={gallery?.require_labels ?? false} disabled={!gallery || !!busy} onChange={event => void perform<Gallery>('Cập nhật chế độ xác thực…', '/settings', { require_labels: event.target.checked }, setGallery, 'PUT')} /> Chỉ hiển thị vật đã xác nhận Label</label></div>
      <p>Nhóm đối tượng đã có mẫu Label sẽ được xác thực ngoại hình tự động. Bật chế độ trên để yêu cầu Label cho cả những nhóm chưa có mẫu.</p>
      {!gallery?.labels.length ? <div className={styles.noSamples}>Chưa có mẫu cho model này. Mẫu Label cũ không được tự nhập lại để tránh kế thừa dữ liệu nhận nhầm.</div> : <div className={styles.cards}>{[...gallery.labels].sort((first, second) => Number(first.negative) - Number(second.negative)).map(item => <article key={item.label}>
        <div><Tag size={16} /><strong>{item.label}</strong><span className={item.negative ? styles.negative : styles.positive}>{item.negative ? 'Mẫu loại' : 'Đã lưu mẫu'}</span></div>
        <p>{item.class_name} · {item.samples} góc nhìn · {item.cameras.length} camera</p>
        <p>{labelStatus(item)}</p>
        <div><button disabled={!!busy} onClick={() => selectLabel(item)}><Plus size={14} /> Thêm góc nhìn</button><button disabled={!!busy} onClick={() => { setViewsLabel(item.label); setViewsOffset(0); }}><Images size={14} /> Xem / xóa góc</button><button disabled={!!busy} aria-label={`Xóa Label ${item.label}`} onClick={() => {
          if (confirm(`Xóa toàn bộ ${item.samples} mẫu của ${item.label}? Không thể hoàn tác.`)) void perform<Gallery>('Xóa Label…', `?label=${encodeURIComponent(item.label)}`, undefined, result => { setGallery(result); setMessage('Đã xóa mẫu khỏi PostgreSQL, SSD và cập nhật gallery.'); }, 'DELETE');
        }}><Trash2 size={14} /></button></div>
      </article>)}</div>}
    </section>
    {viewsLabel && <section className={styles.gallery} aria-label={`Các góc nhìn của ${viewsLabel}`}>
      <div className={styles.toolbar}><h3>Góc nhìn đã lưu · {viewsLabel}</h3><button disabled={!!busy} onClick={() => setViewsLabel('')}>Đóng danh sách</button></div>
      {viewsError && <p role="alert" className={styles.error}>{viewsError}</p>}
      {!savedViews && !viewsError && <p>Đang tải góc nhìn…</p>}
      <div className={styles.cards}>{savedViews?.samples.map(view => <article key={view.id}>
        <div className={styles.savedImage}>
          <img src={`/api/backend/models/${modelId}/labels/samples/${view.id}/image`} alt={`Góc nhìn ${view.label} · ${view.camera_id}`} loading="lazy" />
          <svg viewBox="0 0 1 1" preserveAspectRatio="none" aria-hidden="true"><path d={maskPath(view.mask)} fill="rgba(45,212,191,.23)" stroke="#5eead4" strokeWidth={1} vectorEffect="non-scaling-stroke" fillRule="evenodd" /></svg>
        </div>
        <p>{cameras.find(camera => camera.id === view.camera_id)?.name || view.camera_id} · Frame #{view.frame_id}<br />{new Date(view.created_at).toLocaleString('vi-VN')}</p>
        <button disabled={!!busy} aria-label={`Xóa góc nhìn ${view.id}`} onClick={() => {
          if (confirm(`Xóa riêng góc nhìn này của ${view.label}? Các góc khác vẫn được giữ.`)) void perform<Gallery>('Xóa góc nhìn…', `/samples/${view.id}`, undefined, result => {
            setGallery(result); setViewsOffset(0); setMessage('Đã xóa góc nhìn khỏi PostgreSQL, SSD và bộ đối chiếu GPU.');
          }, 'DELETE');
        }}><Trash2 size={14} /> Xóa góc này</button>
      </article>)}</div>
      {savedViews && <div className={styles.toolbar}><small>{savedViews.total} góc nhìn · trang {Math.floor(viewsOffset / 12) + 1}</small>
        <div><button disabled={!!busy || viewsOffset === 0} onClick={() => setViewsOffset(current => Math.max(0, current - 12))}>Trước</button> <button disabled={!!busy || viewsOffset + 12 >= savedViews.total} onClick={() => setViewsOffset(current => current + 12)}>Sau</button></div></div>}
    </section>}
  </section>;
}
