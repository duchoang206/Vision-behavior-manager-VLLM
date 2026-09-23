'use client';

import { PointerEvent, useEffect, useRef, useState } from 'react';
import { Camera, Check, Pencil, Plus, Save, Trash2, Undo2, X } from 'lucide-react';
import type { VisionModel } from './ModelManager';
import { LearningAnnotation, LearningPoint, LearningSample, learningImage, learningRequest, polygonPath } from '../../lib/active-learning';
import styles from './ActiveLearning.module.css';

export default function ActiveLearningDialog({ cameraId, initialSample, onClose, onSaved }: {
  cameraId?: string; initialSample?: LearningSample; onClose: () => void; onSaved?: () => void;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const [models, setModels] = useState<VisionModel[]>([]);
  const [modelId, setModelId] = useState(initialSample?.model_id || '');
  const [sample, setSample] = useState<LearningSample | null>(null);
  const [annotations, setAnnotations] = useState<LearningAnnotation[]>([]);
  const [selected, setSelected] = useState('');
  const [draft, setDraft] = useState<LearningPoint[]>([]);
  const [drawing, setDrawing] = useState(false);
  const [zoom, setZoom] = useState(1);
  const [complete, setComplete] = useState(false);
  const [changed, setChanged] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [message, setMessage] = useState('');
  const [imageReady, setImageReady] = useState(false);
  const drag = useRef<{ ring: number; point: number } | null>(null);
  const controller = useRef<AbortController | null>(null);
  const model = models.find(item => item.id === modelId);
  const object = annotations.find(item => item.id === selected);
  const editable = sample?.status === 'draft' && !busy;
  const valid = annotations.every(item => item.polygons.length > 0 && item.polygons.every(ring => ring.length >= 3));

  const loadSample = (next: LearningSample) => {
    const proposals: LearningAnnotation[] = next.status === 'draft' ? next.predictions.map(item => ({
      id: String(item.id), class_name: item.class, label: item.label || item.class,
      polygons: item.mask?.frame_id === next.frame_id ? item.mask.polygons : [],
    })) : next.annotations;
    setSample(next); setAnnotations(proposals); setSelected(proposals[0]?.id || '');
    setDraft([]); setDrawing(false); setComplete(false); setChanged(false); setZoom(1); setImageReady(false);
  };

  useEffect(() => {
    const element = dialog.current;
    element?.showModal();
    const abort = new AbortController();
    fetch('/api/backend/models', { cache: 'no-store', signal: abort.signal }).then(async response => {
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || 'Không tải được model.');
      if (abort.signal.aborted) return;
      setModels(payload.models);
      setModelId(current => current || payload.runtime?.model_id || payload.deployment?.model_id || '');
      if (initialSample) loadSample(initialSample);
    }).catch(reason => { if (!abort.signal.aborted) setError(String(reason.message || reason)); });
    return () => { abort.abort(); controller.current?.abort(); element?.close(); };
  }, [initialSample]);

  const capture = async () => {
    if (!cameraId || !modelId || busy) return;
    if (sample?.status === 'draft' && changed && !confirm('Bỏ phần đang sửa để lấy frame khác? Ảnh nháp vẫn nằm trong Dataset.')) return;
    controller.current?.abort();
    const abort = new AbortController(); controller.current = abort;
    setBusy(true); setError(''); setMessage('');
    try {
      const next = await learningRequest<LearningSample>(modelId, '/snapshots', { method: 'POST', signal: abort.signal, body: JSON.stringify({ camera_id: cameraId }) });
      if (!abort.signal.aborted) loadSample(next);
    } catch (reason) { if (!abort.signal.aborted) setError(reason instanceof Error ? reason.message : String(reason)); }
    finally { if (!abort.signal.aborted) setBusy(false); }
  };
  const change = (next: LearningAnnotation[]) => { setAnnotations(next); setChanged(true); setComplete(false); };
  const updateObject = (values: Partial<LearningAnnotation>) => change(annotations.map(item => item.id === selected ? { ...item, ...values } : item));
  const pointAt = (event: PointerEvent<SVGSVGElement>): LearningPoint => {
    const bounds = event.currentTarget.getBoundingClientRect();
    return [Math.max(0, Math.min(1, (event.clientX - bounds.left) / bounds.width)), Math.max(0, Math.min(1, (event.clientY - bounds.top) / bounds.height))];
  };
  const close = () => {
    if (!busy && (!changed || sample?.status !== 'draft' || confirm('Đóng và bỏ phần polygon chưa lưu? Ảnh nháp vẫn được giữ 1 giờ.'))) onClose();
  };
  const finishPolygon = () => {
    if (!object || draft.length < 3) return;
    updateObject({ polygons: [...object.polygons, draft] }); setDraft([]); setDrawing(false);
  };
  const save = async (feedback: 'correct' | 'corrected' | 'reject') => {
    if (!sample || busy) return;
    controller.current?.abort();
    const abort = new AbortController(); controller.current = abort;
    setBusy(true); setError('');
    try {
      const result = await learningRequest<LearningSample>(modelId, `/samples/${sample.id}/review`, {
        method: 'POST', signal: abort.signal, body: JSON.stringify({ annotations: feedback === 'reject' ? [] : annotations, feedback, complete }),
      });
      if (!abort.signal.aborted) {
        setSample(result); setChanged(false);
        setMessage(feedback === 'reject' ? 'Đã loại frame khỏi tập học.' : 'Đã lưu ground truth vào SSD + PostgreSQL. Weights Monitor chưa thay đổi cho đến khi retrain đạt kiểm định.');
        onSaved?.();
      }
    } catch (reason) { if (!abort.signal.aborted) setError(reason instanceof Error ? reason.message : String(reason)); }
    finally { if (!abort.signal.aborted) setBusy(false); }
  };

  return <dialog ref={dialog} className={styles.dialog} aria-labelledby="learning-dialog-title" onCancel={event => { event.preventDefault(); close(); }}>
    <header className={styles.header}><div><h2 id="learning-dialog-title">Active Learning · Đúng / Sửa nhãn</h2><p>Frame DeepStream cố định → kiểm tra tất cả vật → sửa polygon / lớp → lưu dữ liệu học.</p></div>
      <button type="button" autoFocus onClick={close} disabled={busy} aria-label="Đóng Active Learning"><X size={20} /></button></header>
    <div className={styles.body}>
      <div className={styles.toolbar}><label>Model<select value={modelId} disabled={!!sample || busy || !!initialSample} onChange={event => setModelId(event.target.value)}>
        <option value="">Chưa có model deploy</option>{models.filter(item => item.state === 'ready').map(item => <option key={item.id} value={item.id}>{item.name}</option>)}
      </select></label>{cameraId && <button type="button" onClick={capture} disabled={!modelId || busy}><Camera size={15} />{busy ? 'Đang xử lý…' : 'Lấy frame GPU mới'}</button>}</div>
      {!sample && <p>Chọn “Lấy frame GPU mới” để đóng băng một frame cho việc duyệt. Video Monitor vẫn chạy; không ghép polygon của frame cũ lên ảnh mới.</p>}
      {sample && <>
        <p>Camera {sample.camera_id} · source {sample.frame.source_id} · frame #{sample.frame_id} · {new Date(sample.captured_at).toLocaleString('vi-VN')} · version {sample.model_version.slice(0, 12)}</p>
        <div className={styles.editor}><div>
          <div className={styles.viewport}><div className={styles.imageFrame} style={{ width: `${zoom * 100}%`, aspectRatio: `${sample.frame.width} / ${sample.frame.height}` }}>
            <img src={learningImage(sample)} alt={`Frame ${sample.frame_id} để duyệt nhãn`} draggable={false} onLoad={() => setImageReady(true)} onError={() => { setImageReady(false); setError('Không đọc được ảnh trên SSD.'); }} />
            <svg viewBox="0 0 1 1" preserveAspectRatio="none" aria-label="Công cụ chỉnh polygon" style={{ cursor: drawing ? 'crosshair' : 'default' }}
              onPointerDown={event => {
                if (!editable || !imageReady || !drawing || !object || draft.length >= 1024) return;
                const coordinate = pointAt(event);
                setDraft(points => [...points, coordinate]);
              }}
              onPointerMove={event => {
                if (!drag.current || !object || !editable) return;
                const coordinate = pointAt(event), position = drag.current;
                updateObject({ polygons: object.polygons.map((ring, ringIndex) => ringIndex === position.ring ? ring.map((point, pointIndex) => pointIndex === position.point ? coordinate : point) : ring) });
              }} onPointerUp={() => { drag.current = null; }} onPointerCancel={() => { drag.current = null; }}>
              {sample.predictions.filter(item => !annotations.find(annotation => annotation.id === String(item.id))?.polygons.length).map(item => <rect key={item.id} x={item.x} y={item.y} width={item.w} height={item.h} fill="none" stroke="#fbbf24" strokeDasharray="5 3" strokeWidth={1} vectorEffect="non-scaling-stroke" pointerEvents="none" />)}
              {annotations.map(item => <path key={item.id} d={polygonPath(item.polygons)} fill={item.id === selected ? '#2dd4bf40' : '#60a5fa28'} fillRule="evenodd"
                stroke={item.id === selected ? '#5eead4' : '#93c5fd'} strokeWidth={item.id === selected ? 2 : 1} vectorEffect="non-scaling-stroke"
                onPointerDown={event => { if (!drawing) { event.stopPropagation(); setSelected(item.id); } }} />)}
              {editable && !drawing && object?.polygons.map((ring, ringIndex) => ring.map((point, pointIndex) => <ellipse key={`${ringIndex}:${pointIndex}`} cx={point[0]} cy={point[1]}
                rx={.004 / zoom} ry={.004 * sample.frame.width / sample.frame.height / zoom} fill="#e2e8f0" stroke="#0d9488" strokeWidth={1} vectorEffect="non-scaling-stroke"
                onPointerDown={event => { event.stopPropagation(); drag.current = { ring: ringIndex, point: pointIndex }; event.currentTarget.ownerSVGElement?.setPointerCapture(event.pointerId); }} />))}
              {draft.length > 0 && <polyline points={draft.map(point => point.join(',')).join(' ')} fill="none" stroke="#fbbf24" strokeWidth={2} vectorEffect="non-scaling-stroke" pointerEvents="none" />}
            </svg>
          </div></div><div className={styles.toolbar}><label>Zoom<input type="range" min="1" max="4" step=".25" value={zoom} onChange={event => setZoom(Number(event.target.value))} />{Math.round(zoom * 100)}%</label><small>Chấm polygon, hoặc kéo đỉnh để chỉnh.</small></div>
          {drawing && <div className={styles.toolbar}><span>{draft.length} đỉnh</span><button onClick={() => setDraft(points => points.slice(0, -1))}><Undo2 size={14} />Bỏ đỉnh cuối</button><button onClick={finishPolygon} disabled={draft.length < 3}>Khép polygon</button><button onClick={() => { setDraft([]); setDrawing(false); }}>Hủy vẽ</button></div>}
        </div><aside className={styles.objects}>
          <strong>{annotations.length} đối tượng trong frame</strong>
          {annotations.map((item, index) => <button type="button" key={item.id} disabled={drawing} aria-pressed={selected === item.id} onClick={() => setSelected(item.id)}>{index + 1}. {item.label || item.class_name} {item.polygons.length ? '✓' : '· cần polygon'}</button>)}
          {editable && <button disabled={drawing || annotations.length >= 128 || !model} onClick={() => {
            const identifier = `manual-${crypto.getRandomValues(new Uint32Array(4)).join('-')}`; change([...annotations, { id: identifier, class_name: model!.labels[0], label: model!.labels[0], polygons: [] }]); setSelected(identifier);
          }}><Plus size={14} />Thêm vật bị bỏ sót</button>}
          {object && <><label>Lớp YOLO<select value={object.class_name} disabled={!editable} onChange={event => updateObject({ class_name: event.target.value, label: event.target.value })}>{model?.labels.map(name => <option key={name}>{name}</option>)}</select></label>
            <label>Label / tên vật<input value={object.label} maxLength={120} disabled={!editable} onChange={event => updateObject({ label: event.target.value })} /></label>
            {editable && <><button disabled={drawing || !imageReady} onClick={() => { setDrawing(true); setDraft([]); }}><Pencil size={14} />Vẽ polygon / thêm vùng</button>
              <button disabled={drawing} onClick={() => updateObject({ polygons: [] })}>Xóa mask để vẽ lại</button>
              <button disabled={drawing} onClick={() => { change(annotations.filter(item => item.id !== selected)); setSelected(''); }}><Trash2 size={14} />Loại vật nhận nhầm</button></>}
          </>}
          <p>Viền vàng là bbox gợi ý, không phải segmentation chuẩn. Phải vẽ mask cho vật chưa có polygon. Không lưu frame thiếu nhãn.</p>
        </aside></div>
        {editable && <label><input type="checkbox" checked={complete} disabled={drawing || !valid} onChange={event => setComplete(event.target.checked)} />Tôi đã kiểm tra TẤT CẢ vật thuộc các lớp model trong frame, kể cả vật bị bỏ sót.</label>}
      </>}
      {error && <p role="alert" className={styles.error}>{error}</p>}{message && <p role="status" className={styles.message}>{message}</p>}
    </div>
    <footer className={styles.footer}><small>Ground truth trên SSD + PostgreSQL · Không đổi weights ngay khi bấm lưu.</small><div className={styles.toolbar}>
      <button disabled={!editable || !complete || !valid || drawing || changed} onClick={() => save('correct')}><Check size={15} />Đúng</button>
      <button className={styles.primary} disabled={!editable || !complete || !valid || drawing} onClick={() => save('corrected')}><Save size={15} />Lưu nhãn đã sửa</button>
      <button disabled={!editable} onClick={() => { if (confirm('Loại toàn bộ frame này khỏi tập học?')) void save('reject'); }}>Loại frame</button>
    </div></footer>
  </dialog>;
}
