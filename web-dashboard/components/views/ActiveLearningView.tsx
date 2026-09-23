'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import dynamic from 'next/dynamic';
import { BookOpenCheck, Play, RefreshCw, Save, Trash2, Upload } from 'lucide-react';
import type { VisionModel } from './ModelManager';
import { LearningJob, LearningSample, LearningSettings, LearningStatus, learningImage, learningRequest, learningStates, polygonPath } from '../../lib/active-learning';
import styles from './ActiveLearning.module.css';

const ActiveLearningDialog = dynamic(() => import('./ActiveLearningDialog'), { ssr: false });

function LossChart({ job }: { job: LearningJob }) {
  const history = job.metrics.history || [];
  const losses = history.map(row => Number(row['train/box_loss'])).filter(Number.isFinite);
  if (losses.length < 2) return null;
  const maximum = Math.max(...losses, .001), minimum = Math.min(...losses, 0);
  return <><small>Loss bbox qua các epoch (không phải mIoU)</small><svg className={styles.chart} viewBox="0 0 500 120" role="img" aria-label="Biểu đồ train box loss theo epoch">
    <path d="M 30 8 V 100 H 490" stroke="#64748b" fill="none" />
    <polyline points={losses.map((loss, index) => `${30 + index * 450 / (losses.length - 1)},${100 - (loss - minimum) * 90 / Math.max(.001, maximum - minimum)}`).join(' ')} fill="none" stroke="#14b8a6" strokeWidth="2" />
    <text x="35" y="116" fontSize="10" fill="#94a3b8">Epoch 1 → {losses.length} · {losses[0].toFixed(4)} → {losses[losses.length - 1].toFixed(4)}</text>
  </svg></>;
}

function ModelLearning({ model, active }: { model: VisionModel; active: boolean }) {
  const [tab, setTab] = useState<'samples' | 'jobs' | 'settings'>('samples');
  const [status, setStatus] = useState<LearningStatus | null>(null);
  const [settings, setSettings] = useState<LearningSettings | null>(null);
  const [samples, setSamples] = useState<{ samples: LearningSample[]; counts: Record<string, number>; total: number } | null>(null);
  const [filter, setFilter] = useState('');
  const [offset, setOffset] = useState(0);
  const [selected, setSelected] = useState<LearningSample | null>(null);
  const [busy, setBusy] = useState('');
  const [error, setError] = useState('');
  const [message, setMessage] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [trusted, setTrusted] = useState(false);
  const [log, setLog] = useState<{ id: string; content: string } | null>(null);
  const [revision, setRevision] = useState(0);
  const controller = useRef<AbortController | null>(null);
  const alive = useRef(true);
  const reload = () => setRevision(value => value + 1);

  useEffect(() => {
    alive.current = true;
    return () => { alive.current = false; controller.current?.abort(); };
  }, []);
  useEffect(() => {
    if (!active) return;
    const abort = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      try {
        const result = await learningRequest<LearningStatus>(model.id, '', { signal: abort.signal });
        if (abort.signal.aborted) return;
        setStatus(result); setSettings(current => current || result.settings);
      } catch (reason) { if (!abort.signal.aborted) setError(reason instanceof Error ? reason.message : String(reason)); }
      if (!abort.signal.aborted) timer = setTimeout(poll, 5000);
    };
    void poll();
    return () => { abort.abort(); clearTimeout(timer); };
  }, [active, model.id, revision]);
  useEffect(() => {
    if (!active || tab !== 'samples') return;
    const abort = new AbortController();
    const query = `?limit=12&offset=${offset}${filter ? `&status=${filter}` : ''}`;
    learningRequest<{ samples: LearningSample[]; counts: Record<string, number>; total: number }>(model.id, `/samples${query}`, { signal: abort.signal })
      .then(data => { if (!abort.signal.aborted) setSamples(data); })
      .catch(reason => { if (!abort.signal.aborted) setError(String(reason.message || reason)); });
    return () => abort.abort();
  }, [active, model.id, filter, offset, revision, tab]);

  const act = async (label: string, action: () => Promise<unknown>) => {
    if (busy) return;
    setBusy(label); setError(''); setMessage('');
    try { await action(); if (alive.current) { setMessage(label + ' · thành công.'); reload(); } }
    catch (reason) { if (alive.current) setError(reason instanceof Error ? reason.message : String(reason)); }
    finally { if (alive.current) setBusy(''); }
  };
  const upload = () => act('Upload checkpoint', async () => {
    if (!file || !trusted || !file.name.toLowerCase().endsWith('.pt') || file.size > 512 * 1024**2) throw new Error('Chọn file .pt đáng tin cậy, tối đa 512 MiB.');
    const abort = new AbortController(); controller.current = abort;
    const result = await learningRequest<{ id: string }>(model.id, '/weights', { method: 'POST', signal: abort.signal,
      body: JSON.stringify({ filename: file.name, size_bytes: file.size, trusted: true }) });
    for (let position = 0; position < file.size;) {
      const chunk = file.slice(position, position + 2 * 1024**2);
      const response = await learningRequest<{ received_bytes: number }>(model.id, `/weights/${result.id}/file?offset=${position}`, {
        method: 'PUT', signal: abort.signal, headers: { 'Content-Type': 'application/octet-stream' }, body: chunk,
      });
      position = response.received_bytes;
      if (alive.current) setBusy(`Upload ${Math.round(position * 100 / file.size)}%`);
    }
    await learningRequest(model.id, `/weights/${result.id}/complete`, { method: 'POST', signal: abort.signal });
    if (alive.current) setFile(null);
  });
  const jobAction = (job: LearningJob, action: 'cancel' | 'promote') => act(action === 'cancel' ? 'Yêu cầu hủy job' : 'Xếp hàng đổi engine', () => learningRequest(model.id, `/jobs/${job.id}/${action}`, { method: 'POST' }));

  return <>
    {active && selected && <ActiveLearningDialog initialSample={selected} onClose={() => { setSelected(null); reload(); }} onSaved={reload} />}
    <div className={styles.tabs} role="tablist" aria-label="Active Learning">
      <button role="tab" aria-selected={tab === 'samples'} onClick={() => setTab('samples')}>Dataset &amp; phản hồi</button>
      <button role="tab" aria-selected={tab === 'jobs'} onClick={() => setTab('jobs')}>Job &amp; kiểm định</button>
      <button role="tab" aria-selected={tab === 'settings'} onClick={() => setTab('settings')}>Weights &amp; lịch học</button>
      <button onClick={reload} aria-label="Tải lại Active Learning"><RefreshCw size={15} /></button>
    </div>
    {status && <div className={styles.warning}><p>{status.worker.note}</p><p>Fine-tune YOLO detection → ONNX → TensorRT → kiểm định engine → đổi engine. SAM2 giữ nguyên weights; mask/polygon được lưu làm ground truth, không báo mIoU giả.</p></div>}
    {error && <p role="alert" className={styles.error}>{error}</p>}{message && <p role="status" className={styles.message}>{message}</p>}{busy && <p role="status">{busy}…</p>}
    {tab === 'samples' && <>
      <div className={styles.toolbar}><p>{samples?.counts.approved || 0} frame chuẩn · {samples?.counts.draft || 0} ảnh nháp · {samples?.counts.rejected || 0} đã loại</p>
        <label>Lọc<select value={filter} onChange={event => { setFilter(event.target.value); setOffset(0); }}><option value="">Tất cả</option><option value="draft">Ảnh nháp</option><option value="approved">Đã duyệt</option><option value="rejected">Đã loại</option></select></label></div>
      {!samples?.total && <p>Chưa có dữ liệu Active Learning. Mở Monitor → “Đúng / Sửa nhãn” trên camera. Label đa góc nhìn cũ vẫn được giữ riêng, không tự biến thành ground truth huấn luyện.</p>}
      <div className={styles.cards}>{samples?.samples.map(sample => <article className={styles.card} key={sample.id}>
        <div className={styles.thumb}><img src={learningImage(sample)} alt={`Frame ${sample.frame_id} camera ${sample.camera_id}`} loading="lazy" /><svg viewBox="0 0 1 1" preserveAspectRatio="none" aria-hidden="true">{sample.annotations.map(item => <path key={item.id} d={polygonPath(item.polygons)} fill="#2dd4bf35" fillRule="evenodd" stroke="#5eead4" strokeWidth={1} vectorEffect="non-scaling-stroke" />)}</svg></div>
        <p>{sample.camera_id} · #{sample.frame_id} · {sample.status === 'approved' ? 'Đã duyệt' : sample.status === 'draft' ? 'Ảnh nháp' : 'Đã loại'}<br />{new Date(sample.captured_at).toLocaleString('vi-VN')}</p>
        <div className={styles.toolbar}><button onClick={() => setSelected(sample)}>{sample.status === 'draft' ? 'Duyệt / sửa' : 'Xem nhãn'}</button>
          <button aria-label={`Xóa mẫu ${sample.id}`} disabled={!!busy} onClick={() => { if (confirm('Xóa frame và nhãn khỏi dataset? Job dùng mẫu này sẽ không được promote.')) void act('Xóa mẫu', () => learningRequest(model.id, `/samples/${sample.id}`, { method: 'DELETE' })); }}><Trash2 size={14} /></button></div>
      </article>)}</div><div className={styles.toolbar}><small>{samples?.total || 0} mẫu · trang {offset / 12 + 1}</small><div><button disabled={offset === 0} onClick={() => setOffset(value => Math.max(0, value - 12))}>Trước</button> <button disabled={offset + 12 >= (samples?.total || 0)} onClick={() => setOffset(value => value + 12)}>Sau</button></div></div>
    </>}
    {tab === 'jobs' && <>
      <div className={styles.toolbar}><p>Baseline và candidate TensorRT được đo trên cùng tập validation; model kém hơn không thay bản live.</p>
        <button disabled={!!busy || !status || !!status.prerequisites.length} onClick={() => act('Tạo job retrain', () => learningRequest(model.id, '/jobs', { method: 'POST' }))}><Play size={15} />Tạo job retrain</button></div>
      {status?.prerequisites.map(problem => <p key={problem} className={styles.error}>{problem}</p>)}
      {!status?.jobs.length && <p>Chưa có job huấn luyện. Cung cấp checkpoint .pt và dataset gốc ở tab “Weights &amp; lịch học”.</p>}
      <div className={styles.cards}>{status?.jobs.map(job => <article className={styles.card} key={job.id}>
        <h3>{learningStates[job.state] || job.state}</h3><p>{job.id.slice(0, 10)} · {job.trigger} · {new Date(job.created_at).toLocaleString('vi-VN')}</p>
        {(job.reason || job.metrics.reason) && <p>{job.reason || job.metrics.reason}</p>}
        {job.metrics.baseline && job.metrics.candidate && <p>mAP50–95: {(job.metrics.baseline.map50_95 * 100).toFixed(2)}% → {(job.metrics.candidate.map50_95 * 100).toFixed(2)}%<br />Validation: {job.metrics.validation_images} ảnh</p>}
        <LossChart job={job} />
        <div className={styles.toolbar}>
          {['queued', 'waiting_resources', 'preparing', 'training', 'building', 'promotion_queued'].includes(job.state) && <button disabled={!!busy || job.cancel_requested} onClick={() => jobAction(job, 'cancel')}>{job.cancel_requested ? 'Đang hủy…' : 'Hủy job'}</button>}
          {job.state === 'ready' && <button disabled={!!busy} onClick={() => { if (confirm('Áp dụng engine đã vượt kiểm định vào runtime? Bản trước vẫn được giữ để khôi phục.')) void jobAction(job, 'promote'); }}>Áp dụng engine</button>}
          <button disabled={!!busy} onClick={() => act('Tải log', async () => {
            const response = await fetch(`/api/backend/active-learning/models/${model.id}/jobs/${job.id}/log`, { cache: 'no-store' });
            if (!response.ok) throw new Error('Không tải được log.');
            const text = await response.text(); if (alive.current) setLog({ id: job.id, content: text });
          })}>Xem log</button>
        </div>
      </article>)}</div>
      {log && <div><div className={styles.toolbar}><strong>Log · {log.id.slice(0, 10)}</strong><button onClick={() => setLog(null)}>Đóng log</button></div><pre className={styles.log}>{log.content}</pre></div>}
    </>}
    {tab === 'settings' && status && settings && <div className={styles.grid}>
      <section className={styles.panel}><h3>01 · Checkpoint &amp; dữ liệu gốc</h3><p>ONNX không chứa trạng thái huấn luyện. Upload best.pt tương ứng model đã deploy; không tự lấy model pose hay weights không liên quan.</p>
        <div className={styles.fields}><label>Checkpoint .pt<input type="file" accept=".pt" disabled={!!busy} onChange={event => { setFile(event.target.files?.[0] || null); setTrusted(false); }} /></label>
          <label><input type="checkbox" checked={trusted} onChange={event => setTrusted(event.target.checked)} />Tôi tin cậy nguồn .pt này; checkpoint PyTorch có thể chứa mã thực thi.</label>
          <button disabled={!file || !trusted || !!busy} onClick={upload}><Upload size={15} />Upload checkpoint</button></div>
        {status.weights.map(weight => <p key={weight.id}>{weight.filename} · {weight.state} · {(weight.received_bytes / 1024**2).toFixed(1)} MiB</p>)}
        <p>Đặt dataset YOLO gốc trong thư mục SSD của model (đường dẫn trong container):</p><code className={styles.path}>{status.directory}/base/</code>
        <p>Cấu trúc: data.yaml, images/train, images/val, labels/train, labels/val. Validation không trùng train/correction và phải có đủ các lớp.</p>
        <label>YAML tương đối<input value={settings.base_dataset} onChange={event => setSettings({ ...settings, base_dataset: event.target.value })} /></label>
      </section>
      <section className={styles.panel}><h3>02 · Lịch fine-tune</h3><div className={styles.fields}>
        <label><span>Tự xếp hàng khi có dữ liệu mới</span><input type="checkbox" checked={settings.enabled} onChange={event => setSettings({ ...settings, enabled: event.target.checked })} /></label>
        <label>Ngưỡng frame mới<input type="number" min="10" max="100000" value={settings.threshold} onChange={event => setSettings({ ...settings, threshold: Number(event.target.value) })} /></label>
        <label>Giờ chạy hằng ngày<select value={settings.daily_hour ?? ''} onChange={event => setSettings({ ...settings, daily_hour: event.target.value === '' ? null : Number(event.target.value) })}><option value="">Không đặt giờ</option>{Array.from({ length: 24 }, (_, hour) => <option key={hour} value={hour}>{String(hour).padStart(2, '0')}:00</option>)}</select></label>
        <label>Múi giờ<input value={settings.timezone} onChange={event => setSettings({ ...settings, timezone: event.target.value })} /></label>
        <label>Epoch<input type="number" min="1" max="10" value={settings.epochs} onChange={event => setSettings({ ...settings, epochs: Number(event.target.value) })} /></label>
        <label>Learning rate<input type="number" min="0.000001" max="0.001" step="0.000001" value={settings.learning_rate} onChange={event => setSettings({ ...settings, learning_rate: Number(event.target.value) })} /></label>
        <label>Batch<input type="number" min="1" max="8" value={settings.batch} onChange={event => setSettings({ ...settings, batch: Number(event.target.value) })} /></label>
        <label>mAP tăng tối thiểu<input type="number" min="0" max="0.1" step=".001" value={settings.minimum_gain} onChange={event => setSettings({ ...settings, minimum_gain: Number(event.target.value) })} /></label>
        <label><span>Tự áp dụng sau khi đạt kiểm định</span><input type="checkbox" checked={settings.auto_promote} onChange={event => setSettings({ ...settings, auto_promote: event.target.checked })} /></label>
        <button disabled={!!busy} className={styles.primary} onClick={() => act('Lưu lịch học', async () => { const saved = await learningRequest<LearningSettings>(model.id, '/settings', { method: 'PUT', body: JSON.stringify(settings) }); if (alive.current) setSettings(saved); })}><Save size={15} />Lưu cấu hình</button>
      </div></section>
    </div>}
  </>;
}

export default function ActiveLearningView({ active }: { active: boolean }) {
  const [models, setModels] = useState<VisionModel[]>([]);
  const [modelId, setModelId] = useState('');
  const [error, setError] = useState('');
  const load = useCallback(async (signal: AbortSignal) => {
    try {
      const response = await fetch('/api/backend/models', { signal, cache: 'no-store' });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || 'Không tải được model.');
      if (!signal.aborted) { setModels(payload.models); setModelId(current => current || payload.deployment?.model_id || payload.models.find((model: VisionModel) => model.state === 'ready')?.id || ''); }
    } catch (reason) { if (!signal.aborted) setError(reason instanceof Error ? reason.message : String(reason)); }
  }, []);
  useEffect(() => { if (!active) return; const abort = new AbortController(); void load(abort.signal); return () => abort.abort(); }, [active, load]);
  const model = models.find(item => item.id === modelId);
  return <section className={styles.root}><div className={styles.header}><div><h2><BookOpenCheck size={21} />Active Learning</h2><p>Người dùng xác nhận → dataset có kiểm soát → fine-tune → kiểm định → cập nhật model.</p></div>
    <label>Model<select value={modelId} onChange={event => setModelId(event.target.value)}><option value="">Chọn model</option>{models.filter(item => item.state === 'ready').map(item => <option key={item.id} value={item.id}>{item.name}</option>)}</select></label></div>
    {error && <p role="alert" className={styles.error}>{error}</p>}
    {model ? <ModelLearning key={model.id} model={model} active={active} /> : <p>Upload và build model trong Model / TensorRT trước.</p>}
  </section>;
}
