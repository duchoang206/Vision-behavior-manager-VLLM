'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { Cpu, Upload, RefreshCw } from 'lucide-react';
import { useCameras } from '../CameraContext';
import styles from './ModelManager.module.css';

export type VisionModel = { id: string; name: string; filename: string; state: string; size_bytes: number; received_bytes: number; labels: string[]; error?: string; metadata?: { shape?: number[]; sha256?: string } };
type ModelPayload = { models: VisionModel[]; chunk_size: number; max_size: number;
  deployment?: { model_id: string; all_cameras: boolean; camera_ids: string[] } | null;
  runtime: { state?: string; model_id?: string; error?: string; cameras?: Record<string, { frames: number; age_ms: number;
    segmentation?: { ready?: boolean; error?: string; masks?: number; targets?: number; inference_ms?: number; observed_at?: number } }> } };

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api/backend/models${path}`, { ...init, cache: 'no-store' });
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : `HTTP ${response.status}`);
  return data;
}

export default function ModelManager({ active }: { active: boolean }) {
  const { cameras } = useCameras();
  const [allCameras, setAllCameras] = useState(true);
  const [cameraIds, setCameraIds] = useState<string[]>([]);
  const [deploying, setDeploying] = useState(false);
  const [data, setData] = useState<ModelPayload | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [name, setName] = useState('');
  const [labels, setLabels] = useState('');
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState('');
  const [message, setMessage] = useState('');
  const uploadId = useRef<string | null>(null);
  const mounted = useRef(true);
  const refresh = useCallback(async () => {
    try { const payload = await request<ModelPayload>(''); if (mounted.current) setData(payload); }
    catch (reason) { if (mounted.current) setError(String(reason instanceof Error ? reason.message : reason)); }
  }, []);
  useEffect(() => { mounted.current = true; return () => { mounted.current = false; }; }, []);
  useEffect(() => { if (!active) return; void refresh(); const timer = setInterval(refresh, 2500); return () => clearInterval(timer); }, [active, refresh]);
  const labelList = () => labels.split(/[,\n]/).map(label => label.trim()).filter(Boolean);

  const upload = async () => {
    if (!file || busy) return;
    setBusy(true); setError(''); setMessage('');
    try {
      if (!file.name.toLowerCase().endsWith('.onnx') || file.size > (data?.max_size || 512 * 1024**2)) throw new Error('Chọn file ONNX tối đa 512 MiB.');
      const current = uploadId.current ? (await request<ModelPayload>('')).models.find(model => model.id === uploadId.current) : null;
      const model = current || await request<VisionModel>('', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name, filename: file.name, size_bytes: file.size, labels: labelList() }) });
      uploadId.current = model.id;
      const chunkSize = data?.chunk_size || 2 * 1024**2;
      for (let offset = model.received_bytes; offset < file.size;) {
        const chunk = file.slice(offset, offset + chunkSize);
        const result = await request<{ received_bytes: number }>(`/${model.id}/file?offset=${offset}`, { method: 'PUT', headers: { 'Content-Type': 'application/octet-stream' }, body: chunk });
        offset = result.received_bytes;
        setProgress(Math.round(offset * 100 / file.size));
      }
      await request(`/${model.id}/build`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ labels: labelList() }) });
      uploadId.current = null;
      setMessage('Đã nhận ONNX. Backend kiểm tra labels và build TensorRT FP16. Khi Ready, chọn camera rồi bấm Deploy lên Monitor.');
      setFile(null); await refresh();
    } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); }
    finally { setBusy(false); }
  };
  const retry = async (model: VisionModel) => {
    try {
      setError(''); await request(`/${model.id}/build`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ labels: labelList().length ? labelList() : model.labels }) });
      await refresh();
    } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); }
  };

  const deploy = async (model?: VisionModel) => {
    if (deploying) return;
    setDeploying(true); setError(''); setMessage('');
    try {
      if (model) {
        await request(`/${model.id}/deploy`, { method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ all_cameras: allCameras, camera_ids: allCameras ? [] : cameraIds }) });
        setMessage('Đã lưu deployment. Backend tự chạy DeepStream + NvDCF + SAM2 trên camera đã chọn; cấu hình được giữ khi khởi động lại.');
      } else {
        await request('/deployment', { method: 'DELETE' });
        setMessage('Đã dừng deployment trực tiếp. Nếu Workflow vẫn dùng model này, detector tiếp tục phục vụ Workflow.');
      }
      await refresh();
    } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); }
    finally { setDeploying(false); }
  };

  return <section className={styles.root}>
    <div className={styles.header}><div><h2><Cpu size={20} /> Model / TensorRT</h2><p>Upload → TensorRT FP16 → Deploy → DeepStream + NvDCF → SAM2 CUDA → Monitor.</p></div><button onClick={refresh}><RefreshCw size={15} /> Tải lại</button></div>
    <div className={styles.layout}><div className={styles.upload}>
      <label>Tên model<input value={name} maxLength={120} disabled={busy} onChange={event => setName(event.target.value)} placeholder="Robot · Person · Helmet" /></label>
      <label>Weights / best.onnx<input key={file ? 'selected' : 'empty'} type="file" accept=".onnx" disabled={busy} onChange={event => { setFile(event.target.files?.[0] || null); uploadId.current = null; setProgress(0); }} /></label>
      <label>Labels dự phòng / sửa metadata<textarea disabled={busy} value={labels} onChange={event => setLabels(event.target.value)} placeholder="Để trống để đọc names từ ONNX. Hoặc: person, robot, helmet" /></label>
      <p>Tên lớp phải khớp model đã huấn luyện. Robot_2001 được đối chiếu robot 2001 trên FMS nếu camera đã calib. Dùng tab Label để bổ sung nhiều góc nhìn và mẫu dễ nhầm: embedding + triplet xác thực danh tính trước khi gửi mask. Không tự huấn luyện lại weights YOLO.</p>
      <button disabled={!file || busy} onClick={upload}><Upload size={16} />{busy ? `Đang upload ${progress}%` : uploadId.current ? 'Tiếp tục upload / build' : 'Upload & build TensorRT'}</button>
      {(busy || progress > 0) && <progress max={100} value={progress} />}
      <details><summary>Định dạng hỗ trợ</summary><p>YOLOv8 / YOLO11 detection, một input RGB, output [1, 4 + số lớp, anchors]. Không nhận ONNX pose, seg hoặc NMS nhúng.</p><code>yolo export model=best.pt format=onnx imgsz=640 batch=1 dynamic=False half=False nms=False</code><p>Build dùng GPU một lần nên có thể tăng tải tạm thời; không cam kết giữ nguyên FPS trong lúc build. Không bật lại model pose.</p></details>
    </div><div className={styles.models}>
      <div className={styles.runtime}>
        <strong>Camera chạy model</strong>
        <label className={styles.cameraChoice}><input type="checkbox" checked={allCameras} onChange={event => setAllCameras(event.target.checked)} /> Tất cả camera, tự áp dụng cho camera mới</label>
        {!allCameras && cameras.map(camera => <label className={styles.cameraChoice} key={camera.id}><input type="checkbox" checked={cameraIds.includes(camera.id)} onChange={event => setCameraIds(previous => event.target.checked ? [...previous, camera.id] : previous.filter(id => id !== camera.id))} />{camera.name}</label>)}
        {data?.deployment && <p>Đã lưu: {data.models.find(model => model.id === data.deployment?.model_id)?.name || data.deployment.model_id} · {data.deployment.all_cameras ? 'tất cả camera' : data.deployment.camera_ids.map(id => cameras.find(camera => camera.id === id)?.name || id).join(', ')} <button disabled={deploying} onClick={() => deploy()}>Dừng deployment</button></p>}
      </div>
      <div className={styles.runtime}>Custom DeepStream: <strong>{data?.runtime.state || 'idle'}</strong>{data?.runtime.error && <p role="alert">{data.runtime.error}</p>}{Object.entries(data?.runtime.cameras || {}).map(([cameraId, status]) => <span key={cameraId}>{cameras.find(camera => camera.id === cameraId)?.name || cameraId}: {status.frames} frames · metadata {status.age_ms} ms<br />SAM2: {status.segmentation?.error || (status.segmentation?.ready ? `${status.segmentation.masks ?? 0}/${status.segmentation.targets ?? 0} mask · ${status.segmentation.inference_ms ?? 0} ms/lượt` : 'đang nạp CUDA')}</span>)}</div>
      {!data?.models.length && <p>Chưa upload model. Upload và deploy trước khi đăng ký Label hoặc bật mask model trên Monitor.</p>}
      {data?.models.map(model => <article key={model.id}><div className={styles.header}><strong>{model.name}</strong><span data-state={model.state}>{model.state}</span></div><small>{model.filename} · {(model.size_bytes / 1024**2).toFixed(1)} MiB · {model.metadata?.shape?.join(' × ')}</small><p>{model.labels.map((label, index) => `${index}: ${label}`).join(' · ') || 'Labels sẽ được đọc khi upload xong'}</p>{model.error && <details><summary>Lỗi kiểm tra/build</summary><pre>{model.error}</pre></details>}{model.state === 'failed' && <button onClick={() => retry(model)}>Build lại (dùng labels ở ô bên trái nếu nhập)</button>}{model.state === 'ready' && <button disabled={deploying || (!allCameras && !cameraIds.length)} onClick={() => deploy(model)}>{data.deployment?.model_id === model.id ? 'Cập nhật camera deploy' : 'Deploy lên Monitor'}</button>}</article>)}
    </div></div>{error && <p role="alert" className={styles.error}>{error}</p>}{message && <p role="status" className={styles.message}>{message}</p>}
  </section>;
}
