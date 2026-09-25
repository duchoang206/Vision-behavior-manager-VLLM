'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { Cpu, Upload, RefreshCw } from 'lucide-react';
import { useCameras } from '../CameraContext';
import styles from './ModelManager.module.css';

export type VisionModel = { id: string; name: string; filename: string; state: string; size_bytes: number; received_bytes: number; labels: string[]; error?: string; metadata?: { shape?: number[]; sha256?: string; model_type?: string; task?: string } };
type ConfidenceThresholds = {
  default?: Record<string, number>;
  cameras?: Record<string, Record<string, number>>;
};
type ModelDeployment = {
  id: string;
  model_id: string;
  all_cameras: boolean;
  camera_ids: string[];
  enabled: boolean;
  updated_at?: string;
  confidence_thresholds?: ConfidenceThresholds;
};
type ModelPayload = { models: VisionModel[]; chunk_size: number; max_size: number;
  deployment?: ModelDeployment | null; deployments: ModelDeployment[];
  runtime: { state?: string; model_id?: string; model_ids?: string[]; error?: string; workers?: Record<string, { state?: string; error?: string; model_type?: string; cameras?: Record<string, { frames: number; age_ms: number }> }>; cameras?: Record<string, { frames: number; age_ms: number;
    segmentation?: { ready?: boolean; error?: string; masks?: number; targets?: number; inference_ms?: number; observed_at?: number } }> } };

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api/backend/models${path}`, { ...init, cache: 'no-store' });
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : `HTTP ${response.status}`);
  return data;
}

export default function ModelManager({ active }: { active: boolean }) {
  const { cameras } = useCameras();
  const [allCameras, setAllCameras] = useState(false);
  const [cameraIds, setCameraIds] = useState<string[]>([]);
  const [selectedModelIds, setSelectedModelIds] = useState<string[]>([]);
  const [deploying, setDeploying] = useState(false);
  const [data, setData] = useState<ModelPayload | null>(null);
  const [defaultThresholds, setDefaultThresholds] = useState<Record<string, number>>({});
  const [cameraThresholds, setCameraThresholds] = useState<Record<string, Record<string, number>>>({});
  const [selectedCamForThreshold, setSelectedCamForThreshold] = useState<string>('');
  const [file, setFile] = useState<File | null>(null);
  const [name, setName] = useState('');
  const [labels, setLabels] = useState('');
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState('');
  const [message, setMessage] = useState('');
  const [uploadPending, setUploadPending] = useState(false);
  const uploadId = useRef<string | null>(null);
  const hydratedDeployment = useRef<string | null>(null);
  const mounted = useRef(true);
  const refresh = useCallback(async () => {
    try { const payload = await request<ModelPayload>(''); if (mounted.current) setData(payload); }
    catch (reason) { if (mounted.current) setError(String(reason instanceof Error ? reason.message : reason)); }
  }, []);
  useEffect(() => { mounted.current = true; return () => { mounted.current = false; }; }, []);
  useEffect(() => { if (!active) return; const initial = window.setTimeout(() => { void refresh(); }, 0); const timer = setInterval(refresh, 2500); return () => { clearTimeout(initial); clearInterval(timer); }; }, [active, refresh]);

  useEffect(() => {
    if (data?.deployments?.length) {
      const activeDep = data.deployments.find(d => d.enabled) || data.deployments[0];
      const signature = activeDep ? `${activeDep.id}:${activeDep.updated_at || ''}` : null;
      if (activeDep && signature !== hydratedDeployment.current) {
        hydratedDeployment.current = signature;
        setAllCameras(activeDep.all_cameras === true);
        setCameraIds(activeDep.all_cameras ? [] : [...activeDep.camera_ids]);
        setDefaultThresholds(activeDep.confidence_thresholds?.default || {});
        setCameraThresholds(activeDep.confidence_thresholds?.cameras || {});
      }
    }
  }, [data?.deployments]);

  const labelList = () => labels.split(/[,\n]/).map(label => label.trim()).filter(Boolean);

  const getRelevantLabels = (model?: VisionModel): string[] => {
    if (model?.labels?.length) return model.labels;
    const selected = (data?.models || []).filter(m => selectedModelIds.includes(m.id) && m.state === 'ready');
    if (selected.length > 0) {
      const set = new Set<string>();
      selected.forEach(m => m.labels.forEach(l => set.add(l)));
      return Array.from(set);
    }
    const all = new Set<string>();
    (data?.models || []).filter(m => m.state === 'ready').forEach(m => m.labels.forEach(l => all.add(l)));
    return Array.from(all);
  };

  const buildConfidencePayload = (model?: VisionModel): ConfidenceThresholds => {
    const activeLabels = getRelevantLabels(model);
    const def: Record<string, number> = {};
    for (const l of activeLabels) {
      def[l] = defaultThresholds[l] !== undefined ? defaultThresholds[l] : (l.toLowerCase().includes('rack') ? 0.15 : 0.25);
    }
    const cams: Record<string, Record<string, number>> = {};
    for (const [cId, map] of Object.entries(cameraThresholds)) {
      const filtered: Record<string, number> = {};
      for (const [l, val] of Object.entries(map)) {
        if (activeLabels.includes(l) && val !== undefined) {
          filtered[l] = val;
        }
      }
      if (Object.keys(filtered).length > 0) {
        cams[cId] = filtered;
      }
    }
    return { default: def, cameras: cams };
  };

  const upload = async () => {
    if (!file || busy) return;
    setBusy(true); setError(''); setMessage('');
    try {
      if (!file.name.toLowerCase().endsWith('.onnx') || file.size > (data?.max_size || 512 * 1024**2)) throw new Error('Chọn file ONNX tối đa 512 MiB.');
      const current = uploadId.current ? (await request<ModelPayload>('')).models.find(model => model.id === uploadId.current) : null;
      const model = current || await request<VisionModel>('', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name, filename: file.name, size_bytes: file.size, labels: labelList() }) });
      uploadId.current = model.id;
      setUploadPending(true);
      const chunkSize = data?.chunk_size || 2 * 1024**2;
      for (let offset = model.received_bytes; offset < file.size;) {
        const chunk = file.slice(offset, offset + chunkSize);
        const result = await request<{ received_bytes: number }>(`/${model.id}/file?offset=${offset}`, { method: 'PUT', headers: { 'Content-Type': 'application/octet-stream' }, body: chunk });
        offset = result.received_bytes;
        setProgress(Math.round(offset * 100 / file.size));
      }
      await request(`/${model.id}/build`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ labels: labelList() }) });
      uploadId.current = null;
      setUploadPending(false);
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
        const confPayload = buildConfidencePayload(model);
        await request(`/${model.id}/deploy`, { method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ all_cameras: allCameras, camera_ids: allCameras ? [] : cameraIds, confidence_thresholds: confPayload }) });
        setMessage('Đã lưu deployment với ngưỡng độ tự tin tùy chỉnh. Backend chạy DeepStream + NvDCF với ngưỡng riêng cho từng camera/label.');
      } else {
        await request('/deployment', { method: 'DELETE' });
        setMessage('Đã dừng deployment trực tiếp. Nếu Workflow vẫn dùng model này, detector tiếp tục phục vụ Workflow.');
      }
      await refresh();
    } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); }
    finally { setDeploying(false); }
  };

  const deploySelected = async () => {
    const models = data?.models.filter(model => selectedModelIds.includes(model.id) && model.state === 'ready') || [];
    if (!models.length || deploying) return;
    setDeploying(true); setError(''); setMessage('');
    try {
      await Promise.all(models.map(model => request(`/${model.id}/deploy`, { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ all_cameras: allCameras, camera_ids: allCameras ? [] : cameraIds, confidence_thresholds: buildConfidencePayload(model) }) })));
      setMessage(`Đã deploy ${models.length} model với ngưỡng độ tự tin tùy chỉnh.`);
      await refresh();
    } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); }
    finally { setDeploying(false); }
  };

  const setDeploymentEnabled = async (deployment: ModelDeployment, enabled: boolean) => {
    try {
      setDeploying(true); setError('');
      await request(`/deployments/${deployment.id}`, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ enabled }) });
      await refresh();
    } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); }
    finally { setDeploying(false); }
  };

  const removeDeployment = async (deployment: ModelDeployment) => {
    try {
      setDeploying(true); setError('');
      await request(`/deployments/${deployment.id}`, { method: 'DELETE' });
      await refresh();
    } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); }
    finally { setDeploying(false); }
  };

  const deployableLabels = getRelevantLabels();
  const activeDeployments = (data?.deployments || []).filter(deployment => deployment.enabled);
  const activeModelIds = new Set(activeDeployments.map(deployment => deployment.model_id));
  const orderedDeployments = [...(data?.deployments || [])].sort((left, right) => Number(right.enabled) - Number(left.enabled));
  const orderedModels = [...(data?.models || [])].sort((left, right) => Number(activeModelIds.has(right.id)) - Number(activeModelIds.has(left.id)));

  return <section className={styles.root}>
    <div className={styles.header}><div><h2><Cpu size={20} /> Model / TensorRT</h2><p>Upload → TensorRT FP16 → Deploy nhiều model → DeepStream + NvDCF → metadata chuẩn hóa → Monitor.</p></div><button onClick={refresh}><RefreshCw size={15} /> Tải lại</button></div>
    <div className={styles.layout}><div className={styles.upload}>
      <label>Tên model<input value={name} maxLength={120} disabled={busy} onChange={event => setName(event.target.value)} placeholder="Robot · Person · Helmet" /></label>
      <label>Weights / best.onnx<input key={file ? 'selected' : 'empty'} type="file" accept=".onnx" disabled={busy} onChange={event => { setFile(event.target.files?.[0] || null); uploadId.current = null; setUploadPending(false); setProgress(0); }} /></label>
      <label>Labels dự phòng / sửa metadata<textarea disabled={busy} value={labels} onChange={event => setLabels(event.target.value)} placeholder="Để trống để đọc names từ ONNX. Hoặc: person, robot, helmet" /></label>
      <p>Tên lớp phải khớp model đã huấn luyện. Robot_2001 được đối chiếu robot 2001 trên FMS nếu camera đã calib. Dùng tab Label để bổ sung nhiều góc nhìn và mẫu dễ nhầm: embedding + triplet xác thực danh tính trước khi gửi mask. Không tự huấn luyện lại weights YOLO.</p>
      <button disabled={!file || busy} onClick={upload}><Upload size={16} />{busy ? `Đang upload ${progress}%` : uploadPending ? 'Tiếp tục upload / build' : 'Upload & build TensorRT'}</button>
      {(busy || progress > 0) && <progress max={100} value={progress} />}
      <details><summary>Định dạng hỗ trợ</summary><p>YOLO detect, YOLO-Seg native (task=segment) và YOLO Pose 17 điểm COCO (task=pose, đúng một nhãn người). OBB và classifier được giữ trong contract metadata nhưng cần parser DeepStream tương ứng trước khi build.</p><code>yolo export model=best.pt format=onnx imgsz=640 batch=1 dynamic=False half=False nms=False</code><p>Build dùng GPU một lần nên có thể tăng tải tạm thời.</p></details>
    </div><div className={styles.models}>
      <div className={styles.runtime}>
        <div className={styles.activeModels}>
          <strong>Model TensorRT đang chạy</strong>
          {activeDeployments.length ? activeDeployments.map(deployment => {
            const model = data?.models.find(item => item.id === deployment.model_id);
            const cameraNames = deployment.all_cameras
              ? 'Tất cả camera'
              : deployment.camera_ids.map(id => cameras.find(camera => camera.id === id)?.name || id).join(', ');
            return <div className={styles.activeModelRow} key={deployment.id}>
              <span className={styles.liveBadge}>ĐANG CHẠY</span>
              <div><b>{model?.filename || model?.name || deployment.model_id}</b><small>{model?.metadata?.shape?.join(' × ') || 'TensorRT'} · {cameraNames}</small></div>
            </div>;
          }) : <p>Chưa có model trực tiếp nào đang bật.</p>}
        </div>
        <strong>Camera chạy model</strong>
        <label className={styles.cameraChoice}><input type="checkbox" checked={allCameras} onChange={event => setAllCameras(event.target.checked)} /> Tất cả camera, tự áp dụng cho camera mới</label>
        {!allCameras && cameras.map(camera => <label className={styles.cameraChoice} key={camera.id}><input type="checkbox" checked={cameraIds.includes(camera.id)} onChange={event => setCameraIds(previous => event.target.checked ? [...previous, camera.id] : previous.filter(id => id !== camera.id))} />{camera.name}</label>)}

        {deployableLabels.length > 0 && (
          <div className={styles.thresholdSection}>
            <span className={styles.thresholdTitle}>Độ tự tin (Confidence Threshold)</span>
            <p style={{ margin: '0 0 8px 0', fontSize: '11px', color: 'var(--text-muted)' }}>
              Chỉnh ngưỡng phát hiện cho từng Label (mặc định) và tùy chỉnh riêng theo từng Camera.
            </p>

            <strong style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>1. Ngưỡng chung cho tất cả Camera</strong>
            <div className={styles.thresholdGrid}>
              {deployableLabels.map(label => {
                const val = defaultThresholds[label] ?? (label.toLowerCase().includes('rack') ? 0.15 : 0.25);
                return (
                  <div key={label} className={styles.thresholdCard}>
                    <div className={styles.thresholdLabelRow}>
                      <span>{label}</span>
                      <span style={{ color: '#14a08c' }}>{val.toFixed(2)}</span>
                    </div>
                    <div className={styles.thresholdSliderRow}>
                      <input
                        type="range"
                        min={0.01}
                        max={0.99}
                        step={0.01}
                        value={val}
                        onChange={e => {
                          const num = parseFloat(e.target.value);
                          setDefaultThresholds(prev => ({ ...prev, [label]: num }));
                        }}
                      />
                      <input
                        type="number"
                        min={0.01}
                        max={0.99}
                        step={0.01}
                        value={val}
                        onChange={e => {
                          const num = parseFloat(e.target.value) || 0.01;
                          setDefaultThresholds(prev => ({ ...prev, [label]: Math.max(0.01, Math.min(0.99, num)) }));
                        }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>

            <strong style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>2. Tùy chỉnh riêng cho Camera cụ thể</strong>
            <div className={styles.camOverrideBox}>
              <div className={styles.camOverrideHeader}>
                <span>Chọn camera:</span>
                <select
                  value={selectedCamForThreshold}
                  onChange={e => setSelectedCamForThreshold(e.target.value)}
                >
                  <option value="">-- Chọn Camera để đặt ngưỡng riêng --</option>
                  {cameras.map(c => (
                    <option key={c.id} value={c.id}>
                      {c.name} {cameraThresholds[c.id] ? '(Đã có cấu hình riêng)' : ''}
                    </option>
                  ))}
                </select>
                {selectedCamForThreshold && cameraThresholds[selectedCamForThreshold] && (
                  <button
                    type="button"
                    style={{ padding: '4px 8px', fontSize: '11px' }}
                    onClick={() => {
                      setCameraThresholds(prev => {
                        const next = { ...prev };
                        delete next[selectedCamForThreshold];
                        return next;
                      });
                    }}
                  >
                    Xóa cấu hình riêng của camera này
                  </button>
                )}
              </div>

              {selectedCamForThreshold && (
                <div className={styles.thresholdGrid}>
                  {deployableLabels.map(label => {
                    const hasOverride = cameraThresholds[selectedCamForThreshold]?.[label] !== undefined;
                    const defVal = defaultThresholds[label] ?? (label.toLowerCase().includes('rack') ? 0.15 : 0.25);
                    const val = hasOverride ? cameraThresholds[selectedCamForThreshold][label] : defVal;
                    return (
                      <div
                        key={label}
                        className={styles.thresholdCard}
                        style={{ borderColor: hasOverride ? '#14a08c' : 'var(--border)' }}
                      >
                        <div className={styles.thresholdLabelRow}>
                          <span>{label}</span>
                          <span style={{ color: hasOverride ? '#14a08c' : 'var(--text-muted)' }}>
                            {val.toFixed(2)} {hasOverride ? '(Riêng)' : '(Kế thừa)'}
                          </span>
                        </div>
                        <div className={styles.thresholdSliderRow}>
                          <input
                            type="range"
                            min={0.01}
                            max={0.99}
                            step={0.01}
                            value={val}
                            onChange={e => {
                              const num = parseFloat(e.target.value);
                              setCameraThresholds(prev => ({
                                ...prev,
                                [selectedCamForThreshold]: {
                                  ...(prev[selectedCamForThreshold] || {}),
                                  [label]: num,
                                },
                              }));
                            }}
                          />
                          <input
                            type="number"
                            min={0.01}
                            max={0.99}
                            step={0.01}
                            value={val}
                            onChange={e => {
                              const num = parseFloat(e.target.value) || 0.01;
                              setCameraThresholds(prev => ({
                                ...prev,
                                [selectedCamForThreshold]: {
                                  ...(prev[selectedCamForThreshold] || {}),
                                  [label]: Math.max(0.01, Math.min(0.99, num)),
                                },
                              }));
                            }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </div>
        )}

        <button disabled={deploying || !selectedModelIds.length || (!allCameras && !cameraIds.length)} onClick={deploySelected}>Deploy {selectedModelIds.length || ''} model đã chọn</button>
        {orderedDeployments.map(deployment => {
          const defMap = deployment.confidence_thresholds?.default;
          const camMap = deployment.confidence_thresholds?.cameras;
          return (
            <div key={deployment.id} style={{ margin: '8px 0', padding: '6px 0', borderBottom: '1px solid var(--border)' }}>
              <p style={{ margin: 0 }}>Đã lưu: {data?.models.find(model => model.id === deployment.model_id)?.name || deployment.model_id} · {deployment.all_cameras ? 'tất cả camera' : deployment.camera_ids.map(id => cameras.find(camera => camera.id === id)?.name || id).join(', ')} · <strong>{deployment.enabled ? 'đang bật' : 'đã tắt'}</strong> <button disabled={deploying} onClick={() => setDeploymentEnabled(deployment, !deployment.enabled)}>{deployment.enabled ? 'Tắt' : 'Bật'}</button> <button disabled={deploying} onClick={() => removeDeployment(deployment)}>Xóa</button></p>
              {defMap && Object.keys(defMap).length > 0 && (
                <small style={{ display: 'block', color: 'var(--text-muted)', fontSize: '11px', marginTop: 2 }}>
                  Ngưỡng chung: {Object.entries(defMap).map(([k, v]) => `${k}: ${v}`).join(' · ')}
                  {camMap && Object.keys(camMap).length > 0 && (
                    <> | Riêng: {Object.entries(camMap).map(([cId, m]) => {
                      const cName = cameras.find(c => c.id === cId)?.name || cId;
                      return `${cName} (${Object.entries(m).map(([k, v]) => `${k}:${v}`).join(', ')})`;
                    }).join('; ')}</>
                  )}
                </small>
              )}
            </div>
          );
        })}
      </div>
      <div className={styles.runtime}>Custom DeepStream: <strong>{data?.runtime.state || 'idle'}</strong>{data?.runtime.error && <p role="alert">{data.runtime.error}</p>}{Object.entries(data?.runtime.workers || {}).map(([modelId, worker]) => <span key={modelId}>{data?.models.find(model => model.id === modelId)?.name || modelId}: {worker.state || 'idle'} · {worker.model_type || 'detect'}{worker.error && ` · ${worker.error}`}</span>)}</div>
      {!data?.models.length && <p>Chưa upload model. Upload và deploy trước khi bật overlay trên Monitor.</p>}
      {orderedModels.map(model => <article key={model.id}><div className={styles.header}><label><input type="checkbox" disabled={model.state !== 'ready'} checked={selectedModelIds.includes(model.id)} onChange={event => setSelectedModelIds(previous => event.target.checked ? [...previous, model.id] : previous.filter(id => id !== model.id))} /> <strong>{model.name}</strong></label><span data-state={activeModelIds.has(model.id) ? 'live' : model.state}>{activeModelIds.has(model.id) ? 'đang chạy' : model.state}</span></div><small>{model.filename} · {(model.size_bytes / 1024**2).toFixed(1)} MiB · {model.metadata?.shape?.join(' × ')} · {model.metadata?.model_type || model.metadata?.task || 'detect'}</small><p>{model.labels.map((label, index) => `${index}: ${label}`).join(' · ') || 'Labels sẽ được đọc khi upload xong'}</p>{model.error && <details><summary>Lỗi kiểm tra/build</summary><pre>{model.error}</pre></details>}{model.state === 'failed' && <button onClick={() => retry(model)}>Build lại (dùng labels ở ô bên trái nếu nhập)</button>}{model.state === 'ready' && <button disabled={deploying || (!allCameras && !cameraIds.length)} onClick={() => deploy(model)}>{data?.deployments?.some(deployment => deployment.model_id === model.id) ? 'Cập nhật camera & ngưỡng deploy' : 'Deploy lên Monitor'}</button>}</article>)}
    </div></div>{error && <p role="alert" className={styles.error}>{error}</p>}{message && <p role="status" className={styles.message}>{message}</p>}
  </section>;
}
