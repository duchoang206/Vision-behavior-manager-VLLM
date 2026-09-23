'use client';

import { useEffect, useState } from 'react';
import { Activity, Camera as CameraIcon, Crosshair, ExternalLink, Pencil, Plus, RefreshCw, Search, ShieldCheck, Trash2 } from 'lucide-react';
import { Camera, useCameras } from '../CameraContext';
import { useTab } from '../TabContext';
import styles from './CameraSources.module.css';

type Props = { active: boolean; onEdit: (camera: Camera) => void; onDelete: (id: string) => void; onRules: (id: string) => void; onAdd: (name: string, url: string) => void };
export default function CameraSources({ active, onEdit, onDelete, onRules, onAdd }: Props) {
  const { cameras, fetchCameras } = useCameras();
  const { setActiveTab } = useTab();
  const [query, setQuery] = useState('');
  const [filter, setFilter] = useState('all');
  const [selectedId, setSelectedId] = useState('');
  const [adding, setAdding] = useState(false);
  const [name, setName] = useState('');
  const [url, setUrl] = useState('');
  const [preview, setPreview] = useState('');
  const [previewError, setPreviewError] = useState('');
  const [revision, setRevision] = useState(0);
  const [rules, setRules] = useState<{ id: string; name: string; rule_type?: string; type?: string }[]>([]);
  const [model, setModel] = useState('');
  const selected = cameras.find(camera => camera.id === selectedId);
  const redacted = (value: string) => value.replace(/(\w+:\/\/)[^/@\s]+@/g, '$1••••@').replace(/([?&](?:key|api_key|token|password)=)[^&\s]+/gi, '$1••••');
  const online = cameras.filter(camera => camera.status === 'online').length;
  const rows = cameras.filter(camera => `${camera.name} ${camera.id}`.toLocaleLowerCase().includes(query.toLocaleLowerCase()) && (filter === 'all' || camera.status === filter));
  useEffect(() => {
    if (!selected || !active) return;
    const controller = new AbortController();
    const resetTimer = window.setTimeout(() => { setPreview(''); setPreviewError(''); setRules([]); setModel(''); }, 0);
    fetch(`/api/backend/camera/${selected.id}/rules`, { signal: controller.signal }).then(response => response.ok ? response.json() : null).then(result => { if (result) setRules(result.rules || []); }).catch(() => {});
    const load = async () => {
      try {
        const response = await fetch('/api/backend/models', { signal: controller.signal });
        if (!response.ok) throw new Error('Không đọc được runtime AI.');
        const result = await response.json();
        const deployed = result.runtime?.cameras?.[selected.id];
        if (!deployed || !result.runtime?.model_id) { setPreviewError('Nguồn chưa deploy AI. Mở Monitor để xem video; không khởi tạo thêm bộ giải mã CPU.'); return; }
        const modelId = result.runtime.model_id;
        setModel(result.models.find((item: { id: string }) => item.id === modelId)?.name || modelId);
        const snapshot = await fetch(`/api/backend/models/${modelId}/labels/snapshot`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ camera_id: selected.id }), signal: controller.signal });
        const data = await snapshot.json();
        if (!snapshot.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Chưa có frame GPU.');
        if (!controller.signal.aborted) setPreview(`data:image/jpeg;base64,${data.image}`);
      } catch (reason) { if (!controller.signal.aborted) setPreviewError(reason instanceof Error ? reason.message : String(reason)); }
    };
    void load();
    return () => { controller.abort(); window.clearTimeout(resetTimer); };
  }, [selected?.id, active, revision]);
  return <section className={styles.shell} aria-label="Camera Management">
    <header className={styles.header}><div><h1>Camera Management</h1><p>Quản lý nguồn, calibration và pipeline AI tại một nơi.</p></div><div className={styles.actions}><button className={styles.primary} onClick={() => setAdding(!adding)}><Plus size={15} />Add Source</button><label className={styles.search}><Search size={14} /><input aria-label="Tìm camera" placeholder="Tìm nguồn…" value={query} onChange={event => setQuery(event.target.value)} /></label><select aria-label="Lọc trạng thái camera" value={filter} onChange={event => setFilter(event.target.value)}><option value="all">Tất cả trạng thái</option><option value="online">Online</option><option value="offline">Offline</option></select><button aria-label="Làm mới nguồn" onClick={() => void fetchCameras()}><RefreshCw size={14} /></button></div></header>
    {adding && <form className={styles.addForm} onSubmit={event => { event.preventDefault(); onAdd(name.trim(), url.trim()); }}><label>Tên nguồn<input required value={name} onChange={event => setName(event.target.value)} placeholder="Cam kho A" /></label><label>IP / RTSP URL<input required value={url} onChange={event => setUrl(event.target.value)} placeholder="192.168… hoặc rtsp://…" /></label><button className={styles.primary} type="submit">Tiếp tục xác thực</button></form>}
    <div className={styles.metrics}><article><CameraIcon /><div><span>Total Assets</span><strong>{cameras.length}</strong></div></article><article><ShieldCheck /><div><span>Nguồn online</span><strong>{online}/{cameras.length}</strong></div></article><article><Crosshair /><div><span>Đã hiệu chuẩn</span><strong>{cameras.filter(camera => camera.calibration?.matrix).length}</strong></div></article></div>
    <div className={styles.layout}><div className={styles.tableWrap}><table><thead><tr><th>CHANNEL</th><th>SOURCE NAME</th><th>TYPE</th><th>STATUS</th><th>ACTIONS</th></tr></thead><tbody>{rows.map(camera => <tr key={camera.id} className={selectedId === camera.id ? styles.selected : ''} onClick={() => setSelectedId(camera.id)}>
      <td><span className={styles.channel}>CH_{camera.id.slice(0, 6)}</span></td><td><button className={styles.source} onClick={() => setSelectedId(camera.id)}><span className={styles.sourceIcon}><CameraIcon size={19} /></span><span><strong>{camera.name}</strong><small>{redacted(camera.rtsp_url)}</small></span></button></td><td>RTSP</td><td><span className={camera.status === 'online' ? styles.online : styles.offline}>● {camera.status === 'online' ? 'Running' : 'Offline'}</span></td><td><div className={styles.actions}><button title="Chỉnh sửa nguồn" onClick={event => { event.stopPropagation(); onEdit(camera); }}><Pencil size={14} /></button><button title="Xem Monitor" onClick={event => { event.stopPropagation(); setActiveTab('monitor'); }}><ExternalLink size={14} /></button><button title="Xóa nguồn" onClick={event => { event.stopPropagation(); onDelete(camera.id); }}><Trash2 size={14} /></button></div></td>
    </tr>)}</tbody></table>{!rows.length && <p className={styles.empty}>Chưa có nguồn phù hợp. Chọn Add Source để đăng ký.</p>}</div>
      {selected && <aside className={styles.details}><header><strong>Source Details</strong><button onClick={() => setSelectedId('')}>Đóng</button></header><div className={styles.preview}>{preview ? <img src={preview} alt={`Frame GPU ${selected.name}`} /> : <CameraIcon size={32} />}</div>{previewError && <p className={styles.note}>{previewError}</p>}<button disabled={!active} onClick={() => setRevision(value => value + 1)}><RefreshCw size={13} />Lấy một frame mới nhất</button>
        <label>SOURCE NAME<strong>{selected.name}</strong></label><label>RESOURCE URI<code>{redacted(selected.rtsp_url)}</code></label><label>INDEX<strong>{selected.id}</strong></label><hr /><label>ROI CONFIGURATION <span>{rules.length} quy tắc</span></label>{rules.map(rule => <article className={styles.rule} key={rule.id}><strong>{rule.name}</strong><small>{rule.type || rule.rule_type}</small></article>)}<button onClick={() => onRules(selected.id)}>Vẽ / chỉnh ROI</button><hr /><label>3D CALIBRATION<strong>{selected.calibration?.matrix ? 'ĐÃ HIỆU CHỈNH' : 'CHƯA HIỆU CHỈNH'}</strong></label><button onClick={() => setActiveTab('calibration')}><Crosshair size={14} />Mở Calibration</button><hr /><label>AI INFERENCE<strong>{model || 'Chưa deploy model'}</strong></label><p className={styles.note}><Activity size={12} /> Chọn model, chức năng và hiển thị trong Workflow Editor.</p><button onClick={() => setActiveTab('workflow')}>Cấu hình pipeline</button>
      </aside>}
    </div>
  </section>;
}
