'use client';

import { useEffect, useMemo, useState } from 'react';
import { Download, Pause, Play, RefreshCw, Search, Terminal } from 'lucide-react';
import { useCameras } from '../CameraContext';
import styles from './SystemLogsView.module.css';

type Entry = { id: number; created_at: string; level: string; source: string; event: string; message: string; camera_id?: string; request_id?: string; status_code?: number; duration_ms?: number; details: Record<string, unknown> };
type Status = { queue: number; dropped: number; retention_days: number; archive_days: number; database_error?: string; archive_error?: string };
export default function SystemLogsView() {
  const { cameras } = useCameras();
  const [entries, setEntries] = useState<Entry[]>([]);
  const [channel, setChannel] = useState('system');
  const [level, setLevel] = useState('');
  const [source, setSource] = useState('');
  const [search, setSearch] = useState('');
  const [query, setQuery] = useState('');
  const [camera, setCamera] = useState('');
  const [since, setSince] = useState('');
  const [until, setUntil] = useState('');
  const [live, setLive] = useState(true);
  const [before, setBefore] = useState<number>();
  const [more, setMore] = useState(false);
  const [revision, setRevision] = useState(0);
  const [error, setError] = useState('');
  const [status, setStatus] = useState<Status>();
  const [archives, setArchives] = useState<{ name: string; size_bytes: number }[]>([]);
  const [selected, setSelected] = useState<Entry>();
  useEffect(() => { const timer = setTimeout(() => { setQuery(search); setBefore(undefined); }, 300); return () => clearTimeout(timer); }, [search]);
  const params = useMemo(() => {
    const values = new URLSearchParams({ channel, limit: '100' });
    if (level) values.set('level', level);
    if (source) values.set('source', source);
    if (query) values.set('search', query);
    if (camera) values.set('camera_id', camera);
    if (since) values.set('since', new Date(since).toISOString());
    if (until) values.set('until', new Date(until).toISOString());
    if (before) values.set('before', String(before));
    return values.toString();
  }, [channel, level, source, query, camera, since, until, before]);
  useEffect(() => {
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    const load = async () => {
      if (controller.signal.aborted) return;
      if (document.hidden) {
        if (live && !before) timer = setTimeout(load, 2000);
        return;
      }
      try {
        const response = await fetch(`/api/backend/system-logs?${params}`, { signal: controller.signal, cache: 'no-store' });
        if (!response.ok) throw new Error(`Không đọc được log (HTTP ${response.status}).`);
        const result = await response.json();
        if (!controller.signal.aborted) { setEntries(result.logs); setMore(result.has_more); setError(''); }
      } catch (reason) { if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : String(reason)); }
      if (!controller.signal.aborted && live && !before) timer = setTimeout(load, 2000);
    };
    void load();
    return () => { controller.abort(); clearTimeout(timer); };
  }, [params, live, before, revision]);
  useEffect(() => {
    const controller = new AbortController();
    fetch('/api/backend/system-logs/status', { signal: controller.signal }).then(response => response.ok ? response.json() : null).then(value => { if (value) setStatus(value); }).catch(() => {});
    fetch('/api/backend/system-logs/archives', { signal: controller.signal }).then(response => response.ok ? response.json() : null).then(value => { if (value) setArchives(value.archives); }).catch(() => {});
    return () => controller.abort();
  }, [revision]);
  return <section className={styles.shell} aria-label="System Logs">
    <header className={styles.header}><div className={styles.title}><Terminal size={25} /><div><h1>System Logs</h1><p>Nhật ký có cấu trúc · PostgreSQL / SSD · Nén lossless</p></div></div>
      <div className={styles.actions}><button onClick={() => { setBefore(undefined); setRevision(value => value + 1); }}><RefreshCw size={15} />Làm mới</button><a href={`/api/backend/system-logs?${params}&download=true`}><Download size={15} />Tải trang lọc .gz</a><button className={live ? styles.live : ''} onClick={() => { setLive(!live); setBefore(undefined); }}>{live ? <Pause size={14} /> : <Play size={14} />}Auto: {live ? 'ON' : 'OFF'}</button></div>
    </header>
    <div className={styles.filters}>
      <select aria-label="Kênh nhật ký" value={channel} onChange={event => { setChannel(event.target.value); setBefore(undefined); setSelected(undefined); }}><option value="system">Gateway / System</option><option value="workflow">Workflow / MLOps audit</option><option value="audit">Thao tác lưu trữ</option></select>
      <select aria-label="Mức log" value={level} onChange={event => { setLevel(event.target.value); setBefore(undefined); }}><option value="">ALL LEVELS</option>{['info', 'warning', 'error', 'critical'].map(value => <option key={value}>{value}</option>)}</select>
      <select aria-label="Camera log" value={camera} onChange={event => { setCamera(event.target.value); setBefore(undefined); }}><option value="">Tất cả camera</option>{cameras.map(item => <option key={item.id} value={item.id}>{item.name}</option>)}</select>
      <input aria-label="Nguồn log" placeholder="Nguồn (gateway, FMSBridge…)" value={source} onChange={event => { setSource(event.target.value); setBefore(undefined); }} />
      <label className={styles.search}><Search size={15} /><input aria-label="Tìm kiếm log" placeholder="Tìm message, event…" value={search} onChange={event => setSearch(event.target.value)} /></label>
    </div>
    <div className={styles.filters}><label>Từ <input aria-label="Log từ ngày" type="datetime-local" value={since} onChange={event => { setSince(event.target.value); setBefore(undefined); }} /></label><label>Đến <input aria-label="Log đến ngày" type="datetime-local" value={until} onChange={event => { setUntil(event.target.value); setBefore(undefined); }} /></label>
      <select aria-label="Tải lưu trữ theo ngày" value="" onChange={event => { if (event.target.value) window.location.assign(`/api/backend/system-logs/archives/${encodeURIComponent(event.target.value)}`); }}><option value="">Tải log hệ thống theo ngày…</option>{archives.map(item => <option value={item.name} key={item.name}>{item.name} · {(item.size_bytes / 1024).toFixed(1)} KB</option>)}</select>
    </div>
    {error && <p role="alert" className={styles.error}>{error}</p>}
    {(status?.database_error || status?.archive_error || Boolean(status?.dropped)) && <p role="alert" className={styles.error}>Lưu log: DB {status?.database_error || 'OK'} · SSD {status?.archive_error || 'OK'} · Bỏ qua do quá tải: {status?.dropped}</p>}
    <div className={styles.workspace}><div className={styles.console} role="log" aria-label="Kết quả nhật ký">
      {!entries.length && <p className={styles.empty}>Không có bản ghi phù hợp. Log mới xuất hiện sau thao tác trên backend.</p>}
      {entries.map(entry => <button className={`${styles.line} ${selected?.id === entry.id ? styles.selected : ''}`} key={`${channel}:${entry.id}`} onClick={() => setSelected(entry)}>
        <time>{new Date(entry.created_at).toLocaleString('vi-VN')}</time><span className={styles.level} data-level={entry.level}>{entry.level}</span><span className={styles.message}>[{entry.source}] {entry.message}</span><small>{entry.status_code || ''}{entry.duration_ms != null ? ` · ${entry.duration_ms.toFixed(1)}ms` : ''}</small>
      </button>)}
    </div>{selected && <aside className={styles.details}><button onClick={() => setSelected(undefined)}>Đóng chi tiết</button><h3>{selected.event}</h3><p>Request: {selected.request_id || '—'}</p><p>Camera: {selected.camera_id || '—'}</p><pre>{JSON.stringify(selected.details, null, 2)}</pre></aside>}</div>
    <footer className={styles.footer}><span>{entries.length} bản ghi · queue {status?.queue ?? '—'} · DB {status?.retention_days ?? 7} ngày / SSD {status?.archive_days ?? 30} ngày</span><div className={styles.actions}><button onClick={() => { setBefore(undefined); setRevision(value => value + 1); }}>Mới nhất</button><button disabled={!more || !entries.length} onClick={() => { setLive(false); setBefore(entries.at(-1)?.id); }}>Cũ hơn</button></div></footer>
  </section>;
}
