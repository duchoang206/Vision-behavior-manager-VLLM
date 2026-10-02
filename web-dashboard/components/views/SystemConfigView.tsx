'use client';

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Activity, BookOpen, CheckCircle2, Copy, Cpu, Edit3, Package, Play, Plus, RefreshCw, RotateCw, Search, Send,
  SlidersHorizontal, Trash2, X, XCircle,
} from 'lucide-react';
import {
  COMMAND_KIND_LABELS, CommChannel, CommLog, CommMeta, CommStatus, CommandKind, DEVICE_TYPE_LABELS, EVENT_LABELS,
  PROTOCOL_LABELS, SlotState, TestResult, channelAddress, commApi, emptyChannel, formatAgo, isFmsFormat,
} from '../../lib/comm-gateway';
import CommChannelEditor from './CommChannelEditor';
import styles from './SystemConfigView.module.css';

type Tab = 'devices' | 'slots' | 'logs' | 'fms';
type Toast = { ok: boolean; message: string; detail?: string; id: number };

const POLL_MS = 2000;

function runtimeBadge(channel: CommChannel) {
  const runtime = channel.runtime;
  if (!channel.is_enabled) return { label: 'Tắt', cls: '' };
  const state = runtime?.state || 'stopped';
  if (state === 'listening') {
    return runtime?.clients ? { label: `${runtime.clients} client`, cls: styles.badgeOk } : { label: 'Chờ kết nối', cls: styles.badgeWarn };
  }
  if (state === 'online') return { label: 'Online', cls: styles.badgeOk };
  if (state === 'connecting') return { label: 'Đang kết nối', cls: styles.badgeWarn };
  if (state === 'error' || state === 'offline') return { label: state === 'error' ? 'Lỗi' : 'Offline', cls: styles.badgeErr };
  return { label: state, cls: '' };
}

export default function SystemConfigView({ active = true }: { active?: boolean }) {
  const [tab, setTab] = useState<Tab>('devices');
  const [channels, setChannels] = useState<CommChannel[]>([]);
  const [status, setStatus] = useState<CommStatus | null>(null);
  const [meta, setMeta] = useState<CommMeta | null>(null);
  const [slots, setSlots] = useState<SlotState[]>([]);
  const [logs, setLogs] = useState<CommLog[]>([]);
  const [loadError, setLoadError] = useState('');
  const [search, setSearch] = useState('');
  const [editing, setEditing] = useState<{ channel: CommChannel; isNew: boolean } | null>(null);
  const [runner, setRunner] = useState<CommChannel | null>(null);
  const [toast, setToast] = useState<Toast | null>(null);
  const [busy, setBusy] = useState<string>('');
  const lastSeq = useRef(0);

  const notify = useCallback((ok: boolean, message: string, detail?: string) => setToast({ ok, message, detail, id: Date.now() }), []);
  useEffect(() => {
    if (!toast) return;
    const timer = setTimeout(() => setToast(null), toast.ok ? 4500 : 8000);
    return () => clearTimeout(timer);
  }, [toast]);

  const refresh = useCallback(async () => {
    try {
      const [list, state] = await Promise.all([commApi.channels(), commApi.status()]);
      setChannels(list);
      setStatus(state);
      setLoadError('');
    } catch (error) {
      setLoadError(error instanceof Error ? error.message : String(error));
    }
  }, []);

  useEffect(() => { if (active && !meta) commApi.meta().then(setMeta).catch(() => {}); }, [active, meta]);

  useEffect(() => {
    if (!active) return;
    let timer: ReturnType<typeof setTimeout>;
    let cancelled = false;
    let first = true;
    const tick = async () => {
      // Always load once; afterwards skip polling while the page is hidden.
      if (first || !document.hidden) {
        first = false;
        await refresh();
        if (tab === 'slots') commApi.slots().then(s => { if (!cancelled) setSlots(s); }).catch(() => {});
        if (tab === 'logs') {
          commApi.logs(lastSeq.current).then(items => {
            if (cancelled || !items.length) return;
            lastSeq.current = items[items.length - 1].seq;
            const newestFirst = [...items].reverse();
            setLogs(prev => [...newestFirst, ...prev].slice(0, 300));
          }).catch(() => {});
        }
      }
      if (!cancelled) timer = setTimeout(tick, POLL_MS);
    };
    void tick();
    return () => { cancelled = true; clearTimeout(timer); };
  }, [active, tab, refresh]);

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return channels;
    return channels.filter(c => [c.name, c.id, c.description, c.device_type, c.protocol, c.host, String(c.port)]
      .some(v => (v || '').toLowerCase().includes(q)));
  }, [channels, search]);

  const toggleEnabled = async (channel: CommChannel) => {
    setBusy(channel.id);
    try {
      const saved = await commApi.update(channel.id, { is_enabled: !channel.is_enabled });
      notify(true, `${saved.is_enabled ? 'Đã bật' : 'Đã tắt'} "${saved.name}"`);
      await refresh();
    } catch (error) {
      notify(false, 'Không đổi được trạng thái', error instanceof Error ? error.message : String(error));
    } finally { setBusy(''); }
  };

  const remove = async (channel: CommChannel) => {
    if (!window.confirm(`Xóa thiết bị/kênh "${channel.name}"? Kết nối hiện tại sẽ bị đóng.`)) return;
    try {
      await commApi.remove(channel.id);
      notify(true, `Đã xóa "${channel.name}"`);
      await refresh();
    } catch (error) {
      notify(false, 'Xóa thất bại', error instanceof Error ? error.message : String(error));
    }
  };

  const restart = async (channel: CommChannel) => {
    setBusy(channel.id);
    try { await commApi.restart(channel.id); notify(true, `Đã khởi động lại "${channel.name}"`); await refresh(); }
    catch (error) { notify(false, 'Khởi động lại thất bại', error instanceof Error ? error.message : String(error)); }
    finally { setBusy(''); }
  };

  const quickTest = async (channel: CommChannel) => {
    setBusy(channel.id);
    try {
      const result = await commApi.test({ channel_id: channel.id, slot_id: '5', state: 'Car Full' });
      notify(result.ok, result.ok ? `Gửi thử thành công! ${channel.name}` : `Gửi thử thất bại: ${channel.name}`, result.detail);
    } catch (error) {
      notify(false, 'Gửi thử thất bại', error instanceof Error ? error.message : String(error));
    } finally { setBusy(''); }
  };

  const fms = status?.fms;
  const tabs: { id: Tab; label: string; icon: React.ReactNode; count?: number }[] = [
    { id: 'devices', label: 'Thiết bị & Kênh truyền thông', icon: <Cpu size={15} />, count: channels.length },
    { id: 'slots', label: 'Ô hàng FMS', icon: <Package size={15} />, count: status?.slots.total },
    { id: 'logs', label: 'Nhật ký truyền thông', icon: <Activity size={15} /> },
    { id: 'fms', label: 'Kết nối FMS WCS', icon: <BookOpen size={15} /> },
  ];

  return (
    <section className={styles.shell} aria-label="System Config">
      <header className={styles.header}>
        <div className={styles.title}>
          <div className={styles.titleIcon}><SlidersHorizontal size={21} /></div>
          <div><h1>System Config</h1><p>Cổng truyền thông FMS WCS · PLC · thiết bị ngoại vi — cấu hình áp dụng ngay, không cần khởi động lại</p></div>
        </div>
        <div className={styles.pills}>
          <span className={`${styles.pill} ${fms?.connected ? styles.pillOk : ''}`} title={fms?.channels.map(c => c.endpoint).join(', ')}>
            <span className={`${styles.dot} ${fms?.connected ? styles.dotOk : ''}`} />
            FMS WCS: {fms?.connected ? `Đã kết nối (${fms.clients} client)` : fms?.configured ? 'Chờ kết nối…' : 'Chưa cấu hình'}
          </span>
          <span className={styles.pill}><span className={`${styles.dot} ${status?.online ? styles.dotOk : styles.dotWarn}`} />
            Kênh: {status?.online ?? 0}/{status?.enabled ?? 0} hoạt động</span>
          <span className={styles.pill}>Ô hàng: <b style={{ color: 'var(--rose)' }}>{status?.slots.carfull ?? 0}</b> có hàng · <b style={{ color: 'var(--emerald)' }}>{status?.slots.empty ?? 0}</b> trống</span>
          <button type="button" className={`${styles.btn} ${styles.btnSmall}`} onClick={() => void refresh()}><RefreshCw size={13} /> Làm mới</button>
        </div>
      </header>

      {(loadError || status?.store_error) && <p className={styles.errors} role="alert" style={{ listStyle: 'none', paddingLeft: 12, marginBottom: 12 }}>
        {loadError ? `Không tải được cấu hình: ${loadError}` : status?.store_error}</p>}

      <nav className={styles.tabs} role="tablist" aria-label="System Config">
        {tabs.map(t => <button key={t.id} type="button" role="tab" aria-selected={tab === t.id}
          className={`${styles.tab} ${tab === t.id ? styles.tabActive : ''}`} onClick={() => setTab(t.id)}>
          {t.icon}{t.label}{t.count != null && <span className={styles.count}>{t.count}</span>}</button>)}
      </nav>

      {tab === 'devices' && (
        <div className={styles.card}>
          <div className={styles.toolbar}>
            <button type="button" className={`${styles.btn} ${styles.btnPrimary}`} onClick={() => setEditing({ channel: emptyChannel(), isNew: true })}>
              <Plus size={15} /> Thêm thiết bị</button>
            <label className={styles.search}><Search size={14} />
              <input aria-label="Tìm thiết bị" placeholder="Tìm theo tên, IP, giao thức…" value={search} onChange={e => setSearch(e.target.value)} /></label>
            <span className={styles.grow} />
            <span className={styles.muted} style={{ fontSize: 12 }}>Sự kiện đã định tuyến: {status?.stats.dispatched ?? 0} · lỗi template: {status?.stats.render_errors ?? 0}</span>
          </div>
          <div className={styles.tableWrap}>
            <table className={styles.table}>
              <thead><tr>
                <th>No.</th><th>Tên thiết bị</th><th>Loại</th><th>Giao thức · Mode</th><th>Địa chỉ</th>
                <th>Ưu tiên</th><th>Trạng thái</th><th>Bật</th><th style={{ textAlign: 'center' }}>Thao tác</th>
              </tr></thead>
              <tbody>
                {filtered.map((channel, index) => {
                  const badge = runtimeBadge(channel);
                  const runtime = channel.runtime;
                  return (
                    <tr key={channel.id}>
                      <td className={styles.num}>{index + 1}</td>
                      <td className={styles.nameCell}><strong>{channel.name}</strong>
                        <span className={styles.sub} title={channel.description || channel.id}>{channel.description || channel.id}</span></td>
                      <td>{DEVICE_TYPE_LABELS[channel.device_type] || channel.device_type}</td>
                      <td style={{ whiteSpace: 'nowrap' }}>{PROTOCOL_LABELS[channel.protocol]}
                        <span className={styles.sub}>{channel.mode === 'SERVER' ? 'server' : 'client'}{isFmsFormat(channel) ? ' · FMS slots' : ''}</span></td>
                      <td className={`${styles.mono} ${styles.addressCell}`}>{channelAddress(channel)}
                        {runtime?.embedded && <span className={styles.sub}>chung cổng API</span>}</td>
                      <td className={styles.num}>{channel.priority}</td>
                      <td>
                        <span className={`${styles.badge} ${badge.cls}`}>{badge.label}</span>
                        {runtime?.last_error && channel.is_enabled && badge.cls !== styles.badgeOk &&
                          <span className={`${styles.sub} ${styles.ellipsis}`} style={{ maxWidth: 200 }} title={runtime.last_error}>{runtime.last_error}</span>}
                        {!!runtime?.invalid_slot_ids?.length && <span className={styles.sub} style={{ color: 'var(--rose)' }}>Slot không hợp lệ: {runtime.invalid_slot_ids.join(', ')}</span>}
                      </td>
                      <td><button type="button" role="switch" aria-checked={channel.is_enabled} aria-label={`Bật/tắt ${channel.name}`}
                        disabled={busy === channel.id} className={`${styles.switch} ${channel.is_enabled ? styles.switchOn : ''}`}
                        onClick={() => void toggleEnabled(channel)} /></td>
                      <td><div className={styles.actions}>
                        <button type="button" className={styles.linkBtn} onClick={() => setEditing({ channel, isNew: false })}><Edit3 size={13} /> Sửa</button>
                        <button type="button" className={`${styles.linkBtn} ${styles.linkWarn}`} onClick={() => setRunner(channel)}><Cpu size={13} /> Lệnh</button>
                        <button type="button" className={`${styles.linkBtn} ${styles.linkOk}`} disabled={!channel.is_enabled || busy === channel.id}
                          onClick={() => void quickTest(channel)} title="Gửi gói mẫu slot 5 · Car Full"><Send size={13} /> Gửi thử</button>
                        <button type="button" className={styles.linkBtn} disabled={!channel.is_enabled || busy === channel.id}
                          onClick={() => void restart(channel)} aria-label={`Khởi động lại ${channel.name}`}><RotateCw size={13} /></button>
                        <button type="button" className={`${styles.linkBtn} ${styles.linkDanger}`} onClick={() => void remove(channel)}><Trash2 size={13} /> Xóa</button>
                      </div></td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            {!filtered.length && <p className={styles.empty}>{channels.length ? 'Không có thiết bị khớp bộ lọc.' : 'Chưa có thiết bị nào. Bấm “Thêm thiết bị” để tạo kênh truyền thông đầu tiên.'}</p>}
          </div>
        </div>
      )}

      {tab === 'slots' && <SlotsPanel slots={slots} channels={channels} />}
      {tab === 'logs' && <LogsPanel logs={logs} channels={channels} onClear={() => setLogs([])} />}
      {tab === 'fms' && <FmsGuide channels={channels} status={status} notify={notify} />}

      {editing && <CommChannelEditor initial={editing.channel} isNew={editing.isNew} meta={meta} notify={notify}
        onClose={() => setEditing(null)} onSaved={() => { setEditing(null); void refresh(); }} />}
      {runner && <CommandRunner channel={channels.find(c => c.id === runner.id) || runner} notify={notify} onClose={() => setRunner(null)} />}

      {toast && <div key={toast.id} className={`${styles.toast} ${toast.ok ? styles.toastOk : styles.toastErr}`} role={toast.ok ? 'status' : 'alert'}>
        {toast.ok ? <CheckCircle2 size={18} /> : <XCircle size={18} />}
        <div><p>{toast.message}</p>{toast.detail && <small>{toast.detail}</small>}</div>
        <button type="button" className={`${styles.linkBtn}`} onClick={() => setToast(null)} aria-label="Đóng thông báo"><X size={14} /></button>
      </div>}
    </section>
  );
}

function SlotsPanel({ slots, channels }: { slots: SlotState[]; channels: CommChannel[] }) {
  const name = (id: string) => id ? channels.find(c => c.id === id)?.name || id : 'Tất cả kênh';
  return (
    <div className={styles.card}>
      <div className={styles.toolbar}><span className={styles.hint}>Trạng thái ô hàng do AI xác nhận (2 frame → Car Full, 5 frame trống → Empty). Gán Slot ID tại Building → Phân tích Hành vi → Ô chứa hàng.</span></div>
      <div className={styles.tableWrap}>
        <table className={styles.table}>
          <thead><tr><th>Slot ID</th><th>ROI</th><th>Camera</th><th>Trạng thái</th><th>Đối tượng</th><th>Kênh</th><th>Đồng bộ FMS</th><th>Cập nhật</th></tr></thead>
          <tbody>
            {slots.map(slot => (
              <tr key={`${slot.cam_id}:${slot.rule_id}`}>
                <td className={styles.mono}><strong>{slot.slot_id || '—'}</strong>
                  {!slot.slot_id && <span className={styles.sub}>chưa gán (template dùng {slot.effective_slot_id})</span>}
                  {slot.slot_id && !slot.fms_compatible && <span className={styles.sub} style={{ color: 'var(--rose)' }}>FMS cần số nguyên</span>}</td>
                <td>{slot.rule_name}</td>
                <td className={styles.mono}>{slot.cam_id}</td>
                <td><span className={`${styles.badge} ${slot.status === 'CARFULL' ? styles.badgeErr : slot.status === 'EMPTY' ? styles.badgeOk : ''}`}>
                  {slot.status === 'CARFULL' ? 'Car Full' : slot.status === 'EMPTY' ? 'Empty' : 'Chưa xác định'}</span></td>
                <td>{slot.occupant_label || '—'}{slot.status === 'CARFULL' && slot.overlap_ratio > 0 && <span className={styles.sub}>{slot.overlap_ratio.toFixed(1)}%</span>}</td>
                <td>{name(slot.channel_id)}</td>
                <td>{slot.enabled ? <span className={`${styles.badge} ${styles.badgeInfo}`}>Bật</span> : <span className={styles.badge}>Tắt</span>}</td>
                <td className={styles.muted}>{formatAgo(slot.updated_at)}<span className={styles.sub}>{slot.transitions} lần chuyển</span></td>
              </tr>
            ))}
          </tbody>
        </table>
        {!slots.length && <p className={styles.empty}>Chưa có ROI ô chứa hàng nào. Vẽ ROI loại “Ô chứa hàng” và gán Slot ID trong tab Building.</p>}
      </div>
    </div>
  );
}

function LogsPanel({ logs, channels, onClear }: { logs: CommLog[]; channels: CommChannel[]; onClear: () => void }) {
  const [channel, setChannel] = useState('');
  const [onlyErrors, setOnlyErrors] = useState(false);
  const visible = logs.filter(l => (!channel || l.channel_id === channel) && (!onlyErrors || !l.ok));
  return (
    <div className={styles.card}>
      <div className={styles.toolbar}>
        <select className={styles.input} style={{ width: 240 }} aria-label="Lọc theo kênh" value={channel} onChange={e => setChannel(e.target.value)}>
          <option value="">Tất cả kênh</option>{channels.map(c => <option key={c.id} value={c.id}>{c.name}</option>)}</select>
        <label className={styles.check}><input type="checkbox" checked={onlyErrors} onChange={e => setOnlyErrors(e.target.checked)} /> Chỉ lỗi</label>
        <span className={styles.grow} />
        <span className={styles.muted} style={{ fontSize: 12 }}>Tự cập nhật mỗi 2 giây · heartbeat thành công được ẩn</span>
        <button type="button" className={`${styles.btn} ${styles.btnSmall}`} onClick={onClear}>Xóa màn hình</button>
      </div>
      <div role="log" aria-label="Nhật ký truyền thông" style={{ maxHeight: '65vh', overflow: 'auto' }}>
        {visible.map(log => (
          <div className={styles.logLine} key={log.seq}>
            <span className={styles.muted}>{new Date(log.at * 1000).toLocaleTimeString('vi-VN')}</span>
            <span className={styles.mono}>{log.channel_id}</span>
            <span><span className={`${styles.badge} ${log.ok ? styles.badgeOk : styles.badgeErr}`} title={EVENT_LABELS[log.event || '']}>{log.event || '—'}</span></span>
            <span className={styles.mono}>{log.slot_id ? `slot ${log.slot_id}` : log.command || ''}</span>
            <span>{log.detail}</span>
            <span className={`${styles.mono} ${styles.muted}`}>{log.e2e_latency_ms != null ? `${log.e2e_latency_ms.toFixed(1)} ms` : log.latency_ms != null ? `${log.latency_ms.toFixed(1)} ms` : ''}</span>
            {log.payload && <pre>{log.payload}</pre>}
          </div>
        ))}
        {!visible.length && <p className={styles.empty}>Chưa có bản ghi. Sự kiện ô hàng, gửi thử, kết nối client sẽ xuất hiện tại đây.</p>}
      </div>
    </div>
  );
}

function FmsGuide({ channels, status, notify }: { channels: CommChannel[]; status: CommStatus | null; notify: (ok: boolean, m: string, d?: string) => void }) {
  const [host] = useState(() => (typeof window !== 'undefined' ? window.location.hostname : '0.0.0.0'));
  const fmsChannels = channels.filter(isFmsFormat);
  const copy = async (text: string) => {
    try { await navigator.clipboard.writeText(text); notify(true, 'Đã sao chép', text); }
    catch { notify(false, 'Không sao chép được', text); }
  };
  return (
    <div className={styles.card}>
      <div className={styles.guide}>
        <div>
          <h3>Khai báo trên FMS (Building → Call Box → Add)</h3>
          {fmsChannels.length === 0 && <p className={styles.hint}>Chưa có kênh định dạng “FMS WCS (slots)”. Thêm thiết bị loại <b>FMS WCS (CAMERA_AI)</b> ở tab Thiết bị.</p>}
          {fmsChannels.map(channel => {
            const config = JSON.stringify({ path: channel.endpoint_path || '/' });
            const runtime = status?.fms.channels.find(c => c.id === channel.id);
            return (
              <div key={channel.id} style={{ marginBottom: 16 }}>
                <p className={styles.inline} style={{ margin: '0 0 8px' }}><strong>{channel.name}</strong>
                  <span className={`${styles.badge} ${runtime?.clients ? styles.badgeOk : styles.badgeWarn}`}>
                    {runtime?.clients ? `Đã kết nối (${runtime.clients} client)` : channel.is_enabled ? 'Chờ FMS kết nối…' : 'Kênh đang tắt'}</span></p>
                <div className={styles.kv}>
                  <div>Device Name</div><div>CAMERA_AI_VISION</div>
                  <div>Device Type</div><div><code>CAMERA_AI</code></div>
                  <div>Protocol</div><div><code>WebSocket</code> (không chọn TCP/IP)</div>
                  <div>Mode</div><div><code>client</code></div>
                  <div>Device Ip</div><div className={styles.copy}><code>{host}</code><button type="button" className={styles.linkBtn} onClick={() => void copy(host)} aria-label="Sao chép IP"><Copy size={12} /></button></div>
                  <div>Device Port</div><div className={styles.copy}><code>{channel.port}</code><button type="button" className={styles.linkBtn} onClick={() => void copy(String(channel.port))} aria-label="Sao chép cổng"><Copy size={12} /></button></div>
                  <div>Device Configuration</div><div className={styles.copy}><code>{config}</code><button type="button" className={styles.linkBtn} onClick={() => void copy(config)} aria-label="Sao chép cấu hình"><Copy size={12} /></button></div>
                  <div>URL FMS sẽ mở</div><div><code>ws://{host}:{channel.port}{channel.endpoint_path || '/'}</code></div>
                </div>
                {runtime?.peers?.length ? <p className={styles.hint} style={{ marginTop: 6 }}>Client: {runtime.peers.map(p => `${p.peer}${p.path ? ` (${p.path})` : ''}`).join(', ')}</p> : null}
              </div>
            );
          })}
        </div>
        <div>
          <h3>Luồng dữ liệu</h3>
          <ol className={styles.steps}>
            <li>FMS WCS (<code>WsCameraClient</code>) chủ động kết nối WebSocket tới Vision Manager và tự kết nối lại mỗi 3 giây khi rớt mạng.</li>
            <li>Ngay khi kết nối, Vision Manager gửi snapshot toàn bộ ô: <code>{'{"slots":[{"slot_id":"2","state":"Car Full"},{"slot_id":"5","state":"Empty"}]}'}</code>.</li>
            <li>Khi AI xác nhận đổi trạng thái, gửi ngay (&lt; 50 ms): <code>{'{"slot_id":"5","state":"Car Full","slots":[…]}'}</code>; heartbeat gửi lại snapshot theo chu kỳ cấu hình.</li>
            <li>FMS lưu gói mới nhất vào Redis <code>CameraStateWCS</code>, chặn giao hàng vào ô Car Full và kích hoạt task (CALL_CARD) theo cấu hình <b>Config</b> của thiết bị CAMERA_AI cho từng Slot ID.</li>
          </ol>
          <p className={styles.hint} style={{ marginTop: 10 }}>
            Slot ID phải là <b>số nguyên</b> và trùng với số ô cấu hình trong Config của thiết bị CAMERA_AI trên FMS. Trạng thái hợp lệ: <code>Car Full</code> (có hàng) — mọi giá trị khác được FMS hiểu là trống (gửi <code>Empty</code>).
          </p>
          <h3 style={{ marginTop: 16 }}>Kiểm tra nhanh</h3>
          <p className={styles.hint}>Giả lập FMS: <code>wscat -c ws://{host}:{fmsChannels[0]?.port ?? 8000}{fmsChannels[0]?.endpoint_path ?? '/ws/wcs_camera'}</code> — badge phía trên chuyển sang “Đã kết nối (1 client)” và nhận snapshot ngay lập tức.</p>
        </div>
      </div>
    </div>
  );
}

function CommandRunner({ channel, notify, onClose }: { channel: CommChannel; notify: (ok: boolean, m: string, d?: string) => void; onClose: () => void }) {
  const [results, setResults] = useState<(TestResult & { at: number; name: string })[]>([]);
  const [running, setRunning] = useState('');
  const [op, setOp] = useState<{ kind: CommandKind; address: number; value: string; count: number }>({ kind: 'modbus_read_coils', address: 0, value: '1', count: 8 });
  const isModbus = channel.protocol === 'MODBUS_TCP';
  const commands = channel.config.commands || [];
  const runtime = channel.runtime;

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => { if (event.key === 'Escape') onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  const record = (name: string, result: TestResult) => {
    setResults(prev => [{ ...result, name, at: Date.now() }, ...prev].slice(0, 20));
    notify(result.ok, result.ok ? `${name}: thành công` : `${name}: thất bại`, result.detail);
  };
  const run = async (commandId: string, name: string) => {
    setRunning(commandId);
    try { record(name, await commApi.command(channel.id, commandId)); }
    catch (error) { record(name, { ok: false, detail: error instanceof Error ? error.message : String(error) }); }
    finally { setRunning(''); }
  };
  const runModbus = async () => {
    setRunning('__modbus');
    const value = op.kind === 'modbus_coil' ? op.value === '1' : Number(op.value);
    try { record(COMMAND_KIND_LABELS[op.kind], await commApi.modbus(channel.id, { kind: op.kind, address: op.address, value, count: op.count })); }
    catch (error) { record(COMMAND_KIND_LABELS[op.kind], { ok: false, detail: error instanceof Error ? error.message : String(error) }); }
    finally { setRunning(''); }
  };

  return (
    <div className={styles.backdrop} onMouseDown={event => { if (event.target === event.currentTarget) onClose(); }}>
      <div className={`${styles.modal} ${styles.modalNarrow}`} role="dialog" aria-modal="true" aria-labelledby="runner-title">
        <div className={styles.modalHeader}>
          <h2 id="runner-title">Điều khiển · {channel.name}</h2>
          <button type="button" className={`${styles.btn} ${styles.btnGhost} ${styles.btnSmall}`} onClick={onClose} aria-label="Đóng"><X size={16} /></button>
        </div>
        <div className={styles.modalBody}>
          <p className={styles.hint}><code>{channelAddress(channel)}</code> · {runtimeBadge(channel).label}
            {runtime?.last_error ? ` · lỗi gần nhất: ${runtime.last_error}` : ''}</p>
          {!channel.is_enabled && <p className={styles.hint}>Thiết bị đang tắt — lệnh vẫn được gửi qua kết nối tạm (chế độ client).</p>}

          <section className={styles.section}>
            <div className={styles.sectionHead}><span><Play size={14} /> Lệnh đã cấu hình</span></div>
            <div className={styles.sectionBody}>
              {!commands.length && <p className={styles.hint}>Chưa có lệnh. Vào “Sửa” → “Lệnh điều khiển thiết bị ngoại vi” để thêm.</p>}
              <div className={styles.chips}>
                {commands.map(c => <button type="button" key={c.id} className={styles.btn} disabled={!!running} onClick={() => void run(c.id, c.name)}>
                  <Play size={13} /> {running === c.id ? 'Đang chạy…' : c.name}
                  <span className={styles.muted} style={{ fontWeight: 400 }}>{c.kind === 'send' ? '' : `@${c.address}`}</span></button>)}
              </div>
              {Object.entries(channel.config.event_commands || {}).filter(([, ids]) => ids.length).map(([event, ids]) => (
                <p className={styles.hint} key={event}>Tự động: <b>{EVENT_LABELS[event] || event}</b> → {ids.map(id => commands.find(c => c.id === id)?.name || id).join(', ')}</p>
              ))}
            </div>
          </section>

          {isModbus && (
            <section className={styles.section}>
              <div className={styles.sectionHead}><span><Cpu size={14} /> Thao tác Modbus trực tiếp</span></div>
              <div className={styles.sectionBody}>
                <div className={styles.inline} style={{ flexWrap: 'wrap', alignItems: 'flex-end' }}>
                  <label className={styles.field} style={{ width: 230 }}><span>Function</span>
                    <select value={op.kind} onChange={e => setOp({ ...op, kind: e.target.value as CommandKind })}>
                      {(['modbus_read_coils', 'modbus_read_registers', 'modbus_coil', 'modbus_register'] as CommandKind[]).map(k => <option key={k} value={k}>{COMMAND_KIND_LABELS[k]}</option>)}</select></label>
                  <label className={styles.field} style={{ width: 110 }}><span>Địa chỉ</span>
                    <input type="number" min={0} max={65535} value={op.address} onChange={e => setOp({ ...op, address: Number(e.target.value) })} /></label>
                  {op.kind === 'modbus_coil' && <label className={styles.field} style={{ width: 100 }}><span>Giá trị</span>
                    <select value={op.value} onChange={e => setOp({ ...op, value: e.target.value })}><option value="1">ON</option><option value="0">OFF</option></select></label>}
                  {op.kind === 'modbus_register' && <label className={styles.field} style={{ width: 120 }}><span>Giá trị</span>
                    <input type="number" min={0} max={65535} value={op.value} onChange={e => setOp({ ...op, value: e.target.value })} /></label>}
                  {(op.kind === 'modbus_read_coils' || op.kind === 'modbus_read_registers') && <label className={styles.field} style={{ width: 100 }}><span>Số lượng</span>
                    <input type="number" min={1} max={125} value={op.count} onChange={e => setOp({ ...op, count: Number(e.target.value) })} /></label>}
                  <button type="button" className={`${styles.btn} ${styles.btnPrimary}`} disabled={!!running} onClick={() => void runModbus()}><Play size={13} /> Thực thi</button>
                </div>
                {runtime?.poll?.values && <p className={styles.hint}>Poll ({formatAgo(runtime.poll.at)}): <code>{runtime.poll.kind}@{runtime.poll.address} = {JSON.stringify(runtime.poll.values)}</code></p>}
              </div>
            </section>
          )}

          {!!runtime?.inbound?.length && (
            <section className={styles.section}>
              <div className={styles.sectionHead}><span><Activity size={14} /> Dữ liệu nhận từ thiết bị</span></div>
              <div className={styles.sectionBody}>{runtime.inbound.slice().reverse().map((m, i) =>
                <p className={styles.hint} key={i}>{new Date(m.at * 1000).toLocaleTimeString('vi-VN')} · {m.peer}: <code>{m.text}</code></p>)}</div>
            </section>
          )}

          {results.map(r => (
            <div key={r.at} className={`${styles.result} ${r.ok ? styles.resultOk : styles.resultErr}`}>
              {r.ok ? <CheckCircle2 size={16} /> : <XCircle size={16} />}
              <div><strong>{r.name}</strong> · {r.detail}{r.latency_ms != null && <span className={styles.muted}> · {r.latency_ms.toFixed(1)} ms</span>}
                {r.response != null && <pre>{JSON.stringify(r.response)}</pre>}</div>
            </div>
          ))}
        </div>
        <div className={styles.modalFooter}><span /><button type="button" className={styles.btn} onClick={onClose}><X size={14} /> Đóng</button></div>
      </div>
    </div>
  );
}
