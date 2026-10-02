'use client';

import React, { useEffect, useMemo, useRef, useState } from 'react';
import { CheckCircle2, Cpu, FileJson, Globe, KeyRound, Plus, Save, Send, Settings2, Trash2, X, XCircle, Zap } from 'lucide-react';
import {
  COMMAND_KIND_LABELS, CommApiError, CommChannel, CommCommand, CommMeta, CommandKind, DEVICE_TYPE_LABELS, EVENT_LABELS,
  PROTOCOL_LABELS, Protocol, TestResult, commApi, fmsPreview, isClientOnly, isFmsFormat, isFmsSlotId, renderTemplate, sampleContext,
} from '../../lib/comm-gateway';
import styles from './SystemConfigView.module.css';

const DEFAULT_PORTS: Record<Protocol, number> = { WEBSOCKET: 9100, TCPIP: 9000, MQTT: 1883, HTTP_WEBHOOK: 80, MODBUS_TCP: 502 };
const DEFAULT_PATHS: Partial<Record<Protocol, string>> = { WEBSOCKET: '/ws/wcs_camera', MQTT: 'vision/slots', HTTP_WEBHOOK: '/api/slots' };

type Props = {
  initial: CommChannel;
  isNew: boolean;
  meta: CommMeta | null;
  onClose: () => void;
  onSaved: (channel: CommChannel) => void;
  notify: (ok: boolean, message: string, detail?: string) => void;
};

const Switch = ({ on, onChange, label }: { on: boolean; onChange: (value: boolean) => void; label: string }) => (
  <button type="button" role="switch" aria-checked={on} aria-label={label} title={label}
    className={`${styles.switch} ${on ? styles.switchOn : ''}`} onClick={() => onChange(!on)} />
);

export default function CommChannelEditor({ initial, isNew, meta, onClose, onSaved, notify }: Props) {
  const [draft, setDraft] = useState<CommChannel>(() => structuredClone(initial));
  const [errors, setErrors] = useState<string[]>([]);
  const [saving, setSaving] = useState(false);
  const [testing, setTesting] = useState(false);
  const [testSlot, setTestSlot] = useState('5');
  const [testState, setTestState] = useState('Car Full');
  const [testCommand, setTestCommand] = useState('');
  const [testResult, setTestResult] = useState<TestResult | null>(null);
  const templateRef = useRef<HTMLTextAreaElement | null>(null);
  const appPort = meta?.app_port ?? 8000;

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => { if (event.key === 'Escape') onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  const set = <K extends keyof CommChannel>(key: K, value: CommChannel[K]) => setDraft(prev => ({ ...prev, [key]: value }));
  const setConfig = <K extends keyof CommChannel['config']>(key: K, value: CommChannel['config'][K]) =>
    setDraft(prev => ({ ...prev, config: { ...prev.config, [key]: value } }));

  const changeProtocol = (protocol: Protocol) => setDraft(prev => ({
    ...prev,
    protocol,
    mode: isClientOnly(protocol) ? 'CLIENT' : prev.mode,
    port: DEFAULT_PORTS[protocol],
    endpoint_path: DEFAULT_PATHS[protocol] ?? '',
    config: { ...prev.config, payload_format: protocol === 'WEBSOCKET' || protocol === 'TCPIP' ? prev.config.payload_format : 'template' },
  }));

  const changeDeviceType = (deviceType: string) => setDraft(prev => deviceType === 'FMS_WCS' ? {
    ...prev, device_type: deviceType, protocol: 'WEBSOCKET', mode: 'SERVER', host: '0.0.0.0', port: appPort,
    endpoint_path: '/ws/wcs_camera', trigger_events: ['SLOT_CARFULL', 'SLOT_EMPTY'],
    config: { ...prev.config, payload_format: 'fms_wcs_slots', payload_type: 'json', heartbeat_sec: 2, resync_on_connect: true },
  } : deviceType === 'LIGHT_TOWER' || deviceType === 'BOXSENSOR' ? {
    ...prev, device_type: deviceType, protocol: 'MODBUS_TCP', mode: 'CLIENT', port: 502, endpoint_path: '',
  } : { ...prev, device_type: deviceType });

  const usesFms = isFmsFormat(draft);
  const isModbus = draft.protocol === 'MODBUS_TCP';
  const isServer = draft.mode === 'SERVER';
  const context = useMemo(() => sampleContext(testSlot, testState), [testSlot, testState]);
  const preview = useMemo(() => usesFms ? fmsPreview(testSlot, testState)
    : renderTemplate(draft.payload_template, context, draft.config.payload_type), [usesFms, testSlot, testState, draft.payload_template, draft.config.payload_type, context]);

  const insertVariable = (name: string) => {
    const element = templateRef.current;
    const token = `{${name}}`;
    const text = draft.payload_template;
    const start = element?.selectionStart ?? text.length;
    const end = element?.selectionEnd ?? text.length;
    set('payload_template', text.slice(0, start) + token + text.slice(end));
    requestAnimationFrame(() => { element?.focus(); element?.setSelectionRange(start + token.length, start + token.length); });
  };

  const commands = draft.config.commands;
  const setCommands = (next: CommCommand[]) => setDraft(prev => {
    const ids = new Set(next.map(c => c.id));
    const mapping = Object.fromEntries(Object.entries(prev.config.event_commands).map(([event, list]) => [event, list.filter(id => ids.has(id))]));
    return { ...prev, config: { ...prev.config, commands: next, event_commands: mapping } };
  });
  const updateCommand = (index: number, patch: Partial<CommCommand>) => setCommands(commands.map((c, i) => i === index ? { ...c, ...patch } : c));
  const addCommand = () => {
    const kind: CommandKind = isModbus ? 'modbus_coil' : 'send';
    let n = commands.length + 1;
    while (commands.some(c => c.id === `cmd_${n}`)) n += 1;
    setCommands([...commands, { id: `cmd_${n}`, name: `Lệnh ${n}`, kind, ...(kind === 'send' ? { payload: '{"cmd": "ON", "slot": "{slot_id}"}', payload_type: 'json' } : { address: 0, value: true }) }]);
  };
  const toggleMapping = (event: string, commandId: string) => setDraft(prev => {
    const current = prev.config.event_commands[event] || [];
    const next = current.includes(commandId) ? current.filter(id => id !== commandId) : [...current, commandId];
    return { ...prev, config: { ...prev.config, event_commands: { ...prev.config.event_commands, [event]: next } } };
  });

  const save = async () => {
    setSaving(true);
    setErrors([]);
    try {
      const saved = isNew ? await commApi.create(draft) : await commApi.update(initial.id, draft);
      notify(true, `Đã lưu "${saved.name}"`, saved.is_enabled ? `Kênh đang chạy: ${saved.runtime?.target || ''}` : 'Kênh đang tắt');
      onSaved(saved);
    } catch (error) {
      if (error instanceof CommApiError) setErrors(error.errors.length ? error.errors : [error.message]);
      else setErrors([error instanceof Error ? error.message : String(error)]);
    } finally {
      setSaving(false);
    }
  };

  const runTest = async () => {
    setTesting(true);
    setTestResult(null);
    try {
      const result = await commApi.test({
        channel_id: isNew ? undefined : initial.id, slot_id: testSlot, state: testState,
        command_id: testCommand || undefined, channel: draft,
      });
      setTestResult(result);
      notify(result.ok, result.ok ? 'Gửi thử thành công!' : 'Gửi thử thất bại', result.detail);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setTestResult({ ok: false, detail: message });
      notify(false, 'Gửi thử thất bại', message);
    } finally {
      setTesting(false);
    }
  };

  const events = meta?.events ?? ['SLOT_CARFULL', 'SLOT_EMPTY', 'ROI_ALERT'];
  const variables = meta?.variables ?? [];
  const pathLabel = draft.protocol === 'MQTT' ? 'Topic' : draft.protocol === 'HTTP_WEBHOOK' ? 'URL path (hoặc URL đầy đủ)' : 'Endpoint path';
  const showAuth = !isServer && (draft.protocol === 'HTTP_WEBHOOK' || draft.protocol === 'WEBSOCKET' || draft.protocol === 'MQTT');

  return (
    <div className={styles.backdrop} onMouseDown={event => { if (event.target === event.currentTarget) onClose(); }}>
      <div className={styles.modal} role="dialog" aria-modal="true" aria-labelledby="comm-editor-title">
        <div className={styles.modalHeader}>
          <h2 id="comm-editor-title">{isNew ? 'Thêm thiết bị / kênh truyền thông' : `Sửa thiết bị · ${initial.name}`}</h2>
          <button type="button" className={`${styles.btn} ${styles.btnGhost} ${styles.btnSmall}`} onClick={onClose} aria-label="Đóng"><X size={16} /></button>
        </div>

        <div className={styles.modalBody}>
          {errors.length > 0 && <ul className={styles.errors} role="alert">{errors.map(e => <li key={e}>{e}</li>)}</ul>}

          <div className={styles.grid3}>
            <label className={styles.field}><span>Tên thiết bị *</span>
              <input value={draft.name} onChange={e => set('name', e.target.value)} placeholder="VD: PLC Line 1" autoFocus /></label>
            <label className={styles.field}><span>Mô tả</span>
              <input value={draft.description} onChange={e => set('description', e.target.value)} placeholder="Ghi chú vị trí, chức năng…" /></label>
            <label className={styles.field}><span>Loại thiết bị</span>
              <select value={draft.device_type} onChange={e => changeDeviceType(e.target.value)}>
                {(meta?.device_types ?? Object.keys(DEVICE_TYPE_LABELS)).map(t => <option key={t} value={t}>{DEVICE_TYPE_LABELS[t] || t}</option>)}
              </select></label>
            <label className={styles.field}><span>Giao thức</span>
              <select value={draft.protocol} onChange={e => changeProtocol(e.target.value as Protocol)}>
                {(meta?.protocols ?? (Object.keys(PROTOCOL_LABELS) as Protocol[])).map(p => <option key={p} value={p}>{PROTOCOL_LABELS[p]}</option>)}
              </select></label>
            <label className={styles.field}><span>Mức ưu tiên</span>
              <input type="number" min={0} value={draft.priority} onChange={e => set('priority', Number(e.target.value))} /></label>
            <div className={styles.field}><span>Bật thiết bị</span>
              <div className={styles.inline}><Switch on={draft.is_enabled} onChange={v => set('is_enabled', v)} label="Bật thiết bị" />
                <span className={styles.muted}>{draft.is_enabled ? 'Đang bật' : 'Đang tắt'}</span></div></div>
          </div>

          <section className={styles.section}>
            <div className={styles.sectionHead}><span><Globe size={14} /> Kết nối</span>
              {isNew && <label className={styles.inline} style={{ fontWeight: 500 }}>Mã kênh
                <input className={styles.input} style={{ width: 170 }} value={draft.id} onChange={e => set('id', e.target.value.toLowerCase())} placeholder="tự sinh từ tên" /></label>}
              {!isNew && <span className={`${styles.mono} ${styles.muted}`}>id: {draft.id}</span>}
            </div>
            <div className={styles.sectionBody}>
              <div className={styles.grid3}>
                <label className={styles.field}><span>Chế độ</span>
                  <select value={draft.mode} disabled={isClientOnly(draft.protocol)} onChange={e => set('mode', e.target.value as CommChannel['mode'])}>
                    <option value="SERVER">Server (thiết bị kết nối vào đây)</option>
                    <option value="CLIENT">Client (kết nối tới thiết bị)</option>
                  </select></label>
                <label className={styles.field}><span>{isServer ? 'IP lắng nghe' : 'IP / Host thiết bị'}</span>
                  <input value={draft.host} onChange={e => set('host', e.target.value)} placeholder={isServer ? '0.0.0.0' : '192.168.1.188'} /></label>
                <label className={styles.field}><span>Cổng</span>
                  <input type="number" min={1} max={65535} value={draft.port} onChange={e => set('port', Number(e.target.value))} />
                  {isServer && draft.protocol === 'WEBSOCKET' && draft.port === appPort && <small>Dùng chung cổng API {appPort} (route FastAPI)</small>}</label>
              </div>
              {(draft.protocol === 'WEBSOCKET' || draft.protocol === 'MQTT' || draft.protocol === 'HTTP_WEBHOOK') && (
                <div className={styles.grid3}>
                  <label className={styles.field} style={{ gridColumn: 'span 2' }}><span>{pathLabel}</span>
                    <input className={styles.mono} value={draft.endpoint_path} onChange={e => set('endpoint_path', e.target.value)} />
                    {isServer && draft.protocol === 'WEBSOCKET' && draft.port !== appPort && <small>Cổng riêng chấp nhận mọi path (FMS mặc định path “/”) trừ khi bật “Khớp path chính xác”.</small>}</label>
                  {draft.protocol === 'HTTP_WEBHOOK' && <label className={styles.field}><span>HTTP method</span>
                    <select value={draft.config.http_method} onChange={e => setConfig('http_method', e.target.value)}>{['POST', 'PUT', 'PATCH'].map(m => <option key={m}>{m}</option>)}</select></label>}
                  {draft.protocol === 'MQTT' && <label className={styles.field}><span>QoS</span>
                    <select value={draft.config.mqtt_qos} onChange={e => setConfig('mqtt_qos', Number(e.target.value))}>{[0, 1, 2].map(q => <option key={q} value={q}>QoS {q}</option>)}</select></label>}
                  {isServer && draft.protocol === 'WEBSOCKET' && draft.port !== appPort && <label className={styles.check}>
                    <input type="checkbox" checked={draft.config.strict_path} onChange={e => setConfig('strict_path', e.target.checked)} /> Khớp path chính xác</label>}
                </div>
              )}
              <div className={styles.chips}>
                {!isServer && <label className={styles.check}><input type="number" className={styles.input} style={{ width: 70, padding: '4px 6px' }} min={0.2} step={0.5}
                  value={draft.config.connect_timeout_sec} onChange={e => setConfig('connect_timeout_sec', Number(e.target.value))} /> giây connect timeout</label>}
                {draft.protocol === 'HTTP_WEBHOOK' && <label className={styles.check}><input type="number" className={styles.input} style={{ width: 70, padding: '4px 6px' }} min={0.2} step={0.5}
                  value={draft.config.timeout_sec} onChange={e => setConfig('timeout_sec', Number(e.target.value))} /> giây request timeout</label>}
                {!isServer && draft.protocol !== 'TCPIP' && !isModbus && <label className={styles.check}>
                  <input type="checkbox" checked={draft.config.use_tls} onChange={e => setConfig('use_tls', e.target.checked)} /> TLS (wss/https/mqtts)</label>}
                {draft.protocol === 'MQTT' && <label className={styles.check}>
                  <input type="checkbox" checked={draft.config.mqtt_retain} onChange={e => setConfig('mqtt_retain', e.target.checked)} /> Retain</label>}
                {draft.protocol === 'TCPIP' && <>
                  <label className={styles.check}>Kết thúc gói
                    <select className={styles.input} style={{ width: 110, padding: '4px 6px' }} value={draft.config.line_terminator}
                      onChange={e => setConfig('line_terminator', e.target.value as CommChannel['config']['line_terminator'])}>
                      <option value="lf">\n (LF)</option><option value="crlf">\r\n (CRLF)</option><option value="cr">\r (CR)</option><option value="none">Không</option>
                    </select></label>
                  <label className={styles.check}><input type="checkbox" checked={draft.config.hex_payload} onChange={e => setConfig('hex_payload', e.target.checked)} /> Payload dạng HEX</label>
                </>}
                {isModbus && <label className={styles.check}>Unit ID
                  <input type="number" className={styles.input} style={{ width: 70, padding: '4px 6px' }} min={0} max={255}
                    value={draft.config.unit_id} onChange={e => setConfig('unit_id', Number(e.target.value))} /></label>}
              </div>
              {showAuth && <div className={styles.grid3}>
                <label className={styles.field}><span><KeyRound size={12} /> Bearer token</span>
                  <input value={draft.auth_config.bearer_token || ''} onChange={e => set('auth_config', { ...draft.auth_config, bearer_token: e.target.value })} /></label>
                <label className={styles.field}><span>Username</span>
                  <input value={draft.auth_config.username || ''} autoComplete="off" onChange={e => set('auth_config', { ...draft.auth_config, username: e.target.value })} /></label>
                <label className={styles.field}><span>Password</span>
                  <input type="password" autoComplete="new-password" value={draft.auth_config.password || ''} onChange={e => set('auth_config', { ...draft.auth_config, password: e.target.value })} /></label>
              </div>}
              {isModbus && <div className={styles.grid3}>
                <label className={styles.check}><input type="checkbox" checked={!!draft.config.poll?.enabled}
                  onChange={e => setConfig('poll', { kind: 'coils', address: 0, count: 8, interval_sec: 1, ...draft.config.poll, enabled: e.target.checked })} /> Đọc trạng thái định kỳ (poll)</label>
                {draft.config.poll?.enabled && <>
                  <label className={styles.field}><span>Vùng đọc</span>
                    <select value={draft.config.poll.kind || 'coils'} onChange={e => setConfig('poll', { ...draft.config.poll, kind: e.target.value as 'coils' | 'registers' })}>
                      <option value="coils">Coils (FC01)</option><option value="registers">Holding registers (FC03)</option></select></label>
                  <label className={styles.field}><span>Địa chỉ · số lượng · chu kỳ (s)</span>
                    <div className={styles.inline}>
                      <input className={styles.input} type="number" min={0} value={draft.config.poll.address ?? 0} onChange={e => setConfig('poll', { ...draft.config.poll, address: Number(e.target.value) })} />
                      <input className={styles.input} type="number" min={1} max={125} value={draft.config.poll.count ?? 8} onChange={e => setConfig('poll', { ...draft.config.poll, count: Number(e.target.value) })} />
                      <input className={styles.input} type="number" min={0.2} step={0.2} value={draft.config.poll.interval_sec ?? 1} onChange={e => setConfig('poll', { ...draft.config.poll, interval_sec: Number(e.target.value) })} />
                    </div></label>
                </>}
              </div>}
            </div>
          </section>

          {!isModbus && (
            <section className={styles.section}>
              <div className={styles.sectionHead}><span><FileJson size={14} /> Dữ liệu gửi đi</span>
                <div className={styles.chips}>
                  {(draft.protocol === 'WEBSOCKET' || draft.protocol === 'TCPIP') && <>
                    <button type="button" className={`${styles.chipToggle} ${usesFms ? styles.chipToggleOn : ''}`} onClick={() => setConfig('payload_format', 'fms_wcs_slots')}>FMS WCS (slots)</button>
                    <button type="button" className={`${styles.chipToggle} ${!usesFms ? styles.chipToggleOn : ''}`} onClick={() => setConfig('payload_format', 'template')}>Template tùy biến</button>
                  </>}
                </div>
              </div>
              <div className={styles.sectionBody}>
                <div className={styles.inline} style={{ flexWrap: 'wrap' }}>
                  <span className={styles.fieldLabel}>Sự kiện kích hoạt:</span>
                  {events.filter(e => !usesFms || e !== 'ROI_ALERT').map(event => {
                    const on = draft.trigger_events.includes(event);
                    return <button type="button" key={event} className={`${styles.chipToggle} ${on ? styles.chipToggleOn : ''}`} aria-pressed={on}
                      onClick={() => set('trigger_events', on ? draft.trigger_events.filter(e => e !== event) : [...draft.trigger_events, event])}>{EVENT_LABELS[event] || event}</button>;
                  })}
                </div>
                {usesFms ? (
                  <>
                    <p className={styles.hint}>
                      Định dạng gốc mà FMS WCS (<code>WsCameraClient</code>) đọc: mỗi gói chứa toàn bộ trạng thái <code>{'{"slots":[{"slot_id","state"}]}'}</code>,
                      kèm <code>slot_id</code>/<code>state</code> của ô vừa đổi. FMS chỉ hiểu <code>state = &quot;Car Full&quot;</code> là có hàng và đọc <code>slot_id</code> bằng <code>stoi()</code> nên Slot ID phải là số nguyên.
                    </p>
                    <div className={styles.grid3}>
                      <label className={styles.field}><span>Heartbeat gửi lại snapshot (giây, 0 = tắt)</span>
                        <input type="number" min={0} step={0.5} value={draft.config.heartbeat_sec} onChange={e => setConfig('heartbeat_sec', Number(e.target.value))} /></label>
                      <label className={styles.check}><input type="checkbox" checked={draft.config.resync_on_connect} onChange={e => setConfig('resync_on_connect', e.target.checked)} /> Resync toàn bộ ô khi FMS vừa kết nối</label>
                    </div>
                  </>
                ) : (
                  <>
                    <div className={styles.inline} style={{ flexWrap: 'wrap' }}>
                      <label className={styles.inline}>Kiểu
                        <select className={styles.input} style={{ width: 100 }} value={draft.config.payload_type} onChange={e => setConfig('payload_type', e.target.value as 'json' | 'text')}>
                          <option value="json">JSON</option><option value="text">Text</option></select></label>
                      <label className={styles.inline}>Mẫu có sẵn
                        <select className={styles.input} style={{ width: 210 }} value="" onChange={e => {
                          const preset = meta?.presets.find(p => p.id === e.target.value);
                          if (preset) setDraft(prev => ({ ...prev, payload_template: preset.template, config: { ...prev.config, payload_type: preset.payload_type } }));
                        }}><option value="">Chọn preset…</option>{meta?.presets.map(p => <option key={p.id} value={p.id}>{p.name}</option>)}</select></label>
                      {isServer && <label className={styles.check}><input type="checkbox" checked={draft.config.resync_on_connect} onChange={e => setConfig('resync_on_connect', e.target.checked)} /> Gửi lại các ô đang có hàng khi client kết nối</label>}
                    </div>
                    <div className={styles.chips} aria-label="Chèn biến">
                      {variables.map(v => <button type="button" key={v.name} className={styles.chip} title={`${v.description} · ví dụ: ${JSON.stringify(v.example)}`}
                        aria-label={`Chèn biến {${v.name}}: ${v.description}`}
                        onClick={() => insertVariable(v.name)}>+ {`{${v.name}}`}</button>)}
                    </div>
                    <div className={styles.editorGrid}>
                      <label className={styles.field}><span>Payload template</span>
                        <textarea ref={templateRef} spellCheck={false} className={`${styles.code} ${preview.ok ? '' : styles.codeInvalid}`}
                          value={draft.payload_template} onChange={e => set('payload_template', e.target.value)} aria-invalid={!preview.ok} /></label>
                      <div className={styles.field}><span>Live preview {preview.ok
                        ? <span className={`${styles.badge} ${styles.badgeOk}`}><CheckCircle2 size={11} /> hợp lệ</span>
                        : <span className={`${styles.badge} ${styles.badgeErr}`}><XCircle size={11} /> lỗi cú pháp</span>}</span>
                        <pre className={`${styles.preview} ${preview.ok ? '' : styles.previewError}`} aria-live="polite">{preview.ok ? preview.pretty : preview.error}</pre>
                      </div>
                    </div>
                  </>
                )}
              </div>
            </section>
          )}

          <section className={styles.section}>
            <div className={styles.sectionHead}><span><Cpu size={14} /> Lệnh điều khiển thiết bị ngoại vi</span>
              <button type="button" className={`${styles.btn} ${styles.btnSmall}`} onClick={addCommand}><Plus size={13} /> Thêm lệnh</button></div>
            <div className={styles.sectionBody}>
              {commands.length === 0 && <p className={styles.hint}>
                {isModbus ? 'Thêm lệnh ghi coil/thanh ghi (VD: bật đèn đỏ coil 3) rồi gán vào sự kiện bên dưới để tự động điều khiển.'
                  : 'Tùy chọn: định nghĩa lệnh gửi payload riêng (VD: mở cửa, gọi thang) để chạy tay hoặc gán theo sự kiện.'}</p>}
              {commands.map((command, index) => (
                <div className={styles.cmdRow} key={index}>
                  <label className={styles.field}><span>Tên lệnh</span>
                    <input value={command.name} onChange={e => updateCommand(index, { name: e.target.value })} /></label>
                  <label className={styles.field}><span>Loại</span>
                    <select value={command.kind} onChange={e => {
                      const kind = e.target.value as CommandKind;
                      updateCommand(index, kind === 'send' ? { kind, payload: command.payload || 'ON', payload_type: draft.config.payload_type }
                        : { kind, address: command.address ?? 0, value: kind === 'modbus_coil' ? true : kind === 'modbus_register' ? 0 : undefined, count: 1 });
                    }}>
                      {(Object.keys(COMMAND_KIND_LABELS) as CommandKind[]).filter(k => isModbus ? k !== 'send' : k === 'send').map(k => <option key={k} value={k}>{COMMAND_KIND_LABELS[k]}</option>)}
                    </select></label>
                  {command.kind === 'send' ? (
                    <label className={styles.field}><span>Payload ({command.payload_type || 'json'}) · hỗ trợ biến {'{slot_id}'}…</span>
                      <input className={styles.mono} value={command.payload || ''} onChange={e => updateCommand(index, { payload: e.target.value })} /></label>
                  ) : (
                    <div className={styles.field}><span>Địa chỉ · {command.kind === 'modbus_coil' ? 'Giá trị' : command.kind === 'modbus_register' ? 'Giá trị (0-65535)' : 'Số lượng'}</span>
                      <div className={styles.inline}>
                        <input className={styles.input} type="number" min={0} max={65535} value={command.address ?? 0} onChange={e => updateCommand(index, { address: Number(e.target.value) })} aria-label="Địa chỉ" />
                        {command.kind === 'modbus_coil' && <select className={styles.input} value={command.value ? '1' : '0'} onChange={e => updateCommand(index, { value: e.target.value === '1' })} aria-label="Giá trị coil">
                          <option value="1">ON</option><option value="0">OFF</option></select>}
                        {command.kind === 'modbus_register' && <input className={styles.input} type="number" min={0} max={65535} value={Number(command.value ?? 0)} onChange={e => updateCommand(index, { value: Number(e.target.value) })} aria-label="Giá trị thanh ghi" />}
                        {(command.kind === 'modbus_read_coils' || command.kind === 'modbus_read_registers') && <input className={styles.input} type="number" min={1} max={125} value={command.count ?? 1} onChange={e => updateCommand(index, { count: Number(e.target.value) })} aria-label="Số lượng" />}
                      </div></div>
                  )}
                  <button type="button" className={`${styles.btn} ${styles.btnDanger} ${styles.btnSmall}`} aria-label={`Xóa lệnh ${command.name}`}
                    onClick={() => setCommands(commands.filter((_, i) => i !== index))}><Trash2 size={13} /></button>
                </div>
              ))}
              {commands.length > 0 && (
                <div className={styles.field}><span><Zap size={12} /> Tự động chạy lệnh theo sự kiện AI</span>
                  {events.map(event => (
                    <div className={styles.mapRow} key={event}>
                      <span>{EVENT_LABELS[event] || event}</span>
                      <div className={styles.chips}>
                        {commands.map(c => {
                          const on = (draft.config.event_commands[event] || []).includes(c.id);
                          return <button type="button" key={c.id} aria-pressed={on} className={`${styles.chipToggle} ${on ? styles.chipToggleOn : ''}`}
                            onClick={() => toggleMapping(event, c.id)}>{c.name}</button>;
                        })}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </section>

          <section className={styles.section}>
            <div className={styles.sectionHead}><span><Send size={14} /> Gửi thử nghiệm (Send Test)</span></div>
            <div className={styles.sectionBody}>
              <div className={styles.inline} style={{ flexWrap: 'wrap', alignItems: 'flex-end' }}>
                {!isModbus && <>
                  <label className={styles.field} style={{ width: 120 }}><span>Slot ID</span>
                    <input value={testSlot} onChange={e => setTestSlot(e.target.value)} aria-invalid={usesFms && !isFmsSlotId(testSlot)} /></label>
                  <label className={styles.field} style={{ width: 150 }}><span>Trạng thái</span>
                    <select value={testState} onChange={e => setTestState(e.target.value)}><option>Car Full</option><option>Empty</option></select></label>
                </>}
                {commands.length > 0 && <label className={styles.field} style={{ width: 220 }}><span>Gửi</span>
                  <select value={isModbus && !testCommand ? commands[0].id : testCommand} onChange={e => setTestCommand(e.target.value)}>
                    {!isModbus && <option value="">Payload sự kiện (template)</option>}
                    {commands.map(c => <option key={c.id} value={c.id}>Lệnh: {c.name}</option>)}</select></label>}
                <button type="button" className={`${styles.btn} ${styles.btnPrimary}`} onClick={runTest} disabled={testing || (!isModbus && !testCommand && !preview.ok)}>
                  <Send size={14} /> {testing ? 'Đang gửi…' : 'Gửi thử nghiệm'}</button>
              </div>
              {usesFms && <p className={styles.hint}>⚠ FMS xử lý gói thử như dữ liệu thật (có thể tạo task CALL_CARD cho ô). Heartbeat kế tiếp sẽ khôi phục trạng thái thực của các ô đã cấu hình.</p>}
              {testResult && (
                <div className={`${styles.result} ${testResult.ok ? styles.resultOk : styles.resultErr}`} role="status">
                  {testResult.ok ? <CheckCircle2 size={16} /> : <XCircle size={16} />}
                  <div>
                    <strong>{testResult.ok ? 'Gửi thử thành công!' : 'Gửi thử thất bại'}</strong> · {testResult.detail}
                    {testResult.latency_ms != null && <span className={styles.muted}> · {testResult.latency_ms.toFixed(1)} ms</span>}
                    {testResult.payload && <pre>{testResult.payload}</pre>}
                    {testResult.response != null && <pre>Phản hồi: {JSON.stringify(testResult.response)}</pre>}
                  </div>
                </div>
              )}
            </div>
          </section>
        </div>

        <div className={styles.modalFooter}>
          <span className={styles.hint}><Settings2 size={12} /> Lưu là áp dụng ngay — không cần khởi động lại backend.</span>
          <div className={styles.inline}>
            <button type="button" className={styles.btn} onClick={onClose}><X size={14} /> Thoát</button>
            <button type="button" className={`${styles.btn} ${styles.btnPrimary}`} onClick={save} disabled={saving || !draft.name.trim()}>
              <Save size={14} /> {saving ? 'Đang lưu…' : 'Lưu'}</button>
          </div>
        </div>
      </div>
    </div>
  );
}
