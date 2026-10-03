'use client';

// Inspector panel of an "inspection" rule in Building view > Rules: does the
// cell hold goods - the mode and the classes that count as goods - the
// empty-cell baselines (one per lighting condition, captured live or taken from
// the recorded video), what the Monitor shows, the metrology settings of the
// plan (kept), and the Test / Save actions.  InspectionResultPanel shows a test
// run next to the camera canvas.

import React, { useEffect, useRef, useState } from 'react';
import { Camera as CameraIcon, Film, FlaskConical, Save, Trash2 } from 'lucide-react';
import type { ColorPalette } from '../ThemeContext';
import type { CommChannel } from '../../lib/comm-gateway';
import {
  INSPECTION_MODES, INSPECTION_POINT_COUNT, INSPECTION_PRESETS, InspectionApiError, applyPreset, articleFitsCell,
  baselineMatchesRoi, formatMoment, fromDatetimeLocal, inspectionApi, occupancyColor, occupancyText, toDatetimeLocal,
  verdictSummary,
  type BaselineEntry, type BaselineList, type CaptureOptions, type InspectionConfig, type InspectionPreset, type InspectionReport,
} from '../../lib/roi-rules';

type Styles = { input: React.CSSProperties; select: React.CSSProperties; label: React.CSSProperties };

type Props = {
  camId: string;
  ruleId: string;
  points: number[][];
  config: InspectionConfig;
  onConfigChange: (config: InspectionConfig) => void;
  slotId: string;
  onSlotIdChange: (value: string) => void;
  channelId: string;
  onChannelIdChange: (value: string) => void;
  channels: CommChannel[];
  fmsDispatch: boolean;
  onFmsDispatchChange: (value: boolean) => void;
  availableClasses: string[];
  targetClasses: string[];
  onTargetClassesChange: (classes: string[]) => void;
  onResult: (report: InspectionReport | null, clientLatencyMs?: number) => void;
  onSave: () => void;
  saveDisabled: boolean;
  colors: ColorPalette;
  styles: Styles;
};

export const statusColor = (C: ColorPalette, status?: string) =>
  status === 'OK' ? C.emerald : status === 'NG' ? C.rose : status === 'DETECTED' ? C.cyanL : C.amber;

const SOURCE_LABEL: Record<string, string> = { CV: 'Xử lý ảnh', AI: 'AI', 'CV+AI': 'Xử lý ảnh + AI', NONE: '—' };

function Section({ index, title, children, C }: { index: number; title: string; children: React.ReactNode; C: ColorPalette }) {
  return (
    <section style={{ display: 'flex', flexDirection: 'column', gap: '8px', paddingTop: '10px', borderTop: `1px solid ${C.border}` }}>
      <h5 style={{ margin: 0, fontSize: '11px', letterSpacing: '0.06em', color: C.accentL, fontFamily: 'JetBrains Mono, monospace' }}>
        {index}. {title}
      </h5>
      {children}
    </section>
  );
}

export default function InspectionInspector(props: Props) {
  const { camId, ruleId, points, config, onConfigChange, colors: C, styles } = props;
  const [baselines, setBaselines] = useState<BaselineEntry[]>([]);
  const [baselineMax, setBaselineMax] = useState(12);
  const [coverage, setCoverage] = useState<{ from: number; to: number } | null>(null);
  const [thumbs, setThumbs] = useState<Record<string, string>>({});
  const thumbsRef = useRef<Record<string, string>>({});
  const [lightLabel, setLightLabel] = useState('');
  const [moment, setMoment] = useState('');
  const [busy, setBusy] = useState<'capture' | 'record' | 'delete' | 'test' | null>(null);
  const [notice, setNotice] = useState<{ ok: boolean; text: string } | null>(null);
  const [forceCapture, setForceCapture] = useState<CaptureOptions | null>(null);
  const ready = points.length === INSPECTION_POINT_COUNT;
  const cell: [number, number] = [config.roi_width_mm, config.roi_height_mm];
  const usableCount = baselines.filter(b => baselineMatchesRoi(b, points, cell)).length;
  const usesAi = config.mode !== 'CV_ONLY';
  const usesCv = config.mode !== 'AI_ONLY';

  const set = <K extends keyof InspectionConfig>(key: K, value: InspectionConfig[K]) =>
    onConfigChange({ ...config, [key]: value });
  const setNumber = (key: keyof InspectionConfig) => (event: React.ChangeEvent<HTMLInputElement>) => {
    const value = Number(event.target.value);
    if (Number.isFinite(value)) onConfigChange({ ...config, [key]: value, ...(key === 'width_mm' || key === 'height_mm' ? { target_preset: 'custom' } : {}) });
  };

  const applyBaselines = (list: BaselineList) => {
    setBaselines(list.baselines);
    setBaselineMax(list.max);
    setCoverage(list.recordings);
  };

  const loadBaselines = async () => {
    try {
      applyBaselines(await inspectionApi.baselines(camId, ruleId));
    } catch {
      // keep the last list; the capture / delete actions report their own errors
    }
  };

  useEffect(() => {
    let alive = true;
    if (!camId || !ruleId) return;
    inspectionApi.baselines(camId, ruleId)
      .then(list => { if (alive) applyBaselines(list); })
      .catch(() => {});
    return () => { alive = false; };
  }, [camId, ruleId]);

  useEffect(() => {
    let alive = true;
    const missing = baselines.filter(b => !thumbsRef.current[b.id]);
    Promise.all(missing.map(async b => [b.id, await inspectionApi.baselineImage(camId, ruleId, b.id).catch(() => null)] as const))
      .then(results => {
        if (!alive) {
          results.forEach(([, url]) => { if (url) URL.revokeObjectURL(url); });
          return;
        }
        const next = { ...thumbsRef.current };
        results.forEach(([id, url]) => { if (url) next[id] = url; });
        Object.keys(next).forEach(id => {
          if (!baselines.some(b => b.id === id)) { URL.revokeObjectURL(next[id]); delete next[id]; }
        });
        thumbsRef.current = next;
        setThumbs(next);
      });
    return () => { alive = false; };
  }, [baselines, camId, ruleId]);

  useEffect(() => () => { Object.values(thumbsRef.current).forEach(url => URL.revokeObjectURL(url)); }, []);

  useEffect(() => {
    const has = usableCount > 0;
    if (config.has_background_baseline !== has) onConfigChange({ ...config, has_background_baseline: has });
  }, [usableCount]); // eslint-disable-line react-hooks/exhaustive-deps

  const capture = async (options: CaptureOptions) => {
    setBusy(options.at ? 'record' : 'capture'); setNotice(null); setForceCapture(null);
    try {
      const result = await inspectionApi.captureBackground(camId, ruleId, points, cell,
        { label: lightLabel.trim(), targets: props.targetClasses, ...options });
      await loadBaselines();
      setLightLabel('');
      const parts = [`✅ Đã thêm ảnh nền ${result.label ? `"${result.label}" ` : ''}(${result.count}/${baselineMax})`];
      if (result.replaced) parts.push(`đã thay ${result.replaced} ảnh nền của ROI / kích thước ô cũ`);
      setNotice({ ok: !result.warning, text: parts.join(' · ') + (result.warning ? ` – ⚠️ ${result.warning}` : '') });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      if (error instanceof InspectionApiError && error.status === 409 && !options.at && !options.force && message.startsWith('AI đang thấy')) {
        setForceCapture(options);
      }
      setNotice({ ok: false, text: `Không thêm được ảnh nền: ${message}` });
    } finally { setBusy(null); }
  };

  const captureFromRecording = () => {
    const at = fromDatetimeLocal(moment);
    if (at === null) {
      setNotice({ ok: false, text: 'Chọn ngày giờ ô trống trong video ghi hình' });
      return;
    }
    void capture({ at });
  };

  const removeBaseline = async (entry: BaselineEntry) => {
    if (!window.confirm(`Xoá ảnh nền "${entry.label || formatMoment(entry.captured_at)}"?`)) return;
    setBusy('delete'); setNotice(null);
    try {
      await inspectionApi.deleteBaseline(camId, ruleId, entry.id);
      await loadBaselines();
    } catch (error) {
      setNotice({ ok: false, text: `Không xoá được ảnh nền: ${error instanceof Error ? error.message : String(error)}` });
    } finally { setBusy(null); }
  };

  const runTest = async () => {
    setBusy('test'); setNotice(null);
    const started = performance.now();
    try {
      const report = await inspectionApi.test(camId, points, config, ruleId, props.targetClasses);
      props.onResult(report, performance.now() - started);
    } catch (error) {
      props.onResult(null);
      setNotice({ ok: false, text: `Test thất bại: ${error instanceof Error ? error.message : String(error)}` });
    } finally { setBusy(null); }
  };

  const toggleClass = (name: string) => props.onTargetClassesChange(
    props.targetClasses.includes(name) ? props.targetClasses.filter(c => c !== name) : [...props.targetClasses, name]);
  // Selected classes the deployed models no longer list stay visible, so they can be unselected.
  const classChips = [...props.availableClasses, ...props.targetClasses.filter(c => !props.availableClasses.includes(c))];

  const numberField = (id: string, label: string, key: keyof InspectionConfig, step = 1, suffix = 'mm') => (
    <label htmlFor={id} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '8px', fontSize: '12px', color: C.textSub }}>
      <span>{label}</span>
      <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
        <input id={id} data-testid={id} type="number" step={step} value={config[key] as number} onChange={setNumber(key)}
          style={{ ...styles.input, width: '84px', padding: '6px 8px', textAlign: 'right' }} />
        <span style={{ width: '18px', color: C.textMuted }}>{suffix}</span>
      </span>
    </label>
  );

  const dimensionRow = (label: string, valueKey: 'width_mm' | 'height_mm', tolKey: 'tolerance_w_mm' | 'tolerance_h_mm') => (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '6px', fontSize: '12px', color: C.textSub }}>
      <span>{label}</span>
      <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
        <input aria-label={`${label} danh định (mm)`} data-testid={`insp-${valueKey}`} type="number" value={config[valueKey]}
          onChange={setNumber(valueKey)} style={{ ...styles.input, width: '70px', padding: '6px 8px', textAlign: 'right' }} />
        <span>±</span>
        <input aria-label={`Dung sai ${label} (mm)`} data-testid={`insp-${tolKey}`} type="number" value={config[tolKey]}
          onChange={setNumber(tolKey)} style={{ ...styles.input, width: '56px', padding: '6px 8px', textAlign: 'right' }} />
        <span style={{ color: C.textMuted }}>mm</span>
      </span>
    </div>
  );

  const checkbox = (testId: string, label: string, checked: boolean, onChange: (value: boolean) => void, disabled = false) => (
    <label style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px', color: disabled ? C.textMuted : C.textSub, cursor: disabled ? 'not-allowed' : 'pointer' }}>
      <input type="checkbox" data-testid={testId} checked={checked} disabled={disabled} onChange={e => onChange(e.target.checked)}
        style={{ width: '16px', height: '16px', accentColor: C.accent }} />
      {label}
    </label>
  );

  const actionButton = (style: React.CSSProperties): React.CSSProperties => ({
    display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px', padding: '10px 12px', borderRadius: '8px',
    fontSize: '12px', fontWeight: 800, cursor: 'pointer', fontFamily: "'Space Grotesk', sans-serif", ...style,
  });
  const disabledLook = (disabled: boolean): React.CSSProperties =>
    ({ opacity: disabled ? 0.6 : 1, cursor: disabled ? 'not-allowed' : 'pointer' });
  const full = baselines.length >= baselineMax;

  return (
    <div data-testid="inspection-inspector" style={{ display: 'flex', flexDirection: 'column', gap: '10px', padding: '12px', borderRadius: '10px', border: `1px solid ${C.accentBorder}`, background: C.accentDim }}>
      <Section index={1} title="THÔNG TIN QUY TẮC" C={C}>
        <div>
          <label style={styles.label} htmlFor="insp-slot">Mã FMS Slot</label>
          <input id="insp-slot" data-testid="insp-slot" value={props.slotId} onChange={e => props.onSlotIdChange(e.target.value)}
            placeholder="VD: 7" style={styles.input} />
        </div>
        <div>
          <label style={styles.label} htmlFor="insp-channel">Kênh truyền thông</label>
          <select id="insp-channel" value={props.channelId} onChange={e => props.onChannelIdChange(e.target.value)} style={styles.select}>
            <option value="">Mặc định (FMS MQTT)</option>
            {props.channels.map(c => <option key={c.id} value={c.id}>{c.name}</option>)}
          </select>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
          <span style={styles.label}>Kích thước ô sàn thực tế</span>
          {numberField('insp-roi-width', 'Rộng – cạnh (1)→(2)', 'roi_width_mm', 10)}
          {numberField('insp-roi-height', 'Dài – cạnh (2)→(3)', 'roi_height_mm', 10)}
          <span style={{ fontSize: '11px', color: C.textMuted, lineHeight: 1.5 }}>
            Đo thật vạch sơn của ô bằng thước. Ảnh trực giao sẽ là {config.roi_width_mm}×{config.roi_height_mm} px (1 px = 1 mm).
            Đổi ROI hoặc kích thước ô cần chụp lại ảnh nền.
          </span>
        </div>
        <label style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '12px', fontWeight: 700, color: C.textSub, cursor: 'pointer' }}>
          Gửi kết quả sang FMS
          <input type="checkbox" checked={props.fmsDispatch} onChange={e => props.onFmsDispatchChange(e.target.checked)} style={{ width: '18px', height: '18px', accentColor: C.accent }} />
        </label>
      </Section>

      <Section index={2} title="CÁCH PHÁT HIỆN CÓ HÀNG (MODE)" C={C}>
        <div role="radiogroup" aria-label="Cách phát hiện có hàng" style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
          {INSPECTION_MODES.map(mode => {
            const active = config.mode === mode.value;
            return (
              <label key={mode.value} data-testid={`insp-mode-${mode.value}`} style={{
                display: 'flex', alignItems: 'flex-start', gap: '8px', padding: '7px 9px', borderRadius: '7px', cursor: 'pointer', fontSize: '12px',
                border: `1px solid ${active ? C.accentBorder : C.border}`, background: active ? C.accentDim : 'transparent',
                color: active ? C.accentL : C.textSub, fontWeight: active ? 800 : 500,
              }}>
                <input type="radio" name="inspection-mode" value={mode.value} checked={active} onChange={() => set('mode', mode.value)}
                  style={{ accentColor: C.accent, marginTop: '2px' }} />
                <span style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
                  <span>{mode.label}{mode.value === 'HYBRID' && <span aria-hidden> ⭐</span>}
                    <span style={{ color: C.textMuted, fontWeight: 400 }}> – {mode.hint}</span></span>
                  <span style={{ color: active ? C.textSub : C.textMuted, fontWeight: 500, fontSize: '11px' }}>{mode.outcome}</span>
                </span>
              </label>
            );
          })}
        </div>
      </Section>

      <Section index={3} title="ĐỐI TƯỢNG ÁP DỤNG (AI)" C={C}>
        {usesAi ? (
          <>
            <div data-testid="insp-targets" style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
              {classChips.map(name => {
                const active = props.targetClasses.includes(name);
                const unknown = !props.availableClasses.includes(name);
                return (
                  <button type="button" key={name} data-testid={`insp-target-${name}`} aria-pressed={active} onClick={() => toggleClass(name)}
                    title={unknown ? 'Model đang chạy không có class này' : undefined} style={{
                      padding: '4px 12px', borderRadius: '20px', fontSize: '12px', cursor: 'pointer',
                      border: active ? `1px ${unknown ? 'dashed' : 'solid'} ${C.accentBorder}` : `1px solid ${C.borderHard}`,
                      background: active ? C.accentDim : 'transparent', color: active ? C.accentL : C.textSub, fontWeight: active ? 700 : 400,
                    }}>{name}{unknown ? ' *' : ''}</button>
                );
              })}
              {classChips.length === 0 && <span style={{ fontSize: '11px', color: C.textMuted }}>Chưa tải được danh sách class của model</span>}
            </div>
            <span style={{ fontSize: '11px', color: C.textMuted, lineHeight: 1.5 }}>
              AI là model nhận diện đang chạy trên camera này (giống quy tắc Ô chứa hàng). Vật AI nhận ra thuộc các class đã chọn → CÓ HÀNG;
              {config.mode === 'HYBRID' ? ' vật xử lý ảnh thấy mà AI không gọi được tên → KHÔNG XÁC ĐỊNH.' : ' class khác bị bỏ qua.'}
              {props.targetClasses.length === 0 && ' Không chọn class nào = mọi vật AI nhận ra đều tính là hàng.'}
              {classChips.length > props.availableClasses.length && ' (* = model đang chạy không có class này)'}
            </span>
          </>
        ) : (
          <span style={{ fontSize: '11px', color: C.textMuted }}>Chế độ chỉ xử lý ảnh không dùng AI: có vật trong ô là CÓ HÀNG.</span>
        )}
      </Section>

      <Section index={4} title="ẢNH NỀN Ô TRỐNG (NHIỀU ĐIỀU KIỆN ÁNH SÁNG)" C={C}>
        <div data-testid="insp-baseline-status" style={{ fontSize: '12px', fontWeight: 700, color: usableCount ? C.emerald : C.amber }}>
          {usableCount
            ? `✅ ${usableCount} ảnh nền đang dùng${baselines.length > usableCount ? ` · ${baselines.length - usableCount} ảnh không khớp ROI hiện tại` : ''} (tối đa ${baselineMax})`
            : usesCv ? '⚠️ Chưa có ảnh nền khớp ROI – chụp khi ô trống hoàn toàn' : 'Chế độ chỉ AI không cần ảnh nền'}
        </div>
        <span style={{ fontSize: '11px', color: C.textMuted, lineHeight: 1.5 }}>
          Mỗi điều kiện ánh sáng ô gặp (sáng, trưa nắng, đèn ca đêm…) chụp một ảnh lúc ô trống. Mỗi khung hình được so với ảnh nền khớp ánh sáng nhất.
        </span>
        {baselines.length > 0 && (
          <div data-testid="insp-baselines" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(118px, 1fr))', gap: '8px' }}>
            {baselines.map(entry => {
              const matches = baselineMatchesRoi(entry, points, cell);
              return (
                <figure key={entry.id} data-testid={`insp-baseline-${entry.id}`} style={{ margin: 0, padding: '6px', borderRadius: '8px', border: `1px solid ${matches ? C.border : C.amber}`, background: C.card, position: 'relative' }}>
                  {thumbs[entry.id]
                    // eslint-disable-next-line @next/next/no-img-element
                    ? <img src={thumbs[entry.id]} alt={`Ảnh nền ${entry.label || formatMoment(entry.captured_at)}`} style={{ width: '100%', aspectRatio: `${entry.size_px?.[0] || 1} / ${entry.size_px?.[1] || 1}`, objectFit: 'cover', borderRadius: '5px', opacity: matches ? 1 : 0.5 }} />
                    : <div style={{ width: '100%', aspectRatio: '1 / 1', borderRadius: '5px', background: C.cardAlt }} />}
                  <figcaption style={{ marginTop: '4px', fontSize: '11px', color: C.textSub, lineHeight: 1.35 }}>
                    <div style={{ fontWeight: 700, color: C.textPrimary, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {entry.source === 'recording' ? '🎞' : '📷'} {entry.label || 'Không tên'}
                    </div>
                    <div>{formatMoment(entry.captured_at)}</div>
                    <div style={{ color: C.textMuted }}>Độ sáng {Math.round(entry.brightness)}</div>
                    {!matches && <div style={{ color: C.amber }}>Không khớp ROI</div>}
                  </figcaption>
                  <button type="button" data-testid={`insp-baseline-delete-${entry.id}`} aria-label={`Xoá ảnh nền ${entry.label || formatMoment(entry.captured_at)}`}
                    disabled={busy !== null} onClick={() => removeBaseline(entry)}
                    style={{ position: 'absolute', top: '10px', right: '10px', padding: '4px', borderRadius: '6px', border: `1px solid ${C.roseBorder}`, background: C.card, color: C.rose, display: 'flex', ...disabledLook(busy !== null) }}>
                    <Trash2 size={12} />
                  </button>
                </figure>
              );
            })}
          </div>
        )}
        <div>
          <label style={styles.label} htmlFor="insp-light-label">Điều kiện ánh sáng (tên ảnh nền)</label>
          <input id="insp-light-label" data-testid="insp-light-label" value={lightLabel} maxLength={48} onChange={e => setLightLabel(e.target.value)}
            placeholder="VD: Sáng, Trưa nắng, Đèn ca đêm" style={styles.input} />
        </div>
        <button type="button" data-testid="insp-capture" onClick={() => capture({})} disabled={!ready || busy !== null || full}
          style={actionButton({ background: C.cyanDim, color: C.cyanL, border: `1px solid ${C.cyanBorder}`, ...disabledLook(!ready || busy !== null || full) })}>
          <CameraIcon size={14} /> {busy === 'capture' ? 'Đang chụp…' : '📸 Chụp ngay – ô đang trống'}
        </button>
        {forceCapture && (
          <button type="button" data-testid="insp-capture-force" onClick={() => capture({ ...forceCapture, force: true })} disabled={busy !== null}
            style={actionButton({ background: 'transparent', color: C.amber, border: `1px solid ${C.amber}`, ...disabledLook(busy !== null) })}>
            Vẫn lưu – tôi chắc ô đang trống
          </button>
        )}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', padding: '8px', borderRadius: '8px', border: `1px dashed ${C.borderHard}` }}>
          <label style={{ ...styles.label, marginBottom: 0 }} htmlFor="insp-moment">Hoặc lấy từ video ghi hình lúc ô trống</label>
          <input id="insp-moment" data-testid="insp-moment" type="datetime-local" step={1} value={moment} onChange={e => setMoment(e.target.value)}
            min={coverage ? toDatetimeLocal(coverage.from) : undefined} max={coverage ? toDatetimeLocal(coverage.to) : undefined}
            style={{ ...styles.input, colorScheme: 'dark light' }} />
          <span style={{ fontSize: '11px', color: C.textMuted }}>
            {coverage ? `Video ghi hình có từ ${formatMoment(coverage.from)} đến ${formatMoment(coverage.to)}` : 'Camera này chưa có video ghi hình'}
          </span>
          <button type="button" data-testid="insp-capture-recording" onClick={captureFromRecording} disabled={!ready || busy !== null || full || !moment}
            style={actionButton({ background: 'transparent', color: C.cyanL, border: `1px solid ${C.cyanBorder}`, ...disabledLook(!ready || busy !== null || full || !moment) })}>
            <Film size={14} /> {busy === 'record' ? 'Đang lấy ảnh từ video…' : '🎞 Lấy ảnh nền tại thời điểm này'}
          </button>
        </div>
      </Section>

      <Section index={5} title="HIỂN THỊ TRÊN MONITOR" C={C}>
        {checkbox('insp-monitor-overlay', 'Vẽ ô kiểm định trên camera', config.monitor_overlay, value => set('monitor_overlay', value))}
        {checkbox('insp-monitor-label', 'Nhãn trạng thái (CÓ HÀNG / TRỐNG / KHÔNG XÁC ĐỊNH)', config.monitor_label, value => set('monitor_label', value), !config.monitor_overlay)}
        {checkbox('insp-monitor-object', 'Khung vật thể xử lý ảnh / AI tìm thấy', config.monitor_object, value => set('monitor_object', value), !config.monitor_overlay)}
      </Section>

      <Section index={6} title="ĐO KÍCH THƯỚC (THAM KHẢO)" C={C}>
        <details data-testid="insp-metrology">
          <summary style={{ fontSize: '12px', color: C.textSub, cursor: 'pointer' }}>
            Mẫu hàng, dung sai cơ khí và đối chất – {INSPECTION_PRESETS[config.target_preset]?.label || 'Tùy chỉnh'}
          </summary>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginTop: '8px' }}>
            <div>
              <label style={styles.label} htmlFor="insp-preset">Mẫu chuẩn</label>
              <select id="insp-preset" data-testid="insp-preset" value={config.target_preset}
                onChange={e => onConfigChange(applyPreset(config, e.target.value as InspectionPreset))} style={styles.select}>
                {(Object.keys(INSPECTION_PRESETS) as InspectionPreset[]).map(key => (
                  <option key={key} value={key}>{INSPECTION_PRESETS[key].label}</option>
                ))}
              </select>
            </div>
            {dimensionRow('Chiều rộng', 'width_mm', 'tolerance_w_mm')}
            {dimensionRow('Chiều dài', 'height_mm', 'tolerance_h_mm')}
            {numberField('insp-offset', 'Lệch tâm max', 'max_center_offset_mm')}
            {numberField('insp-rotation', 'Góc xoay max', 'max_rotation_deg', 0.1, '°')}
            {numberField('insp-margin', 'Khoảng an toàn mép ô', 'safe_margin_mm')}
            {!articleFitsCell(config) && (
              <div data-testid="insp-fit-warning" style={{ fontSize: '11px', color: C.rose, lineHeight: 1.5 }}>
                Kiện {config.width_mm}×{config.height_mm} mm + khoảng an toàn {config.safe_margin_mm} mm mỗi bên không vừa ô {config.roi_width_mm}×{config.roi_height_mm} mm – phần đo kích thước sẽ luôn NG lấn vạch.
              </div>
            )}
            {checkbox('insp-enable_ai_assisted_cv', 'Bật AI chỉ điểm khi CV tiệp màu', config.enable_ai_assisted_cv, value => set('enable_ai_assisted_cv', value))}
            {checkbox('insp-enable_shadow_filter', 'Bật bộ lọc bóng râm & Dị vật FOD', config.enable_shadow_filter, value => set('enable_shadow_filter', value))}
          </div>
        </details>
      </Section>

      <Section index={7} title="HÀNH ĐỘNG" C={C}>
        {!ready && (
          <div style={{ fontSize: '11px', color: C.amber, lineHeight: 1.5 }}>
            Chấm đúng {INSPECTION_POINT_COUNT} đỉnh của ô sàn theo thứ tự (1) trên-trái → (2) trên-phải → (3) dưới-phải → (4) dưới-trái ({points.length}/{INSPECTION_POINT_COUNT}).
          </div>
        )}
        <button type="button" data-testid="insp-test" onClick={runTest} disabled={!ready || busy !== null}
          style={actionButton({ background: 'linear-gradient(135deg, #6366f1 0%, #4f46e5 100%)', color: '#fff', border: '1px solid rgba(99,102,241,0.5)', ...disabledLook(!ready || busy !== null) })}>
          <FlaskConical size={14} /> {busy === 'test' ? 'Đang kiểm tra…' : '🧪 TEST THỬ TRÊN KHUNG HÌNH'}
        </button>
        <button type="button" data-testid="insp-save" onClick={props.onSave} disabled={props.saveDisabled}
          style={actionButton({ background: props.saveDisabled ? C.cardAlt : C.emeraldDim, color: props.saveDisabled ? C.textMuted : C.emerald, border: `1px solid ${props.saveDisabled ? C.border : C.emeraldBorder}`, cursor: props.saveDisabled ? 'not-allowed' : 'pointer' })}>
          <Save size={14} /> 💾 LƯU QUY TẮC KIỂM ĐỊNH
        </button>
        {notice && (
          <div role="status" data-testid="insp-notice" style={{ fontSize: '12px', fontWeight: 700, lineHeight: 1.5, color: notice.ok ? C.emerald : notice.text.startsWith('✅') ? C.amber : C.rose }}>{notice.text}</div>
        )}
      </Section>
    </div>
  );
}

export function InspectionResultPanel({ report, clientLatencyMs, colors: C }: { report: InspectionReport; clientLatencyMs?: number; colors: ColorPalette }) {
  const color = statusColor(C, report.status);
  const occupancy = report.occupancy;
  const occColor = occupancyColor(occupancy?.state);
  const choice = report.baseline_choice;
  const checks = report.checks || {};
  const rows: [string, keyof typeof checks, string][] = [
    ['Chiều rộng', 'width', 'mm'], ['Chiều dài', 'height', 'mm'], ['Lệch tâm', 'center_offset', 'mm'],
    ['Góc xoay', 'rotation', '°'], ['Cách mép ô', 'safe_margin', 'mm'],
  ];
  return (
    <div data-testid="inspection-result" style={{ marginTop: '14px', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 260px), 1fr))', gap: '14px', padding: '14px', borderRadius: '10px', border: `1px solid ${occColor}`, background: C.cardAlt }}>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', minWidth: 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap' }}>
          <span data-testid="inspection-result-occupancy" data-state={occupancy?.state || 'UNKNOWN'}
            style={{ padding: '5px 14px', borderRadius: '6px', fontWeight: 900, fontSize: '15px', color: '#06070a', background: occColor }}>
            {occupancyText(occupancy)}
          </span>
          {occupancy && (
            <span style={{ fontSize: '11px', color: C.textSub, border: `1px solid ${C.border}`, borderRadius: '5px', padding: '2px 7px' }}>
              {SOURCE_LABEL[occupancy.source] || occupancy.source}
            </span>
          )}
        </div>
        <div data-testid="inspection-result-reason" style={{ fontSize: '13px', color: C.textPrimary, fontWeight: 700 }}>{occupancy?.reason || report.message}</div>
        {choice && (
          <div data-testid="inspection-result-baseline" style={{ fontSize: '12px', color: C.textSub, lineHeight: 1.5 }}>
            Ảnh nền khớp ánh sáng nhất: <b>{choice.label || formatMoment(choice.captured_at)}</b>
            {choice.score != null ? ` (khác biệt ${(choice.score * 100).toFixed(1)}% diện tích ô)` : ''} · dùng {choice.usable}/{choice.count} ảnh nền
            {choice.candidates.length > 1 && (
              <span style={{ color: C.textMuted }}> – {choice.candidates.slice(1).map(c => `${c.label || formatMoment(c.captured_at)} ${c.score != null ? `${(c.score * 100).toFixed(1)}%` : ''}`).join(', ')}</span>
            )}
          </div>
        )}
        <div style={{ fontSize: '12px', color: C.textSub }} title={report.ai?.error || undefined}>
          AI: {report.ai ? (report.ai.available
            ? (report.ai.detections.map(d => `${d.class_name} ${Math.round(d.confidence * 100)}%${d.role === 'other' ? ' (không thuộc đối tượng)' : ''}`).join(', ') || `không thấy vật nào (${report.ai.model})`)
            : 'không khả dụng') : 'không dùng ở chế độ này'}
        </div>
        <details data-testid="inspection-result-metrology">
          <summary style={{ fontSize: '12px', color: C.textSub, cursor: 'pointer' }}>
            Đo kích thước (tham khảo): <b data-testid="inspection-result-status" style={{ color }}>{report.status}</b>{' '}
            <span style={{ fontFamily: 'monospace', color }}>{report.code}</span>
          </summary>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', marginTop: '6px' }}>
            <div style={{ fontSize: '12px', color: C.textPrimary }}>{report.message}</div>
            <div data-testid="inspection-result-summary" style={{ fontSize: '12px', color: C.textLabel, fontFamily: 'JetBrains Mono, monospace', lineHeight: 1.6 }}>{verdictSummary(report)}</div>
            {report.measurement && (
              <table style={{ borderCollapse: 'collapse', fontSize: '12px', color: C.textSub }}>
                <tbody>
                  {rows.map(([label, key, unit]) => {
                    const check = checks[key];
                    if (!check) return null;
                    return (
                      <tr key={key}>
                        <td style={{ padding: '2px 8px 2px 0' }}>{label}</td>
                        <td style={{ padding: '2px 8px', fontFamily: 'monospace', color: C.textPrimary }}>{check.value.toFixed(key === 'rotation' ? 2 : 1)} {unit}</td>
                        <td style={{ padding: '2px 8px', color: C.textMuted }}>{key === 'width' || key === 'height' ? `${check.nominal}±${check.limit}` : key === 'safe_margin' ? `≥ ${check.limit}` : `≤ ${check.limit}`} {unit}</td>
                        <td style={{ padding: '2px 0', fontWeight: 800, color: check.ok ? C.emerald : C.rose }}>{check.ok ? 'PASS' : 'NG'}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            )}
            {report.arbitration?.steps?.length > 0 && (
              <ol style={{ margin: 0, paddingLeft: '18px', fontSize: '11px', color: C.textMuted, lineHeight: 1.5 }}>
                {report.arbitration.steps.map((step, index) => <li key={index}>{step}</li>)}
              </ol>
            )}
          </div>
        </details>
        <div style={{ fontSize: '11px', color: C.textMuted, fontFamily: 'monospace' }}>
          {clientLatencyMs !== undefined && <>Phản hồi: {clientLatencyMs.toFixed(0)} ms · </>}
          Pipeline: {report.timing_ms?.total?.toFixed(1)} ms · Server: {report.server_latency_ms?.toFixed(1)} ms
        </div>
      </div>
      {report.overlay?.bev_image && (
        // eslint-disable-next-line @next/next/no-img-element
        <img data-testid="inspection-result-bev" src={report.overlay.bev_image} alt="Ảnh trực giao BEV của ô với kết quả"
          style={{ width: '100%', aspectRatio: '1 / 1', objectFit: 'contain', borderRadius: '8px', border: `1px solid ${C.border}` }} />
      )}
    </div>
  );
}
