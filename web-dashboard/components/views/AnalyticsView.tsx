'use client';

import { useState, useEffect } from 'react';
import { useAppTheme } from '../ThemeContext';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, Cell
} from 'recharts';
import {
  ArrowRightLeft, UserCheck, History, MapPin, Activity, Database,
  Cpu, GitMerge, Video, Download, Play, X, ShieldAlert, Clock
} from 'lucide-react';

type ThemeColors = {
  accent: string;
  accentBorder: string;
  accentDim: string;
  accentGlow: string;
  accentL: string;
  amber: string;
  bg: string;
  border: string;
  borderHard: string;
  card: string;
  cardAlt: string;
  chart: string[];
  cyan: string;
  cyanBorder: string;
  cyanDim: string;
  cyanL: string;
  elevated: string;
  rose: string;
  roseBorder: string;
  roseDim: string;
  textLabel: string;
  textMuted: string;
  textPrimary: string;
  textSub: string;
  violet: string;
  violetBorder: string;
  violetDim: string;
};
type TooltipPayload = { value: number };

// ─── Custom Tooltip ───────────────────────────────────────────────────────────
const CinderTooltip = ({
  active,
  payload,
  label,
  colors
}: {
  active?: boolean;
  payload?: TooltipPayload[];
  label?: string;
  colors: ThemeColors;
}) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: colors.card, border: `1px solid ${colors.borderHard}`,
      borderRadius: '8px', padding: '8px 12px',
      boxShadow: '0 12px 32px rgba(0,0,0,0.25)',
    }}>
      <p style={{ color: colors.textLabel, fontWeight: 700, fontSize: '12px', marginBottom: '3px' }}>{label}</p>
      {payload.map((p, i) => (
        <p key={i} style={{ color: colors.chart[i % colors.chart.length], fontSize: '13px', fontWeight: 600 }}>
          Số lượng: {p.value} sự kiện
        </p>
      ))}
    </div>
  );
};

// ─── KPI Card Data ────────────────────────────────────────────────────────────
type KpiDef = {
  label: string; value: React.ReactNode; sub: string;
  icon: React.ReactNode; color: string; dim: string; border: string; glow: string;
};
const getKpis = (data: DashboardAnalytics, C: ThemeColors): KpiDef[] => [
  {
    label: 'SỐ LUỒNG CAMERA', value: data.active_cameras ?? 0, sub: 'WHEP · WebRTC Active',
    icon: <Activity size={20} />,
    color: C.accentL, dim: C.accentDim, border: C.accentBorder, glow: C.accentGlow,
  },
  {
    label: 'TỔNG CẢNH BÁO', value: data.total_alerts ?? 0, sub: 'Intrusion / Dwell / Tripwire',
    icon: <Database size={20} />,
    color: C.rose, dim: C.roseDim, border: C.roseBorder, glow: C.roseBorder,
  },
  {
    label: 'ĐỘ TRỄ GPU', value: data.gpu_latency_ms != null ? `${data.gpu_latency_ms} ms` : 'N/A', sub: 'TensorRT FP16 Zero-Copy',
    icon: <Cpu size={20} />,
    color: C.cyanL, dim: C.cyanDim, border: C.cyanBorder, glow: C.cyanBorder,
  },
  {
    label: 'MTMC FUSION', value: data.system_efficiency != null ? `${data.system_efficiency}%` : 'N/A', sub: 'Hungarian Re-ID',
    icon: <GitMerge size={20} />,
    color: C.violet, dim: C.violetDim, border: C.violetBorder, glow: C.violetBorder,
  },
];

// ─── Event Interface ─────────────────────────────────────────────────────────
interface EventRecord {
  camera: string;
  type: string;
  roi_status?: string;
  description: string;
  severity?: string;
  time: string;
  video_file?: string;
  global_id?: number;
}

type TrackStep = {
  cam_id: string;
  floor_x?: number;
  floor_y?: number;
  floor_pos?: [number, number];
  timestamp?: number | string;
};

type JourneyData = {
  global_id: number | string;
  trajectory?: TrackStep[];
};

type DashboardAnalytics = {
  total_objects: number;
  active_cameras: number;
  total_alerts: number;
  system_efficiency: number | null;
  gpu_latency_ms?: number | null;
  class_distribution: Array<{ name: string; value: number }>;
  alerts_trend: Array<{ date: string; alerts: number }>;
  tripwire_stats: Array<{ rule_id: string; cam_id: string; entry: number; exit: number }>;
  recent_events: EventRecord[];
};

const EMPTY_ANALYTICS: DashboardAnalytics = {
  total_objects: 0,
  active_cameras: 0,
  total_alerts: 0,
  system_efficiency: null,
  gpu_latency_ms: null,
  class_distribution: [],
  alerts_trend: [],
  tripwire_stats: [],
  recent_events: []
};

const legacySampleVideoPattern = /^20260826_/;
const legacySampleDescriptions = [
  'Phát hiện đối tượng/xe hàng chiếm dụng Vùng cấm Cửa Kho A',
  'Đối tượng Global ID #105 đi vào Vùng cấm Cửa Thoát Hiểm',
  'Xe chở hàng cắt qua Vạch ảo Cổng Ra Vào',
  'Robot dừng chờ quá 20s tại Khu vực Bốc Dỡ'
];
const legacySampleTripwires = new Set(['Line Cổng Kho A', 'Line Hành Lang B']);

const isLegacySampleEvent = (event: EventRecord) => {
  const videoFile = event.video_file || '';
  return legacySampleVideoPattern.test(videoFile)
    || event.time.startsWith('2026-08-26')
    || legacySampleDescriptions.some(description => event.description.includes(description));
};

const isLegacySampleDistribution = (distribution: DashboardAnalytics['class_distribution']) => {
  const expected = new Map([
    ['Intrusion', 2],
    ['Tripwire', 3],
    ['Dwell Time', 1],
    ['Crowd Density', 1]
  ]);
  return distribution.length === expected.size
    && distribution.every(item => expected.get(item.name) === item.value);
};

const normalizeDashboardAnalytics = (raw: Partial<DashboardAnalytics>): DashboardAnalytics => {
  const classDistribution = raw.class_distribution ?? [];
  const recentEvents = raw.recent_events ?? [];
  const tripwireStats = raw.tripwire_stats ?? [];
  const filteredEvents = recentEvents.filter(event => !isLegacySampleEvent(event));
  const filteredTripwires = tripwireStats.filter(item => !legacySampleTripwires.has(item.rule_id));

  const looksLikeLegacyFallback =
    (raw.total_alerts ?? raw.total_objects) === 7
    && raw.system_efficiency === 99.4
    && isLegacySampleDistribution(classDistribution)
    && recentEvents.length > 0
    && filteredEvents.length === 0;

  if (looksLikeLegacyFallback) {
    return EMPTY_ANALYTICS;
  }

  const filteredDistribution = filteredEvents.length === recentEvents.length
    ? classDistribution
    : [];

  return {
    total_objects: filteredEvents.length === recentEvents.length ? (raw.total_objects ?? 0) : filteredEvents.length,
    active_cameras: raw.active_cameras ?? 0,
    total_alerts: filteredEvents.length === recentEvents.length ? (raw.total_alerts ?? 0) : filteredEvents.length,
    system_efficiency: raw.system_efficiency ?? null,
    gpu_latency_ms: raw.gpu_latency_ms ?? null,
    class_distribution: filteredDistribution,
    alerts_trend: raw.alerts_trend ?? [],
    tripwire_stats: filteredTripwires,
    recent_events: filteredEvents
  };
};

const formatCoord = (value?: number) => value == null ? 'N/A' : value.toFixed(2);

const formatStepTime = (timestamp?: number | string) => {
  if (timestamp == null) return 'N/A';
  const raw = typeof timestamp === 'number'
    ? timestamp * (timestamp < 1e12 ? 1000 : 1)
    : timestamp;
  const date = new Date(raw);
  return Number.isNaN(date.getTime()) ? 'N/A' : date.toLocaleTimeString();
};

// ─── Component ────────────────────────────────────────────────────────────────
export default function AnalyticsView() {
  const { colors: C, isDark } = useAppTheme();
  const [data, setData] = useState<DashboardAnalytics>(EMPTY_ANALYTICS);
  const [searchGid, setSearchGid] = useState('');
  const [journeyData, setJourneyData] = useState<JourneyData | null>(null);
  const [searchLoading, setSearchLoading] = useState(false);
  const [selectedVideoEvent, setSelectedVideoEvent] = useState<EventRecord | null>(null);

  const fetchData = async () => {
    try {
      const res = await fetch('/api/backend/analytics/dashboard');
      if (res.ok) {
        const d = await res.json();
        setData(normalizeDashboardAnalytics(d));
      } else {
        setData(EMPTY_ANALYTICS);
      }
    } catch (e) {
      console.error(e);
      setData(EMPTY_ANALYTICS);
    }
  };

  useEffect(() => {
    const firstLoad = window.setTimeout(fetchData, 0);
    const iv = setInterval(fetchData, 4000);
    return () => {
      window.clearTimeout(firstLoad);
      clearInterval(iv);
    };
  }, []);

  const handleSearchJourney = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!searchGid) return;
    setSearchLoading(true);
    try {
      const res = await fetch(`/api/backend/tracks/${searchGid}/history`);
      setJourneyData(res.ok ? (await res.json()).data : null);
    } catch { setJourneyData(null); } finally { setSearchLoading(false); }
  };

  const handleDownloadVideo = (e: React.MouseEvent, vfile: string) => {
    e.preventDefault();
    alert(`[BẢO MẬT VMS]\nFile: ${vfile}\nVideo đang được lưu an toàn tại máy chủ lưu trữ (/var/vms/recordings/).\nTính năng tải file trực tiếp về máy cục bộ yêu cầu quyền Quản trị viên (Admin Level 2).`);
  };

  // Shared styles
  const sectionCard: React.CSSProperties = {
    background: C.card, borderRadius: '12px',
    border: `1px solid ${C.border}`, padding: '20px',
    boxShadow: isDark ? '0 4px 20px rgba(0,0,0,0.4)' : '0 1px 3px rgba(0,0,0,0.05), 0 1px 2px rgba(0,0,0,0.03)',
  };

  return (
    <div style={{ backgroundColor: C.bg, minHeight: 'calc(100vh - 80px)', padding: '24px', transition: 'background-color 0.2s ease' }}>
      <div style={{ maxWidth: '1440px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '22px' }}>

        {/* ── Page Header ─────────────────────────────────────────────────── */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <div>
            <h1 style={{ fontSize: '22px', fontWeight: 700, color: C.textPrimary, margin: 0, letterSpacing: '-0.02em' }}>
              AI Video Analytics
              <span style={{ color: C.textMuted, fontWeight: 400, fontSize: '16px' }}> & Storage Dashboard</span>
            </h1>
            <p style={{ fontSize: '12px', color: C.textSub, margin: '5px 0 0', fontFamily: 'monospace' }}>
              Dữ liệu sự kiện lưu trữ & truy vấn thời gian thực · Lưu trữ video chứng cứ MP4 ·{' '}
              <span style={{ color: C.cyanL, fontWeight: 600 }}>PostgreSQL 15</span>
            </p>
          </div>

          {/* Journey Search */}
          <form onSubmit={handleSearchJourney} style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
            <div style={{ position: 'relative' }}>
              <input
                type="number" value={searchGid}
                onChange={e => setSearchGid(e.target.value)}
                placeholder="Tra cứu Global ID..."
                style={{
                  padding: '9px 12px 9px 36px',
                  borderRadius: '8px', border: `1px solid ${C.borderHard}`,
                  fontSize: '13px', width: '230px',
                  background: C.elevated, color: C.textPrimary,
                  fontFamily: "'Space Grotesk', sans-serif", outline: 'none',
                  transition: 'border-color 0.2s, box-shadow 0.2s',
                }}
                onFocus={e => { e.target.style.borderColor = C.accent; e.target.style.boxShadow = `0 0 0 3px ${C.accentGlow}`; }}
                onBlur={e => { e.target.style.borderColor = C.borderHard; e.target.style.boxShadow = 'none'; }}
              />
              <UserCheck size={14} color={C.textMuted} style={{ position: 'absolute', left: '10px', top: '50%', transform: 'translateY(-50%)' }} />
            </div>
            <button type="submit" style={{
              padding: '9px 18px',
              background: 'linear-gradient(135deg, #6366f1 0%, #4f46e5 100%)',
              color: '#fff', border: '1px solid rgba(99,102,241,0.5)',
              borderRadius: '8px', fontWeight: 600, fontSize: '13px',
              cursor: 'pointer', fontFamily: "'Space Grotesk', sans-serif",
              boxShadow: isDark ? '0 4px 16px rgba(99,102,241,0.3)' : '0 2px 8px rgba(99,102,241,0.25)',
              transition: 'all 0.2s',
            }}>
              {searchLoading ? '...' : 'Tra cứu vết'}
            </button>
          </form>
        </div>

        {/* ── Journey Result ───────────────────────────────────────────────── */}
        {journeyData && (
          <div style={{
            ...sectionCard,
            borderLeft: `3px solid ${C.accent}`,
            borderColor: C.accentBorder,
            background: C.cardAlt,
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontWeight: 700, color: C.accentL, fontSize: '14px' }}>
                <History size={16} /> Hành trình Global ID #{journeyData.global_id}
              </div>
              <button onClick={() => setJourneyData(null)} style={{
                background: 'none', border: 'none', color: C.textMuted, cursor: 'pointer', fontSize: '18px',
              }}>✕</button>
            </div>
            <div style={{ display: 'flex', gap: '10px', overflowX: 'auto', paddingBottom: '4px' }}>
              {(journeyData.trajectory || []).map((step, idx) => (
                <div key={idx} style={{
                  background: C.card, padding: '12px', borderRadius: '8px',
                  border: `1px solid ${C.borderHard}`, minWidth: '164px', fontSize: '12px',
                  transition: 'border-color 0.2s',
                }}>
                  <div style={{ fontWeight: 700, color: C.textPrimary, display: 'flex', alignItems: 'center', gap: '4px', marginBottom: '5px' }}>
                    <MapPin size={12} color={C.accent} /> Camera {step.cam_id}
                  </div>
                  <div style={{ color: C.textSub, fontFamily: 'monospace', fontSize: '11px' }}>
                    ({formatCoord(step.floor_pos?.[0] ?? step.floor_x)},&nbsp;
                     {formatCoord(step.floor_pos?.[1] ?? step.floor_y)})
                  </div>
                  <div style={{ color: C.textMuted, fontSize: '10px', marginTop: '4px', fontFamily: 'monospace' }}>
                    {formatStepTime(step.timestamp)}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ── KPI Cards ────────────────────────────────────────────────────── */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '14px' }}>
          {getKpis(data, C).map((k, i) => (
            <div key={i} style={{
              background: isDark ? 'rgba(13,17,23,0.8)' : 'rgba(255,255,255,0.95)',
              backdropFilter: 'blur(12px)',
              WebkitBackdropFilter: 'blur(12px)',
              border: `1px solid ${k.border}`,
              borderRadius: '14px', padding: '20px',
              boxShadow: isDark
                ? `0 0 28px ${k.glow}, inset 0 1px 0 rgba(255,255,255,0.04)`
                : `0 4px 16px rgba(0,0,0,0.04), 0 1px 3px rgba(0,0,0,0.03), 0 0 12px ${k.glow}`,
              position: 'relative', overflow: 'hidden',
              transition: 'transform 0.2s, box-shadow 0.2s',
            }}>
              {/* Corner glow */}
              <div style={{
                position: 'absolute', top: '-30px', right: '-30px',
                width: '100px', height: '100px', borderRadius: '50%',
                background: k.glow, filter: 'blur(30px)', pointerEvents: 'none',
              }} />
              {/* Icon + Label */}
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '12px' }}>
                <span style={{ fontSize: '11px', color: C.textMuted, fontFamily: 'JetBrains Mono, monospace', letterSpacing: '0.06em', fontWeight: 600 }}>
                  {k.label}
                </span>
                <span style={{ color: k.color, opacity: 0.9 }}>{k.icon}</span>
              </div>
              {/* Big number */}
              <div style={{
                fontSize: '34px', fontWeight: 700, color: k.color,
                lineHeight: 1, marginBottom: '6px',
                fontFamily: "'Space Grotesk', sans-serif",
                textShadow: isDark ? `0 0 20px ${k.glow}` : 'none',
              }}>
                {k.value}
              </div>
              {/* Sub */}
              <div style={{ fontSize: '11px', color: C.textMuted, fontFamily: 'JetBrains Mono, monospace' }}>
                {k.sub}
              </div>
            </div>
          ))}
        </div>

        {/* ── Main 2-col grid ──────────────────────────────────────────────── */}
        <div style={{ display: 'grid', gridTemplateColumns: '1.05fr 1.35fr', gap: '16px' }}>

          {/* Left column */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>

            {/* Bar Chart: Phân bố sự kiện hành vi */}
            <div style={sectionCard}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
                <h3 style={{ fontSize: '13px', fontWeight: 700, color: C.textLabel, margin: 0, letterSpacing: '0.02em' }}>
                  PHÂN BỐ SỰ KIỆN HÀNH VI
                </h3>
                <span style={{
                  fontSize: '11px', fontWeight: 700, color: C.accentL,
                  background: C.accentDim, padding: '3px 8px', borderRadius: '6px',
                  border: `1px solid ${C.accentBorder}`, fontFamily: 'monospace'
                }}>
                  {data.class_distribution.reduce((acc, cur) => acc + (cur.value || 0), 0)} Tổng sự kiện
                </span>
              </div>

              {/* Stat breakdown badges */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '8px', marginBottom: '16px' }}>
                {data.class_distribution.map((item, idx) => {
                  const color = C.chart[idx % C.chart.length];
                  return (
                    <div key={idx} style={{
                      background: C.cardAlt, padding: '8px 12px', borderRadius: '8px',
                      border: `1px solid ${C.border}`, display: 'flex', justifyContent: 'space-between', alignItems: 'center'
                    }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                        <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: color }} />
                        <span style={{ fontSize: '12px', fontWeight: 600, color: C.textLabel }}>{item.name}</span>
                      </div>
                      <span style={{ fontSize: '14px', fontWeight: 700, color, fontFamily: 'monospace' }}>
                        {item.value}
                      </span>
                    </div>
                  );
                })}
              </div>

              <div style={{ height: '180px', minWidth: 0, minHeight: 180 }}>
                <ResponsiveContainer width="100%" height="100%" minWidth={0} minHeight={180}>
                  <BarChart data={data.class_distribution} barCategoryGap="30%">
                    <CartesianGrid strokeDasharray="2 4" vertical={false} stroke={isDark ? 'rgba(51,65,85,0.4)' : 'rgba(229,231,235,0.8)'} />
                    <XAxis dataKey="name" tick={{ fontSize: 11, fill: C.textMuted, fontFamily: 'JetBrains Mono, monospace' }} axisLine={false} tickLine={false} />
                    <YAxis tick={{ fontSize: 11, fill: C.textMuted, fontFamily: 'JetBrains Mono, monospace' }} axisLine={false} tickLine={false} allowDecimals={false} />
                    <RechartsTooltip content={<CinderTooltip colors={C} />} cursor={{ fill: C.accentDim }} />
                    <Bar dataKey="value" radius={[6, 6, 0, 0]}>
                      {data.class_distribution.map((_, i) => (
                        <Cell key={i} fill={C.chart[i % C.chart.length]} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Tripwire Counters */}
            <div style={sectionCard}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
                <h3 style={{ fontSize: '13px', fontWeight: 700, color: C.textLabel, margin: 0, letterSpacing: '0.02em' }}>
                  THỐNG KÊ TRIPWIRE (VẠCH ẢO)
                </h3>
                <ArrowRightLeft size={16} color={C.cyanL} />
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {(data.tripwire_stats || []).length === 0 ? (
                  <div style={{
                    border: `1px dashed ${C.borderHard}`,
                    borderRadius: '10px', padding: '24px 20px',
                    textAlign: 'center', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '8px',
                  }}>
                    <ArrowRightLeft size={24} color={C.textMuted} strokeWidth={1.5} />
                    <p style={{ color: C.textLabel, fontSize: '13px', fontWeight: 600, margin: 0 }}>
                      Chưa có vạch ảo nào ghi nhận
                    </p>
                    <p style={{ color: C.textMuted, fontSize: '11px', fontFamily: 'monospace', margin: 0 }}>
                      Vào tab <span style={{ color: C.accentL, fontWeight: 600 }}>Building → ROI/Tripwire</span> để vẽ vạch ảo
                    </p>
                  </div>
                ) : (
                  (data.tripwire_stats || []).map((t, i) => (
                    <div key={i} style={{
                      display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                      padding: '11px 14px', background: C.cardAlt,
                      borderRadius: '8px', border: `1px solid ${C.border}`,
                      transition: 'border-color 0.2s',
                    }}>
                      <div>
                        <div style={{ fontWeight: 600, color: C.textLabel, fontFamily: 'JetBrains Mono, monospace', fontSize: '12px' }}>
                          {t.rule_id}
                        </div>
                        <div style={{ fontSize: '10px', color: C.textMuted, fontFamily: 'monospace' }}>
                          Nguồn: Camera {t.cam_id}
                        </div>
                      </div>
                      <div style={{ display: 'flex', gap: '12px' }}>
                        <span style={{
                          background: 'rgba(74,222,128,0.12)', color: isDark ? '#4ade80' : '#15803d',
                          border: '1px solid rgba(74,222,128,0.25)', padding: '3px 9px', borderRadius: '6px',
                          fontWeight: 700, fontFamily: 'monospace', fontSize: '12px'
                        }}>
                          VÀO (IN): {t.entry}
                        </span>
                        <span style={{
                          background: 'rgba(251,113,133,0.12)', color: isDark ? '#fb7185' : '#b91c1c',
                          border: '1px solid rgba(251,113,133,0.25)', padding: '3px 9px', borderRadius: '6px',
                          fontWeight: 700, fontFamily: 'monospace', fontSize: '12px'
                        }}>
                          RA (OUT): {t.exit}
                        </span>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>

          </div>

          {/* Right column — Event Logs (Nhật ký sự kiện & lưu file MP4) */}
          <div style={{ ...sectionCard, display: 'flex', flexDirection: 'column' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <ShieldAlert size={18} color={C.rose} />
                <h3 style={{ fontSize: '14px', fontWeight: 700, color: C.textPrimary, margin: 0, letterSpacing: '0.02em' }}>
                  NHẬT KÝ SỰ KIỆN & CHỨNG CỨ MP4
                </h3>
              </div>
              <span style={{
                fontSize: '11px', color: C.accentL, fontFamily: 'monospace', fontWeight: 600,
                background: C.accentDim, padding: '3px 8px', borderRadius: '6px', border: `1px solid ${C.accentBorder}`
              }}>
                PostgreSQL Sync · {data.recent_events.length} Bản ghi
              </span>
            </div>

            <div style={{ flex: 1, overflowY: 'auto', maxHeight: '520px', display: 'flex', flexDirection: 'column', gap: '10px', paddingRight: '4px' }}>
              {(data.recent_events || []).length === 0 ? (
                <div style={{
                  border: `1px dashed ${C.borderHard}`, borderRadius: '10px',
                  padding: '40px 20px', textAlign: 'center',
                  display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '12px',
                }}>
                  <Database size={32} color={C.textMuted} strokeWidth={1.5} />
                  <p style={{ color: C.textLabel, fontWeight: 600, fontSize: '13px', margin: 0 }}>
                    Chưa có nhật ký sự kiện
                  </p>
                  <p style={{ color: C.textMuted, fontSize: '12px', fontFamily: 'monospace', margin: 0 }}>
                    Hệ thống đang chờ sự kiện từ DeepStream pipeline
                  </p>
                </div>
              ) : (
                data.recent_events.map((ev: EventRecord, i: number) => {
                  const isIntrusion = ev.type === 'intrusion';
                  const isTripwire = ev.type === 'tripwire';
                  const isDwell = ev.type === 'dwell_time';
                  const accentColor = isIntrusion ? C.rose : (isTripwire ? C.cyan : (isDwell ? C.amber : C.violet));
                  const accentDim = isIntrusion ? C.roseDim : (isTripwire ? C.cyanDim : (isDwell ? 'rgba(245,158,11,0.1)' : C.violetDim));
                  const accentBorder = isIntrusion ? C.roseBorder : (isTripwire ? C.cyanBorder : (isDwell ? 'rgba(245,158,11,0.3)' : C.violetBorder));
                  const roiStatus = ev.roi_status;
                  const videoFileName = ev.video_file;

                  return (
                    <div key={i} style={{
                      padding: '13px 15px', borderRadius: '10px',
                      background: C.cardAlt, border: `1px solid ${C.border}`,
                      borderLeft: `4px solid ${accentColor}`,
                      display: 'flex', flexDirection: 'column', gap: '8px',
                      boxShadow: isDark ? '0 2px 8px rgba(0,0,0,0.3)' : '0 1px 3px rgba(0,0,0,0.04)',
                      transition: 'border-color 0.2s, transform 0.15s',
                    }}>
                      {/* Top row: Type badge, Status badge, Timestamp */}
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                          <span style={{
                            fontWeight: 700, color: accentColor, fontSize: '11px',
                            textTransform: 'uppercase', fontFamily: 'JetBrains Mono, monospace', letterSpacing: '0.06em',
                            background: accentDim, padding: '2px 8px', borderRadius: '4px',
                            border: `1px solid ${accentBorder}`,
                          }}>
                            {ev.type}
                          </span>

                          {roiStatus && (
                            <span style={{
                              fontWeight: 700, fontSize: '11px',
                              fontFamily: 'JetBrains Mono, monospace',
                              background: roiStatus.includes('CARFULL') || roiStatus.includes('CRITICAL') ? 'rgba(244,63,94,0.18)' : (roiStatus.includes('IN') || roiStatus.includes('EMPTY') ? 'rgba(16,185,129,0.15)' : 'rgba(245,158,11,0.15)'),
                              color: roiStatus.includes('CARFULL') || roiStatus.includes('CRITICAL') ? (isDark ? '#fb7185' : '#e11d48') : (roiStatus.includes('IN') || roiStatus.includes('EMPTY') ? (isDark ? '#34d399' : '#059669') : (isDark ? '#fbbf24' : '#d97706')),
                              padding: '2px 8px', borderRadius: '4px',
                              border: `1px solid ${roiStatus.includes('CARFULL') ? 'rgba(244,63,94,0.4)' : 'rgba(245,158,11,0.3)'}`
                            }}>
                              STATUS: {roiStatus}
                            </span>
                          )}
                        </div>

                        {/* Timestamp */}
                        <div style={{ display: 'flex', alignItems: 'center', gap: '4px', color: C.textMuted, fontSize: '11px', fontFamily: 'monospace' }}>
                          <Clock size={12} />
                          <span>{ev.time}</span>
                        </div>
                      </div>

                      {/* Description */}
                      <div style={{ color: C.textPrimary, fontSize: '13px', fontWeight: 500, lineHeight: 1.45 }}>
                        {ev.description}
                      </div>

                      {/* Source Camera & GID */}
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '11px', color: C.textSub, fontFamily: 'monospace' }}>
                        <span>Nguồn: <b>{ev.camera}</b> {ev.global_id ? `· Global ID #${ev.global_id}` : ''}</span>
                        <span style={{ color: C.accentL, fontWeight: 600 }}>Mức độ: {(ev.severity || 'unknown').toUpperCase()}</span>
                      </div>

                      {/* MP4 Attachment bar */}
                      <div style={{
                        marginTop: '4px', padding: '8px 10px', borderRadius: '7px',
                        background: isDark ? 'rgba(9,10,15,0.8)' : 'rgba(241,245,249,0.9)',
                        border: `1px solid ${C.borderHard}`,
                        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                      }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '7px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                          <Video size={14} color={C.cyanL} />
                          <span style={{
                            fontSize: '11px', color: C.textLabel, fontFamily: 'JetBrains Mono, monospace',
                            fontWeight: 600, letterSpacing: '0.02em', overflow: 'hidden', textOverflow: 'ellipsis', maxWidth: '280px'
                          }}>
                            {videoFileName || 'Chưa có file MP4'}
                          </span>
                        </div>

                        {/* Actions: Play Video / Download */}
                        {videoFileName && (
                          <div style={{ display: 'flex', gap: '6px' }}>
                            <button
                              onClick={() => setSelectedVideoEvent({ ...ev, video_file: videoFileName })}
                              style={{
                                display: 'flex', alignItems: 'center', gap: '4px',
                                background: C.accentDim, border: `1px solid ${C.accentBorder}`,
                                color: C.accentL, padding: '4px 9px', borderRadius: '5px',
                                cursor: 'pointer', fontSize: '11px', fontWeight: 600,
                                fontFamily: "'Space Grotesk', sans-serif", transition: 'all 0.15s',
                              }}
                            >
                              <Play size={11} fill={C.accentL} /> Xem MP4
                            </button>
                            <button
                              onClick={(e) => handleDownloadVideo(e, videoFileName)}
                              style={{
                                display: 'flex', alignItems: 'center', gap: '4px',
                                background: 'transparent', border: `1px solid ${C.borderHard}`,
                                color: C.textSub, padding: '4px 8px', borderRadius: '5px',
                                cursor: 'pointer', fontSize: '11px', fontWeight: 500,
                                transition: 'all 0.15s',
                              }}
                              title="Tải video về máy"
                            >
                              <Download size={11} />
                            </button>
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })
              )}
            </div>
          </div>

        </div>
      </div>

      {/* ── VIDEO PLAYER MODAL ────────────────────────────────────────────── */}
      {selectedVideoEvent && (
        <div style={{
          position: 'fixed', inset: 0,
          backgroundColor: 'rgba(0,0,0,0.85)',
          backdropFilter: 'blur(8px)', WebkitBackdropFilter: 'blur(8px)',
          zIndex: 2000, display: 'flex', alignItems: 'center', justifyContent: 'center',
          padding: '20px'
        }}>
          <div style={{
            background: C.card, borderRadius: '16px', border: `1px solid ${C.borderHard}`,
            width: '680px', maxWidth: '100%', overflow: 'hidden',
            boxShadow: '0 32px 80px rgba(0,0,0,0.8)',
            display: 'flex', flexDirection: 'column'
          }}>
            {/* Modal Header */}
            <div style={{
              padding: '16px 20px', background: C.cardAlt, borderBottom: `1px solid ${C.border}`,
              display: 'flex', justifyContent: 'space-between', alignItems: 'center'
            }}>
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Video size={18} color={C.accentL} />
                  <h3 style={{ margin: 0, fontSize: '15px', fontWeight: 700, color: C.textPrimary }}>
                    Phát Video Chứng Cứ Sự Kiện
                  </h3>
                </div>
                <div style={{ fontSize: '11px', color: C.textMuted, fontFamily: 'monospace', marginTop: '2px' }}>
                  {selectedVideoEvent.video_file}
                </div>
              </div>
              <button
                onClick={() => setSelectedVideoEvent(null)}
                style={{ background: 'none', border: 'none', color: C.textMuted, cursor: 'pointer', padding: '4px' }}
              >
                <X size={20} />
              </button>
            </div>

            {/* HTML5 Video Player Frame */}
            <div style={{
              position: 'relative', width: '100%', aspectRatio: '16/9', background: '#020306',
              display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden'
            }}>
              <video
                key={selectedVideoEvent.video_file}
                src={`/recordings/${selectedVideoEvent.video_file}`}
                controls
                autoPlay
                loop
                style={{ width: '100%', height: '100%', objectFit: 'contain' }}
              />

              {/* Watermark & Bounding Overlay */}
                <div style={{
                position: 'absolute', top: '12px', left: '14px', pointerEvents: 'none',
                background: 'rgba(9,10,15,0.85)', padding: '4px 10px', borderRadius: '6px',
                border: '1px solid rgba(244,63,94,0.4)', color: '#fb7185',
                fontSize: '11px', fontWeight: 700, fontFamily: 'monospace', letterSpacing: '0.04em'
              }}>
                ● REC · {selectedVideoEvent.type.toUpperCase()} · {selectedVideoEvent.roi_status || 'N/A'}
              </div>

              <div style={{
                position: 'absolute', top: '12px', right: '14px', pointerEvents: 'none',
                background: 'rgba(9,10,15,0.85)', padding: '4px 10px', borderRadius: '6px',
                border: '1px solid rgba(51,65,85,0.6)', color: '#cbd5e1',
                fontSize: '11px', fontFamily: 'monospace'
              }}>
                {selectedVideoEvent.time}
              </div>
            </div>

            {/* Modal Footer Info */}
            <div style={{ padding: '16px 20px', background: C.card, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <div style={{ color: C.textPrimary, fontSize: '13px', fontWeight: 600 }}>
                  {selectedVideoEvent.description}
                </div>
                <div style={{ color: C.textSub, fontSize: '11px', marginTop: '2px', fontFamily: 'monospace' }}>
                  Nguồn: {selectedVideoEvent.camera} · Trạng thái: {selectedVideoEvent.roi_status || 'N/A'}
                </div>
              </div>
              <button
                onClick={(e) => selectedVideoEvent.video_file && handleDownloadVideo(e, selectedVideoEvent.video_file)}
                style={{
                  display: 'flex', alignItems: 'center', gap: '6px',
                  background: 'linear-gradient(135deg, #6366f1 0%, #4f46e5 100%)',
                  color: '#ffffff', border: '1px solid rgba(99,102,241,0.5)',
                  padding: '9px 16px', borderRadius: '8px', cursor: 'pointer',
                  fontWeight: 600, fontSize: '12px', fontFamily: "'Space Grotesk', sans-serif",
                  boxShadow: '0 4px 16px rgba(99,102,241,0.35)'
                }}
              >
                <Download size={14} /> Tải File MP4
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
