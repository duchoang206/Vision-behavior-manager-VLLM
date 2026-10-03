// Building view rules shared types, and the client of the inspection-station API
// (backend/routers/inspection.py).  InspectionConfig mirrors
// backend/core/inspection/config.py field for field.

export const INSPECTION_API = '/api/backend';

export type InspectionMode = 'CV_ONLY' | 'AI_ONLY' | 'HYBRID';
export type InspectionPreset = 'carton_800x600' | 'pallet_1200x1000' | 'pallet_1100x1100' | 'custom';
export type LightProfile = 'adaptive' | 'low' | 'normal' | 'high';

export type InspectionConfig = {
  mode: InspectionMode;
  // Real size of the floor cell outlined by the 4 ROI vertices:
  // width along edge 1→2, length along edge 2→3 (1000 x 1000 is only the default).
  roi_width_mm: number;
  roi_height_mm: number;
  target_preset: InspectionPreset;
  width_mm: number;
  height_mm: number;
  tolerance_w_mm: number;
  tolerance_h_mm: number;
  max_center_offset_mm: number;
  max_rotation_deg: number;
  safe_margin_mm: number;
  light_profile: LightProfile;
  enable_ai_assisted_cv: boolean;
  enable_shadow_filter: boolean;
  temporal_window_frames: number;
  has_background_baseline?: boolean;
  // Monitor view display of the station (no effect on any decision).
  monitor_overlay: boolean;
  monitor_label: boolean;
  monitor_object: boolean;
};

export type StoredRule = {
  id: string;
  cam_id: string;
  type: 'occupancy' | 'intrusion' | 'tripwire' | 'direction' | 'inspection';
  name: string;
  points: number[][];
  fms_slot_id?: string | null;
  comm_channel_id?: string | null;
  enable_fms_dispatch?: boolean;
  target_objects?: string[];
  inspection_config?: InspectionConfig;
};

export const INSPECTION_PRESETS: Record<InspectionPreset, { label: string; width_mm?: number; height_mm?: number }> = {
  carton_800x600: { label: 'Thùng Carton (800x600)', width_mm: 800, height_mm: 600 },
  pallet_1200x1000: { label: 'Pallet EUR (1200x1000)', width_mm: 1200, height_mm: 1000 },
  pallet_1100x1100: { label: 'Pallet vuông (1100x1100)', width_mm: 1100, height_mm: 1100 },
  custom: { label: 'Tùy chỉnh' },
};

// What each mode answers to "does the cell hold goods?" (backend occupancy.py).
export const INSPECTION_MODES: { value: InspectionMode; label: string; hint: string; outcome: string }[] = [
  { value: 'CV_ONLY', label: 'Chỉ xử lý ảnh', hint: 'so với ảnh nền ô trống, không cần GPU',
    outcome: 'Có vật → CÓ HÀNG · không có → TRỐNG' },
  { value: 'AI_ONLY', label: 'Chỉ AI', hint: 'model nhận diện đang chạy trên camera',
    outcome: 'AI thấy đối tượng áp dụng trong ô → CÓ HÀNG · không → TRỐNG' },
  { value: 'HYBRID', label: 'Xử lý ảnh + AI', hint: 'xử lý ảnh phát hiện, AI gọi tên (khuyên dùng)',
    outcome: 'Có vật: AI nhận ra đối tượng áp dụng → CÓ HÀNG · AI không biết → KHÔNG XÁC ĐỊNH · không có vật → TRỐNG' },
];

export const DEFAULT_INSPECTION_CONFIG: InspectionConfig = {
  mode: 'HYBRID',
  roi_width_mm: 1000,
  roi_height_mm: 1000,
  target_preset: 'carton_800x600',
  width_mm: 800,
  height_mm: 600,
  tolerance_w_mm: 30,
  tolerance_h_mm: 30,
  max_center_offset_mm: 50,
  max_rotation_deg: 5,
  safe_margin_mm: 30,
  light_profile: 'adaptive',
  enable_ai_assisted_cv: true,
  enable_shadow_filter: true,
  temporal_window_frames: 3,
  has_background_baseline: false,
  monitor_overlay: true,
  monitor_label: true,
  monitor_object: true,
};

export const INSPECTION_POINT_COUNT = 4;

export function applyPreset(config: InspectionConfig, preset: InspectionPreset): InspectionConfig {
  const dims = INSPECTION_PRESETS[preset];
  return {
    ...config,
    target_preset: preset,
    width_mm: dims.width_mm ?? config.width_mm,
    height_mm: dims.height_mm ?? config.height_mm,
  };
}

const finite = (value: unknown, fallback: number) => (typeof value === 'number' && Number.isFinite(value) ? value : fallback);

export function normalizeInspectionConfig(raw?: Partial<InspectionConfig> | null): InspectionConfig {
  const base = { ...DEFAULT_INSPECTION_CONFIG, ...(raw || {}) };
  const mode: InspectionMode = (['CV_ONLY', 'AI_ONLY', 'HYBRID'] as const).includes(base.mode) ? base.mode : 'HYBRID';
  const preset: InspectionPreset = base.target_preset in INSPECTION_PRESETS ? base.target_preset : 'custom';
  return {
    ...base,
    mode,
    target_preset: preset,
    roi_width_mm: finite(base.roi_width_mm, 1000),
    roi_height_mm: finite(base.roi_height_mm, 1000),
    width_mm: finite(base.width_mm, 800),
    height_mm: finite(base.height_mm, 600),
    tolerance_w_mm: finite(base.tolerance_w_mm, 30),
    tolerance_h_mm: finite(base.tolerance_h_mm, 30),
    max_center_offset_mm: finite(base.max_center_offset_mm, 50),
    max_rotation_deg: finite(base.max_rotation_deg, 5),
    safe_margin_mm: finite(base.safe_margin_mm, 30),
    temporal_window_frames: Math.max(1, Math.round(finite(base.temporal_window_frames, 3))),
    enable_ai_assisted_cv: base.enable_ai_assisted_cv !== false,
    enable_shadow_filter: base.enable_shadow_filter !== false,
    monitor_overlay: base.monitor_overlay !== false,
    monitor_label: base.monitor_label !== false,
    monitor_object: base.monitor_object !== false,
  };
}

export const newInspectionRuleId = () => `insp_${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`;

// ── API types ────────────────────────────────────────────────────────────────
export type InspectionStatus = 'OK' | 'NG' | 'UNCERTAIN' | 'DETECTED';
export type OccupancyState = 'CARFULL' | 'EMPTY' | 'UNKNOWN';

export type OccupancyDecision = {
  state: OccupancyState;
  text: string;
  reason: string;
  source: 'CV' | 'AI' | 'CV+AI' | 'NONE';
  label: string | null;
  confidence: number | null;
  cv_object: boolean;
  cv_size_mm: number[] | null;
  ai_objects: { class_name: string; confidence: number; role: string; label: string; source: string }[];
};

export type InspectionMeasurement = {
  width_mm: number;
  height_mm: number;
  center_x_mm: number;
  center_y_mm: number;
  offset_x_mm: number;
  offset_y_mm: number;
  offset_mm: number;
  rotation_deg: number;
  min_margin_mm: number;
  corners_mm: number[][];
  refined: boolean;
};

export type InspectionCheck = { value: number; limit: number; nominal?: number; ok: boolean };

export type BaselineCandidate = { id: string; label: string; captured_at: number | null; score: number | null };
export type BaselineChoice = BaselineCandidate & { count: number; usable: number; candidates: BaselineCandidate[] };

export type InspectionOverlayData = {
  cell_center: number[];
  obb: number[][] | null;
  center: number[] | null;
  ai_boxes: { label: string; role: string; source?: string; polygon: number[][] }[];
  objects?: number[][][];
  bev_image?: string;
};

export type InspectionReport = {
  status: InspectionStatus;
  slot_state: 'EMPTY' | 'OCCUPIED' | 'UNKNOWN';
  code: string;
  message: string;
  errors: string[];
  mode: InspectionMode;
  ai_class?: string | null;
  ai_confidence?: number | null;
  measurement: InspectionMeasurement | null;
  checks: Partial<Record<'width' | 'height' | 'center_offset' | 'rotation' | 'safe_margin', InspectionCheck>>;
  arbitration: { dispute: string | null; resolution: string | null; steps: string[] };
  cv: Record<string, unknown> | null;
  ai: { available: boolean; model: string; error: string; detections: { class_name: string; confidence: number; role: string; source?: string }[] } | null;
  timing_ms: Record<string, number>;
  overlay: InspectionOverlayData | null;
  occupancy?: OccupancyDecision | null;
  baseline_choice?: BaselineChoice | null;
  baseline: { rule_id: string; timestamp: number; count?: number } | null;
  targets?: string[];
  server_latency_ms?: number;
};

export type BaselineInfo = { url: string; timestamp: number | null };

export type BaselineEntry = {
  id: string;
  label: string;
  source: 'live' | 'recording';
  captured_at: number;
  timestamp: number;
  brightness: number;
  laplacian_var: number;
  frames_used: number;
  size_px: number[];
  cell_mm: number[];
  points: number[][];
  frame_shape: number[];
  recording?: { file: string; at: number } | null;
};

export type BaselineList = { baselines: BaselineEntry[]; max: number; recordings: { from: number; to: number } | null };

export type CaptureResult = {
  status: string; id: string; label: string; source: string; captured_at: number; timestamp: number;
  size_px: number[]; cell_mm: number[]; brightness: number; count: number; replaced: number; warning: string | null;
};

export type InspectionStation = {
  cam_id: string;
  rule_id: string;
  name: string;
  fms_slot_id?: string | null;
  mode: InspectionMode;
  points?: number[][];
  state: {
    confirmed: (Partial<InspectionReport> & { operator_decision?: { approved: boolean } }) | null;
    confirmed_status: InspectionStatus | 'PENDING';
    confirmed_at: number | null;
    interlock: boolean;
    agv_permission: boolean;
    unsettled_for_sec: number;
    operator_hold: boolean;
  };
  occupancy?: OccupancyDecision | null;
  occupancy_since?: number | null;
  latest_occupancy?: OccupancyDecision | null;
  latest: Partial<InspectionReport> | null;
  overlay?: InspectionOverlayData | null;
  baseline_choice?: BaselineChoice | null;
  display?: { overlay: boolean; label: boolean; object: boolean };
  targets?: string[];
  frame_size?: number[] | null;
  last_payload: Record<string, unknown> | null;
  snapshot: string | null;
  baseline: boolean;
  baseline_count?: number;
  latency_ms: { mean: number | null; max: number | null };
};

export class InspectionApiError extends Error {
  status: number;

  constructor(message: string, status = 0) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${INSPECTION_API}${path}`, {
    cache: 'no-store',
    ...init,
    headers: init?.body ? { 'Content-Type': 'application/json', ...(init.headers || {}) } : init?.headers,
  });
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = data?.detail;
    if (typeof detail === 'string') throw new InspectionApiError(detail, response.status);
    if (Array.isArray(detail)) throw new InspectionApiError(detail.map((d: { msg?: string }) => d.msg).join('; '), response.status);
    throw new InspectionApiError(`HTTP ${response.status}`, response.status);
  }
  return data as T;
}

const camPath = (camId: string) => `/camera/${encodeURIComponent(camId)}/inspection`;

export type CaptureOptions = { label?: string; at?: number | null; targets?: string[]; force?: boolean };

export const inspectionApi = {
  captureBackground: (camId: string, ruleId: string, points: number[][], cellMm: [number, number] = [1000, 1000],
    options: CaptureOptions = {}) =>
    request<CaptureResult>(`${camPath(camId)}/capture-background`, {
      method: 'POST',
      body: JSON.stringify({
        rule_id: ruleId, points, roi_width_mm: cellMm[0], roi_height_mm: cellMm[1],
        label: options.label || '', at: options.at ?? null, target_objects: options.targets ?? null, force: !!options.force,
      }),
    }),
  test: (camId: string, points: number[][], config: InspectionConfig, ruleId?: string, targets?: string[]) =>
    request<InspectionReport>(`${camPath(camId)}/test`, {
      method: 'POST', body: JSON.stringify({ points, config, rule_id: ruleId, target_objects: targets ?? null }),
    }),
  baseline: async (camId: string, ruleId: string): Promise<BaselineInfo | null> => {
    const response = await fetch(`${INSPECTION_API}${camPath(camId)}/baseline/${encodeURIComponent(ruleId)}?optional=1`, { cache: 'no-store' });
    if (!response.ok || response.status === 204) return null;
    const stamp = Number(response.headers.get('X-Baseline-Timestamp'));
    return { url: URL.createObjectURL(await response.blob()), timestamp: Number.isFinite(stamp) ? stamp : null };
  },
  baselines: (camId: string, ruleId: string) =>
    request<BaselineList>(`${camPath(camId)}/baselines/${encodeURIComponent(ruleId)}`),
  baselineImage: async (camId: string, ruleId: string, baselineId: string): Promise<string | null> => {
    const response = await fetch(`${INSPECTION_API}${camPath(camId)}/baseline/${encodeURIComponent(ruleId)}?optional=1&bid=${encodeURIComponent(baselineId)}`,
      { cache: 'no-store' });
    if (!response.ok || response.status === 204) return null;
    return URL.createObjectURL(await response.blob());
  },
  deleteBaseline: (camId: string, ruleId: string, baselineId: string) =>
    request<{ status: string; count: number }>(`${camPath(camId)}/baseline/${encodeURIComponent(ruleId)}/${encodeURIComponent(baselineId)}`,
      { method: 'DELETE' }),
  states: () => request<{ stations: InspectionStation[] }>('/inspection/states').then(r => r.stations),
  decide: (camId: string, ruleId: string, approve: boolean, operator = '') =>
    request<{ status: string }>(`${camPath(camId)}/${encodeURIComponent(ruleId)}/decision`, {
      method: 'POST', body: JSON.stringify({ approve, operator }),
    }),
};

// ── presentation helpers ─────────────────────────────────────────────────────
export const STATUS_LABEL: Record<string, string> = {
  OK: 'OK', NG: 'NG', UNCERTAIN: 'UNCERTAIN', DETECTED: 'DETECTED', PENDING: 'ĐANG CHỜ',
};

export const OCCUPANCY_LABEL: Record<OccupancyState, string> = {
  CARFULL: 'CÓ HÀNG', EMPTY: 'TRỐNG', UNKNOWN: 'KHÔNG XÁC ĐỊNH',
};

/** Same colours as the storage-slot chips of the Monitor view (CAR FULL red, EMPTY green). */
export const occupancyColor = (state?: string | null) =>
  state === 'CARFULL' ? '#f43f5e' : state === 'EMPTY' ? '#10b981' : '#f59e0b';

export function occupancyText(decision?: OccupancyDecision | null): string {
  if (!decision) return 'ĐANG CHỜ';
  const what = decision.label ? ` · ${decision.label}${decision.confidence != null ? ` ${Math.round(decision.confidence * 100)}%` : ''}` : '';
  return `${OCCUPANCY_LABEL[decision.state] || decision.state}${what}`;
}

export function verdictSummary(report: InspectionReport): string {
  const c = report.checks || {};
  const pass = (check?: InspectionCheck) => (check ? (check.ok ? 'PASS' : 'NG') : '—');
  const m = report.measurement;
  const parts: string[] = [];
  if (m) {
    parts.push(`Chiều rộng: ${m.width_mm.toFixed(0)}mm (${pass(c.width)})`);
    parts.push(`Chiều dài: ${m.height_mm.toFixed(0)}mm (${pass(c.height)})`);
    parts.push(`Lệch tâm: ${m.offset_mm.toFixed(0)}mm (${pass(c.center_offset)})`);
    parts.push(`Góc: ${m.rotation_deg.toFixed(1)}° (${pass(c.rotation)})`);
  }
  if (report.ai) {
    parts.push(report.ai.available
      ? `AI: ${report.ai_class ? `${report.ai_class} ${Math.round((report.ai_confidence || 0) * 100)}%` : 'không thấy class nào'}`
      : 'AI: không khả dụng');
  }
  return `${parts.join(' | ')}${parts.length ? ' => ' : ''}KẾT QUẢ: ${report.status}`;
}

/** Would an article of the configured size (plus the safe margin) fit in the cell at all? */
export function articleFitsCell(config: InspectionConfig): boolean {
  const margin = 2 * config.safe_margin_mm;
  return config.width_mm + margin <= config.roi_width_mm && config.height_mm + margin <= config.roi_height_mm;
}

/** Keep the first vertex, make the traversal clockwise on screen (homography_rectifier.order_clockwise). */
export function orderClockwise(points: number[][]): number[][] {
  if (points.length !== 4) return points;
  let signed = 0;
  for (let i = 0; i < 4; i++) {
    const [x1, y1] = points[i];
    const [x2, y2] = points[(i + 1) % 4];
    signed += x1 * y2 - x2 * y1;
  }
  return signed < 0 ? [points[0], points[3], points[2], points[1]] : points;
}

/** Was this baseline captured for the drawn ROI and cell size?  (The backend ignores those that were not.) */
export function baselineMatchesRoi(entry: Pick<BaselineEntry, 'points' | 'cell_mm'>, points: number[][], cellMm: [number, number]): boolean {
  if (points.length !== 4 || !entry.points || entry.points.length !== 4) return false;
  if (Math.abs((entry.cell_mm?.[0] ?? 1000) - cellMm[0]) > 0.5 || Math.abs((entry.cell_mm?.[1] ?? 1000) - cellMm[1]) > 0.5) return false;
  const a = orderClockwise(entry.points);
  const b = orderClockwise(points);
  return a.every((p, i) => Math.abs(p[0] - b[i][0]) <= 2e-3 && Math.abs(p[1] - b[i][1]) <= 2e-3);
}

const pad = (value: number) => String(value).padStart(2, '0');

/** Epoch seconds → value of an <input type="datetime-local"> in the browser's time zone. */
export function toDatetimeLocal(epochSec: number): string {
  const d = new Date(epochSec * 1000);
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}

/** Value of an <input type="datetime-local"> (browser time zone) → epoch seconds, null when invalid. */
export function fromDatetimeLocal(value: string): number | null {
  const match = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})(?::(\d{2}))?$/.exec(value || '');
  if (!match) return null;
  const [, y, mo, d, h, mi, s] = match.map(Number);
  const date = new Date(y, mo - 1, d, h, mi, Number.isFinite(s) ? s : 0);
  return Number.isNaN(date.getTime()) ? null : date.getTime() / 1000;
}

export function formatMoment(epochSec?: number | null): string {
  if (!epochSec) return '—';
  const d = new Date(epochSec * 1000);
  return `${pad(d.getHours())}:${pad(d.getMinutes())} ${pad(d.getDate())}/${pad(d.getMonth() + 1)}/${d.getFullYear()}`;
}
