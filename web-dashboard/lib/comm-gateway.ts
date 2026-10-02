// Client side of the communication gateway (System Config tab).
// The template renderer mirrors backend/core/comm_gateway/template_renderer.py
// so Live Preview updates on every keystroke without a round trip.

export const COMM_API = '/api/backend/comm';

export type Protocol = 'WEBSOCKET' | 'TCPIP' | 'MQTT' | 'HTTP_WEBHOOK' | 'MODBUS_TCP';
export type Mode = 'SERVER' | 'CLIENT';
export type CommandKind = 'send' | 'modbus_coil' | 'modbus_register' | 'modbus_read_coils' | 'modbus_read_registers';

export type CommCommand = {
  id: string;
  name: string;
  kind: CommandKind;
  payload?: string;
  payload_type?: 'json' | 'text';
  address?: number;
  value?: boolean | number;
  count?: number;
};

export type CommConfig = {
  payload_format: 'template' | 'fms_wcs_slots';
  payload_type: 'json' | 'text';
  heartbeat_sec: number;
  resync_on_connect: boolean;
  strict_path: boolean;
  line_terminator: 'lf' | 'crlf' | 'cr' | 'none';
  encoding: string;
  hex_payload: boolean;
  unit_id: number;
  connect_timeout_sec: number;
  timeout_sec: number;
  http_method: string;
  mqtt_qos: number;
  mqtt_retain: boolean;
  use_tls: boolean;
  commands: CommCommand[];
  event_commands: Record<string, string[]>;
  poll: { enabled?: boolean; kind?: 'coils' | 'registers'; address?: number; count?: number; interval_sec?: number };
};

export type CommPeer = { peer: string; path?: string; connected_at?: number; sent?: number };

export type CommRuntime = {
  state: string;
  target?: string;
  clients: number;
  peers: CommPeer[];
  sent?: number;
  failed?: number;
  dropped?: number;
  queue?: number;
  last_error?: string;
  last_error_at?: number;
  last_sent_at?: number;
  last_latency_ms?: number;
  embedded?: boolean;
  fms_slots?: number;
  invalid_slot_ids?: string[];
  poll?: { at?: number; kind?: string; address?: number; values?: (number | boolean)[] };
  inbound?: { at: number; peer: string; text: string }[];
};

export type CommChannel = {
  id: string;
  name: string;
  description: string;
  device_type: string;
  protocol: Protocol;
  mode: Mode;
  host: string;
  port: number;
  endpoint_path: string;
  auth_config: { bearer_token?: string; username?: string; password?: string; headers?: Record<string, string> };
  payload_template: string;
  trigger_events: string[];
  is_enabled: boolean;
  priority: number;
  config: CommConfig;
  runtime?: CommRuntime;
  created_at?: string;
  updated_at?: string;
};

export type TemplateVariable = { name: string; type: string; description: string; example: unknown };
export type TemplatePreset = { id: string; name: string; payload_type: 'json' | 'text'; template: string };

export type CommMeta = {
  protocols: Protocol[];
  modes: Mode[];
  device_types: string[];
  events: string[];
  variables: TemplateVariable[];
  presets: TemplatePreset[];
  app_port: number;
};

export type SlotState = {
  cam_id: string;
  rule_id: string;
  rule_name: string;
  slot_id: string;
  effective_slot_id: string;
  channel_id: string;
  enabled: boolean;
  status: 'UNKNOWN' | 'CARFULL' | 'EMPTY';
  state: string | null;
  occupant_label: string;
  overlap_ratio: number;
  updated_at: number;
  transitions: number;
  fms_compatible: boolean;
};

export type CommLog = {
  seq: number;
  at: number;
  channel_id?: string;
  event?: string;
  slot_id?: string;
  command?: string;
  ok: boolean;
  detail: string;
  latency_ms?: number;
  e2e_latency_ms?: number;
  payload?: string;
};

export type FmsStatus = {
  configured: boolean;
  connected: boolean;
  clients: number;
  channels: { id: string; name: string; enabled: boolean; clients: number; peers: CommPeer[]; endpoint: string }[];
};

export type CommStatus = {
  started: boolean;
  store_error: string;
  app_port: number;
  channels: number;
  enabled: number;
  online: number;
  fms: FmsStatus;
  slots: { total: number; carfull: number; empty: number; unknown: number };
  stats: Record<string, number>;
};

export type TestResult = {
  ok: boolean;
  detail: string;
  target?: string;
  delivered?: number;
  latency_ms?: number;
  payload?: string;
  response?: unknown;
};

export const PROTOCOL_LABELS: Record<Protocol, string> = {
  WEBSOCKET: 'WebSocket',
  TCPIP: 'TCP/IP',
  MQTT: 'MQTT',
  HTTP_WEBHOOK: 'HTTP Webhook',
  MODBUS_TCP: 'Modbus TCP',
};

export const DEVICE_TYPE_LABELS: Record<string, string> = {
  FMS_WCS: 'FMS WCS (CAMERA_AI)',
  PLC: 'PLC',
  ELEVATOR: 'Thang máy',
  CALLER: 'Call Box',
  BOXSENSOR: 'Box Sensor',
  LIGHT_TOWER: 'Đèn tháp / Còi',
  DOOR: 'Cửa cuốn / Cửa tự động',
  WMS: 'WMS / ERP',
  GENERIC: 'Thiết bị khác',
};

export const EVENT_LABELS: Record<string, string> = {
  SLOT_CARFULL: 'Ô có hàng (Car Full)',
  SLOT_EMPTY: 'Ô trống (Empty)',
  ROI_ALERT: 'Cảnh báo ROI (xâm nhập / lảng vãng / mật độ)',
};

export const COMMAND_KIND_LABELS: Record<CommandKind, string> = {
  send: 'Gửi payload (text/JSON)',
  modbus_coil: 'Modbus FC05 · Ghi coil',
  modbus_register: 'Modbus FC06 · Ghi thanh ghi',
  modbus_read_coils: 'Modbus FC01 · Đọc coil',
  modbus_read_registers: 'Modbus FC03 · Đọc thanh ghi',
};

export const DEFAULT_TEMPLATE = '{\n  "slot_id": "{slot_id}",\n  "state": "{state}"\n}';

export const DEFAULT_CONFIG: CommConfig = {
  payload_format: 'template',
  payload_type: 'json',
  heartbeat_sec: 2,
  resync_on_connect: true,
  strict_path: false,
  line_terminator: 'lf',
  encoding: 'utf-8',
  hex_payload: false,
  unit_id: 1,
  connect_timeout_sec: 3,
  timeout_sec: 3,
  http_method: 'POST',
  mqtt_qos: 0,
  mqtt_retain: false,
  use_tls: false,
  commands: [],
  event_commands: {},
  poll: {},
};

export const emptyChannel = (appPort = 8000): CommChannel => ({
  id: '',
  name: '',
  description: '',
  device_type: 'GENERIC',
  protocol: 'TCPIP',
  mode: 'CLIENT',
  host: '',
  port: 502,
  endpoint_path: '',
  auth_config: {},
  payload_template: DEFAULT_TEMPLATE,
  trigger_events: ['SLOT_CARFULL', 'SLOT_EMPTY'],
  is_enabled: true,
  priority: 1,
  config: { ...DEFAULT_CONFIG },
  runtime: { state: 'new', clients: 0, peers: [] },
  ...(appPort ? {} : {}),
});

export const isClientOnly = (protocol: Protocol) => protocol === 'MQTT' || protocol === 'HTTP_WEBHOOK' || protocol === 'MODBUS_TCP';
export const isFmsFormat = (channel: Pick<CommChannel, 'config'>) => channel.config.payload_format === 'fms_wcs_slots';
export const isFmsSlotId = (value: string) => /^-?\d{1,9}$/.test(value.trim());

export function channelAddress(channel: CommChannel): string {
  const host = channel.host || '0.0.0.0';
  switch (channel.protocol) {
    case 'WEBSOCKET': return `${channel.config.use_tls ? 'wss' : 'ws'}://${host}:${channel.port}${channel.endpoint_path || '/'}`;
    case 'HTTP_WEBHOOK': return channel.endpoint_path?.startsWith('http') ? channel.endpoint_path
      : `${channel.config.use_tls ? 'https' : 'http'}://${host}:${channel.port}${channel.endpoint_path || '/'}`;
    case 'MQTT': return `mqtt://${host}:${channel.port}/${channel.endpoint_path || ''}`;
    case 'MODBUS_TCP': return `modbus://${host}:${channel.port} · unit ${channel.config.unit_id}`;
    default: return `tcp://${host}:${channel.port}`;
  }
}

// ── API ─────────────────────────────────────────────────────────────────────
export class CommApiError extends Error {
  constructor(message: string, readonly errors: string[] = []) { super(message); }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${COMM_API}${path}`, {
    cache: 'no-store',
    ...init,
    headers: init?.body ? { 'Content-Type': 'application/json', ...(init.headers || {}) } : init?.headers,
  });
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = data?.detail;
    if (detail && typeof detail === 'object' && Array.isArray(detail.errors)) throw new CommApiError(detail.message, detail.errors);
    if (typeof detail === 'string') throw new CommApiError(detail);
    if (Array.isArray(detail)) throw new CommApiError(detail.map((d: { msg?: string }) => d.msg).join('; '));
    throw new CommApiError(`HTTP ${response.status}`);
  }
  return data as T;
}

export const commApi = {
  meta: () => request<CommMeta>('/meta'),
  status: () => request<CommStatus>('/status'),
  channels: () => request<{ channels: CommChannel[] }>('/channels').then(r => r.channels),
  create: (channel: Partial<CommChannel>) => request<CommChannel>('/channels', { method: 'POST', body: JSON.stringify(stripRuntime(channel)) }),
  update: (id: string, channel: Partial<CommChannel>) =>
    request<CommChannel>(`/channels/${encodeURIComponent(id)}`, { method: 'PUT', body: JSON.stringify(stripRuntime(channel)) }),
  remove: (id: string) => request<{ status: string }>(`/channels/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  restart: (id: string) => request<CommChannel>(`/channels/${encodeURIComponent(id)}/restart`, { method: 'POST' }),
  test: (body: { channel_id?: string; slot_id?: string; state?: string; command_id?: string; channel?: Partial<CommChannel> }) =>
    request<TestResult>('/test_dispatch', { method: 'POST', body: JSON.stringify({ ...body, channel: body.channel ? stripRuntime(body.channel) : undefined }) }),
  command: (id: string, commandId: string, context: Record<string, unknown> = {}) =>
    request<TestResult>(`/channels/${encodeURIComponent(id)}/commands/${encodeURIComponent(commandId)}`, { method: 'POST', body: JSON.stringify({ context }) }),
  modbus: (id: string, op: { kind: CommandKind; address: number; value?: unknown; count?: number }) =>
    request<TestResult>(`/channels/${encodeURIComponent(id)}/modbus`, { method: 'POST', body: JSON.stringify(op) }),
  slots: () => request<{ slots: SlotState[] }>('/slots').then(r => r.slots),
  logs: (after = 0) => request<{ logs: CommLog[] }>(`/logs?after=${after}`).then(r => r.logs),
};

function stripRuntime(channel: Partial<CommChannel>): Partial<CommChannel> {
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const { runtime, created_at, updated_at, ...rest } = channel;
  return rest;
}

// ── Template rendering (mirror of the backend) ─────────────────────────────
const IDENT = /\{([A-Za-z_][A-Za-z0-9_]*)\}/y;
const IDENT_GLOBAL = /\{([A-Za-z_][A-Za-z0-9_]*)\}/g;

export type RenderResult = { ok: true; value: string; pretty: string } | { ok: false; error: string };

const textValue = (value: unknown) => value === null || value === undefined ? '' : String(value);

export function sampleContext(slotId = '5', state = 'Car Full'): Record<string, unknown> {
  const occupied = ['car full', 'carfull', 'full', 'occupied'].includes(state.trim().toLowerCase());
  const now = Date.now();
  return {
    event: occupied ? 'SLOT_CARFULL' : 'SLOT_EMPTY',
    slot_id: slotId,
    state: occupied ? 'Car Full' : 'Empty',
    status: occupied ? 'CARFULL' : 'EMPTY',
    is_occupied: occupied,
    camera_id: 'cam_4',
    camera_name: 'Cam 4',
    rule_id: 'rule_demo',
    rule_name: 'Ô A1',
    occupant_label: occupied ? 'rack #400' : '',
    occupant_id: occupied ? 400 : null,
    overlap_ratio: 87.5,
    confidence: 0.94,
    description: '',
    timestamp_iso: new Date(now).toISOString(),
    timestamp_ms: now,
  };
}

export function renderTemplate(template: string, context: Record<string, unknown>, payloadType: 'json' | 'text' = 'json'): RenderResult {
  if (!template || !template.trim()) return { ok: false, error: 'Template rỗng.' };
  const names = new Set(Object.keys(context));
  if (payloadType === 'text') {
    const value = template.replace(IDENT_GLOBAL, (match, name: string) => names.has(name) ? textValue(context[name]) : match);
    return { ok: true, value, pretty: value };
  }
  let out = '';
  let inString = false;
  for (let i = 0; i < template.length;) {
    const ch = template[i];
    if (ch === '{') {
      IDENT.lastIndex = i;
      const match = IDENT.exec(template);
      if (match && names.has(match[1])) {
        const value = context[match[1]];
        out += inString ? JSON.stringify(textValue(value)).slice(1, -1) : JSON.stringify(value ?? null);
        i = IDENT.lastIndex;
        continue;
      }
    }
    if (inString && ch === '\\' && i + 1 < template.length) { out += template.slice(i, i + 2); i += 2; continue; }
    if (ch === '"') inString = !inString;
    out += ch;
    i += 1;
  }
  try {
    const document = JSON.parse(out);
    return { ok: true, value: JSON.stringify(document), pretty: JSON.stringify(document, null, 2) };
  } catch (error) {
    return { ok: false, error: `JSON không hợp lệ: ${error instanceof Error ? error.message : String(error)}` };
  }
}

export function fmsPreview(slotId: string, state: string): RenderResult {
  if (!isFmsSlotId(slotId)) return { ok: false, error: `Slot ID "${slotId}" phải là số nguyên — FMS WCS đọc bằng stoi().` };
  const slot = { slot_id: slotId.trim(), state: state === 'Empty' ? 'Empty' : 'Car Full' };
  const document = { ...slot, slots: [slot] };
  return { ok: true, value: JSON.stringify(document), pretty: JSON.stringify(document, null, 2) };
}

export const formatAgo = (epochSeconds?: number) => {
  if (!epochSeconds) return '—';
  const seconds = Math.max(0, Date.now() / 1000 - epochSeconds);
  if (seconds < 60) return `${Math.round(seconds)}s trước`;
  if (seconds < 3600) return `${Math.round(seconds / 60)} phút trước`;
  return new Date(epochSeconds * 1000).toLocaleString('vi-VN');
};
