'use client';

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Activity, AlertTriangle, Check, CircleStop, Clock3, Code2, Database,
  GitBranch, GripVertical, Pause, Play, Plus, RefreshCw, Save, Send, Settings2,
  Trash2, Webhook, X, Zap,
} from 'lucide-react';
import styles from './WorkflowView.module.css';
import WorkflowGeometry from './WorkflowGeometry';

type WorkflowTab = 'deployed' | 'deploy';
type BlockGroup = 'source' | 'processing' | 'output' | 'unavailable';
type CameraResource = { id: string; name: string; status?: string; calibrated?: boolean };
type CatalogBlock = { type: string; title: string; description: string; group: BlockGroup; available: boolean };
type Catalog = {
  blocks: CatalogBlock[];
  connectors: string[];
  cameras: CameraResource[];
  labels: string[];
  recording_enabled: boolean;
  deepstream_active: boolean;
  max_active: number;
  models?: Array<{ id: string; name: string; filename: string; state: string; labels: string[]; metadata?: { shape?: number[] } }>;
};
type WorkflowNode = { id: string; type: string; config: Record<string, unknown>; x: number; y: number };
type WorkflowEdge = { source: string; target: string };
type WorkflowDefinition = { name: string; description: string; nodes: WorkflowNode[]; edges: WorkflowEdge[] };
type Pipeline = { id: string; name: string; definition: WorkflowDefinition; revision: number; actor: string; created_at: string; updated_at: string };
type Deployment = { id: string; pipeline_id: string; revision: number; definition: WorkflowDefinition; status: 'running' | 'paused' | 'stopped'; actor: string; created_at: string; updated_at: string; runtime?: RuntimeStatus };
type RuntimeStatus = { state?: string; error?: string; cameras?: Record<string, { frames?: number; events?: number; last_frame_at?: number; nodes?: Record<string, { count: number; signal: boolean; total_crossings?: number; objects?: Array<{ id?: string; label?: string; class?: string }> }> }> };
type WorkflowPayload = { pipelines: Pipeline[]; deployments: Deployment[]; runtime: { healthy: boolean; error?: string; replaced_frames?: number; dropped_events?: number; queued_actions?: number } };
type Validation = { valid: boolean; errors: string[]; warnings: string[]; order: string[]; camera_ids: string[] };
type LogEntry = { id: number; level: string; message: string; details: Record<string, unknown>; created_at: string };

type DragState = { id: string; offsetX: number; offsetY: number };

const icons: Record<string, React.ComponentType<{ size?: number; strokeWidth?: number }>> = {
  source: Activity,
  detector: Zap,
  roi: GitBranch,
  line: GitBranch,
  dwell: Clock3,
  matcher: Settings2,
  classifier: Code2,
  counter: Database,
  proximity: AlertTriangle,
  display: Activity,
  event: Database,
  webhook: Webhook,
  ppe: AlertTriangle,
  anomaly: AlertTriangle,
  attendance: Clock3,
  email: Webhook,
};

const defaultDefinition = (): WorkflowDefinition => ({
  name: 'Vision pipeline mới',
  description: 'Luồng Vision realtime dùng metadata tracking hiện có.',
  nodes: [
    { id: 'source_1', type: 'source', config: { camera_ids: [] }, x: 90, y: 160 },
    { id: 'detector_1', type: 'detector', config: { classes: ['person', 'robot', 'rack'], confidence: 0.35, model_id: '' }, x: 390, y: 160 },
    { id: 'display_1', type: 'display', config: {}, x: 700, y: 160 },
  ],
  edges: [{ source: 'source_1', target: 'detector_1' }, { source: 'detector_1', target: 'display_1' }],
});

const cloneDefinition = (definition: WorkflowDefinition): WorkflowDefinition => JSON.parse(JSON.stringify(definition));
const nodeLabel = (node: WorkflowNode, catalog: Catalog | null) => catalog?.blocks.find(block => block.type === node.type)?.title || node.type;
const groupLabel: Record<BlockGroup, string> = { source: 'Nguồn', processing: 'Xử lý', output: 'Đầu ra', unavailable: 'Chưa tích hợp' };
const formatTime = (value?: string | number) => value ? new Date(typeof value === 'number' ? value * 1000 : value).toLocaleString('vi-VN') : '—';
const formatAge = (value?: number) => value ? `${Math.max(0, (Date.now() / 1000 - value) * 1000).toFixed(0)} ms` : 'chưa có frame';
const deploymentState = (deployment: Deployment) => {
  if (deployment.status !== 'running') return deployment.status;
  if (deployment.runtime?.state === 'blocked') return 'blocked';
  const cameraIds = deployment.definition.nodes.filter(node => node.type === 'source').flatMap(node => node.config.camera_ids as string[] || []);
  const fresh = cameraIds.filter(cameraId => Date.now() / 1000 - (deployment.runtime?.cameras?.[cameraId]?.last_frame_at || 0) <= 3).length;
  return fresh === cameraIds.length && fresh ? 'running' : fresh ? 'partial' : 'waiting_metadata';
};

async function apiJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, { ...init, credentials: 'same-origin', headers: { 'Content-Type': 'application/json', ...(init?.headers || {}) } });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : data.error || `HTTP ${response.status}`);
  return data as T;
}

function makeNode(block: CatalogBlock, index: number): WorkflowNode {
  const config: Record<string, unknown> = {};
  if (block.type === 'detector') Object.assign(config, { classes: ['person', 'robot', 'rack'], confidence: 0.35, model_id: '' });
  if (block.type === 'classifier') config.classes = ['person'];
  if (block.type === 'matcher') config.labels = [];
  if (block.type === 'roi') config.points = [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]];
  if (block.type === 'line') Object.assign(config, { points: [[0.2, 0.5], [0.8, 0.5]], direction: 'both', hysteresis: 0.005 });
  if (block.type === 'dwell') config.seconds = 5;
  if (block.type === 'counter') Object.assign(config, { operator: 'gte', count: 1 });
  if (block.type === 'proximity') config.distance_m = 1.5;
  if (block.type === 'event') Object.assign(config, { severity: 'warning', message: 'Workflow phát hiện sự kiện', cooldown_seconds: 10, evidence: false });
  if (block.type === 'webhook') Object.assign(config, { connector: '', message: 'Vision workflow event', cooldown_seconds: 10, require_fms: false });
  if (block.type === 'display') config.monitor = true;
  if (block.type === 'source') config.camera_ids = [];
  return { id: `${block.type}_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`, type: block.type, config, x: 120 + (index % 3) * 280, y: Math.min(600, 90 + Math.floor(index / 3) * 150) };
}

export default function WorkflowView({ active }: { active: boolean }) {
  const [tab, setTab] = useState<WorkflowTab>('deployed');
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [payload, setPayload] = useState<WorkflowPayload | null>(null);
  const [definition, setDefinition] = useState<WorkflowDefinition>(defaultDefinition);
  const [pipelineId, setPipelineId] = useState<string | null>(null);
  const [revision, setRevision] = useState<number | null>(null);
  const [selectedId, setSelectedId] = useState('source_1');
  const [connectSource, setConnectSource] = useState<string | null>(null);
  const [filter, setFilter] = useState('');
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const [validation, setValidation] = useState<Validation | null>(null);
  const [busy, setBusy] = useState(false);
  const [logsFor, setLogsFor] = useState<string | null>(null);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [drag, setDrag] = useState<DragState | null>(null);
  const [dirty, setDirty] = useState(false);
  const [startPaused, setStartPaused] = useState(false);
  const [template, setTemplate] = useState('monitor');
  const [logRuntime, setLogRuntime] = useState<RuntimeStatus | null>(null);
  const [hasMoreLogs, setHasMoreLogs] = useState(false);
  const canvasRef = useRef<HTMLDivElement>(null);

  const load = useCallback(async (signal?: AbortSignal) => {
    if (!active) return;
    try {
      const [nextCatalog, nextPayload] = await Promise.all([
        apiJson<Catalog>('/api/backend/workflows/catalog', { signal, cache: 'no-store' }),
        apiJson<WorkflowPayload>('/api/backend/workflows', { signal, cache: 'no-store' }),
      ]);
      if (signal?.aborted) return;
      setCatalog(nextCatalog);
      setPayload(nextPayload);
    } catch (reason) {
      if (!signal?.aborted) setError(reason instanceof Error ? reason.message : String(reason));
    }
  }, [active]);

  useEffect(() => {
    if (!active) return;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      if (!document.hidden) await load(controller.signal);
      if (!controller.signal.aborted) timer = setTimeout(poll, 5000);
    };
    poll();
    return () => { controller.abort(); clearTimeout(timer); };
  }, [active, load]);

  useEffect(() => {
    if (!logsFor || !active) return;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    setLogs([]); setLogRuntime(null); setHasMoreLogs(false);
    const loadLogs = async () => {
      try {
        const data = await apiJson<{ logs: LogEntry[]; runtime: RuntimeStatus; has_more: boolean }>(`/api/backend/workflows/deployments/${logsFor}/logs`, { signal: controller.signal, cache: 'no-store' });
        if (!controller.signal.aborted) {
          setLogs(current => [...new Map([...current, ...data.logs].map(entry => [entry.id, entry])).values()].sort((left, right) => right.id - left.id).slice(0, 1000));
          setLogRuntime(data.runtime);
          if (data.has_more) setHasMoreLogs(true);
        }
      } catch (reason) {
        if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : String(reason));
      }
      if (!controller.signal.aborted) timer = setTimeout(loadLogs, 5000);
    };
    loadLogs();
    return () => { controller.abort(); clearTimeout(timer); };
  }, [active, logsFor]);

  const selectedNode = definition.nodes.find(node => node.id === selectedId) || null;
  const visibleBlocks = useMemo(() => {
    const query = filter.trim().toLocaleLowerCase();
    return (catalog?.blocks || []).filter(block => !query || `${block.title} ${block.description} ${block.type}`.toLocaleLowerCase().includes(query));
  }, [catalog, filter]);

  const edit = (change: (current: WorkflowDefinition) => WorkflowDefinition) => {
    setDefinition(current => change(cloneDefinition(current)));
    setDirty(true);
    setValidation(null);
    setMessage('');
  };

  const selectCamera = (cameraId: string) => edit(current => {
    const source = current.nodes.find(node => node.type === 'source');
    if (!source) return current;
    const cameraIds = Array.isArray(source.config.camera_ids) ? [...source.config.camera_ids as string[]] : [];
    source.config.camera_ids = cameraIds.includes(cameraId) ? cameraIds.filter(id => id !== cameraId) : [...cameraIds, cameraId];
    return current;
  });

  const addBlock = (block: CatalogBlock) => {
    if (!block.available) return;
    if (block.type === 'source' && definition.nodes.some(node => node.type === 'source')) { setError('Pipeline chỉ có một nguồn; chọn nhiều camera trong khối nguồn.'); return; }
    const node = makeNode(block, definition.nodes.length + 1);
    edit(current => { current.nodes.push(node); return current; });
    setSelectedId(node.id);
    setTab('deploy');
  };

  const removeSelected = () => {
    if (!selectedNode || selectedNode.type === 'source') return;
    edit(current => ({ ...current, nodes: current.nodes.filter(node => node.id !== selectedNode.id), edges: current.edges.filter(edge => edge.source !== selectedNode.id && edge.target !== selectedNode.id) }));
    setSelectedId(definition.nodes.find(node => node.type === 'source')?.id || '');
  };

  const connect = (targetId: string) => {
    if (!connectSource || connectSource === targetId) return;
    if (definition.nodes.find(node => node.id === targetId)?.type === 'source') { setError('Không thể nối đầu vào của khối nguồn.'); return; }
    edit(current => {
      if (!current.edges.some(edge => edge.source === connectSource && edge.target === targetId)) current.edges.push({ source: connectSource, target: targetId });
      return current;
    });
    setConnectSource(null);
  };

  const onPointerMove = (event: React.PointerEvent<HTMLDivElement>) => {
    if (!drag || !canvasRef.current) return;
    const bounds = canvasRef.current.getBoundingClientRect();
    const x = Math.max(18, Math.min(980, event.clientX - bounds.left + canvasRef.current.scrollLeft - drag.offsetX));
    const y = Math.max(18, Math.min(610, event.clientY - bounds.top + canvasRef.current.scrollTop - drag.offsetY));
    edit(current => { const node = current.nodes.find(item => item.id === drag.id); if (node) { node.x = x; node.y = y; } return current; });
  };

  const saveDraft = async () => {
    setBusy(true); setError(''); setMessage('');
    try {
      const data = await apiJson<Pipeline>(pipelineId ? `/api/backend/workflows/${pipelineId}` : '/api/backend/workflows', {
        method: pipelineId ? 'PUT' : 'POST', body: JSON.stringify({ definition, revision: pipelineId ? revision : undefined }),
      });
      setPipelineId(data.id); setRevision(data.revision); setDefinition(cloneDefinition(data.definition)); setDirty(false); setMessage(`Đã lưu bản nháp revision ${data.revision}.`); await load();
    } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); } finally { setBusy(false); }
  };

  const validateDraft = async () => {
    setBusy(true); setError(''); setMessage('');
    try { const result = await apiJson<Validation>('/api/backend/workflows/validate', { method: 'POST', body: JSON.stringify(definition) }); setValidation(result); setMessage(result.valid ? 'Pipeline hợp lệ để deploy.' : 'Pipeline còn lỗi, chưa thể deploy.'); }
    catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); } finally { setBusy(false); }
  };

  const deployDraft = async () => {
    setBusy(true); setError(''); setMessage('');
    try {
      const result = await apiJson<Validation>('/api/backend/workflows/validate', { method: 'POST', body: JSON.stringify(definition) });
      setValidation(result);
      if (!result.valid) { setError('Sửa các lỗi cấu hình trước khi deploy.'); return; }
      let savedId = pipelineId;
      let savedRevision = revision;
      if (!savedId || dirty) {
        const data = await apiJson<Pipeline>(savedId ? `/api/backend/workflows/${savedId}` : '/api/backend/workflows', {
          method: savedId ? 'PUT' : 'POST', body: JSON.stringify({ definition, revision: savedId ? revision : undefined }),
        });
        savedId = data.id; savedRevision = data.revision;
        setPipelineId(data.id); setRevision(data.revision); setDefinition(cloneDefinition(data.definition)); setDirty(false);
      }
      const deployed = await apiJson<Deployment>(`/api/backend/workflows/${savedId}/deploy`, { method: 'POST', body: JSON.stringify({ revision: savedRevision, start_paused: startPaused }) });
      setMessage(`Đã deploy ${deployed.id.slice(0, 8)} (${startPaused ? 'tạm dừng' : 'chạy ngay'}).`);
      setTab('deployed'); await load();
    } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); } finally { setBusy(false); }
  };

  const actionDeployment = async (deploymentId: string, action: 'pause' | 'resume' | 'stop') => {
    if (action === 'stop' && !window.confirm('Dừng deployment này? Tracking trên Monitor và FFmpeg vẫn tiếp tục chạy.')) return;
    setBusy(true); setError('');
    try { await apiJson(`/api/backend/workflows/deployments/${deploymentId}/action`, { method: 'POST', body: JSON.stringify({ action }) }); setMessage(`Đã thực hiện ${action}.`); await load(); }
    catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); } finally { setBusy(false); }
  };

  const loadPipeline = (pipeline: Pipeline) => {
    if (dirty && !window.confirm('Bỏ các thay đổi chưa lưu trong editor?')) return;
    const deployment = payload?.deployments.find(item => item.pipeline_id === pipeline.id && item.status !== 'stopped');
    setPipelineId(pipeline.id); setRevision(pipeline.revision); setDefinition(cloneDefinition(pipeline.definition)); setSelectedId(pipeline.definition.nodes[0]?.id || ''); setDirty(false); setValidation(null); setMessage(''); setError(''); setTab('deploy');
    if (deployment) setLogsFor(deployment.id);
  };

  const startDrag = (event: React.PointerEvent<HTMLDivElement>, node: WorkflowNode) => {
    if (connectSource) return;
    const bounds = canvasRef.current?.getBoundingClientRect();
    if (!bounds) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    setSelectedId(node.id);
    setDrag({ id: node.id, offsetX: event.clientX - bounds.left + (canvasRef.current?.scrollLeft || 0) - node.x, offsetY: event.clientY - bounds.top + (canvasRef.current?.scrollTop || 0) - node.y });
  };

  const finishDrag = () => setDrag(null);
  const sourceCameraIds = ((definition.nodes.find(node => node.type === 'source')?.config.camera_ids || []) as string[]);
  const detectorNode = definition.nodes.find(node => node.type === 'detector');
  const displayNodes = definition.nodes.filter(node => node.type === 'display');
  const chooseModel = (modelId: string) => edit(current => {
    current.nodes.filter(node => node.type === 'detector').forEach(node => {
      node.config.model_id = modelId;
      node.config.classes = catalog?.models?.find(model => model.id === modelId)?.labels || ['person', 'robot', 'rack'];
    });
    return current;
  });

  const newPipeline = () => {
    if (dirty && !window.confirm('Bỏ các thay đổi chưa lưu?')) return;
    const next = defaultDefinition();
    if (template !== 'monitor') {
      const type = template === 'intrusion' ? 'roi' : template === 'traffic' ? 'line' : template === 'identity' ? 'matcher' : 'proximity';
      const block = catalog?.blocks.find(item => item.type === type);
      if (block) {
        const filterNode = makeNode(block, 3);
        filterNode.x = 390; filterNode.y = 300;
        const eventNode = makeNode(catalog!.blocks.find(item => item.type === 'event')!, 4);
        eventNode.x = 700; eventNode.y = 300;
        next.nodes.push(filterNode, eventNode);
        next.edges.push({ source: 'detector_1', target: filterNode.id }, { source: filterNode.id, target: eventNode.id });
      }
    }
    setDefinition(next); setPipelineId(null); setRevision(null); setSelectedId('source_1'); setValidation(null); setDirty(true); setTab('deploy'); setError(''); setMessage('');
  };

  const deletePipeline = async (pipeline: Pipeline) => {
    if (!window.confirm(`Xóa pipeline “${pipeline.name}”? Nhật ký đã ghi vẫn được giữ.`)) return;
    setBusy(true);
    try {
      await apiJson(`/api/backend/workflows/${pipeline.id}?revision=${pipeline.revision}`, { method: 'DELETE' });
      if (pipelineId === pipeline.id) { setPipelineId(null); setRevision(null); setDirty(true); }
      setMessage('Đã xóa pipeline.'); await load();
    } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); }
    finally { setBusy(false); }
  };

  const loadOlderLogs = async () => {
    if (!logsFor || !logs.length) return;
    try {
      const data = await apiJson<{ logs: LogEntry[]; has_more: boolean }>(`/api/backend/workflows/deployments/${logsFor}/logs?before=${logs[logs.length - 1].id}`);
      setLogs(current => [...current, ...data.logs.filter(entry => !current.some(existing => existing.id === entry.id))]); setHasMoreLogs(data.has_more);
    } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); }
  };

  return (
    <section className={styles.shell} aria-label="Workflow Editor">
      <header className={styles.pageHeader}>
        <div className={styles.titleGroup}><div className={styles.titleIcon}><GitBranch size={22} /></div><div><p className={styles.eyebrow}>VISION PIPELINE</p><h1>Workflow Editor</h1><p className={styles.subtitle}>Metadata realtime · Detector TensorRT tùy chọn · Giữ mask SAM2 đã đăng ký trên Monitor.</p></div></div>
        <div className={styles.headerStatus}><span className={`${styles.statusDot} ${catalog?.deepstream_active ? styles.good : styles.warn}`} />{catalog ? catalog.deepstream_active ? 'DeepStream đang chạy' : 'DeepStream GST đang tắt · dùng tracking hiện tại' : 'Đang tải runtime'}<button type="button" className={styles.iconButton} onClick={() => load()} title="Làm mới"><RefreshCw size={16} /></button></div>
      </header>

      <div className={styles.tabs} role="tablist">
        <button type="button" role="tab" aria-selected={tab === 'deployed'} className={tab === 'deployed' ? styles.activeTab : ''} onClick={() => setTab('deployed')}><Activity size={16} />Pipeline đã deploy <span>{payload?.deployments.filter(item => item.status !== 'stopped').length || 0}</span></button>
        <button type="button" role="tab" aria-selected={tab === 'deploy'} className={tab === 'deploy' ? styles.activeTab : ''} onClick={() => setTab('deploy')}><Send size={16} />Chọn cách deploy</button>
      </div>

      {error && <div className={styles.errorBanner}><AlertTriangle size={16} /><span>{error}</span><button type="button" onClick={() => setError('')}><X size={15} /></button></div>}
      {message && <div className={styles.messageBanner}><Check size={16} /><span>{message}</span></div>}

      {tab === 'deployed' && <div className={styles.deployedLayout}>
        <div className={styles.panel}>
          <div className={styles.panelHeading}><div><p className={styles.panelKicker}>CONTROL PLANE</p><h2>Pipeline & bản nháp</h2></div><button type="button" className={styles.primaryButton} onClick={newPipeline}><Plus size={16} />Tạo pipeline</button></div>
          <div className={styles.runtimeBar}><span className={`${styles.statusDot} ${payload?.runtime.healthy ? styles.good : styles.bad}`} />Runtime {payload?.runtime.healthy ? 'healthy' : 'chưa sẵn sàng'}<span>• Latest-only metadata</span><span>• queue {payload?.runtime.queued_actions ?? 0}</span><span>• dropped {payload?.runtime.dropped_events ?? 0}</span></div>
          {payload?.runtime.error && <p role="alert" className={styles.badText}>{payload.runtime.error}</p>}
          <div className={styles.pipelineList}>
            {!payload?.pipelines.length && <div className={styles.empty}><GitBranch size={28} /><strong>Chưa có pipeline</strong><span>Chọn cách deploy để tạo workflow đầu tiên.</span></div>}
            {payload?.pipelines.map(pipeline => {
              const deployments = payload.deployments.filter(item => item.pipeline_id === pipeline.id);
              return <article key={pipeline.id} className={styles.pipelineCard}>
                <div className={styles.cardTitle}><div className={styles.cardIcon}><GitBranch size={17} /></div><div><h3>{pipeline.name}</h3><p>revision {pipeline.revision} · cập nhật {formatTime(pipeline.updated_at)}</p></div><span className={styles.revision}>v{pipeline.revision}</span></div>
                <p className={styles.cardDescription}>{pipeline.definition.description || 'Không có mô tả.'}</p>
                <div className={styles.cardMeta}><span><Database size={13} />{pipeline.definition.nodes.length} khối</span><span><Activity size={13} />{pipeline.definition.nodes.filter(node => node.type === 'source').flatMap(node => (node.config.camera_ids || []) as string[]).length} camera</span><span><Code2 size={13} />{deployments.length} lần deploy</span></div>
                <div className={styles.deployments}>{deployments.length ? deployments.map(deployment => <div key={deployment.id} className={styles.deploymentRow}><span title={deployment.runtime?.error || deploymentState(deployment)} className={`${styles.stateBadge} ${styles[`state_${deploymentState(deployment)}`]}`}>{deploymentState(deployment)}</span><span>v{deployment.revision} · {formatTime(deployment.created_at)}</span><div className={styles.rowActions}>{deployment.status === 'running' && <button type="button" onClick={() => actionDeployment(deployment.id, 'pause')} disabled={busy} title="Tạm dừng"><Pause size={14} /></button>}{deployment.status === 'paused' && <button type="button" onClick={() => actionDeployment(deployment.id, 'resume')} disabled={busy} title="Tiếp tục"><Play size={14} /></button>}{deployment.status !== 'stopped' && <button type="button" onClick={() => actionDeployment(deployment.id, 'stop')} disabled={busy} title="Dừng"><CircleStop size={14} /></button>}<button type="button" onClick={() => setLogsFor(logsFor === deployment.id ? null : deployment.id)} title="Nhật ký"><Activity size={14} /></button></div></div>) : <span className={styles.muted}>Chưa deploy</span>}</div>
                <div className={styles.cardFooter}><button type="button" className={styles.secondaryButton} onClick={() => loadPipeline(pipeline)}><Settings2 size={14} />Mở editor</button><button type="button" className={styles.dangerButton} onClick={() => deletePipeline(pipeline)} disabled={busy || deployments.some(item => item.status !== 'stopped')} title="Xóa pipeline đã dừng"><Trash2 size={15} /></button></div>
              </article>;
            })}
          </div>
        </div>
        <aside className={styles.logPanel}>{logsFor ? <>
          <div className={styles.panelHeading}><div><p className={styles.panelKicker}>LIVE OUTPUT & AUDIT</p><h2>Kết quả & nhật ký</h2></div><button type="button" className={styles.iconButton} onClick={() => setLogsFor(null)} title="Đóng nhật ký"><X size={16} /></button></div>
          {logRuntime?.error && <p className={styles.badText}>{logRuntime.error}</p>}
          {Object.entries(logRuntime?.cameras || {}).map(([cameraId, camera]) => <div className={styles.liveResult} key={cameraId}>
            <strong>{catalog?.cameras.find(item => item.id === cameraId)?.name || cameraId}</strong>
            <small>{Date.now() / 1000 - (camera.last_frame_at || 0) > 3 ? 'Mất metadata / đang chờ' : 'Đang nhận metadata'} · frame gần nhất cách {formatAge(camera.last_frame_at)}</small>
            <small>{camera.frames || 0} frame xử lý · {camera.events || 0} tác vụ gửi hàng đợi</small>
            {Date.now() / 1000 - (camera.last_frame_at || 0) <= 3 && Object.entries(camera.nodes || {}).map(([nodeId, node]) => <div key={nodeId}><div className={styles.outputRow}><span>{nodeId}</span><strong>{node.count}</strong><small>{node.signal ? 'pass' : 'no match'}</small></div>{node.total_crossings != null && <small>Tổng qua vạch (phiên này): {node.total_crossings}</small>}{!!node.objects?.length && <small>{node.objects.map(obj => obj.label || `${obj.class || 'object'} #${obj.id}`).join(', ')}</small>}</div>)}
          </div>)}
          {logs.length ? <div className={styles.logList}>{logs.map(log => <div key={log.id} className={styles.logEntry}><span className={`${styles.logLevel} ${log.level === 'error' ? styles.badText : ''}`}>{log.level}</span><div><strong>{log.message}</strong><small>{formatTime(log.created_at)}</small><details><summary>Chi tiết</summary><pre>{JSON.stringify(log.details, null, 2)}</pre></details></div></div>)}{hasMoreLogs && <button type="button" className={styles.secondaryButton} onClick={loadOlderLogs}>Nhật ký cũ hơn</button>}</div> : <div className={styles.empty}><Activity size={24} /><span>Chưa có log runtime.</span></div>}
        </> : <div className={styles.logEmpty}><Activity size={30} /><strong>Chọn một deployment</strong><span>Bấm biểu tượng nhật ký để xem số frame, kết quả từng khối và lịch sử lệnh.</span></div>}</aside>
      </div>}

      {tab === 'deploy' && <>
        <div className={styles.templateBar}>
          <label>Mẫu pipeline<select value={template} onChange={event => setTemplate(event.target.value)} className={styles.selectInput}><option value="monitor">Giám sát / Display</option><option value="intrusion">Xâm nhập vùng ROI</option><option value="traffic">Đếm qua vạch</option><option value="identity">Theo nhãn đã đăng ký</option><option value="safety">Khoảng cách người–robot</option></select></label>
          <button type="button" className={styles.secondaryButton} onClick={newPipeline}><Plus size={15} />Tạo từ mẫu</button>
          <label>Triển khai<select className={styles.selectInput} value={startPaused ? 'paused' : 'running'} onChange={event => setStartPaused(event.target.value === 'paused')}><option value="running">Lưu và chạy ngay trên máy biên</option><option value="paused">Lưu ở trạng thái tạm dừng</option></select></label>
          <span className={styles.fieldHelp}>Chỉ có máy biên hiện tại; không giả lập server/cluster chưa kết nối.</span>
        </div>
        <div className={styles.quickSetup} inert={busy}>
          <div><span className={styles.panelKicker}>01 / CAMERA</span><details><summary>{sourceCameraIds.length} camera được chọn</summary><div className={styles.quickCameras}>{catalog?.cameras.map(camera => <label key={camera.id}><input type="checkbox" checked={sourceCameraIds.includes(camera.id)} onChange={() => selectCamera(camera.id)} />{camera.name}<small>{camera.status}</small></label>)}</div></details></div>
          <div><label className={styles.panelKicker} htmlFor="workflow-model">02 / MODEL TENSORRT</label><select id="workflow-model" className={styles.selectInput} disabled={!detectorNode} value={String(detectorNode?.config.model_id || '')} onChange={event => chooseModel(event.target.value)}><option value="">Dùng metadata đang chạy</option>{catalog?.models?.map(model => <option key={model.id} value={model.id} disabled={model.state !== 'ready'}>{model.name} · {model.state}</option>)}</select></div>
          <div><span className={styles.panelKicker}>03 / CHỨC NĂNG</span><div className={styles.functionChips}>{definition.nodes.filter(node => !['source', 'display', 'detector'].includes(node.type)).map(node => <button key={node.id} onClick={() => setSelectedId(node.id)}>{nodeLabel(node, catalog)}</button>)}<span>Thêm khối từ thư viện bên dưới</span></div></div>
          <div><span className={styles.panelKicker}>04 / ĐẦU RA</span><label className={styles.checkRow}><input type="checkbox" disabled={!displayNodes.length} checked={displayNodes.some(node => node.config.monitor !== false)} onChange={event => edit(current => { current.nodes.filter(node => node.type === 'display').forEach(node => { node.config.monitor = event.target.checked; }); return current; })} /><span><strong>Hiển thị trên Monitor</strong><small>Không ảnh hưởng deployment trực tiếp ở Model / TensorRT hoặc workflow khác đang bật hiển thị.</small></span></label></div>
        </div>
        <div className={styles.editorLayout} inert={busy}>
        <aside className={styles.libraryPanel}>
          <div className={styles.panelHeading}><div><p className={styles.panelKicker}>BLOCK LIBRARY</p><h2>Thư viện khối</h2></div></div>
          <input className={styles.searchInput} value={filter} onChange={event => setFilter(event.target.value)} placeholder="Tìm khối..." />
          {(['source', 'processing', 'output', 'unavailable'] as BlockGroup[]).map(group => <div key={group} className={styles.blockGroup}><p className={styles.groupTitle}>{groupLabel[group]}</p>{visibleBlocks.filter(block => block.group === group).map(block => { const Icon = icons[block.type] || Settings2; return <button key={block.type} type="button" className={`${styles.libraryBlock} ${!block.available ? styles.disabledBlock : ''}`} onClick={() => addBlock(block)} disabled={!block.available}><span className={styles.blockIcon}><Icon size={15} /></span><span><strong>{block.title}</strong><small>{block.description}</small></span><Plus size={14} /></button>; })}</div>)}
          <div className={styles.libraryNote}><AlertTriangle size={14} /><span>Upload model ở Building → Model / TensorRT. Một custom detector dùng chung cho các workflow; chỉ model Ready được deploy.</span></div>
        </aside>

        <main className={styles.editorPanel}>
          <div className={styles.editorToolbar}><div><input aria-label="Tên pipeline" maxLength={120} className={styles.nameInput} value={definition.name} onChange={event => edit(current => { current.name = event.target.value; return current; })} /><p className={styles.editorHint}>{pipelineId ? `Pipeline ${pipelineId.slice(0, 8)} · revision ${revision}` : 'Bản nháp mới'}{dirty ? ' · chưa lưu' : ''} · Chọn nguồn → Nối khối → bấm đích</p><input aria-label="Mô tả pipeline" className={styles.textInput} maxLength={1000} placeholder="Mô tả nghiệp vụ" value={definition.description} onChange={event => edit(current => { current.description = event.target.value; return current; })} /></div><div className={styles.toolbarActions}><button type="button" disabled={!selectedNode || ['display', 'event', 'webhook'].includes(selectedNode.type)} className={connectSource ? styles.connectActive : styles.secondaryButton} onClick={() => setConnectSource(connectSource ? null : selectedId)}><GitBranch size={15} />{connectSource ? 'Hủy nối' : 'Nối khối'}</button><button type="button" className={styles.secondaryButton} onClick={validateDraft} disabled={busy}><Check size={15} />Kiểm tra</button><button type="button" className={styles.secondaryButton} onClick={saveDraft} disabled={busy}><Save size={15} />Lưu nháp</button><button type="button" className={styles.primaryButton} onClick={deployDraft} disabled={busy}><Send size={15} />Deploy pipeline</button></div></div>
          <div className={styles.canvas} ref={canvasRef} onPointerMove={onPointerMove} onPointerUp={finishDrag} onPointerCancel={finishDrag}>
            <div className={styles.canvasGrid} />
            <svg className={styles.edges} viewBox="0 0 1200 720" preserveAspectRatio="none" aria-hidden="true">{definition.edges.map((edge, index) => { const source = definition.nodes.find(node => node.id === edge.source); const target = definition.nodes.find(node => node.id === edge.target); if (!source || !target) return null; return <path key={`${edge.source}-${edge.target}-${index}`} d={`M ${source.x + 210} ${source.y + 44} C ${source.x + 260} ${source.y + 44}, ${target.x - 45} ${target.y + 44}, ${target.x} ${target.y + 44}`} />; })}</svg>
            {definition.nodes.map(node => { const Icon = icons[node.type] || Settings2; const isSelected = selectedId === node.id; return <div key={node.id} role="button" tabIndex={0} aria-label={`Khối ${nodeLabel(node, catalog)}`} onKeyDown={event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); if (connectSource) connect(node.id); else setSelectedId(node.id); } }} className={`${styles.graphNode} ${isSelected ? styles.selectedNode : ''} ${connectSource === node.id ? styles.connectNode : ''}`} style={{ left: node.x, top: node.y }} onPointerDown={event => startDrag(event, node)} onClick={() => connectSource ? connect(node.id) : setSelectedId(node.id)}><div className={styles.nodeHeader}><span className={styles.nodeIcon}><Icon size={14} /></span><span><small>{node.type}</small><strong>{nodeLabel(node, catalog)}</strong></span><span className={styles.nodePort} /></div><p>{nodeSummary(node)}</p></div>; })}
            {!definition.nodes.length && <div className={styles.emptyCanvas}><GitBranch size={32} /><strong>Thêm khối từ thư viện</strong><span>Graph sẽ chạy theo thứ tự đã nối.</span></div>}
            <div className={styles.canvasLegend}><span><span className={styles.legendLine} />Realtime metadata</span><span><GripVertical size={13} />Kéo thả khối</span></div>
          </div>
        </main>

        <aside className={styles.configPanel}>
          <div className={styles.panelHeading}><div><p className={styles.panelKicker}>NODE CONFIG</p><h2>Cấu hình khối</h2></div>{selectedNode && selectedNode.type !== 'source' && <button type="button" className={styles.dangerButton} onClick={removeSelected} title="Xóa khối"><Trash2 size={15} /></button>}</div>
          {selectedNode ? <NodeConfig key={`${pipelineId}:${selectedNode.id}`} node={selectedNode} catalog={catalog} sourceCameraIds={sourceCameraIds} edit={edit} selectCamera={selectCamera} /> : <div className={styles.empty}><Settings2 size={25} /><span>Chọn một khối trên graph.</span></div>}
          {selectedNode && <div className={styles.edgeList}><strong>Kết nối</strong>{definition.edges.filter(edge => edge.source === selectedId || edge.target === selectedId).map(edge => <div key={`${edge.source}-${edge.target}`}><span>{edge.source} → {edge.target}</span><button type="button" className={styles.iconButton} title="Xóa kết nối" onClick={() => edit(current => ({ ...current, edges: current.edges.filter(item => item.source !== edge.source || item.target !== edge.target) }))}><X size={13} /></button></div>)}</div>}
          {validation && <div className={`${styles.validationBox} ${validation.valid ? styles.validationGood : styles.validationBad}`}><strong>{validation.valid ? 'Sẵn sàng deploy' : 'Chưa thể deploy'}</strong>{validation.errors.map(errorItem => <span key={errorItem}>• {errorItem}</span>)}{validation.valid && validation.warnings.map(warning => <span key={warning} className={styles.warningText}>• {warning}</span>)}</div>}
        </aside>
      </div></>}
    </section>
  );
}

function nodeSummary(node: WorkflowNode) {
  const config = node.config;
  if (node.type === 'source') return `${(config.camera_ids as string[] || []).length} camera nguồn`;
  if (node.type === 'detector') return `${(config.classes as string[] || []).join(', ') || 'chưa chọn lớp'} · ${config.model_id ? 'custom TensorRT' : 'DeepStream mặc định'} · conf ${config.confidence ?? '—'}`;
  if (node.type === 'matcher') return `${(config.labels as string[] || []).length} nhãn đã chọn`;
  if (node.type === 'event') return `${config.severity || 'warning'} · ${config.evidence ? 'MP4' : 'event log'}`;
  if (node.type === 'display') return config.monitor === false ? 'chỉ xem trong workflow' : 'hiển thị trên Monitor';
  if (node.type === 'webhook') return String(config.connector || 'chưa chọn connector');
  if (node.type === 'dwell') return `${config.seconds || 5}s liên tục`;
  if (node.type === 'proximity') return `${config.distance_m || 1.5}m`;
  if (node.type === 'counter') return `${config.operator || 'gte'} ${config.count ?? 1}`;
  return 'metadata realtime';
}

function NodeConfig({ node, catalog, sourceCameraIds, edit, selectCamera }: { node: WorkflowNode; catalog: Catalog | null; sourceCameraIds: string[]; edit: (change: (current: WorkflowDefinition) => WorkflowDefinition) => void; selectCamera: (cameraId: string) => void }) {
  const setValue = (key: string, value: unknown) => edit(current => { const target = current.nodes.find(item => item.id === node.id); if (target) target.config[key] = value; return current; });
  const setTextList = (key: string, value: string) => setValue(key, value.split(',').map(item => item.trim()).filter(Boolean));
  const selectModel = (modelId: string) => edit(current => {
    const target = current.nodes.find(item => item.id === node.id);
    if (target) { target.config.model_id = modelId; target.config.classes = catalog?.models?.find(model => model.id === modelId)?.labels || ['person', 'robot', 'rack']; }
    return current;
  });
  if (node.type === 'source') return <div className={styles.configContent}><label className={styles.fieldLabel}>Camera nguồn <small>camera mới đăng ký sẽ xuất hiện sau khi tải lại</small></label><div className={styles.cameraList}>{catalog?.cameras.map(camera => <label key={camera.id} className={styles.checkRow}><input type="checkbox" checked={sourceCameraIds.includes(camera.id)} onChange={() => selectCamera(camera.id)} /><span><strong>{camera.name}</strong><small>{camera.status || 'offline'} {camera.calibrated ? '· đã calib' : '· chưa calib'}</small></span><span className={`${styles.statusDot} ${camera.status === 'online' ? styles.good : styles.warn}`} /></label>)}</div><p className={styles.fieldHelp}>Pipeline nhận frame mới nhất theo camera; không giữ queue cũ gây trễ.</p></div>;
  if (node.type === 'detector' || node.type === 'classifier') return <div className={styles.configContent}><label className={styles.fieldLabel}>Class được giữ lại</label><ClassInput key={String(node.config.model_id || 'shared')} values={(node.config.classes as string[]) || []} onChange={value => setTextList('classes', value)} />{node.type === 'detector' && <><label className={styles.fieldLabel}>Model TensorRT</label><select className={styles.selectInput} value={String(node.config.model_id || '')} onChange={event => selectModel(event.target.value)}><option value="">Metadata hiện có (không tải model)</option>{(catalog?.models || []).map(model => <option key={model.id} value={model.id} disabled={model.state !== 'ready'}>{model.name} · {model.state} · {(model.labels || []).join(', ')}</option>)}</select><p className={styles.fieldHelp}>Chỉ chọn model Ready. Khi deploy, backend khởi chạy detector DeepStream GPU riêng và chuyển metadata frame mới nhất. Build lỗi không làm dừng Monitor.</p><label className={styles.fieldLabel}>Confidence tối thiểu</label><input className={styles.numberInput} type="number" min="0.01" max="1" step="0.01" value={Number(node.config.confidence ?? .35)} onChange={event => setValue('confidence', Number(event.target.value))} /></>}<p className={styles.fieldHelp}>Detector custom hỗ trợ YOLOv8/YOLO11 detection raw; classifier vẫn lọc metadata hiện có.</p></div>;
  if (node.type === 'matcher') return <div className={styles.configContent}><label className={styles.fieldLabel}>Nhãn đã đăng ký</label><div className={styles.cameraList}>{catalog?.labels.map(label => <label key={label} className={styles.checkRow}><input type="checkbox" checked={((node.config.labels as string[]) || []).includes(label)} onChange={() => setValue('labels', ((node.config.labels as string[]) || []).includes(label) ? (node.config.labels as string[]).filter(item => item !== label) : [...((node.config.labels as string[]) || []), label])} /><span><strong>{label}</strong><small>registered identity</small></span></label>)}</div>{!catalog?.labels.length && <p className={styles.fieldHelp}>Chưa có nhãn robot/kệ đã đăng ký.</p>}</div>;
  if (node.type === 'roi' || node.type === 'line') return <div className={styles.configContent}><WorkflowGeometry key={`${node.id}:${sourceCameraIds.join(',')}`} cameraId={sourceCameraIds.length === 1 ? sourceCameraIds[0] : undefined} type={node.type} points={node.config.points as number[][] || []} onChange={points => setValue('points', points)} /><p className={styles.fieldHelp}>Mỗi vùng/đường kẻ thuộc một camera.</p>{node.type === 'line' && <><label className={styles.fieldLabel}>Hướng qua vạch</label><select className={styles.selectInput} value={String(node.config.direction || 'both')} onChange={event => setValue('direction', event.target.value)}><option value="both">Hai chiều</option><option value="a_to_b">Phía âm → phía dương</option><option value="b_to_a">Phía dương → phía âm</option></select><p className={styles.fieldHelp}>Phía được tính theo tích có hướng của đoạn điểm 1 → 2 trong hệ tọa độ ảnh (Y hướng xuống).</p></>}</div>;
  if (node.type === 'dwell') return <div className={styles.configContent}><label className={styles.fieldLabel}>Thời gian đứng (giây)</label><input className={styles.numberInput} type="number" min="0.1" max="86400" step="0.5" value={Number(node.config.seconds ?? 5)} onChange={event => setValue('seconds', Number(event.target.value))} /></div>;
  if (node.type === 'counter') return <div className={styles.configContent}><label className={styles.fieldLabel}>Điều kiện số lượng</label><select className={styles.selectInput} value={String(node.config.operator || 'gte')} onChange={event => setValue('operator', event.target.value)}><option value="gte">≥</option><option value="lte">≤</option><option value="eq">=</option></select><input className={styles.numberInput} type="number" min="0" max="10000" step="1" value={Number(node.config.count ?? 1)} onChange={event => setValue('count', Number(event.target.value))} /></div>;
  if (node.type === 'proximity') return <div className={styles.configContent}><label className={styles.fieldLabel}>Khoảng cách cảnh báo (m)</label><input className={styles.numberInput} type="number" min="0.05" max="100" step="0.05" value={Number(node.config.distance_m ?? 1.5)} onChange={event => setValue('distance_m', Number(event.target.value))} /><p className={styles.fieldHelp}>Dùng vị trí Vision chiếu lên mặt phẳng FMS đã calib. Chỉ tính bên trong vùng hiệu chuẩn; đây không phải hệ thống an toàn được chứng nhận.</p></div>;
  if (node.type === 'display') return <div className={styles.configContent}><label className={styles.checkRow}><input type="checkbox" checked={node.config.monitor !== false} onChange={event => setValue('monitor', event.target.checked)} /><span><strong>Hiển thị tracking trên Monitor</strong><small>Tắt hiển thị của workflow này, không tắt AI hoặc ghi sự kiện.</small></span></label><p className={styles.fieldHelp}>Áp dụng sau deploy. Deployment trực tiếp trong Model / TensorRT hoặc workflow khác bật hiển thị vẫn được ưu tiên. Đây là bật/tắt lớp tracking camera; bộ lọc từng nhánh được xem trong chi tiết pipeline.</p></div>;
  if (node.type === 'event') return <div className={styles.configContent}><label className={styles.fieldLabel}>Mức cảnh báo</label><select className={styles.selectInput} value={String(node.config.severity || 'warning')} onChange={event => setValue('severity', event.target.value)}><option value="info">Info</option><option value="warning">Warning</option><option value="critical">Critical</option></select><label className={styles.fieldLabel}>Nội dung</label><input className={styles.textInput} value={String(node.config.message || '')} onChange={event => setValue('message', event.target.value)} /><label className={styles.fieldLabel}>Cooldown (giây)</label><input className={styles.numberInput} type="number" min="1" max="86400" value={Number(node.config.cooldown_seconds ?? 10)} onChange={event => setValue('cooldown_seconds', Number(event.target.value))} /><label className={styles.checkRow}><input type="checkbox" checked={Boolean(node.config.evidence)} onChange={event => setValue('evidence', event.target.checked)} /><span><strong>Lưu chứng cứ MP4</strong><small>Liên kết đoạn FFmpeg đang ghi</small></span></label></div>;
  if (node.type === 'webhook') return <div className={styles.configContent}><label className={styles.fieldLabel}>Connector backend</label><select className={styles.selectInput} value={String(node.config.connector || '')} onChange={event => setValue('connector', event.target.value)}><option value="">Chọn connector</option>{catalog?.connectors.map(connector => <option key={connector} value={connector}>{connector}</option>)}</select><label className={styles.fieldLabel}>Cooldown (giây)</label><input className={styles.numberInput} type="number" min="1" max="86400" value={Number(node.config.cooldown_seconds ?? 10)} onChange={event => setValue('cooldown_seconds', Number(event.target.value))} /><label className={styles.checkRow}><input type="checkbox" checked={Boolean(node.config.require_fms)} onChange={event => setValue('require_fms', event.target.checked)} /><span><strong>Chỉ gửi khi có FMS</strong><small>Yêu cầu camera đã calib</small></span></label></div>;
  return <div className={styles.unavailable}><AlertTriangle size={20} /><strong>Khối chưa sẵn sàng</strong><span>Backend sẽ chặn deploy cho đến khi model/runtime tương ứng được tích hợp.</span></div>;
}

function ClassInput({ values, onChange }: { values: string[]; onChange: (text: string) => void }) {
  const [text, setText] = useState(values.join(', '));
  return <input className={styles.textInput} aria-label="Các lớp đối tượng" value={text} placeholder="person, robot, rack" onChange={event => { setText(event.target.value); onChange(event.target.value); }} />;
}
