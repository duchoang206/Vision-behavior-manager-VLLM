'use client';

import React, { useCallback, useEffect, useRef, useState } from 'react';
import { useCameras } from '../CameraContext';
import { useAppTheme } from '../ThemeContext';
import { RegisteredMask, validRegisteredMask, maskPath } from '../../lib/registered-mask';
import {
  Eye, EyeOff, ShieldAlert, Plus, Trash2,
  Camera as CameraIcon, RefreshCw, MapPin, Pencil,
  Tag, Crop, Bot, Package
} from 'lucide-react';

// ─── Types ────────────────────────────────────────────────────────────────────
type Rule = {
  id: string; type?: string; rule_type?: string; name: string;
  points: number[][]; target_objects?: string[]; threshold?: number; direction?: string;
  camera_points?: number[][]; fms_points?: number[][]; coordinate_space?: string;
};

type CropRect = { x: number; y: number; w: number; h: number };
type DrawSpace = 'camera' | 'fms';
type FmsLayout = {
  id?: string;
  name?: string;
  slam_map?: {
    href?: string;
    rect?: [number, number, number, number];
    pixel_size?: [number, number];
  };
  size?: {
    width?: number;
    depth?: number;
  };
};
type TargetRegistryItem = {
  label: string;
  key?: string;
  cam_id?: string;
  category?: string;
  fms_robot_id?: number;
  last_cam?: string;
  samples_count?: number;
  bbox?: number[] | null;
  has_crop_image?: boolean;
  mask_samples_count?: number;
};
type EmptyStateColors = {
  borderHard: string;
  textMuted: string;
  textLabel: string;
};

const getLayoutFrame = (layout: FmsLayout | null) => ({
  minX: 0,
  minZ: 0,
  width: layout?.size?.width ?? 26,
  depth: layout?.size?.depth ?? 18,
});

const getSlamMapRect = (layout: FmsLayout | null): [number, number, number, number] => {
  const frame = getLayoutFrame(layout);
  return layout?.slam_map?.rect ?? [frame.minX, frame.minZ, frame.minX + frame.width, frame.minZ + frame.depth];
};

const pointsAttr = (points: number[][]) => points.map(p => `${p[0]},${p[1]}`).join(' ');
const getRuleType = (rule: Rule) => rule.type || rule.rule_type || 'intrusion';
const getRuleCameraPoints = (rule: Rule) => {
  if (rule.camera_points && rule.camera_points.length > 0) return rule.camera_points;
  return (rule.coordinate_space || 'camera') === 'camera' ? (rule.points || []) : [];
};
const getRuleFmsPoints = (rule: Rule) => {
  if (rule.fms_points && rule.fms_points.length > 0) return rule.fms_points;
  return (rule.coordinate_space || 'camera') === 'fms' ? (rule.points || []) : [];
};
const isValidPointPolygon = (points?: number[][]) =>
  Array.isArray(points) && points.length >= 3 && points.every(p => Array.isArray(p) && p.length >= 2 && Number.isFinite(p[0]) && Number.isFinite(p[1]));
const isValidNormPolygon = (points?: number[][]) =>
  isValidPointPolygon(points) && points!.every(p => p[0] >= 0 && p[0] <= 1 && p[1] >= 0 && p[1] <= 1);
const isRealConfiguredRule = (rule: Rule) =>
  getRuleType(rule) !== 'occupancy'
  || (isValidNormPolygon(getRuleCameraPoints(rule)) && isValidPointPolygon(getRuleFmsPoints(rule)));
const filterRealConfiguredRules = (rules: Rule[] = []) => rules.filter(isRealConfiguredRule);

// ─── Empty State Component ────────────────────────────────────────────────────
const EmptyState = ({ icon, title, hint, colors }: { icon: React.ReactNode; title: string; hint: string; colors: EmptyStateColors }) => (
  <div style={{
    border: `1px dashed ${colors.borderHard}`, borderRadius: '10px',
    padding: '32px 20px', textAlign: 'center',
    display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '10px',
  }}>
    <div style={{ color: colors.textMuted, opacity: 0.7 }}>{icon}</div>
    <p style={{ color: colors.textLabel, fontWeight: 700, fontSize: '13px', margin: 0 }}>{title}</p>
    <p style={{ color: colors.textMuted, fontSize: '12px', fontFamily: 'monospace', margin: 0, lineHeight: 1.6 }}
      dangerouslySetInnerHTML={{ __html: hint }} />
  </div>
);

// ─── Gradient primary button (Kinpaku Gold) ──────────────────────────────────
const GradientBtn = ({
  children,
  onClick,
  disabled,
  style,
  isDark,
}: {
  children: React.ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  style?: React.CSSProperties;
  isDark: boolean;
}) => (
  <button onClick={onClick} disabled={disabled} style={{
    padding: '10px 20px', borderRadius: '8px', fontWeight: 700,
    fontSize: '13px', cursor: disabled ? 'not-allowed' : 'pointer',
    fontFamily: "'Space Grotesk', sans-serif",
    background: disabled
      ? 'var(--bg-card-alt)'
      : 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)',
    color: disabled ? 'var(--text-muted)' : '#06070a',
    border: disabled ? '1px solid var(--border)' : '1px solid rgba(245, 158, 11, 0.6)',
    boxShadow: disabled ? 'none' : (isDark ? '0 4px 18px rgba(245, 158, 11, 0.35)' : '0 2px 10px rgba(245, 158, 11, 0.25)'),
    opacity: disabled ? 0.6 : 1,
    transition: 'all 0.2s cubic-bezier(0.16, 1, 0.3, 1)',
    display: 'flex', alignItems: 'center', gap: '6px',
    ...style,
  }}>
    {children}
  </button>
);

// ─── Main Component ───────────────────────────────────────────────────────────
export default function BuildingView() {
  const { cameras, fetchCameras, deleteCamera: handleDeleteCameraCtx, updateCamera: handleUpdateCameraCtx } = useCameras();
  const { colors: C, isDark } = useAppTheme();
  const [activeTab, setActiveTab] = useState<'camera' | 'rules' | 'label'>('camera');
  const [snapshotTimestamp, setSnapshotTimestamp] = useState<number>(() => Date.now());
  const labelImageRef = useRef<HTMLImageElement | null>(null);
  const [newCamName, setNewCamName] = useState('');
  const [newCamUrl, setNewCamUrl] = useState('');
  const [editCamModal, setEditCamModal] = useState({ open: false, id: '', name: '', url: '' });
  const [showAuthModal, setShowAuthModal] = useState(false);
  const [cameraBrand, setCameraBrand] = useState('hikvision');
  const [authUsername, setAuthUsername] = useState('');
  const [authPassword, setAuthPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [selectedCamId, setSelectedCamId] = useState('');
  const [rulesByCam, setRulesByCam] = useState<Record<string, Rule[]>>({});
  const [currentRuleType, setCurrentRuleType] = useState<string>('intrusion');
  const [currentRuleName, setCurrentRuleName] = useState('');
  const [currentThreshold, setCurrentThreshold] = useState(15);
  const [currentPoints, setCurrentPoints] = useState<number[][]>([]);
  const [currentFmsPoints, setCurrentFmsPoints] = useState<number[][]>([]);
  const [ruleDrawSpace, setRuleDrawSpace] = useState<DrawSpace>('camera');
  const [editingRuleId, setEditingRuleId] = useState<string | null>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [availableClasses, setAvailableClasses] = useState<string[]>([]);
  const [currentTargetClasses, setCurrentTargetClasses] = useState<string[]>(['robot', 'rack']);
  const [fmsLayout, setFmsLayout] = useState<FmsLayout | null>(null);
  const [labelCamId, setLabelCamId] = useState('');
  const [labelName, setLabelName] = useState('Robot_9001');
  const [labelCategory, setLabelCategory] = useState<'robot' | 'rack' | 'person'>('robot');
  const [labelCrop, setLabelCrop] = useState<CropRect | null>(null);
  const [labelMask, setLabelMask] = useState<RegisteredMask | null>(null);
  const [maskPoints, setMaskPoints] = useState<{ point: number[]; label: number }[]>([]);
  const [maskTool, setMaskTool] = useState<'box' | 'add' | 'remove'>('box');
  const [maskBusy, setMaskBusy] = useState(false);
  const [maskAvailable, setMaskAvailable] = useState(false);
  const [labelFrameReady, setLabelFrameReady] = useState(false);
  const maskRequest = useRef(0);
  const [cropStart, setCropStart] = useState<{ x: number; y: number } | null>(null);
  const [isSavingLabel, setIsSavingLabel] = useState(false);
  const [labelStatus, setLabelStatus] = useState('');
  const [registeredTargets, setRegisteredTargets] = useState<TargetRegistryItem[]>([]);

  useEffect(() => {
    if (activeTab !== 'label') return;
    let cancelled = false;
    fetch('/api/backend/registry/mask/status').then(response => response.json()).then(data => {
      if (!cancelled) setMaskAvailable(Boolean(data.available));
    }).catch(() => { if (!cancelled) setMaskAvailable(false); });
    return () => { cancelled = true; maskRequest.current += 1; };
  }, [activeTab]);

  // ─── Shared element styles derived from dynamic theme ───────────────────────
  const inputStyle: React.CSSProperties = {
    width: '100%', padding: '10px 14px',
    borderRadius: '8px', border: `1px solid ${C.borderHard}`,
    fontSize: '13px', outline: 'none',
    background: C.elevated, color: C.textPrimary,
    fontFamily: "'Space Grotesk', sans-serif",
    transition: 'border-color 0.2s, box-shadow 0.2s',
    caretColor: C.accentL,
  };
  const selectStyle: React.CSSProperties = {
    ...inputStyle, appearance: 'none', WebkitAppearance: 'none', cursor: 'pointer',
  };
  const labelStyle: React.CSSProperties = {
    display: 'block', fontSize: '11px', color: C.textLabel,
    marginBottom: '6px', fontFamily: 'JetBrains Mono, monospace',
    letterSpacing: '0.06em', textTransform: 'uppercase', fontWeight: 600,
  };
  const cardStyle: React.CSSProperties = {
    background: C.card, borderRadius: '12px',
    border: `1px solid ${C.border}`, padding: '20px',
    boxShadow: isDark ? '0 4px 20px rgba(0,0,0,0.4)' : '0 1px 3px rgba(0,0,0,0.05), 0 1px 2px rgba(0,0,0,0.03)',
  };
  const focusHandlers = {
    onFocus: (e: React.FocusEvent<HTMLInputElement | HTMLSelectElement>) => {
      e.target.style.borderColor = C.accent;
      e.target.style.boxShadow = `0 0 0 3px ${C.accentGlow}`;
    },
    onBlur: (e: React.FocusEvent<HTMLInputElement | HTMLSelectElement>) => {
      e.target.style.borderColor = C.borderHard;
      e.target.style.boxShadow = 'none';
    },
  };

  const fetchRegisteredTargets = useCallback((camId?: string) => {
    const url = camId ? `/api/backend/registry/targets?cam_id=${encodeURIComponent(camId)}` : '/api/backend/registry/targets';
    fetch(url)
      .then(r => r.ok ? r.json() : null)
      .then(d => { if (Array.isArray(d?.targets)) setRegisteredTargets(d.targets); })
      .catch(() => {});
  }, []);

  const resetLabelMaskState = (newFrame = true) => {
    maskRequest.current += 1;
    setLabelMask(null);
    setMaskPoints([]);
    setMaskTool('box');
    setMaskBusy(false);
    if (newFrame) setLabelFrameReady(false);
    setCropStart(null);
    setLabelCrop(null);
  };

  useEffect(() => {
    if (activeTab === 'label') fetchRegisteredTargets(labelCamId);
  }, [activeTab, labelCamId, fetchRegisteredTargets]);

  useEffect(() => {
    fetch('/api/backend/model/classes').then(r => r.json())
      .then(d => { if (d.classes?.length > 0) setAvailableClasses(d.classes); }).catch(() => {});
  }, []);

  useEffect(() => {
    let mounted = true;
    const sources = ['/api/backend/fms/layout', '/maps/warehouse_layout.json'];
    (async () => {
      for (const src of sources) {
        try {
          const res = await fetch(src);
          if (!res.ok) continue;
          const data = await res.json();
          if (mounted && data && !data.error) {
            setFmsLayout(data);
            return;
          }
        } catch {}
      }
    })();
    return () => { mounted = false; };
  }, []);

  useEffect(() => {
    const firstCameraId = cameras[0]?.id;
    if (!firstCameraId) return;
    queueMicrotask(() => {
      setSelectedCamId(prev => prev || firstCameraId);
      setLabelCamId(prev => prev || firstCameraId);
    });
  }, [cameras]);
  useEffect(() => {
    if (selectedCamId) {
      fetch(`/api/backend/camera/${selectedCamId}/rules`).then(r => r.ok ? r.json() : null)
        .then(d => { if (Array.isArray(d?.rules)) setRulesByCam(p => ({ ...p, [selectedCamId]: filterRealConfiguredRules(d.rules) })); }).catch(() => {});
    }
  }, [selectedCamId]);

  const handleAddCamera = (e: React.FormEvent) => { e.preventDefault(); if (!newCamName || !newCamUrl) return; setShowAuthModal(true); };

  const handleApplyCameraAuth = async () => {
    const cleanIpOrUrl = newCamUrl.trim();
    let fullRtspUrl = '';
    if (cleanIpOrUrl.startsWith('rtsp://') || cleanIpOrUrl.startsWith('http://') || cleanIpOrUrl.startsWith('https://')) {
      fullRtspUrl = cleanIpOrUrl;
    } else {
      const uEnc = authUsername ? encodeURIComponent(authUsername) : '';
      const pEnc = authPassword ? encodeURIComponent(authPassword) : '';
      const auth = (uEnc && pEnc) ? `${uEnc}:${pEnc}@` : (uEnc ? `${uEnc}@` : '');
      let hp = cleanIpOrUrl.replace(/^https?:\/\//, '').replace(/^rtsp:\/\//, '');
      let ps = '';
      if (hp.includes('/')) { const i = hp.indexOf('/'); ps = hp.slice(i); hp = hp.slice(0, i); }
      if (!hp.includes(':')) hp = `${hp}:554`;
      fullRtspUrl = cameraBrand === 'custom' ? `rtsp://${auth}${hp}${ps}` :
        cameraBrand === 'dahua' ? `rtsp://${auth}${hp}/cam/realmonitor?channel=1&subtype=0` :
        `rtsp://${auth}${hp}/Streaming/Channels/101`;
    }
    try {
      const res = await fetch('/api/backend/camera/add', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: newCamName.trim(), rtsp_url: fullRtspUrl })
      });
      if (res.ok) {
        setNewCamName(''); setNewCamUrl(''); setAuthUsername(''); setAuthPassword('');
        setShowAuthModal(false); fetchCameras();
      } else { const e = await res.json().catch(() => null); alert(e?.detail || `Lỗi HTTP ${res.status}`); }
    } catch (e: unknown) {
      alert(`Lỗi Backend: ${e instanceof Error ? e.message : String(e)}`);
    }
  };

  const handleDeleteCamera = async (id: string) => {
    if (confirm('Xóa Camera này khỏi hệ thống?')) await handleDeleteCameraCtx(id);
  };
  const handleSaveEditCamera = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!editCamModal.name || !editCamModal.url) return;
    const ok = await handleUpdateCameraCtx(editCamModal.id, editCamModal.name.trim(), editCamModal.url.trim());
    if (ok) setEditCamModal({ open: false, id: '', name: '', url: '' }); else alert('Lỗi cập nhật');
  };

  const persistCameraRules = async (camId: string, rules: Rule[]) => {
    await fetch(`/api/backend/camera/${camId}/rules`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ rules })
    });
  };

  const resetRuleDraft = () => {
    setCurrentPoints([]);
    setCurrentFmsPoints([]);
    setIsDrawing(false);
    setCurrentRuleName('');
    setEditingRuleId(null);
    setRuleDrawSpace('camera');
  };

  const handleSaveCurrentRule = async () => {
    const min = currentRuleType === 'tripwire' ? 2 : 3;
    const isStorageSlot = currentRuleType === 'occupancy';
    if (!selectedCamId) { alert('Chọn camera trước khi lưu quy tắc.'); return; }
    if (isStorageSlot && (currentPoints.length < 3 || currentFmsPoints.length < 3)) {
      alert('Ô chứa hàng cần vẽ đủ polygon trên cả Camera và FMS Map.');
      return;
    }
    if (!isStorageSlot && currentPoints.length < min) { alert(`Cần ít nhất ${min} điểm!`); return; }
    const rules = rulesByCam[selectedCamId] || [];
    const ruleId = editingRuleId || `rule_${Date.now().toString().slice(-4)}`;
    const nr: Rule = {
      id: ruleId, type: currentRuleType,
      name: currentRuleName || `${currentRuleType.toUpperCase()} #${rules.length + 1}`,
      points: isStorageSlot ? currentFmsPoints : currentPoints,
      camera_points: isStorageSlot ? currentPoints : currentPoints,
      fms_points: isStorageSlot ? currentFmsPoints : [],
      coordinate_space: isStorageSlot ? 'hybrid' : 'camera',
      target_objects: currentTargetClasses,
      threshold: currentThreshold
    };
    const updated = editingRuleId
      ? rules.map(rule => rule.id === editingRuleId ? nr : rule)
      : [...rules, nr];
    setRulesByCam({ ...rulesByCam, [selectedCamId]: updated });
    resetRuleDraft();
    try { await persistCameraRules(selectedCamId, updated); } catch {}
  };

  const handleEditRule = (rule: Rule) => {
    const rType = getRuleType(rule);
    setEditingRuleId(rule.id);
    setCurrentRuleType(rType);
    setCurrentRuleName(rule.name || '');
    setCurrentThreshold(rule.threshold ?? 15);
    setCurrentTargetClasses(rule.target_objects?.length ? rule.target_objects : ['robot', 'rack']);
    setCurrentPoints(getRuleCameraPoints(rule));
    setCurrentFmsPoints(getRuleFmsPoints(rule));
    setRuleDrawSpace('camera');
    setIsDrawing(false);
  };

  const handleDeleteRule = async (ruleId: string) => {
    const updated = (rulesByCam[selectedCamId] || []).filter(r => r.id !== ruleId);
    setRulesByCam({ ...rulesByCam, [selectedCamId]: updated });
    if (editingRuleId === ruleId) resetRuleDraft();
    try {
      const res = await fetch(`/api/backend/camera/${selectedCamId}/rules/${encodeURIComponent(ruleId)}`, { method: 'DELETE' });
      if (!res.ok) await persistCameraRules(selectedCamId, updated);
    } catch {
      try { await persistCameraRules(selectedCamId, updated); } catch {}
    }
  };

  const handleClearRules = async () => {
    if (!selectedCamId || !confirm('Xóa toàn bộ quy tắc của camera này?')) return;
    setRulesByCam({ ...rulesByCam, [selectedCamId]: [] });
    resetRuleDraft();
    try { await persistCameraRules(selectedCamId, []); } catch {}
  };
  const getNormPoint = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    return {
      x: Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width)),
      y: Math.min(1, Math.max(0, (e.clientY - rect.top) / rect.height)),
    };
  };

  const updateLabelCrop = (start: { x: number; y: number }, end: { x: number; y: number }) => {
    const x = Math.min(start.x, end.x);
    const y = Math.min(start.y, end.y);
    const w = Math.abs(end.x - start.x);
    const h = Math.abs(end.y - start.y);
    setLabelCrop(w > 0.005 && h > 0.005 ? { x, y, w, h } : null);
  };

  const getLabelFrame = () => {
    const image = labelImageRef.current;
    if (!labelFrameReady || !image || !image.complete || !image.naturalWidth) throw new Error('Ảnh chưa tải xong.');
    const frameCanvas = document.createElement('canvas');
    frameCanvas.width = image.naturalWidth;
    frameCanvas.height = image.naturalHeight;
    const context = frameCanvas.getContext('2d');
    if (!context) throw new Error('Không đọc được ảnh camera.');
    context.drawImage(image, 0, 0);
    return frameCanvas.toDataURL('image/jpeg', 0.95);
  };

  const generateLabelMask = async (corrections = maskPoints) => {
    if (!labelCrop || labelCategory === 'person' || !labelName.trim()) return;
    const requestId = ++maskRequest.current;
    setMaskBusy(true);
    setLabelMask(null);
    try {
      const response = await fetch('/api/backend/registry/mask/preview', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ cam_id: labelCamId, label: labelName.trim(), category: labelCategory,
          frame_image: getLabelFrame(), bbox: [labelCrop.x, labelCrop.y, labelCrop.w, labelCrop.h],
          points: corrections.map(item => item.point), point_labels: corrections.map(item => item.label) }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Không tạo được mask.');
      if (!validRegisteredMask(data.mask)) throw new Error('Mask trả về không hợp lệ.');
      if (requestId !== maskRequest.current) return;
      setLabelMask(data.mask);
      setLabelStatus('Kiểm tra mask. Dùng + Vật / − Nền để chỉnh, sau đó lưu góc nhìn.');
    } catch (error) {
      if (requestId === maskRequest.current) setLabelStatus(`Lỗi mask: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      if (requestId === maskRequest.current) setMaskBusy(false);
    }
  };

  const handleSaveLabelCrop = async () => {
    const img = labelImageRef.current;
    if (!img || !labelCrop || !labelName.trim() || !labelCamId || !labelFrameReady || maskBusy) return;
    if (labelCategory !== 'person' && !validRegisteredMask(labelMask)) {
      setLabelStatus('Lỗi: hãy tạo và kiểm tra mask trước khi lưu nhãn robot/kệ.');
      return;
    }

    const canvas = document.createElement('canvas');
    canvas.width = Math.max(1, Math.round(labelCrop.w * img.naturalWidth));
    canvas.height = Math.max(1, Math.round(labelCrop.h * img.naturalHeight));
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.drawImage(
      img,
      labelCrop.x * img.naturalWidth,
      labelCrop.y * img.naturalHeight,
      labelCrop.w * img.naturalWidth,
      labelCrop.h * img.naturalHeight,
      0,
      0,
      canvas.width,
      canvas.height
    );

    setIsSavingLabel(true);
    setLabelStatus('');
    try {
      const res = await fetch('/api/backend/registry/register-crop', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          cam_id: labelCamId,
          label: labelName.trim(),
          category: labelCategory,
          crop_image: canvas.toDataURL('image/jpeg', 0.9),
          bbox: [labelCrop.x, labelCrop.y, labelCrop.w, labelCrop.h],
          ...(labelCategory !== 'person' && labelMask ? { mask: labelMask, frame_image: getLabelFrame() } : {}),
        }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => null);
        throw new Error(err?.detail || `HTTP ${res.status}`);
      }

      const payload = await res.json().catch(() => null);
      setLabelStatus(payload?.message || `Đã gán nhãn ${labelName.trim()} và đưa vào registry tracking.`);
      setLabelCrop(null);
      setLabelMask(null);
      setMaskPoints([]);
      fetchRegisteredTargets(labelCamId);
    } catch (e: unknown) {
      setLabelStatus(`Lỗi gán nhãn: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setIsSavingLabel(false);
    }
  };

  const handleDeleteRegisteredTarget = async (label: string, camId?: string) => {
    const targetCamId = camId || labelCamId;
    const camName = cameras.find(c => c.id === targetCamId)?.name || 'Camera này';
    if (!confirm(`Xóa nhãn '${label}' trên ${camName}? (Thao tác này chỉ xóa trên ${camName}, không ảnh hưởng đến camera khác)`)) return;
    try {
      const res = await fetch(`/api/backend/camera/${encodeURIComponent(targetCamId)}/registry/target/${encodeURIComponent(label)}`, { method: 'DELETE' });
      if (res.ok) fetchRegisteredTargets(targetCamId);
    } catch {}
  };

  const handleUseRegisteredTarget = (target: TargetRegistryItem) => {
    setLabelName(target.label);
    if (target.category === 'robot' || target.category === 'rack' || target.category === 'person') {
      setLabelCategory(target.category);
    }
    if (target.last_cam && cameras.some(c => c.id === target.last_cam)) {
      setLabelCamId(target.last_cam);
    }
    resetLabelMaskState();
    setLabelStatus('');
    setSnapshotTimestamp(prev => prev + 1);
  };

  // Tab style helper
  const tabStyle = (active: boolean): React.CSSProperties => ({
    padding: '10px 20px', borderRadius: '9px', cursor: 'pointer',
    fontWeight: active ? 700 : 500, fontSize: '13px',
    fontFamily: "'Space Grotesk', sans-serif",
    border: active ? `1px solid ${C.accentBorder}` : `1px solid ${C.border}`,
    background: active ? C.accentDim : 'transparent',
    color: active ? C.accentL : C.textSub,
    boxShadow: active ? `0 0 16px ${C.accentGlow}` : 'none',
    transition: 'all 0.2s',
    display: 'flex', alignItems: 'center', gap: '7px',
  });

  // ── Modal Overlay ──────────────────────────────────────────────────────────
  const modalOverlay: React.CSSProperties = {
    position: 'fixed', inset: 0,
    backgroundColor: isDark ? 'rgba(0,0,0,0.85)' : 'rgba(15,23,42,0.4)',
    backdropFilter: 'blur(6px)', WebkitBackdropFilter: 'blur(6px)',
    zIndex: 1000, display: 'flex', alignItems: 'center', justifyContent: 'center',
  };
  const modalCard: React.CSSProperties = {
    background: C.card, padding: '28px', borderRadius: '14px',
    border: `1px solid ${C.borderHard}`,
    boxShadow: isDark ? `0 32px 80px rgba(0,0,0,0.7), 0 0 0 1px ${C.accentBorder}` : '0 20px 60px rgba(0,0,0,0.15)',
  };
  const isStorageRule = currentRuleType === 'occupancy';
  const activeDrawPoints = ruleDrawSpace === 'fms' ? currentFmsPoints : currentPoints;
  const ruleMinPoints = currentRuleType === 'tripwire' ? 2 : 3;
  const saveRuleDisabled = isStorageRule
    ? currentPoints.length < 3 || currentFmsPoints.length < 3
    : currentPoints.length < ruleMinPoints;
  const slamMapRect = getSlamMapRect(fmsLayout);
  const slamMapWidth = Math.max(0.001, slamMapRect[2] - slamMapRect[0]);
  const slamMapHeight = Math.max(0.001, slamMapRect[3] - slamMapRect[1]);
  const slamPixelSize = fmsLayout?.slam_map?.pixel_size ?? [465, 301];
  const slamMapHref = fmsLayout?.slam_map?.href || '/maps/fms_map.png';

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div style={{ backgroundColor: C.bg, minHeight: 'calc(100vh - 80px)', padding: '24px', transition: 'background-color 0.2s ease' }}>
      <div style={{ maxWidth: '1400px', margin: '0 auto' }}>

        {/* Top Tabs */}
        <div style={{
          display: 'flex', gap: '8px',
          borderBottom: `1px solid ${C.border}`,
          paddingBottom: '18px', marginBottom: '24px',
        }}>
          <button onClick={() => setActiveTab('camera')} style={tabStyle(activeTab === 'camera')}>
            <CameraIcon size={14} /> Quản lý Camera ({cameras.length})
          </button>
          <button onClick={() => setActiveTab('rules')} style={tabStyle(activeTab === 'rules')}>
            <ShieldAlert size={14} /> Phân tích Hành vi (ROI / Tripwire)
          </button>
          <button onClick={() => setActiveTab('label')} style={tabStyle(activeTab === 'label')}>
            <Tag size={14} /> Label
          </button>
        </div>

        {/* ── TAB 1: CAMERA MANAGEMENT ───────────────────────────────────── */}
        {activeTab === 'camera' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>

            {/* Add Camera Form */}
            <div style={cardStyle}>
              <h2 style={{ fontSize: '15px', fontWeight: 700, color: C.textPrimary, marginBottom: '18px', letterSpacing: '-0.01em' }}>
                Đăng ký Luồng Camera Mới
              </h2>
              <form onSubmit={handleAddCamera} style={{ display: 'flex', gap: '12px', alignItems: 'flex-end' }}>
                <div style={{ flex: 1 }}>
                  <label style={labelStyle}>Tên Camera</label>
                  <input type="text" value={newCamName} onChange={e => setNewCamName(e.target.value)}
                    placeholder="VD: Cổng chính, Kho A..." style={inputStyle} {...focusHandlers} />
                </div>
                <div style={{ flex: 2 }}>
                  <label style={labelStyle}>IP Address hoặc RTSP URL</label>
                  <input type="text" value={newCamUrl} onChange={e => setNewCamUrl(e.target.value)}
                    placeholder="192.168.1.100 hoặc rtsp://..." style={inputStyle} {...focusHandlers} />
                </div>
                <GradientBtn isDark={isDark} style={{ whiteSpace: 'nowrap', alignSelf: 'flex-end', height: '42px' }}>
                  <Plus size={15} /> Thêm Camera
                </GradientBtn>
              </form>
            </div>

            {/* Camera Table */}
            {cameras.length === 0 ? (
              <EmptyState
                colors={C}
                icon={<CameraIcon size={36} strokeWidth={1.2} />}
                title="Chưa có camera nào được đăng ký"
                hint={`Điền tên và IP/RTSP URL vào form ở trên, sau đó bấm <span style='color:${C.accentL};font-weight:600'>+ Thêm Camera</span>`}
              />
            ) : (
              <div style={{ background: C.card, borderRadius: '12px', border: `1px solid ${C.border}`, overflow: 'hidden' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
                  <thead>
                    <tr style={{ background: C.cardAlt, borderBottom: `1px solid ${C.border}` }}>
                      {['ID', 'TÊN CAMERA', 'RTSP URL', 'TRẠNG THÁI', 'THAO TÁC'].map((h, i) => (
                        <th key={h} style={{
                          padding: '13px 16px', fontSize: '11px', fontWeight: 700,
                          color: C.textLabel, fontFamily: 'JetBrains Mono, monospace',
                          letterSpacing: '0.06em', textAlign: i === 4 ? 'right' : 'left',
                        }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {cameras.map(c => (
                      <tr key={c.id} style={{ borderBottom: `1px solid ${C.borderSubtle}`, transition: 'background 0.15s' }}
                        onMouseEnter={e => (e.currentTarget.style.background = C.accentDim)}
                        onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}>
                        <td style={{ padding: '14px 16px', color: C.textMuted, fontFamily: 'monospace', fontSize: '12px' }}>#{c.id}</td>
                        <td style={{ padding: '14px 16px', fontWeight: 700, color: C.textPrimary, fontSize: '13px' }}>{c.name}</td>
                        <td style={{ padding: '14px 16px', color: C.textSub, fontFamily: 'monospace', fontSize: '11px', maxWidth: '260px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{c.rtsp_url}</td>
                        <td style={{ padding: '14px 16px' }}>
                          <span style={{
                            background: C.cyanDim, color: C.cyanL,
                            padding: '4px 10px', borderRadius: '20px',
                            fontSize: '11px', fontWeight: 700, fontFamily: 'monospace',
                            border: `1px solid ${C.cyanBorder}`,
                            boxShadow: isDark ? '0 0 8px rgba(6,182,212,0.15)' : 'none',
                          }}>● Live · WebRTC</span>
                        </td>
                        <td style={{ padding: '14px 16px', textAlign: 'right' }}>
                          <div style={{ display: 'flex', gap: '8px', justifyContent: 'flex-end' }}>
                            <button onClick={() => setEditCamModal({ open: true, id: c.id, name: c.name, url: c.rtsp_url })}
                              style={{
                                background: C.accentDim, border: `1px solid ${C.accentBorder}`,
                                color: C.accentL, padding: '6px 14px', borderRadius: '6px',
                                cursor: 'pointer', fontWeight: 600, fontSize: '12px',
                                fontFamily: "'Space Grotesk', sans-serif", transition: 'all 0.15s',
                              }}>Sửa</button>
                            <button onClick={() => handleDeleteCamera(c.id)}
                              style={{
                                background: C.roseDim, border: `1px solid ${C.roseBorder}`,
                                color: C.rose, padding: '6px 14px', borderRadius: '6px',
                                cursor: 'pointer', fontWeight: 600, fontSize: '12px',
                                fontFamily: "'Space Grotesk', sans-serif", transition: 'all 0.15s',
                              }}>Xóa</button>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}

        {/* ── TAB 2: RULES ───────────────────────────────────────────────── */}
        {activeTab === 'rules' && (
          <div style={{ display: 'grid', gridTemplateColumns: '300px 1fr', gap: '20px' }}>

            {/* Left Panel */}
            <div style={{ ...cardStyle, display: 'flex', flexDirection: 'column', gap: '14px' }}>

              <div>
                <label style={labelStyle}>Chọn Camera</label>
                <select value={selectedCamId} onChange={e => setSelectedCamId(e.target.value)}
                  style={selectStyle} {...focusHandlers}>
                  {cameras.map(c => <option key={c.id} value={c.id}>{c.name}</option>)}
                </select>
              </div>

	              <div>
	                <label style={labelStyle}>Loại Quy Tắc Hành Vi</label>
	                <select value={currentRuleType}
	                  onChange={e => {
	                    const nextType = e.target.value;
	                    setCurrentRuleType(nextType);
	                    setCurrentPoints([]);
	                    setCurrentFmsPoints([]);
	                    setRuleDrawSpace('camera');
	                    setIsDrawing(false);
	                  }}
	                  style={selectStyle} {...focusHandlers}>
                  <option value="occupancy">Ô chứa hàng / Vị trí lưu kho (Storage Slot ROI)</option>
                  <option value="intrusion">Vùng cấm xâm nhập (Intrusion ROI)</option>
                  <option value="tripwire">Vạch ảo 2 chiều (Tripwire Line)</option>
                  <option value="dwell_time">Lảng vãng / Dừng chờ (Dwell Time)</option>
                  <option value="density">Mật độ đám đông (Crowd Density)</option>
	                </select>
	              </div>

	              {isStorageRule && (
	                <div>
	                  <label style={labelStyle}>Mặt phẳng tọa độ</label>
	                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
	                    {([
	                      ['camera', 'Camera', <CameraIcon key="camera" size={14} />, currentPoints.length],
	                      ['fms', 'FMS Map', <MapPin key="fms" size={14} />, currentFmsPoints.length],
	                    ] as const).map(([space, label, icon, count]) => {
	                      const active = ruleDrawSpace === space;
	                      return (
	                        <button key={space} type="button" onClick={() => { setRuleDrawSpace(space); setIsDrawing(false); }} style={{
	                          padding: '9px 8px', borderRadius: '8px', cursor: 'pointer',
	                          display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px',
	                          background: active ? C.accentDim : 'transparent',
	                          border: `1px solid ${active ? C.accentBorder : C.borderHard}`,
	                          color: active ? C.accentL : C.textSub,
	                          fontSize: '12px', fontWeight: 700,
	                        }}>
	                          {icon}{label} · {count}
	                        </button>
	                      );
	                    })}
	                  </div>
	                </div>
	              )}

              <div>
                <label style={labelStyle}>Tên Khu Vực / Vạch</label>
                <input type="text" value={currentRuleName} onChange={e => setCurrentRuleName(e.target.value)}
                  placeholder="VD: Cửa thoát hiểm..." style={inputStyle} {...focusHandlers} />
              </div>

              <div>
                <label style={labelStyle}>Đối Tượng Áp Dụng</label>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                  {availableClasses.map(c => {
                    const active = currentTargetClasses.includes(c);
                    return (
                      <div key={c} onClick={() => {
                        if (active) setCurrentTargetClasses(currentTargetClasses.filter(x => x !== c));
                        else setCurrentTargetClasses([...currentTargetClasses, c]);
                      }} style={{
                        padding: '4px 12px', borderRadius: '20px', fontSize: '12px', cursor: 'pointer',
                        border: active ? `1px solid ${C.accentBorder}` : `1px solid ${C.borderHard}`,
                        background: active ? C.accentDim : 'transparent',
                        color: active ? C.accentL : C.textSub,
                        fontWeight: active ? 700 : 400,
                        transition: 'all 0.15s',
                      }}>{c}</div>
                    );
                  })}
                </div>
              </div>

              {(currentRuleType === 'dwell_time' || currentRuleType === 'density') && (
                <div>
                  <label style={labelStyle}>
                    {currentRuleType === 'dwell_time' ? 'Ngưỡng dừng (giây)' : 'Số người tối đa'}
                  </label>
                  <input type="number" value={currentThreshold}
                    onChange={e => setCurrentThreshold(Number(e.target.value))}
                    style={inputStyle} {...focusHandlers} />
                </div>
              )}

	              <div style={{ display: 'flex', gap: '8px', marginTop: '4px' }}>
	                <button onClick={() => {
	                  setIsDrawing(true);
	                  if (ruleDrawSpace === 'fms') setCurrentFmsPoints([]);
	                  else setCurrentPoints([]);
	                }} style={{
	                  flex: 1, padding: '10px 8px', borderRadius: '8px', fontWeight: 700,
	                  cursor: 'pointer', fontSize: '12px',
                  fontFamily: "'Space Grotesk', sans-serif",
                  background: isDrawing ? C.accentDim : 'linear-gradient(135deg, #6366f1 0%, #4f46e5 100%)',
                  color: isDrawing ? C.accentL : '#ffffff',
                  border: isDrawing ? `1px solid ${C.accentBorder}` : '1px solid rgba(99,102,241,0.5)',
                  boxShadow: isDrawing ? 'none' : '0 4px 16px rgba(99,102,241,0.3)',
	                  transition: 'all 0.2s',
	                }}>
	                  {isDrawing ? `● ${ruleDrawSpace === 'fms' ? 'FMS' : 'Camera'} (${activeDrawPoints.length})` : '+ Chấm Tọa Độ'}
	                </button>
	                <button onClick={handleSaveCurrentRule} disabled={saveRuleDisabled} style={{
	                  flex: 1, padding: '10px 8px', borderRadius: '8px', fontWeight: 700,
	                  cursor: saveRuleDisabled ? 'not-allowed' : 'pointer', fontSize: '12px',
	                  fontFamily: "'Space Grotesk', sans-serif",
	                  background: saveRuleDisabled ? C.cardAlt : C.cyanDim,
	                  color: saveRuleDisabled ? C.textMuted : C.cyanL,
	                  border: `1px solid ${saveRuleDisabled ? C.border : C.cyanBorder}`,
	                  transition: 'all 0.2s', opacity: saveRuleDisabled ? 0.6 : 1,
	                }}>
	                  {editingRuleId ? 'Cập Nhật' : 'Lưu Quy Tắc'}
	                </button>
	              </div>
	              {editingRuleId && (
	                <button type="button" onClick={resetRuleDraft} style={{
	                  padding: '8px 10px', borderRadius: '8px', cursor: 'pointer',
	                  background: 'transparent', color: C.textMuted, border: `1px solid ${C.borderHard}`,
	                  fontSize: '12px', fontWeight: 700,
	                }}>
	                  Hủy sửa
	                </button>
	              )}

	              {/* Saved Rules */}
	              <div style={{ borderTop: `1px solid ${C.border}`, paddingTop: '14px' }}>
	                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
	                  <h4 style={{ ...labelStyle, marginBottom: 0 }}>Quy tắc đã lưu</h4>
	                  {(rulesByCam[selectedCamId] || []).length > 0 && (
	                    <button type="button" onClick={handleClearRules} style={{
	                      background: C.roseDim, border: `1px solid ${C.roseBorder}`, color: C.rose,
	                      borderRadius: '6px', padding: '5px 8px', cursor: 'pointer',
	                      fontSize: '10px', fontWeight: 800, fontFamily: 'monospace',
	                    }}>
	                      XÓA HẾT
	                    </button>
	                  )}
	                </div>
	                <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', maxHeight: '200px', overflowY: 'auto' }}>
                  {(rulesByCam[selectedCamId] || []).length === 0 ? (
                    <div style={{ textAlign: 'center', color: C.textMuted, fontSize: '12px', padding: '16px', fontFamily: 'monospace' }}>
                      Chưa có quy tắc nào
                    </div>
	                  ) : (
	                    (rulesByCam[selectedCamId] || []).map(r => {
	                      const rType = getRuleType(r);
	                      const isIntrusion = rType === 'intrusion';
	                      const cameraCount = getRuleCameraPoints(r).length;
	                      const fmsCount = getRuleFmsPoints(r).length;
	                      return (
                        <div key={r.id} style={{
                          display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start',
                          padding: '9px 11px', background: C.cardAlt, borderRadius: '8px',
                          border: `1px solid ${C.border}`,
                          borderLeft: `3px solid ${isIntrusion ? C.rose : C.cyanL}`,
                        }}>
	                          <div>
	                            <div style={{ fontWeight: 700, color: C.textPrimary, fontSize: '12px' }}>{r.name}</div>
	                            <div style={{ color: C.textMuted, textTransform: 'uppercase', fontSize: '10px', fontFamily: 'monospace', marginBottom: '4px' }}>
	                              {rType}{rType === 'occupancy' ? ` · CAM ${cameraCount} · FMS ${fmsCount}` : ''}
	                            </div>
                            {r.target_objects?.length ? (
                              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
                                {r.target_objects.map(c => (
                                  <span key={c} style={{ background: C.accentDim, color: C.accentL, padding: '2px 6px', borderRadius: '4px', fontSize: '10px', fontWeight: 600 }}>{c}</span>
                                ))}
                              </div>
                            ) : null}
                          </div>
	                          <div style={{ display: 'flex', gap: '6px' }}>
	                            <button onClick={() => handleEditRule(r)} style={{ background: C.accentDim, border: `1px solid ${C.accentBorder}`, color: C.accentL, cursor: 'pointer', padding: '5px', borderRadius: '6px', display: 'flex' }}>
	                              <Pencil size={13} />
	                            </button>
	                            <button onClick={() => handleDeleteRule(r.id)} style={{ background: 'none', border: 'none', color: C.textMuted, cursor: 'pointer', padding: '5px', transition: 'color 0.15s' }}
	                              onMouseEnter={e => e.currentTarget.style.color = C.rose}
	                              onMouseLeave={e => e.currentTarget.style.color = C.textMuted}>
	                              <Trash2 size={13} />
	                            </button>
	                          </div>
	                        </div>
                      );
                    })
                  )}
                </div>
              </div>
            </div>

	            {/* Right Canvas */}
	            <div style={{ ...cardStyle, display: 'flex', flexDirection: 'column' }}>
	              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px', gap: '10px' }}>
	                <span style={{ fontSize: '12px', fontWeight: 600, color: C.textLabel }}>
	                  Vẽ ROI Camera + FMS Map
	                </span>
	                <button type="button" onClick={() => setSnapshotTimestamp(Date.now())} style={{
	                  display: 'flex', alignItems: 'center', gap: '5px',
	                  padding: '6px 12px', fontSize: '11px', fontWeight: 600,
	                  color: C.accentL, background: C.accentDim,
	                  border: `1px solid ${C.accentBorder}`, borderRadius: '7px', cursor: 'pointer',
	                  fontFamily: "'Space Grotesk', sans-serif",
	                }}>
	                  <RefreshCw size={12} /> Chụp lại Frame
	                </button>
	              </div>

	              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '14px', alignItems: 'start' }}>
	                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
	                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', color: C.textMuted, fontSize: '11px', fontFamily: 'monospace' }}>
	                    <span>CAMERA ROI</span>
	                    <span>{currentPoints.length} điểm</span>
	                  </div>
	                  <div style={{ position: 'relative', width: '100%', aspectRatio: '16/9', background: '#020306', borderRadius: '10px', overflow: 'hidden', border: `1px solid ${ruleDrawSpace === 'camera' ? C.accentBorder : C.borderHard}` }}>
	                    {selectedCamId ? (
	                      <img key={`rule-snap-${selectedCamId}-${snapshotTimestamp}`}
	                        src={`/api/backend/camera/${selectedCamId}/snapshot?t=${snapshotTimestamp}`}
	                        alt="Camera Snapshot" style={{ width: '100%', height: '100%', objectFit: 'fill', position: 'absolute', top: 0, left: 0 }} />
	                    ) : (
	                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: C.textMuted, fontSize: '13px', flexDirection: 'column', gap: '8px' }}>
	                        <CameraIcon size={32} strokeWidth={1} />
	                        <span>Chọn camera để hiển thị frame</span>
	                      </div>
	                    )}
	                    {isDrawing && ruleDrawSpace === 'camera' && (
	                      <div style={{ position: 'absolute', top: '10px', left: '50%', transform: 'translateX(-50%)', background: 'rgba(99,102,241,0.95)', color: '#fff', padding: '4px 14px', borderRadius: '20px', fontSize: '11px', fontWeight: 700, fontFamily: 'monospace', zIndex: 20, letterSpacing: '0.04em', boxShadow: '0 4px 12px rgba(0,0,0,0.3)' }}>
	                        ● CAMERA ({currentPoints.length})
	                      </div>
	                    )}
	                    <svg viewBox="0 0 1 1" preserveAspectRatio="none"
	                      style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', cursor: isDrawing && ruleDrawSpace === 'camera' ? 'crosshair' : 'default', zIndex: 10 }}
	                      onClick={e => {
	                        if (!isDrawing || ruleDrawSpace !== 'camera') return;
	                        const r = e.currentTarget.getBoundingClientRect();
	                        setCurrentPoints(prev => [...prev, [
	                          Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)),
	                          Math.min(1, Math.max(0, (e.clientY - r.top) / r.height)),
	                        ]]);
	                      }}>
	                      {(rulesByCam[selectedCamId] || []).map(r => {
	                        const rType = getRuleType(r);
	                        const pts = getRuleCameraPoints(r);
	                        if (rType === 'tripwire' && pts.length >= 2) return (
	                          <g key={r.id}>
	                            <line x1={pts[0][0]} y1={pts[0][1]} x2={pts[1][0]} y2={pts[1][1]} stroke="#06b6d4" strokeWidth="0.005" />
	                            <circle cx={pts[0][0]} cy={pts[0][1]} r="0.008" fill="#06b6d4" />
	                            <circle cx={pts[1][0]} cy={pts[1][1]} r="0.008" fill="#06b6d4" />
	                            <text x={pts[0][0]} y={pts[0][1] - 0.022} fill="#22d3ee" fontSize="0.028" fontWeight="bold">{r.name}</text>
	                          </g>
	                        );
	                        if (pts.length >= 3) return (
	                          <g key={r.id}>
	                            <polygon points={pointsAttr(pts)} fill={rType === 'occupancy' ? 'rgba(34,211,238,0.14)' : 'rgba(244,63,94,0.18)'} stroke={rType === 'occupancy' ? '#22d3ee' : '#f43f5e'} strokeWidth="0.004" />
	                            <text x={pts[0][0]} y={pts[0][1] - 0.022} fill={rType === 'occupancy' ? '#22d3ee' : '#fb7185'} fontSize="0.028" fontWeight="bold">{r.name}</text>
	                          </g>
	                        );
	                        return null;
	                      })}
	                      {currentPoints.length > 0 && (
	                        <>
	                          {currentPoints.map((pt, i) => (
	                            <circle key={i} cx={pt[0]} cy={pt[1]} r="0.009" fill="#6366f1" stroke="#ffffff" strokeWidth="0.002" />
	                          ))}
	                          <polyline points={pointsAttr(currentPoints)} fill="none" stroke="#818cf8" strokeWidth="0.004" strokeDasharray="0.013" />
	                        </>
	                      )}
	                    </svg>
	                  </div>
	                </div>

	                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
	                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', color: C.textMuted, fontSize: '11px', fontFamily: 'monospace' }}>
	                    <span>FMS MAP ROI</span>
	                    <span>{currentFmsPoints.length} điểm</span>
	                  </div>
	                  <div style={{ position: 'relative', width: '100%', aspectRatio: `${slamPixelSize[0]} / ${slamPixelSize[1]}`, background: isDark ? '#0f172a' : '#f8fafc', borderRadius: '10px', overflow: 'hidden', border: `1px solid ${ruleDrawSpace === 'fms' ? C.accentBorder : C.borderHard}` }}>
	                    {isDrawing && ruleDrawSpace === 'fms' && (
	                      <div style={{ position: 'absolute', top: '10px', left: '50%', transform: 'translateX(-50%)', background: 'rgba(14,165,233,0.95)', color: '#fff', padding: '4px 14px', borderRadius: '20px', fontSize: '11px', fontWeight: 700, fontFamily: 'monospace', zIndex: 20, letterSpacing: '0.04em', boxShadow: '0 4px 12px rgba(0,0,0,0.3)' }}>
	                        ● FMS ({currentFmsPoints.length})
	                      </div>
	                    )}
	                    <svg viewBox={`${slamMapRect[0]} ${slamMapRect[1]} ${slamMapWidth} ${slamMapHeight}`} preserveAspectRatio="none"
	                      style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', cursor: isDrawing && ruleDrawSpace === 'fms' ? 'crosshair' : 'default', zIndex: 10 }}
	                      onClick={e => {
	                        if (!isDrawing || ruleDrawSpace !== 'fms' || !isStorageRule) return;
	                        const r = e.currentTarget.getBoundingClientRect();
	                        const x = slamMapRect[0] + ((e.clientX - r.left) / r.width) * slamMapWidth;
	                        const y = slamMapRect[1] + ((e.clientY - r.top) / r.height) * slamMapHeight;
	                        setCurrentFmsPoints(prev => [...prev, [Number(x.toFixed(3)), Number(y.toFixed(3))]]);
	                      }}>
	                      <rect x={slamMapRect[0]} y={slamMapRect[1]} width={slamMapWidth} height={slamMapHeight} fill={isDark ? '#111827' : '#eef2f7'} />
	                      <image href={slamMapHref} x={slamMapRect[0]} y={slamMapRect[1]} width={slamMapWidth} height={slamMapHeight} preserveAspectRatio="none" opacity="0.82" />
	                      {(rulesByCam[selectedCamId] || []).map(r => {
	                        if (getRuleType(r) !== 'occupancy') return null;
	                        const pts = getRuleFmsPoints(r);
	                        if (pts.length < 3) return null;
	                        return (
	                          <g key={`fms-${r.id}`}>
	                            <polygon points={pointsAttr(pts)} fill="rgba(16,185,129,0.22)" stroke="#10b981" strokeWidth="0.045" />
	                            <text x={pts[0][0]} y={pts[0][1] - 0.18} fill="#047857" fontSize="0.34" fontWeight="bold">{r.name}</text>
	                          </g>
	                        );
	                      })}
	                      {currentFmsPoints.length > 0 && (
	                        <>
	                          {currentFmsPoints.map((pt, i) => (
	                            <circle key={i} cx={pt[0]} cy={pt[1]} r="0.14" fill="#0ea5e9" stroke="#ffffff" strokeWidth="0.035" />
	                          ))}
	                          <polyline points={pointsAttr(currentFmsPoints)} fill="none" stroke="#0ea5e9" strokeWidth="0.05" strokeDasharray="0.16" />
	                        </>
	                      )}
	                    </svg>
	                  </div>
	                </div>
	              </div>
	            </div>
          </div>
        )}

        {/* ── TAB 3: LABEL ─────────────────────────────────────────────── */}
        {activeTab === 'label' && (
          <div style={{ display: 'grid', gridTemplateColumns: '340px 1fr', gap: '20px' }}>
            <div style={{ ...cardStyle, display: 'flex', flexDirection: 'column', gap: '14px' }}>
              <div>
                <h3 style={{ fontSize: '14px', fontWeight: 700, color: C.textPrimary, margin: '0 0 8px 0' }}>
                  Gán nhãn Tracking
                </h3>
                <div style={{ background: C.cardAlt, border: `1px solid ${C.border}`, borderRadius: '8px', padding: '12px', fontSize: '12px', color: C.textSub, lineHeight: 1.7 }}>
                  Khoanh sát vật → Tạo mask → chỉnh + Vật / − Nền → lưu nhãn. Chọn lại cùng nhãn để thêm góc nhìn. Chỉ nhãn robot/kệ đã đăng ký được bám mask; nhận diện người giữ nguyên.
                </div>
              </div>

              <div>
                <label style={labelStyle}>Chọn Camera</label>
                <select disabled={isSavingLabel} value={labelCamId} onChange={e => {
                  const newCamId = e.target.value;
                  setLabelCamId(newCamId);
                  resetLabelMaskState();
                  setLabelStatus('');
                  fetchRegisteredTargets(newCamId);
                }} style={selectStyle} {...focusHandlers}>
                  {cameras.map(c => <option key={c.id} value={c.id}>{c.name}</option>)}
                </select>
              </div>

              <div>
                <label style={labelStyle}>Loại đối tượng</label>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '8px' }}>
                  {([
                    ['robot', 'Robot', <Bot key="robot" size={14} />],
                    ['rack', 'Kệ', <Package key="rack" size={14} />],
                    ['person', 'Người', <Tag key="person" size={14} />],
                  ] as const).map(([value, label, icon]) => {
                    const active = labelCategory === value;
                    return (
                      <button key={value} disabled={isSavingLabel} type="button" onClick={() => {
                        setLabelCategory(value);
                        resetLabelMaskState(false);
                        if (!labelName.trim() || ['Robot_9001', 'Rack_A1', 'Person_01'].includes(labelName)) {
                          setLabelName(value === 'robot' ? 'Robot_9001' : value === 'rack' ? 'Rack_A1' : 'Person_01');
                        }
                      }} style={{
                        padding: '9px 8px', borderRadius: '8px', cursor: 'pointer',
                        display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px',
                        background: active ? C.accentDim : 'transparent',
                        border: `1px solid ${active ? C.accentBorder : C.borderHard}`,
                        color: active ? C.accentL : C.textSub,
                        fontSize: '12px', fontWeight: 700,
                      }}>
                        {icon}{label}
                      </button>
                    );
                  })}
                </div>
              </div>

              <div>
                <label style={labelStyle}>Tên định danh / Label</label>
                <input disabled={isSavingLabel} value={labelName} onChange={e => setLabelName(e.target.value)} placeholder="VD: Robot_9001, Rack_A1..." style={inputStyle} {...focusHandlers} />
              </div>

              <div style={{ display: 'flex', gap: '8px' }}>
                <button type="button" disabled={isSavingLabel} onClick={() => { resetLabelMaskState(); setSnapshotTimestamp(Date.now()); setLabelStatus(''); }} style={{
                  flex: 1, padding: '10px', borderRadius: '8px', cursor: 'pointer',
                  background: C.cardAlt, color: C.textLabel, border: `1px solid ${C.borderHard}`,
                  display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px',
                  fontSize: '12px', fontWeight: 700,
                }}>
                  <RefreshCw size={14} /> Chụp frame
                </button>
                <GradientBtn isDark={isDark} onClick={handleSaveLabelCrop} disabled={!labelCrop || !labelName.trim() || isSavingLabel || maskBusy || !labelFrameReady || (labelCategory !== 'person' && !labelMask)} style={{ flex: 1, justifyContent: 'center' }}>
                  <Crop size={14} /> {isSavingLabel ? 'Đang lưu...' : 'Lưu góc nhìn'}
                </GradientBtn>
              </div>

              {labelCategory !== 'person' && <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                <GradientBtn isDark={isDark} onClick={() => generateLabelMask()} disabled={!maskAvailable || !labelCrop || labelCrop.w <= 0 || labelCrop.h <= 0 || !labelName.trim() || !labelFrameReady || maskBusy || isSavingLabel}>
                  <Crop size={14} /> {maskBusy ? 'Đang tạo mask…' : 'Tạo mask'}
                </GradientBtn>
                <div style={{ display: 'flex', gap: '6px' }}>
                  {([['box', 'Khoanh lại'], ['add', '+ Vật'], ['remove', '− Nền']] as const).map(([tool, title]) =>
                    <button key={tool} type="button" disabled={maskBusy || isSavingLabel} onClick={() => setMaskTool(tool)}
                      style={{ ...tabStyle(maskTool === tool), padding: '7px 9px', fontSize: '11px' }}>{title}</button>)}
                </div>
                <div style={{ fontSize: '11px', color: maskAvailable ? C.textMuted : C.amber }}>
                  {maskAvailable ? `${maskPoints.length}/64 điểm chỉnh · SAM2 theo nhãn · không huấn luyện tự động` : 'Model mask chưa sẵn sàng. Các nhãn cũ vẫn được giữ.'}
                </div>
              </div>}

              {labelStatus && (
                <div style={{ padding: '11px 14px', background: labelStatus.startsWith('Lỗi') ? C.roseDim : C.emeraldDim, color: labelStatus.startsWith('Lỗi') ? C.rose : C.emerald, borderRadius: '8px', fontSize: '12px', border: `1px solid ${labelStatus.startsWith('Lỗi') ? C.roseBorder : C.emeraldBorder}` }}>
                  {labelStatus}
                </div>
              )}

              <div style={{ borderTop: `1px solid ${C.border}`, paddingTop: '14px' }}>
                <h4 style={{ ...labelStyle, marginBottom: '10px' }}>
                  Label của {cameras.find(c => c.id === labelCamId)?.name || 'Camera này'} ({registeredTargets.length})
                </h4>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', maxHeight: '290px', overflowY: 'auto' }}>
                  {registeredTargets.length === 0 ? (
                    <EmptyState colors={C} icon={<Tag size={28} strokeWidth={1.3} />} title="Chưa có label nào trên Camera này" hint="Crop frame camera rồi lưu label để gán nhãn riêng cho camera này." />
                  ) : (
                    registeredTargets.map(tgt => {
                      const camName = cameras.find(c => c.id === (tgt.cam_id || tgt.last_cam || ''))?.name || 'Camera';
                      return (
                        <div key={tgt.key || tgt.label} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '10px', padding: '10px 12px', borderRadius: '8px', background: C.cardAlt, border: `1px solid ${C.border}` }}>
                          <div>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                              <span style={{ fontSize: '13px', fontWeight: 800, color: C.textPrimary, fontFamily: 'monospace' }}>{tgt.label}</span>
                              <span style={{ fontSize: '10px', padding: '2px 6px', borderRadius: '4px', background: C.accentDim, color: C.accentL, fontWeight: 700, border: `1px solid ${C.accentBorder}` }}>
                                {(tgt.samples_count || 1) > 1 ? `${tgt.samples_count} góc nhìn` : '1 góc nhìn'}
                              </span>
                            </div>
                            <div style={{ fontSize: '11px', color: C.textMuted, marginTop: '3px' }}>
                              {tgt.category || 'object'}{tgt.fms_robot_id ? ` · FMS ${tgt.fms_robot_id}` : ''} · {camName}
                              {` · ${tgt.mask_samples_count || 0} mẫu mask`}
                            </div>
                            {(!tgt.bbox || !tgt.has_crop_image) && (
                              <div style={{ fontSize: '10px', color: C.amber, marginTop: '4px', fontWeight: 800, fontFamily: 'monospace' }}>
                                CHƯA CÓ MẪU CROP ĐỂ TRACKING
                              </div>
                            )}
                          </div>
                          <div style={{ display: 'flex', gap: '6px' }}>
                            <button type="button" title="Sử dụng mẫu này" onClick={() => handleUseRegisteredTarget(tgt)} style={{ background: C.cyanDim, border: `1px solid ${C.cyanBorder}`, color: C.cyanL, borderRadius: '6px', padding: '7px', cursor: 'pointer', display: 'flex' }}>
                              <Crop size={14} />
                            </button>
                            <button
                              type="button"
                              title={`Xóa nhãn '${tgt.label}' trên ${camName}`}
                              onClick={() => handleDeleteRegisteredTarget(tgt.label, tgt.cam_id || labelCamId)}
                              style={{ background: C.roseDim, border: `1px solid ${C.roseBorder}`, color: C.rose, borderRadius: '6px', padding: '7px', cursor: 'pointer', display: 'flex' }}>
                              <Trash2 size={14} />
                            </button>
                          </div>
                        </div>
                      );
                    })
                  )}
                </div>
              </div>
            </div>

            <div style={{ ...cardStyle, display: 'flex', flexDirection: 'column', minHeight: '560px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                <span style={{ fontSize: '12px', fontWeight: 700, color: C.textLabel, display: 'flex', alignItems: 'center', gap: '7px' }}>
                  <Crop size={15} /> {labelCategory === 'person' ? 'Frame Crop Label' : 'Mask nhãn đăng ký'}
                </span>
                {labelCrop && (
                  <span style={{ fontSize: '11px', color: C.textMuted, fontFamily: 'monospace' }}>
                    x:{labelCrop.x.toFixed(2)} y:{labelCrop.y.toFixed(2)} w:{labelCrop.w.toFixed(2)} h:{labelCrop.h.toFixed(2)}
                  </span>
                )}
              </div>

              <div style={{ position: 'relative', width: '100%', aspectRatio: '16/9', background: '#020306', borderRadius: '10px', overflow: 'hidden', border: `1px solid ${C.borderHard}` }}>
                {labelCamId ? (
                  <img
                    ref={labelImageRef}
                    key={`label-${labelCamId}-${snapshotTimestamp}`}
                    src={`/api/backend/camera/${labelCamId}/snapshot?t=${snapshotTimestamp}`}
                    alt="Label Snapshot"
                    onLoad={() => setLabelFrameReady(true)}
                    onError={() => { setLabelFrameReady(false); setLabelStatus('Lỗi tải ảnh camera. Hãy chụp lại frame.'); }}
                    style={{ width: '100%', height: '100%', objectFit: 'fill', position: 'absolute', inset: 0 }}
                  />
                ) : (
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: C.textMuted, fontSize: '13px', flexDirection: 'column', gap: '8px' }}>
                    <CameraIcon size={32} strokeWidth={1} />
                    <span>Chọn camera để chụp frame</span>
                  </div>
                )}

                <svg
                  viewBox="0 0 1 1"
                  preserveAspectRatio="none"
                  style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', cursor: 'crosshair', zIndex: 5 }}
                  onMouseDown={e => {
                    if (!labelFrameReady || maskBusy || isSavingLabel) return;
                    const p = getNormPoint(e);
                    if (labelCategory !== 'person' && maskTool !== 'box') {
                      if (!labelCrop || maskPoints.length >= 64) return;
                      const corrections = [...maskPoints, { point: [p.x, p.y], label: maskTool === 'add' ? 1 : 0 }];
                      setMaskPoints(corrections);
                      generateLabelMask(corrections);
                      return;
                    }
                    maskRequest.current += 1;
                    setLabelMask(null);
                    setMaskPoints([]);
                    setCropStart(p);
                    setLabelCrop({ x: p.x, y: p.y, w: 0, h: 0 });
                  }}
                  onMouseMove={e => {
                    if (!cropStart) return;
                    updateLabelCrop(cropStart, getNormPoint(e));
                  }}
                  onMouseUp={e => {
                    if (!cropStart) return;
                    updateLabelCrop(cropStart, getNormPoint(e));
                    setCropStart(null);
                  }}
                  onMouseLeave={() => setCropStart(null)}
                >
                  {labelCrop && (
                    <>
                      <rect x="0" y="0" width="1" height="1" fill="rgba(0,0,0,0.28)" />
                      {labelMask ? <path d={maskPath(labelMask)} fill="rgba(34,211,238,0.28)" fillRule="evenodd" stroke="#22d3ee" strokeWidth="0.002" />
                        : <rect x={labelCrop.x} y={labelCrop.y} width={labelCrop.w} height={labelCrop.h} fill="rgba(34,211,238,0.12)" stroke="#22d3ee" strokeWidth="0.004" />}
                      <text x={labelCrop.x} y={Math.max(0.03, labelCrop.y - 0.012)} fill="#22d3ee" fontSize="0.028" fontWeight="bold">{labelName || 'Label'}</text>
                    </>
                  )}
                  {maskPoints.map((item, index) => <circle key={index} cx={item.point[0]} cy={item.point[1]} r="0.006" fill={item.label ? '#22c55e' : '#ef4444'} stroke="white" strokeWidth="0.001" />)}
                </svg>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* ── AUTH MODAL ──────────────────────────────────────────────────────── */}
      {showAuthModal && (
        <div style={modalOverlay}>
          <form onSubmit={e => { e.preventDefault(); handleApplyCameraAuth(); }} style={{ ...modalCard, width: '420px' }}>
            <h3 style={{ margin: '0 0 20px', fontSize: '16px', color: C.textPrimary, fontWeight: 700 }}>
              Xác thực luồng RTSP
            </h3>
            <div style={{ marginBottom: '14px' }}>
              <label style={labelStyle}>Hãng Camera</label>
              <select value={cameraBrand} onChange={e => setCameraBrand(e.target.value)} style={selectStyle} {...focusHandlers}>
                <option value="hikvision">Hikvision</option>
                <option value="dahua">Dahua</option>
                <option value="custom">Tùy chỉnh (Link RTSP đầy đủ)</option>
              </select>
            </div>
            {cameraBrand !== 'custom' ? (
              <>
                <div style={{ marginBottom: '14px' }}>
                  <label style={labelStyle}>Tài khoản</label>
                  <input type="text" value={authUsername} onChange={e => setAuthUsername(e.target.value)} style={inputStyle} {...focusHandlers} />
                </div>
                <div style={{ marginBottom: '22px' }}>
                  <label style={labelStyle}>Mật khẩu</label>
                  <div style={{ position: 'relative' }}>
                    <input type={showPassword ? 'text' : 'password'} value={authPassword} onChange={e => setAuthPassword(e.target.value)} style={inputStyle} {...focusHandlers} />
                    <button type="button" onClick={() => setShowPassword(!showPassword)} style={{ position: 'absolute', right: '12px', top: '50%', transform: 'translateY(-50%)', background: 'none', border: 'none', cursor: 'pointer', color: C.textMuted }}>
                      {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
                    </button>
                  </div>
                </div>
              </>
            ) : (
              <div style={{ marginBottom: '22px', padding: '12px', background: C.accentDim, borderRadius: '8px', fontSize: '12px', color: C.accentL, border: `1px solid ${C.accentBorder}`, fontFamily: 'monospace' }}>
                Sử dụng nguyên bản đường dẫn RTSP bạn đã nhập.
              </div>
            )}
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px' }}>
              <button type="button" onClick={() => setShowAuthModal(false)} style={{ padding: '9px 18px', borderRadius: '8px', border: `1px solid ${C.borderHard}`, background: 'transparent', cursor: 'pointer', color: C.textLabel, fontFamily: "'Space Grotesk', sans-serif", fontWeight: 600, fontSize: '13px' }}>Hủy</button>
              <GradientBtn isDark={isDark}>Xác nhận</GradientBtn>
            </div>
          </form>
        </div>
      )}

      {/* ── EDIT MODAL ──────────────────────────────────────────────────────── */}
      {editCamModal.open && (
        <div style={modalOverlay}>
          <form onSubmit={handleSaveEditCamera} style={{ ...modalCard, width: '460px' }}>
            <h3 style={{ margin: '0 0 20px', fontSize: '16px', color: C.textPrimary, fontWeight: 700 }}>
              Chỉnh sửa thông tin Camera
            </h3>
            <div style={{ marginBottom: '14px' }}>
              <label style={labelStyle}>Tên Camera</label>
              <input type="text" value={editCamModal.name} onChange={e => setEditCamModal({ ...editCamModal, name: e.target.value })} style={inputStyle} {...focusHandlers} required />
            </div>
            <div style={{ marginBottom: '22px' }}>
              <label style={labelStyle}>RTSP URL / IP Address</label>
              <input type="text" value={editCamModal.url} onChange={e => setEditCamModal({ ...editCamModal, url: e.target.value })} style={inputStyle} {...focusHandlers} required />
            </div>
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px' }}>
              <button type="button" onClick={() => setEditCamModal({ open: false, id: '', name: '', url: '' })} style={{ padding: '9px 18px', borderRadius: '8px', border: `1px solid ${C.borderHard}`, background: 'transparent', cursor: 'pointer', color: C.textLabel, fontFamily: "'Space Grotesk', sans-serif", fontWeight: 600, fontSize: '13px' }}>Hủy</button>
              <GradientBtn isDark={isDark}>Lưu Thay Đổi</GradientBtn>
            </div>
          </form>
        </div>
      )}
    </div>
  );
}
