'use client';

import React, { useState } from 'react';
import {
  ShieldAlert,
  ShieldCheck,
  AlertTriangle,
  Clock,
  Trash2,
  Search,
  Eye,
  Camera as CameraIcon,
  Activity,
  Layers,
  ArrowRight
} from 'lucide-react';
import { useAppTheme } from '../ThemeContext';
import { Camera } from '../CameraContext';

export type LiveAlertItem = {
  cam_id: string;
  global_id: number;
  rule_type: string;
  severity: string;
  description: string;
  timestamp: number;
};

interface MonitorAlertsPanelProps {
  alerts: LiveAlertItem[];
  cameras: Camera[];
  onSelectCameraTab?: (camId: string) => void;
  onClearAlerts?: () => void;
}

function formatRelativeTime(timestamp: number): string {
  const diffSec = Math.max(0, Math.floor((Date.now() - timestamp) / 1000));
  if (diffSec < 10) return 'Vừa xong';
  if (diffSec < 60) return `${diffSec}s trước`;
  const diffMin = Math.floor(diffSec / 60);
  if (diffMin < 60) return `${diffMin}m trước`;
  const diffHour = Math.floor(diffMin / 60);
  return `${diffHour}h trước`;
}

function formatRuleType(ruleType: string): string {
  const normalized = (ruleType || '').toLowerCase();
  if (normalized.includes('tripwire') || normalized.includes('line')) {
    return 'Vượt Vạch';
  }
  if (normalized.includes('roi') || normalized.includes('intrusion') || normalized.includes('forbidden')) {
    return 'Xâm Nhập Vùng';
  }
  if (normalized.includes('dwell')) {
    return 'Dừng Đỗ Quá Hạn';
  }
  if (normalized.includes('fall')) {
    return 'Phát Hiện Té Ngã';
  }
  return ruleType.toUpperCase();
}

export default function MonitorAlertsPanel({
  alerts,
  cameras,
  onSelectCameraTab,
  onClearAlerts,
}: MonitorAlertsPanelProps) {
  const { colors: C, isDark } = useAppTheme();
  const [filterSeverity, setFilterSeverity] = useState<'ALL' | 'HIGH' | 'WARNING'>('ALL');
  const [searchQuery, setSearchQuery] = useState('');

  const getCameraName = (camId: string) => {
    const cam = cameras.find(c => c.id === camId);
    return cam ? cam.name : `Cam ${camId.slice(0, 6)}`;
  };

  const filteredAlerts = alerts.filter(alert => {
    const s = (alert.severity || '').toLowerCase();
    const isHigh = s === 'emergency' || s === 'critical' || s === 'high';
    const isWarn = s === 'warning' || s === 'medium';

    if (filterSeverity === 'HIGH' && !isHigh) return false;
    if (filterSeverity === 'WARNING' && !isWarn) return false;

    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      const camName = getCameraName(alert.cam_id).toLowerCase();
      const desc = (alert.description || '').toLowerCase();
      const rule = (alert.rule_type || '').toLowerCase();
      const idStr = String(alert.global_id);
      return camName.includes(q) || desc.includes(q) || rule.includes(q) || idStr.includes(q);
    }
    return true;
  });

  const highCount = alerts.filter(a => {
    const s = (a.severity || '').toLowerCase();
    return s === 'emergency' || s === 'critical' || s === 'high';
  }).length;

  const warnCount = alerts.filter(a => {
    const s = (a.severity || '').toLowerCase();
    return s === 'warning' || s === 'medium';
  }).length;

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      height: '100%',
      minWidth: '340px',
      background: C.surface,
      borderRadius: '14px',
      border: `1px solid ${C.border}`,
      boxShadow: isDark
        ? '0 8px 24px -6px rgba(0, 0, 0, 0.45), 0 0 1px rgba(255, 255, 255, 0.08) inset'
        : '0 4px 16px -2px rgba(0, 0, 0, 0.06)',
      overflow: 'hidden'
    }}>
      {/* Header */}
      <div style={{
        padding: '14px 16px',
        borderBottom: `1px solid ${C.border}`,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        background: isDark ? 'rgba(255,255,255,0.02)' : 'rgba(0,0,0,0.01)'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div style={{
            width: '32px',
            height: '32px',
            borderRadius: '9px',
            background: 'rgba(244, 63, 94, 0.15)',
            border: '1px solid rgba(244, 63, 94, 0.35)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}>
            <ShieldAlert size={18} color="#f43f5e" />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span style={{ fontSize: '14px', fontWeight: 700, color: C.textPrimary }}>
                Cảnh Báo Vi Phạm
              </span>
              <span style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '4px',
                padding: '2px 6px',
                borderRadius: '6px',
                background: 'rgba(244, 63, 94, 0.15)',
                color: '#f43f5e',
                fontSize: '10px',
                fontWeight: 800,
                letterSpacing: '0.04em'
              }}>
                <span style={{
                  width: '6px',
                  height: '6px',
                  borderRadius: '50%',
                  background: '#f43f5e',
                  boxShadow: '0 0 6px #f43f5e'
                }} />
                LIVE
              </span>
            </div>
            <span style={{ fontSize: '11px', color: C.textMuted }}>
              Nhật ký sự kiện AI & Quy tắc hành vi
            </span>
          </div>
        </div>

        {alerts.length > 0 && onClearAlerts && (
          <button
            onClick={onClearAlerts}
            title="Xóa danh sách cảnh báo hiện tại"
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '5px',
              padding: '5px 9px',
              borderRadius: '6px',
              background: 'transparent',
              border: `1px solid ${C.border}`,
              color: C.textMuted,
              fontSize: '11px',
              fontWeight: 500,
              cursor: 'pointer',
              transition: 'all 0.15s ease'
            }}
          >
            <Trash2 size={12} />
            <span>Xóa log</span>
          </button>
        )}
      </div>

      {/* Filter and Search Bar */}
      <div style={{
        padding: '10px 14px',
        borderBottom: `1px solid ${C.border}`,
        display: 'flex',
        flexDirection: 'column',
        gap: '8px',
        background: isDark ? 'rgba(0,0,0,0.1)' : 'rgba(0,0,0,0.01)'
      }}>
        {/* Severity Filter Pills */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <button
            onClick={() => setFilterSeverity('ALL')}
            style={{
              padding: '4px 10px',
              borderRadius: '6px',
              fontSize: '11px',
              fontWeight: filterSeverity === 'ALL' ? 700 : 500,
              background: filterSeverity === 'ALL'
                ? (isDark ? 'rgba(245, 158, 11, 0.18)' : 'rgba(245, 158, 11, 0.12)')
                : 'transparent',
              color: filterSeverity === 'ALL' ? C.accentGlow : C.textSecondary,
              border: filterSeverity === 'ALL' ? `1px solid ${C.accentBorder}` : `1px solid ${C.border}`,
              cursor: 'pointer'
            }}
          >
            Tất cả ({alerts.length})
          </button>

          <button
            onClick={() => setFilterSeverity('HIGH')}
            style={{
              padding: '4px 10px',
              borderRadius: '6px',
              fontSize: '11px',
              fontWeight: filterSeverity === 'HIGH' ? 700 : 500,
              background: filterSeverity === 'HIGH'
                ? 'rgba(244, 63, 94, 0.2)'
                : 'transparent',
              color: filterSeverity === 'HIGH' ? '#f43f5e' : C.textSecondary,
              border: filterSeverity === 'HIGH' ? '1px solid rgba(244, 63, 94, 0.4)' : `1px solid ${C.border}`,
              cursor: 'pointer'
            }}
          >
            Nguy hiểm ({highCount})
          </button>

          <button
            onClick={() => setFilterSeverity('WARNING')}
            style={{
              padding: '4px 10px',
              borderRadius: '6px',
              fontSize: '11px',
              fontWeight: filterSeverity === 'WARNING' ? 700 : 500,
              background: filterSeverity === 'WARNING'
                ? 'rgba(245, 158, 11, 0.2)'
                : 'transparent',
              color: filterSeverity === 'WARNING' ? '#f59e0b' : C.textSecondary,
              border: filterSeverity === 'WARNING' ? '1px solid rgba(245, 158, 11, 0.4)' : `1px solid ${C.border}`,
              cursor: 'pointer'
            }}
          >
            Cảnh báo ({warnCount})
          </button>
        </div>

        {/* Search Input */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          padding: '6px 10px',
          borderRadius: '8px',
          background: C.card,
          border: `1px solid ${C.border}`
        }}>
          <Search size={13} color={C.textMuted} />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Tìm theo Camera, hành vi, ID..."
            style={{
              background: 'transparent',
              border: 'none',
              outline: 'none',
              fontSize: '11px',
              color: C.textPrimary,
              width: '100%'
            }}
          />
          {searchQuery && (
            <button
              onClick={() => setSearchQuery('')}
              style={{ background: 'none', border: 'none', color: C.textMuted, cursor: 'pointer', fontSize: '11px' }}
            >
              ×
            </button>
          )}
        </div>
      </div>

      {/* Alerts Scroll List */}
      <div style={{
        flex: 1,
        overflowY: 'auto',
        padding: '12px',
        display: 'flex',
        flexDirection: 'column',
        gap: '10px'
      }}>
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          height: '100%',
          padding: '30px 16px',
          textAlign: 'center',
          gap: '12px'
        }}>
          <div style={{
            width: '54px',
            height: '54px',
            borderRadius: '16px',
            background: 'rgba(16, 185, 129, 0.12)',
            border: '1px solid rgba(16, 185, 129, 0.3)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}>
            <ShieldCheck size={26} color="#10b981" />
          </div>
          <div>
            <p style={{ fontSize: '14px', fontWeight: 600, color: C.textPrimary }}>
              {alerts.length === 0 ? 'Không Có Cảnh Báo Vi Phạm' : `Đang tải ${filteredAlerts.length} cảnh báo...`}
            </p>
            <p style={{ fontSize: '12px', color: C.textMuted, marginTop: '4px', maxWidth: '260px' }}>
              {alerts.length === 0
                ? 'Hệ thống AI đang giám sát liên tục theo thời gian thực và chưa ghi nhận hành vi bất thường.'
                : 'Dữ liệu sự kiện đang được cập nhật thời gian thực.'}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
