'use client';

// Monitor view HUD of the inspection stations (plan Phase 6): one chip per
// station with its confirmed occupancy (CÓ HÀNG / TRỐNG / KHÔNG XÁC ĐỊNH) and,
// below it, the metrology state and AGV permission; the switch showing the
// cells on the camera tiles; and a yellow pop-up for every confirmed UNCERTAIN
// so a supervisor approves or rejects it (target.md Test 5.3).

import React, { useState } from 'react';
import { useAppTheme } from '../ThemeContext';
import { inspectionApi, occupancyColor, occupancyText, STATUS_LABEL, type InspectionStation } from '../../lib/roi-rules';
import {
  refreshInspectionStations, setInspectionOverlayVisible, useInspectionOverlayVisible, useInspectionStations,
} from '../../lib/inspection-stations';

export default function InspectionMonitorHud({ active }: { active: boolean }) {
  const { colors: C } = useAppTheme();
  const { stations, error: pollError } = useInspectionStations(active);
  const overlayVisible = useInspectionOverlayVisible();
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState('');

  if (!active || stations.length === 0) return null;
  const color = (status: string) =>
    status === 'OK' ? C.emerald : status === 'NG' ? C.rose : status === 'DETECTED' ? C.cyanL : C.amber;
  const pending = stations.filter(s => s.state.confirmed_status === 'UNCERTAIN' && !s.state.operator_hold);

  const decide = async (station: InspectionStation, approve: boolean) => {
    setBusy(station.rule_id);
    try {
      await inspectionApi.decide(station.cam_id, station.rule_id, approve, 'monitor');
      setError('');
      await refreshInspectionStations();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally { setBusy(null); }
  };

  return (
    <>
      <div data-testid="inspection-hud" style={{
        position: 'fixed', left: '16px', bottom: '16px', zIndex: 900, maxWidth: 'min(440px, calc(100vw - 32px))',
        maxHeight: 'calc(100vh - 32px)', overflowY: 'auto',
        display: 'flex', flexDirection: 'column', gap: '6px', padding: '10px', borderRadius: '10px',
        background: C.card, border: `1px solid ${C.borderHard}`, boxShadow: '0 8px 30px rgba(0,0,0,0.35)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '10px' }}>
          <span style={{ fontSize: '11px', fontWeight: 800, letterSpacing: '0.06em', color: C.textLabel }}>📐 TRẠM KIỂM ĐỊNH Ô HÀNG</span>
          <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '11px', color: C.textSub, cursor: 'pointer' }}>
            <input type="checkbox" data-testid="inspection-hud-overlay-toggle" checked={overlayVisible}
              onChange={e => setInspectionOverlayVisible(e.target.checked)} style={{ accentColor: C.accent }} />
            Hiện ô trên camera
          </label>
        </div>
        {(error || pollError) && <div style={{ fontSize: '11px', color: C.rose }}>{error || pollError}</div>}
        {stations.map(station => {
          const status = station.state.confirmed_status;
          const confirmed = station.state.confirmed;
          const occupancy = station.occupancy;
          const occColor = occupancyColor(occupancy?.state);
          return (
            <div key={`${station.cam_id}:${station.rule_id}`} data-testid={`inspection-hud-${station.rule_id}`} style={{
              display: 'grid', gridTemplateColumns: 'auto 1fr', gap: '3px 10px', alignItems: 'center', padding: '7px 9px',
              borderRadius: '8px', background: C.cardAlt, borderLeft: `4px solid ${occColor}`,
            }}>
              <span data-testid={`inspection-hud-occupancy-${station.rule_id}`} data-state={occupancy?.state || 'PENDING'}
                style={{ gridRow: 'span 3', padding: '4px 8px', borderRadius: '5px', fontSize: '11px', fontWeight: 900, color: '#06070a', background: occColor, textAlign: 'center', maxWidth: '140px' }}>
                {occupancyText(occupancy)}
              </span>
              <span style={{ fontSize: '12px', fontWeight: 700, color: C.textPrimary }}>
                {station.name}{station.fms_slot_id ? ` · Slot ${station.fms_slot_id}` : ''} · {station.mode}
              </span>
              <span style={{ fontSize: '11px', color: C.textSub, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
                title={occupancy?.reason || ''}>
                {occupancy?.reason || 'Đang chờ đủ khung hình để chốt trạng thái'}
              </span>
              <span style={{ fontSize: '10px', color: C.textMuted, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                <b data-testid={`inspection-hud-status-${station.rule_id}`} style={{ color: color(status) }}>{STATUS_LABEL[status] || status}</b>
                {' · '}{station.state.interlock ? '🔒 Khóa an toàn' : station.state.agv_permission ? '✅ AGV được phép' : '⛔ AGV không vào'}
                {station.baseline_choice?.label ? ` · nền: ${station.baseline_choice.label}` : ''}
                {confirmed?.message ? ` · ${confirmed.message}` : ''}
              </span>
            </div>
          );
        })}
      </div>

      {pending.map(station => (
        <div key={`popup-${station.rule_id}`} role="alertdialog" aria-modal="true" aria-labelledby={`insp-popup-${station.rule_id}`}
          data-testid={`inspection-uncertain-popup-${station.rule_id}`}
          style={{ position: 'fixed', inset: 0, zIndex: 1100, display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'rgba(0,0,0,0.55)' }}>
          <div style={{ width: 'min(520px, calc(100vw - 32px))', maxHeight: 'calc(100vh - 32px)', overflowY: 'auto', borderRadius: '14px', padding: '20px', background: '#422006', border: '2px solid #f59e0b', boxShadow: '0 24px 70px rgba(0,0,0,0.6)', color: '#fef3c7' }}>
            <h3 id={`insp-popup-${station.rule_id}`} style={{ margin: '0 0 6px', fontSize: '16px', color: '#fbbf24' }}>
              ⚠️ CẦN XÁC NHẬN – {station.name}{station.fms_slot_id ? ` (Slot ${station.fms_slot_id})` : ''}
            </h3>
            <p style={{ margin: '0 0 10px', fontSize: '13px' }}>
              {station.state.confirmed?.message} · {station.state.confirmed?.code}. AGV đang dừng chờ tại vạch an toàn.
            </p>
            {station.snapshot && (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={station.snapshot} alt="Ảnh cận cảnh ô kiểm định" style={{ width: '100%', borderRadius: '8px', border: '1px solid #f59e0b', marginBottom: '12px' }} />
            )}
            <div style={{ display: 'flex', gap: '10px', justifyContent: 'flex-end' }}>
              <button type="button" data-testid={`inspection-reject-${station.rule_id}`} disabled={busy === station.rule_id}
                onClick={() => decide(station, false)}
                style={{ padding: '10px 16px', borderRadius: '8px', border: '1px solid #f43f5e', background: '#4c0519', color: '#fecdd3', fontWeight: 800, cursor: 'pointer' }}>
                Từ chối
              </button>
              <button type="button" data-testid={`inspection-approve-${station.rule_id}`} disabled={busy === station.rule_id}
                onClick={() => decide(station, true)}
                style={{ padding: '10px 16px', borderRadius: '8px', border: '1px solid #10b981', background: '#064e3b', color: '#d1fae5', fontWeight: 800, cursor: 'pointer' }}>
                Chấp thuận
              </button>
            </div>
          </div>
        </div>
      ))}
    </>
  );
}
