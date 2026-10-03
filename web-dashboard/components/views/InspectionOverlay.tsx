'use client';

// SVG layer drawn inside the Building view camera canvas (viewBox 0..1):
// numbered ROI vertices of an inspection draft, and the result of "Test thử" -
// the occupancy answer (CÓ HÀNG / TRỐNG / KHÔNG XÁC ĐỊNH), what image processing
// and the AI found, and the metrology read-out (rotated OBB, centre-offset
// vector, mm), all mapped back from the BEV to camera coordinates by the backend.

import React from 'react';
import { occupancyColor, occupancyText, type InspectionReport } from '../../lib/roi-rules';

const COLORS: Record<string, string> = { OK: '#10b981', NG: '#f43f5e', UNCERTAIN: '#f59e0b', DETECTED: '#22d3ee' };
const pointsAttr = (points: number[][]) => points.map(p => `${p[0]},${p[1]}`).join(' ');

export function InspectionVertexLabels({ points }: { points: number[][] }) {
  return (
    <g data-testid="inspection-vertex-labels">
      {points.map((point, index) => (
        <text key={index} x={point[0] + 0.012} y={point[1] - 0.012} fill="#fbbf24" fontSize="0.03" fontWeight="bold"
          stroke="#06070a" strokeWidth="0.004" paintOrder="stroke">({index + 1})</text>
      ))}
    </g>
  );
}

export default function InspectionOverlay({ report }: { report: InspectionReport }) {
  const overlay = report.overlay;
  if (!overlay) return null;
  const color = COLORS[report.status] || '#ffffff';
  const occupancy = report.occupancy;
  const occColor = occupancyColor(occupancy?.state);
  const badge = occupancy ? occupancyText(occupancy) : report.status;
  const m = report.measurement;
  const obb = overlay.obb;
  return (
    <g data-testid="inspection-overlay">
      {overlay.ai_boxes.map((box, index) => (
        <g key={`ai-${index}`}>
          <polygon points={pointsAttr(box.polygon)} fill="none" stroke="#38bdf8" strokeWidth="0.003" strokeDasharray="0.01" />
          <text x={box.polygon[0][0]} y={box.polygon[0][1] - 0.008} fill="#38bdf8" fontSize="0.022" fontWeight="bold"
            stroke="#06070a" strokeWidth="0.003" paintOrder="stroke">{box.label}</text>
        </g>
      ))}
      {(overlay.objects || []).slice(obb ? 1 : 0).map((polygon, index) => (
        <polygon key={`cv-${index}`} points={pointsAttr(polygon)} fill={`${occColor}1f`} stroke={occColor}
          strokeWidth="0.004" strokeDasharray="0.012 0.008" />
      ))}
      {obb && (
        <polygon data-testid="inspection-obb" points={pointsAttr(obb)} fill={`${color}22`} stroke={color} strokeWidth="0.005" />
      )}
      <circle cx={overlay.cell_center[0]} cy={overlay.cell_center[1]} r="0.006" fill="none" stroke="#ffffff" strokeWidth="0.003" />
      {overlay.center && (
        <>
          <line x1={overlay.cell_center[0]} y1={overlay.cell_center[1]} x2={overlay.center[0]} y2={overlay.center[1]}
            stroke={color} strokeWidth="0.004" />
          <circle cx={overlay.center[0]} cy={overlay.center[1]} r="0.007" fill={color} />
        </>
      )}
      {m && obb && (
        <text x={Math.min(...obb.map(p => p[0]))} y={Math.max(...obb.map(p => p[1])) + 0.035} fill={color} fontSize="0.026"
          fontWeight="bold" stroke="#06070a" strokeWidth="0.004" paintOrder="stroke">
          {`${m.width_mm.toFixed(0)}×${m.height_mm.toFixed(0)} mm · Δ=${m.offset_mm.toFixed(0)} mm · θ=${m.rotation_deg.toFixed(1)}°`}
        </text>
      )}
      <g transform={`translate(${overlay.cell_center[0]}, ${Math.max(0.03, overlay.cell_center[1] - 0.2)})`}>
        <text data-testid="inspection-overlay-badge" data-state={occupancy?.state || report.status} x="0" y="0.032"
          textAnchor="middle" fill={occColor} fontSize="0.034" fontWeight="900" stroke="#06070a" strokeWidth="0.008"
          paintOrder="stroke">{badge}</text>
      </g>
    </g>
  );
}
