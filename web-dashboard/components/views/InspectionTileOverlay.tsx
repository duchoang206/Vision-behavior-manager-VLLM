'use client';

// Inspection cells drawn on a Monitor camera tile: the cell outline coloured by
// its confirmed occupancy (CÓ HÀNG / TRỐNG / KHÔNG XÁC ĐỊNH), the label, and what
// image processing and the AI found in it.  Each station picks what it shows
// (Building view > "Hiển thị trên Monitor"); the viewer can hide them all from
// the Monitor HUD.  The SVG viewBox is the video's pixel size with
// preserveAspectRatio "xMidYMid meet", which letterboxes exactly like the
// <video style="object-fit: contain"> underneath.

import React from 'react';
import { occupancyColor, occupancyText, orderClockwise } from '../../lib/roi-rules';
import { useInspectionOverlayVisible, useInspectionStations } from '../../lib/inspection-stations';

const DEFAULT_SIZE = [1920, 1080];

export default function InspectionTileOverlay({ camId, active }: { camId: string; active: boolean }) {
  const { stations } = useInspectionStations(active);
  const visible = useInspectionOverlayVisible();
  if (!active || !visible) return null;
  const mine = stations.filter(s => s.cam_id === camId && s.display?.overlay !== false && s.points?.length === 4);
  if (!mine.length) return null;
  const [width, height] = mine.find(s => s.frame_size?.length === 2)?.frame_size || DEFAULT_SIZE;
  const unit = Math.max(width, height) / 1000;
  const toPoints = (polygon: number[][]) => polygon.map(p => `${p[0] * width},${p[1] * height}`).join(' ');

  return (
    <svg data-testid={`inspection-tile-${camId}`} viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="xMidYMid meet"
      style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none', zIndex: 11 }}>
      {mine.map(station => {
        const occupancy = station.occupancy;
        const color = occupancyColor(occupancy?.state);
        const outline = orderClockwise(station.points as number[][]);
        const anchor = outline.reduce((top, p) => (p[1] < top[1] ? p : top), outline[0]);
        const overlay = station.overlay;
        const label = `${station.name || station.rule_id}${station.fms_slot_id ? ` [Slot ${station.fms_slot_id}]` : ''} · ${occupancyText(occupancy)}`;
        return (
          <g key={`${station.cam_id}:${station.rule_id}`} data-testid={`inspection-tile-cell-${station.rule_id}`}
            data-state={occupancy?.state || 'PENDING'}>
            <polygon points={toPoints(outline)} fill={`${color}26`} stroke={color} strokeWidth={3 * unit} strokeLinejoin="round" />
            {station.display?.object !== false && overlay && (
              <>
                {(overlay.objects || []).map((polygon, index) => (
                  <polygon key={`cv-${index}`} points={toPoints(polygon)} fill="none" stroke={color}
                    strokeWidth={2 * unit} strokeDasharray={`${9 * unit} ${6 * unit}`} />
                ))}
                {overlay.ai_boxes.map((box, index) => (
                  <g key={`ai-${index}`}>
                    <polygon points={toPoints(box.polygon)} fill="none" stroke="#38bdf8" strokeWidth={2 * unit} />
                    <text x={box.polygon[0][0] * width + 4 * unit} y={box.polygon[0][1] * height - 6 * unit}
                      fill="#38bdf8" fontSize={16 * unit} fontWeight={800} stroke="#06070a" strokeWidth={4 * unit}
                      paintOrder="stroke">{box.label}</text>
                  </g>
                ))}
              </>
            )}
            {station.display?.label !== false && (
              <text data-testid={`inspection-tile-label-${station.rule_id}`} x={anchor[0] * width} y={anchor[1] * height - 10 * unit}
                fill={color} fontSize={20 * unit} fontWeight={900} stroke="#06070a" strokeWidth={5 * unit}
                paintOrder="stroke">📐 {label}</text>
            )}
          </g>
        );
      })}
    </svg>
  );
}
