import { RegisteredMask } from './registered-mask';

const PALETTE = [
  [45, 212, 191], [96, 165, 250], [192, 132, 252],
  [251, 191, 36], [244, 114, 182], [163, 230, 53],
];

export function segmentationColors(identity: string, alert = false) {
  let hash = 0;
  for (const character of identity.trim().toLowerCase()) hash = ((hash << 5) - hash + character.charCodeAt(0)) | 0;
  const channels = alert ? [248, 113, 113] : PALETTE[(hash >>> 0) % PALETTE.length];
  return { stroke: `rgb(${channels.join(',')})`, fill: `rgba(${channels.join(',')},0.24)` };
}

export function segmentationPath(mask: RegisteredMask, width: number, height: number, offsetX: number, offsetY: number) {
  const path = new Path2D();
  for (const ring of mask.polygons) {
    const points = ring.map(point => [offsetX + point[0] * width, offsetY + point[1] * height]);
    points.forEach((point, index) => {
      const previous = points[(index + points.length - 1) % points.length];
      const next = points[(index + 1) % points.length];
      const incoming = Math.hypot(previous[0] - point[0], previous[1] - point[1]);
      const outgoing = Math.hypot(next[0] - point[0], next[1] - point[1]);
      const radius = Math.min(.65, incoming / 4, outgoing / 4);
      const startX = point[0] + (previous[0] - point[0]) * radius / (incoming || 1);
      const startY = point[1] + (previous[1] - point[1]) * radius / (incoming || 1);
      if (index === 0) path.moveTo(startX, startY);
      else path.lineTo(startX, startY);
      path.quadraticCurveTo(point[0], point[1],
        point[0] + (next[0] - point[0]) * radius / (outgoing || 1),
        point[1] + (next[1] - point[1]) * radius / (outgoing || 1));
    });
    path.closePath();
  }
  return path;
}
