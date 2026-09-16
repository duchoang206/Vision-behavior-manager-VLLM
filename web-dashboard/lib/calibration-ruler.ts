import { Point2 } from './calibration-points';

export type ImageViewport = [number, number, number, number];
export type RulerCalibration = {
  matrix?: number[][];
  method?: string;
  coverage_polygon?: Point2[];
  calibration_updated_at?: number;
  calibrated_at?: number;
};
export type RulerMeasurement = {
  camera_id: string;
  units: 'm';
  points: { camera: Point2; floor: Point2; fms: Point2 | null; inside_calibrated_area: boolean | null }[];
  segments: { from: number; to: number; distance_m: number }[];
  total_distance_m: number;
  direct_distance_m: number;
  warnings: string[];
  calibration: { method: string; updated_at: number | null; point_count: number; reprojection_error_m: number | null };
};

export function clampImageViewport(view: ImageViewport, size: Point2): ImageViewport {
  const width = Math.min(size[0], Math.max(size[0] / 16, view[2]));
  const height = width * size[1] / size[0];
  return [Math.max(0, Math.min(size[0] - width, view[0])), Math.max(0, Math.min(size[1] - height, view[1])), width, height];
}

export function zoomImageViewport(view: ImageViewport, size: Point2, factor: number, center?: Point2): ImageViewport {
  const width = Math.min(size[0], Math.max(size[0] / 16, view[2] * factor));
  const anchor = center || [view[0] + view[2] / 2, view[1] + view[3] / 2];
  const ratio = width / view[2];
  return clampImageViewport([anchor[0] - (anchor[0] - view[0]) * ratio, anchor[1] - (anchor[1] - view[1]) * ratio, width, view[3] * ratio], size);
}

export function calibrationMethodLabel(method?: string) {
  if (method === 'auto_robot_fms') return 'Robot + FMS';
  if (method === 'manual_camera_fms_click') return 'Chấm điểm Camera ↔ FMS';
  return method || 'Homography';
}
