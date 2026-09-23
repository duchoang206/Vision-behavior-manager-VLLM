import { useState } from 'react';
import { CalibrationPair, Point2 } from './calibration-points';

export type LengthConstraint = { points: Point2[]; distance_m: number };
export type DeepCalibProfile = {
  image_width: number; image_height: number; focal_length_px: number; distortion_xi: number;
  rectified_width?: number; rectified_height?: number; rectified_focal_px?: number;
  rectified_cx?: number; rectified_cy?: number; geometry_version?: number;
  focal_confidence?: number; distortion_confidence?: number;
};
export type DeepCalibPreview = {
  camera_id: string; preview_id: string; rectified_image: string;
  profile: DeepCalibProfile; snapshot: string;
};
export type DeepCalibStatus = {
  name: string; weights_available: boolean; legacy_runtime_available: boolean;
  repository_url?: string; revision?: string; reason?: string;
};

export function useCalibrationDraft() {
  const [pairs, setPairs] = useState<CalibrationPair[]>([]);
  const [pending, setPending] = useState<Point2 | null>(null);
  const [lengthConstraints, setLengthConstraints] = useState<LengthConstraint[]>([]);
  const [lengthPoints, setLengthPoints] = useState<Point2[]>([]);
  const [lengthMode, setLengthMode] = useState(false);
  const [dirty, setDirty] = useState(false);
  return { pairs, setPairs, pending, setPending, lengthConstraints, setLengthConstraints,
    lengthPoints, setLengthPoints, lengthMode, setLengthMode, dirty, setDirty };
}

export async function snapshotDataUrl(snapshot: string, signal: AbortSignal): Promise<string> {
  const response = await fetch(snapshot, { signal });
  if (!response.ok) throw new Error('Không đọc được snapshot hiện tại. Hãy lấy frame mới.');
  const blob = await response.blob();
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(new Error('Không mã hóa được snapshot.'));
    reader.readAsDataURL(blob);
  });
}

export function visibleRectifiedPoint(point: Point2, profile: DeepCalibProfile): boolean {
  const width = profile.rectified_width ?? profile.image_width;
  const height = profile.rectified_height ?? profile.image_height;
  const focal = profile.rectified_focal_px ?? profile.focal_length_px;
  const rayX = (point[0] * width - (profile.rectified_cx ?? width / 2)) / focal;
  const rayY = (point[1] * height - (profile.rectified_cy ?? height / 2)) / focal;
  const denominator = 1 + profile.distortion_xi * Math.sqrt(1 + rayX * rayX + rayY * rayY);
  const rawX = rayX * profile.focal_length_px / denominator + profile.image_width / 2;
  const rawY = rayY * profile.focal_length_px / denominator + profile.image_height / 2;
  return rawX >= 0 && rawX <= profile.image_width && rawY >= 0 && rawY <= profile.image_height;
}
