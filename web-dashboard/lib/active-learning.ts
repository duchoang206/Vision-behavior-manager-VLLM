export type LearningPoint = [number, number];
export type LearningAnnotation = { id: string; class_name: string; label: string; polygons: LearningPoint[][] };
export type LearningPrediction = { id: number; class: string; label?: string; x: number; y: number; w: number; h: number;
  mask?: { polygons: LearningPoint[][]; frame_id?: number } };
export type LearningSample = {
  id: string; model_id: string; camera_id: string; frame_id: number; captured_at: number;
  status: 'draft' | 'approved' | 'rejected'; created_at: string; model_version: string;
  frame: { width: number; height: number; source_id: number; frame_pts_ns: number; generation: string };
  predictions: LearningPrediction[]; annotations: LearningAnnotation[];
};
export type LearningSettings = { enabled: boolean; threshold: number; daily_hour: number | null; timezone: string;
  epochs: number; learning_rate: number; batch: number; minimum_gain: number; max_class_drop: number;
  auto_promote: boolean; base_dataset: string };
export type LearningMetrics = { map50_95: number; map50: number; per_class: Record<string, number> };
export type LearningJob = { id: string; state: string; reason?: string; created_at: string; trigger: string;
  cancel_requested: boolean; metrics: { baseline?: LearningMetrics; candidate?: LearningMetrics;
    history?: Record<string, string>[]; accepted?: boolean; reason?: string; validation_images?: number };
  version?: { id: string; reload?: { mode: string } } };
export type LearningStatus = { settings: LearningSettings; prerequisites: string[]; directory: string; jobs: LearningJob[];
  weights: { id: string; filename: string; state: string; received_bytes: number; size_bytes: number }[];
  worker: { running: boolean; job_active?: boolean; state?: string; note: string; error?: string; gpu_policy: string } };

export async function learningRequest<Type>(modelId: string, path: string, init?: RequestInit): Promise<Type> {
  const response = await fetch(`/api/backend/active-learning/models/${encodeURIComponent(modelId)}${path}`, {
    ...init, headers: { ...(typeof init?.body === 'string' ? { 'Content-Type': 'application/json' } : {}), ...init?.headers }, cache: 'no-store',
  });
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : `Yêu cầu không hợp lệ (${response.status}).`);
  return data;
}

export const learningImage = (sample: LearningSample) => `/api/backend/active-learning/models/${sample.model_id}/samples/${sample.id}/image`;
export const polygonPath = (polygons: LearningPoint[][]) => polygons.map(ring => `M ${ring.map(point => point.join(' ')).join(' L ')} Z`).join(' ');
export const learningStates: Record<string, string> = {
  queued: 'Đã xếp hàng', waiting_resources: 'Chờ GPU rảnh', preparing: 'Chuẩn bị dataset', training: 'Đang fine-tune',
  building: 'Export / build / kiểm định', ready: 'Đạt kiểm định — chờ deploy', rejected: 'Không đạt kiểm định',
  promotion_queued: 'Chờ đổi engine', promoting: 'Đang đổi engine', completed: 'Đã áp dụng', failed: 'Thất bại', cancelled: 'Đã hủy',
};
