import { RegisteredMask, validRegisteredMask } from './registered-mask';

export const MASK_BRIDGE_MS = 1200;
export const MODEL_MASK_MAX_AGE_MS = 700;
export const MASK_POSITION_MAX_AGE_MS = 350;

type MaskBox = [number, number, number, number];

type MaskObservation = {
  id?: number;
  model_id?: string;
  category?: string;
  label?: string;
  class: string;
  x: number;
  y: number;
  w: number;
  h: number;
  mask?: RegisteredMask | null;
  observed_at?: number;
  tracking_state?: string;
  mask_stale?: boolean;
  mask_revoked?: boolean;
  identity_verified?: boolean;
};

export type LiveRegisteredMask = {
  identity: string;
  mask: RegisteredMask;
  box: MaskBox;
  receivedAt: number;
  positionedAt: number;
  observedAt?: number;
};

export function isLiveRegisteredMask(frame: LiveRegisteredMask | null | undefined, now: number, maxAgeMs = MODEL_MASK_MAX_AGE_MS): frame is LiveRegisteredMask {
  return Boolean(frame && now - frame.receivedAt <= maxAgeMs
    && now - frame.positionedAt <= MASK_POSITION_MAX_AGE_MS);
}

export function metadataReceivedAt(now: number, observedAt?: number, sentAt?: number, transportAgeMs = 0): number {
  const age = Number.isFinite(observedAt) && Number.isFinite(sentAt)
    ? Math.max(0, sentAt! - observedAt!) : 0;
  return now - age - Math.max(0, transportAgeMs);
}

export function updateLiveRegisteredMask(
  previous: LiveRegisteredMask | null | undefined,
  observation: MaskObservation,
  now: number,
  sentAt?: number,
  transportAgeMs = 0,
): LiveRegisteredMask | null {
  const identity = observation.model_id && observation.id !== undefined
    ? `${observation.model_id}:${observation.id}:${observation.class}` : observation.label?.trim().toLowerCase();
  const box: MaskBox = [observation.x, observation.y, observation.w, observation.h];
  const receivedAt = metadataReceivedAt(now, observation.observed_at, sentAt, transportAgeMs);
  if (!identity || !/robot|rack|agv|amr|forklift|shelf|pallet|kệ/i.test(observation.category || observation.class)
    || observation.mask_revoked || observation.identity_verified === false
    || /lost|removed|deleted/i.test(observation.tracking_state || '')
    || !box.every(Number.isFinite) || box[2] <= 0 || box[3] <= 0
    || now - receivedAt > MASK_BRIDGE_MS) return null;

  const retained = previous?.identity === identity ? previous : null;
  if (retained?.observedAt !== undefined && observation.observed_at !== undefined
    && observation.observed_at < retained.observedAt) {
    return isLiveRegisteredMask(retained, now, observation.model_id ? MODEL_MASK_MAX_AGE_MS : MASK_BRIDGE_MS) ? retained : null;
  }

  const propagated = observation.mask?.source === 'sam2_optical_flow';
  if (validRegisteredMask(observation.mask) && (!observation.mask_stale || propagated)
    && (!observation.model_id || Number.isFinite(observation.mask.observed_at))) {
    const repeated = retained?.observedAt !== undefined && retained.observedAt === observation.observed_at;
    const attachedAt = Number.isFinite(observation.mask.attached_at)
      ? observation.mask.attached_at : observation.mask.observed_at;
    const maskReceivedAt = observation.model_id
      ? metadataReceivedAt(now, observation.mask.observed_at, sentAt, transportAgeMs)
      : repeated ? retained.receivedAt : receivedAt;
    const positionedAt = metadataReceivedAt(now, attachedAt, sentAt, transportAgeMs);
    const frame = { identity, mask: observation.mask, box, receivedAt: maskReceivedAt, positionedAt, observedAt: observation.observed_at };
    return isLiveRegisteredMask(frame, now, observation.model_id ? MODEL_MASK_MAX_AGE_MS : MASK_BRIDGE_MS) ? frame : null;
  }

  const maxAge = observation.model_id ? MODEL_MASK_MAX_AGE_MS : MASK_BRIDGE_MS;
  if (!retained || now - retained.receivedAt > maxAge
    || now - receivedAt > MASK_POSITION_MAX_AGE_MS
    || (observation.model_id && (observation.identity_verified !== true || !Number.isFinite(observation.observed_at)))) return null;
  const scaleX = box[2] / retained.box[2];
  const scaleY = box[3] / retained.box[3];
  const distance = Math.hypot(box[0] + box[2] / 2 - retained.box[0] - retained.box[2] / 2,
    box[1] + box[3] / 2 - retained.box[1] - retained.box[3] / 2);
  if (scaleX < .5 || scaleX > 2 || scaleY < .5 || scaleY > 2
    || distance > Math.hypot(retained.box[2], retained.box[3]) * .75) return null;

  const clamp = (value: number) => Math.max(0, Math.min(1, value));
  return {
    ...retained,
    box,
    positionedAt: receivedAt,
    observedAt: observation.observed_at,
    mask: {
      ...retained.mask,
      polygons: retained.mask.polygons.map(ring => ring.map(point => [
        clamp(box[0] + (point[0] - retained.box[0]) * scaleX),
        clamp(box[1] + (point[1] - retained.box[1]) * scaleY),
      ])),
    },
  };
}
