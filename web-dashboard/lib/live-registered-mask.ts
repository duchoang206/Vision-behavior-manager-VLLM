import { RegisteredMask, validRegisteredMask } from './registered-mask';

export const MASK_BRIDGE_MS = 1200;

type MaskBox = [number, number, number, number];

type MaskObservation = {
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
};

export type LiveRegisteredMask = {
  identity: string;
  mask: RegisteredMask;
  box: MaskBox;
  receivedAt: number;
  observedAt?: number;
};

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
  const identity = observation.label?.trim().toLowerCase();
  const box: MaskBox = [observation.x, observation.y, observation.w, observation.h];
  const receivedAt = metadataReceivedAt(now, observation.observed_at, sentAt, transportAgeMs);
  if (!identity || !/robot|rack|agv|amr/i.test(observation.class)
    || /lost|removed|deleted/i.test(observation.tracking_state || '')
    || !box.every(Number.isFinite) || box[2] <= 0 || box[3] <= 0
    || now - receivedAt > MASK_BRIDGE_MS) return null;

  const retained = previous?.identity === identity ? previous : null;
  if (retained?.observedAt !== undefined && observation.observed_at !== undefined
    && observation.observed_at < retained.observedAt) {
    return now - retained.receivedAt <= MASK_BRIDGE_MS ? retained : null;
  }

  const propagated = observation.mask?.source === 'sam2_optical_flow';
  if (validRegisteredMask(observation.mask) && (!observation.mask_stale || propagated)) {
    const repeated = retained?.observedAt !== undefined && retained.observedAt === observation.observed_at;
    const maskReceivedAt = repeated ? retained.receivedAt : receivedAt;
    if (now - maskReceivedAt > MASK_BRIDGE_MS) return null;
    return { identity, mask: observation.mask, box, receivedAt: maskReceivedAt, observedAt: observation.observed_at };
  }

  if (!retained || now - retained.receivedAt > MASK_BRIDGE_MS) return null;
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
    mask: {
      ...retained.mask,
      polygons: retained.mask.polygons.map(ring => ring.map(point => [
        clamp(box[0] + (point[0] - retained.box[0]) * scaleX),
        clamp(box[1] + (point[1] - retained.box[1]) * scaleY),
      ])),
    },
  };
}
