export type RegisteredMask = {
  polygons: number[][][];
  source?: string;
  confidence?: number;
  observed_at?: number;
};

export function validRegisteredMask(mask?: RegisteredMask | null): mask is RegisteredMask {
  return Boolean(mask && Array.isArray(mask.polygons) && mask.polygons.length > 0 && mask.polygons.length <= 32
    && mask.polygons.every(ring => Array.isArray(ring) && ring.length >= 3
      && ring.every(point => Array.isArray(point) && point.length === 2
        && point.every(value => Number.isFinite(value) && value >= 0 && value <= 1))));
}

export function maskPath(mask: RegisteredMask): string {
  return mask.polygons.map(ring => `M ${ring.map(point => point.join(' ')).join(' L ')} Z`).join(' ');
}
