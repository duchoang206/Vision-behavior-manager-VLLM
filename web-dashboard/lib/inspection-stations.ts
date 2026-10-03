'use client';

// One shared poll of the inspection stations (/api/inspection/states) for every
// Monitor consumer - the HUD and each camera tile - and the viewer's switch to
// show or hide the inspection cells on the camera tiles (kept per browser).

import { useEffect, useSyncExternalStore } from 'react';
import { inspectionApi, type InspectionStation } from './roi-rules';

const POLL_MS = 1000;
const OVERLAY_KEY = 'rsky.inspection.monitorOverlay';

type Snapshot = { stations: InspectionStation[]; error: string };

const EMPTY: Snapshot = { stations: [], error: '' };
let snapshot: Snapshot = EMPTY;
let overlayVisible: boolean | null = null;
let timer: ReturnType<typeof setInterval> | null = null;
let consumers = 0;
const listeners = new Set<() => void>();

const emit = () => listeners.forEach(listener => listener());

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => { listeners.delete(listener); };
}

async function poll(force = false) {
  if (!force && typeof document !== 'undefined' && document.hidden) return;
  try {
    snapshot = { stations: await inspectionApi.states(), error: '' };
  } catch {
    snapshot = { ...snapshot, error: 'Mất kết nối trạm kiểm định' };
  }
  emit();
}

export const refreshInspectionStations = () => poll(true);

/** Stations of every camera, polled once per second while at least one consumer is active. */
export function useInspectionStations(active: boolean): Snapshot {
  useEffect(() => {
    if (!active) return;
    consumers += 1;
    if (consumers === 1) {
      void poll(true);
      timer = setInterval(() => { void poll(); }, POLL_MS);
    }
    return () => {
      consumers -= 1;
      if (consumers === 0 && timer) {
        clearInterval(timer);
        timer = null;
      }
    };
  }, [active]);
  return useSyncExternalStore(subscribe, () => snapshot, () => EMPTY);
}

function readOverlayVisible(): boolean {
  if (overlayVisible === null) {
    try {
      overlayVisible = window.localStorage.getItem(OVERLAY_KEY) !== '0';
    } catch {
      overlayVisible = true;
    }
  }
  return overlayVisible;
}

export function setInspectionOverlayVisible(value: boolean) {
  overlayVisible = value;
  try {
    window.localStorage.setItem(OVERLAY_KEY, value ? '1' : '0');
  } catch {
    // Private mode / blocked storage: the switch still works for this page.
  }
  emit();
}

export function useInspectionOverlayVisible(): boolean {
  return useSyncExternalStore(subscribe, readOverlayVisible, () => true);
}
