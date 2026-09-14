import 'server-only';

import type { NextRequest } from 'next/server';

export function isSameOriginRequest(request: NextRequest) {
  if (request.headers.get('sec-fetch-site') === 'cross-site') return false;
  const origin = request.headers.get('origin');
  if (!origin) return true;
  try {
    const parsed = new URL(origin);
    return ['http:', 'https:'].includes(parsed.protocol) && parsed.host === request.headers.get('host');
  } catch {
    return false;
  }
}

export function safeNextPath(value: unknown) {
  if (typeof value !== 'string' || !value.startsWith('/') || value.startsWith('//') || /[\\\u0000-\u0020]/.test(value)) return '/';
  try {
    const destination = new URL(value, 'http://rskyview.local');
    if (destination.origin !== 'http://rskyview.local' || !['/', '/monitor', '/building', '/analytics'].includes(destination.pathname)) return '/';
    return `${destination.pathname}${destination.search}${destination.hash}`;
  } catch {
    return '/';
  }
}
