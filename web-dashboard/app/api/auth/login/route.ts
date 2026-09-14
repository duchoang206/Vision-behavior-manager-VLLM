import { NextRequest, NextResponse } from 'next/server';
import { credentialsMatch, setSession } from '../../../../lib/auth';
import { isSameOriginRequest } from '../../../../lib/auth-request';

export const dynamic = 'force-dynamic';

const authState = globalThis as typeof globalThis & { rskyviewLoginWindow?: { count: number; resetsAt: number } };

export async function POST(request: NextRequest) {
  const headers = { 'Cache-Control': 'no-store' };
  if (!isSameOriginRequest(request)) return NextResponse.json({ error: 'Invalid request origin.' }, { status: 403, headers });
  if (!request.headers.get('content-type')?.startsWith('application/json')) return NextResponse.json({ error: 'JSON required.' }, { status: 415, headers });
  const contentLength = Number(request.headers.get('content-length') || 0);
  if (contentLength > 4096) return NextResponse.json({ error: 'Invalid request.' }, { status: 400, headers });

  let body: { username?: unknown; password?: unknown };
  try {
    const text = await request.text();
    if (text.length > 4096) return NextResponse.json({ error: 'Invalid request.' }, { status: 400, headers });
    body = JSON.parse(text);
    if (!body || typeof body !== 'object' || Array.isArray(body)) return NextResponse.json({ error: 'Invalid request.' }, { status: 400, headers });
  } catch {
    return NextResponse.json({ error: 'Invalid request.' }, { status: 400, headers });
  }
  const username = typeof body.username === 'string' ? body.username.trim() : '';
  const password = typeof body.password === 'string' ? body.password : '';
  if (!username || !password || username.length > 128 || password.length > 256) {
    return NextResponse.json({ error: 'Invalid credentials.' }, { status: 401, headers });
  }
  if (!authState.rskyviewLoginWindow || authState.rskyviewLoginWindow.resetsAt <= Date.now()) {
    authState.rskyviewLoginWindow = { count: 0, resetsAt: Date.now() + 60_000 };
  }
  if (authState.rskyviewLoginWindow.count >= 8) {
    return NextResponse.json({ error: 'Too many attempts.' }, {
      status: 429, headers: { ...headers, 'Retry-After': String(Math.ceil((authState.rskyviewLoginWindow.resetsAt - Date.now()) / 1000)) },
    });
  }
  authState.rskyviewLoginWindow.count += 1;
  try {
    if (!(await credentialsMatch(username, password))) return NextResponse.json({ error: 'Invalid credentials.' }, { status: 401, headers });
    const secure = request.nextUrl.protocol === 'https:' || request.headers.get('x-forwarded-proto') === 'https';
    await setSession(username, secure);
    authState.rskyviewLoginWindow.count = 0;
    return NextResponse.json({ success: true }, { headers });
  } catch {
    return NextResponse.json({ error: 'Sign-in service unavailable.' }, { status: 503, headers });
  }
}
