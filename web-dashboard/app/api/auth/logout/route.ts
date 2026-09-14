import { NextRequest, NextResponse } from 'next/server';
import { deleteSession } from '../../../../lib/auth';
import { isSameOriginRequest } from '../../../../lib/auth-request';

export const dynamic = 'force-dynamic';

export async function POST(request: NextRequest) {
  const headers = { 'Cache-Control': 'no-store' };
  if (!isSameOriginRequest(request)) return NextResponse.json({ error: 'Invalid request origin.' }, { status: 403, headers });
  if (!request.headers.get('content-type')?.startsWith('application/json')) return NextResponse.json({ error: 'JSON required.' }, { status: 415, headers });
  await deleteSession();
  return NextResponse.json({ success: true }, { headers });
}
