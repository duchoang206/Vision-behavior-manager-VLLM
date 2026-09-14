import { NextResponse } from 'next/server';
import { getSession } from '../../../../lib/auth';

export const dynamic = 'force-dynamic';

export async function GET() {
  const session = await getSession();
  const headers = { 'Cache-Control': 'no-store' };
  return session ? NextResponse.json({ authenticated: true, username: session.username, expiresAt: session.expiresAt }, { headers }) : NextResponse.json({ authenticated: false }, { status: 401, headers });
}
