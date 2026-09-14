import { NextRequest, NextResponse } from 'next/server';
import { SESSION_COOKIE, verifySessionToken } from './lib/auth';
import { isSameOriginRequest } from './lib/auth-request';

const publicPaths = new Set(['/login', '/api/auth/login', '/api/auth/logout', '/favicon.ico', '/icon.png', '/logo.png', '/login-assets/studio-loop.mp4', '/login-assets/studio-poster.jpg']);

export function proxy(request: NextRequest) {
  const { pathname, search } = request.nextUrl;
  const headers = { 'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff', 'X-Frame-Options': 'DENY' };
  if (publicPaths.has(pathname)) return NextResponse.next({ headers: pathname === '/login' ? headers : undefined });
  if (verifySessionToken(request.cookies.get(SESSION_COOKIE)?.value)) {
    if (!['GET', 'HEAD', 'OPTIONS'].includes(request.method) && !isSameOriginRequest(request)) {
      return NextResponse.json({ error: 'Invalid request origin.' }, { status: 403, headers });
    }
    return NextResponse.next({ headers });
  }

  if (pathname.startsWith('/api/')) return NextResponse.json({ error: 'Authentication required.' }, { status: 401, headers });
  const loginUrl = new URL('/login', request.url);
  loginUrl.searchParams.set('next', `${pathname}${search}`);
  return NextResponse.redirect(loginUrl, { headers });
}

export const config = {
  matcher: '/((?!_next/static|_next/image|_next/webpack-hmr).*)',
};
