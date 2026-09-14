'use client';

import { useEffect } from 'react';

export default function AuthSessionGuard({ expiresAt, children }: { expiresAt: number; children: React.ReactNode }) {
  useEffect(() => {
    let active = true;
    let redirecting = false;
    function navigateToLogin(path: string) {
      if (!active || redirecting) return;
      redirecting = true;
      window.location.replace(path);
    }
    function returnToLogin() {
      navigateToLogin(`/login?next=${encodeURIComponent(window.location.pathname + window.location.search)}`);
    }
    function handleSignOut() {
      navigateToLogin('/login');
    }
    function handleStorage(event: StorageEvent) {
      if (event.key === 'rskyview-signout') void checkSession();
    }
    async function checkSession() {
      try {
        const response = await fetch('/api/auth/session', { cache: 'no-store', signal: AbortSignal.timeout(8000) });
        if (response.status === 401) returnToLogin();
      } catch {}
    }
    void checkSession();
    const interval = window.setInterval(checkSession, 60_000);
    const expiry = window.setTimeout(returnToLogin, Math.max(0, expiresAt * 1000 - Date.now()));
    window.addEventListener('focus', checkSession);
    window.addEventListener('pageshow', checkSession);
    window.addEventListener('storage', handleStorage);
    window.addEventListener('rskyview-signout', handleSignOut);
    return () => {
      active = false;
      window.clearInterval(interval);
      window.clearTimeout(expiry);
      window.removeEventListener('focus', checkSession);
      window.removeEventListener('pageshow', checkSession);
      window.removeEventListener('storage', handleStorage);
      window.removeEventListener('rskyview-signout', handleSignOut);
    };
  }, [expiresAt]);
  return children;
}
