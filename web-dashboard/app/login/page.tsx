import { redirect } from 'next/navigation';
import LoginScreen from '../../components/LoginScreen';
import { getSession } from '../../lib/auth';
import { safeNextPath } from '../../lib/auth-request';

export const metadata = { title: 'Welcome back | R-SkyView · RTC Technology' };

export default async function LoginPage({ searchParams }: { searchParams: Promise<{ next?: string | string[] }> }) {
  const nextPath = safeNextPath((await searchParams).next);
  if (await getSession()) redirect(nextPath);
  return <LoginScreen nextPath={nextPath} />;
}
