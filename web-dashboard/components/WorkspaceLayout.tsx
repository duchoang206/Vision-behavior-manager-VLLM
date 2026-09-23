'use client';

import { usePathname } from 'next/navigation';
import { CameraProvider } from './CameraContext';
import { TabProvider } from './TabContext';
import DashboardShell from './DashboardShell';
import type { ReactNode } from 'react';

export default function WorkspaceLayout({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  if (pathname === '/login') return <>{children}</>;
  return <CameraProvider><TabProvider><DashboardShell>{children}</DashboardShell></TabProvider></CameraProvider>;
}
