import type { Metadata } from 'next';
import './globals.css';
import React from 'react';
import DashboardShell from '../components/DashboardShell';
import { LanguageProvider } from '../components/LanguageContext';
import { ThemeProvider } from '../components/ThemeContext';
import { TabProvider } from '../components/TabContext';
import { CameraProvider } from '../components/CameraContext';

export const metadata: Metadata = {
  title: 'VMS-RTC',
  description: 'Vision AI YOLO Backend System',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <ThemeProvider>
          <LanguageProvider>
            <CameraProvider>
              <TabProvider>
                <DashboardShell>{children}</DashboardShell>
            </TabProvider>
          </CameraProvider>
        </LanguageProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
