import type { Metadata } from 'next';
import './globals.css';
import React from 'react';
import WorkspaceLayout from '../components/WorkspaceLayout';
import { LanguageProvider } from '../components/LanguageContext';
import { ThemeProvider } from '../components/ThemeContext';

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
            <WorkspaceLayout>{children}</WorkspaceLayout>
          </LanguageProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
