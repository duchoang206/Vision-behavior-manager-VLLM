'use client';

import React, { createContext, useContext, useState, useEffect } from 'react';

// ─── Neo-Kinpaku Dark palette (Dark Lacquer + Kinpaku Gold + Verdigris Patina) ─────────────
export const DARK_COLORS = {
  bg:           '#06070a', // Lacquer Black ground
  elevated:     '#0c0f16', // Lacquer Deep
  surface:      '#0c0f16',
  card:         '#11151f', // Raised Lacquer
  cardAlt:      '#161b27', // Graphite
  border:       'rgba(255, 255, 255, 0.09)',
  borderHard:   'rgba(245, 158, 11, 0.35)', // Kinpaku hairline
  borderSubtle: 'rgba(255, 255, 255, 0.05)',
  textPrimary:  '#f8fafc',
  textLabel:    '#e2e8f0',
  textSecondary:'#cbd5e1',
  textSub:      '#94a3b8',
  textMuted:    '#64748b',
  // Primary brand anchor: Kinpaku Gold
  accent:       '#f59e0b',
  accentL:      '#fbbf24',
  accentGlow:   '#fbbf24',
  accentAlt:    '#d97706',
  accentDim:    'rgba(245, 158, 11, 0.12)',
  accentBorder: 'rgba(245, 158, 11, 0.38)',
  // Secondary brand anchor: Verdigris Patina
  cyan:         '#06b6d4',
  cyanL:        '#22d3ee',
  cyanDim:      'rgba(6, 182, 212, 0.12)',
  cyanBorder:   'rgba(6, 182, 212, 0.35)',
  // States
  rose:         '#f43f5e',
  roseDim:      'rgba(244, 63, 94, 0.12)',
  roseBorder:   'rgba(244, 63, 94, 0.35)',
  emerald:      '#10b981',
  emeraldDim:   'rgba(16, 185, 129, 0.12)',
  emeraldBorder:'rgba(16, 185, 129, 0.35)',
  violet:       '#8b5cf6',
  violetDim:    'rgba(139, 92, 246, 0.12)',
  violetBorder: 'rgba(139, 92, 246, 0.35)',
  amber:        '#f59e0b',
  orange:       '#f97316',
  // Chart Palette (harmonious Kinpaku & Patina tones)
  chart: ['#f59e0b', '#06b6d4', '#10b981', '#f43f5e', '#8b5cf6'],
};

// ─── Light palette (Tally / Warm Paper) ─────────────────────────────────────────────────
export const LIGHT_COLORS = {
  bg:           '#f8f6f0', // Warm paper
  elevated:     '#ffffff',
  surface:      '#ffffff',
  card:         '#ffffff',
  cardAlt:      '#f3efe6',
  border:       'rgba(214, 206, 192, 0.85)',
  borderHard:   'rgba(180, 115, 20, 0.4)',
  borderSubtle: 'rgba(230, 224, 212, 0.9)',
  textPrimary:  '#181511',
  textLabel:    '#2d2720',
  textSecondary:'#473f35',
  textSub:      '#6e6355',
  textMuted:    '#948777',
  accent:       '#d97706',
  accentL:      '#b45309',
  accentGlow:   '#d97706',
  accentAlt:    '#92400e',
  accentDim:    'rgba(217, 119, 6, 0.09)',
  accentBorder: 'rgba(217, 119, 6, 0.32)',
  cyan:         '#0891b2',
  cyanL:        '#0e7490',
  cyanDim:      'rgba(8, 145, 178, 0.08)',
  cyanBorder:   'rgba(8, 145, 178, 0.28)',
  rose:         '#e11d48',
  roseDim:      'rgba(225, 29, 72, 0.08)',
  roseBorder:   'rgba(225, 29, 72, 0.25)',
  emerald:      '#059669',
  emeraldDim:   'rgba(5, 150, 105, 0.08)',
  emeraldBorder:'rgba(5, 150, 105, 0.25)',
  violet:       '#7c3aed',
  violetDim:    'rgba(124, 58, 237, 0.08)',
  violetBorder: 'rgba(124, 58, 237, 0.25)',
  amber:        '#d97706',
  orange:       '#ea580c',
  chart: ['#d97706', '#0891b2', '#059669', '#e11d48', '#7c3aed'],
};

export type ThemeMode = 'dark' | 'light';
export type ColorPalette = typeof DARK_COLORS;

interface ThemeContextValue {
  mode: ThemeMode;
  colors: ColorPalette;
  theme: ThemeMode;
  toggleTheme: () => void;
  isDark: boolean;
}

const ThemeContext = createContext<ThemeContextValue>({
  mode: 'dark',
  colors: DARK_COLORS,
  theme: 'dark',
  toggleTheme: () => {},
  isDark: true,
});

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [mode, setMode] = useState<ThemeMode>('dark');

  // Load from localStorage on mount
  useEffect(() => {
    try {
      const saved = localStorage.getItem('vms-theme') as ThemeMode | null;
      if (saved === 'light' || saved === 'dark') setMode(saved);
    } catch {}
  }, []);

  // Apply data-theme attribute to <html> for CSS class-based theming
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', mode);
    try { localStorage.setItem('vms-theme', mode); } catch {}
  }, [mode]);

  const toggleTheme = () => setMode(prev => (prev === 'dark' ? 'light' : 'dark'));
  const colors = mode === 'dark' ? DARK_COLORS : LIGHT_COLORS;

  return (
    <ThemeContext.Provider value={{ mode, colors, theme: mode, toggleTheme, isDark: mode === 'dark' }}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useAppTheme() {
  return useContext(ThemeContext);
}
