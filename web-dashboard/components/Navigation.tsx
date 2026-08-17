'use client';

import React from 'react';
import { useLanguage } from './LanguageContext';
import { useTab } from './TabContext';
import { Video, Building2, Bot, BarChart3 } from 'lucide-react';

export default function Navigation() {
  const { t } = useLanguage();
  const { activeTab, setActiveTab } = useTab();

  const navItems = [
    {
      id: 'monitor',
      label: t.nav.monitor,
      icon: Video,
      isLive: false,
    },
    {
      id: 'building',
      label: t.nav.building,
      icon: Building2,
      isLive: false,
    },
    {
      id: 'robot_map',
      label: t.nav.robotMap,
      icon: Bot,
      isLive: true, // Show dynamic live telemetry beacon
    },
    {
      id: 'analytics',
      label: t.nav.analytics,
      icon: BarChart3,
      isLive: false,
    },
  ];

  return (
    <nav className="fms-nav" aria-label="Main Navigation">
      {navItems.map((item) => {
        const IconComponent = item.icon;
        const isActive = activeTab === item.id;

        return (
          <button
            key={item.id}
            type="button"
            onClick={() => setActiveTab(item.id as any)}
            className={`fms-nav-item ${isActive ? 'active' : ''}`}
            aria-current={isActive ? 'page' : undefined}
          >
            <IconComponent
              size={17}
              style={{
                color: isActive ? 'var(--accent)' : 'inherit',
                transition: 'color 0.2s ease, transform 0.2s ease',
                transform: isActive ? 'scale(1.1)' : 'scale(1)',
              }}
            />
            <span>{item.label}</span>

            {item.isLive && (
              <span
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '4px',
                  padding: '2px 6px',
                  borderRadius: '10px',
                  fontSize: '9px',
                  fontWeight: 700,
                  letterSpacing: '0.06em',
                  fontFamily: "'JetBrains Mono', monospace",
                  background: 'rgba(16, 185, 129, 0.15)',
                  color: '#10b981',
                  border: '1px solid rgba(16, 185, 129, 0.35)',
                  marginLeft: '2px',
                }}
              >
                <span
                  style={{
                    width: '5px',
                    height: '5px',
                    borderRadius: '50%',
                    background: '#10b981',
                    animation: 'liveDotBlink 1.4s infinite ease-in-out',
                    boxShadow: '0 0 6px #10b981',
                  }}
                />
                LIVE
              </span>
            )}
          </button>
        );
      })}
    </nav>
  );
}
