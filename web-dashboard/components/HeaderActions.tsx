'use client';

import React, { useState, useRef, useEffect } from 'react';
import { useLanguage } from './LanguageContext';
import { useAppTheme } from './ThemeContext';
import { Sun, Moon, Bell, User, FileText, ChevronDown, Check, Activity } from 'lucide-react';

export default function HeaderActions() {
  const { language, t, changeLanguage } = useLanguage();
  const { isDark, toggleTheme, colors } = useAppTheme();
  const [showLangMenu, setShowLangMenu] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setShowLangMenu(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  return (
    <div className="fms-actions">
      {/* Live Engine Heartbeat Indicator */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          padding: '6px 12px',
          borderRadius: '8px',
          background: 'var(--bg-card)',
          border: '1px solid var(--border)',
          marginRight: '4px',
        }}
        title="FMS MQTT Broker & YOLO Analytics Engine Active"
      >
        <span
          style={{
            width: '7px',
            height: '7px',
            borderRadius: '50%',
            background: '#10b981',
            boxShadow: '0 0 8px #10b981',
            animation: 'beaconPulse 2.2s infinite ease-out',
          }}
        />
        <span
          style={{
            fontSize: '11px',
            fontFamily: "'JetBrains Mono', monospace",
            fontWeight: 600,
            color: 'var(--text-sub)',
            letterSpacing: '0.04em',
          }}
        >
          SYS ONLINE
        </span>
      </div>

      {/* Theme Toggle (Neo-Kinpaku Dark <-> Warm Paper Light) */}
      <button
        className="action-btn"
        onClick={toggleTheme}
        title={isDark ? 'Giao diện sáng (Warm Paper)' : 'Giao diện tối (Dark Lacquer)'}
        aria-label="Toggle Theme"
        style={{
          color: isDark ? '#fbbf24' : '#d97706',
          borderColor: isDark ? 'rgba(251, 191, 36, 0.25)' : 'rgba(217, 119, 6, 0.25)',
        }}
      >
        {isDark ? (
          <Sun size={18} style={{ transition: 'transform 0.4s ease' }} />
        ) : (
          <Moon size={18} style={{ transition: 'transform 0.4s ease' }} />
        )}
      </button>

      {/* Alert Bell */}
      <button className="action-btn" title={t.header.alerts} aria-label="Alerts">
        <Bell size={18} style={{ color: 'var(--text-sub)' }} />
        {/* Notification dot */}
        <span
          style={{
            position: 'absolute',
            top: '8px',
            right: '8px',
            width: '6px',
            height: '6px',
            borderRadius: '50%',
            background: 'var(--rose)',
            boxShadow: '0 0 8px var(--rose)',
          }}
        />
      </button>

      {/* Documents */}
      <button className="action-btn" title={t.header.documents} aria-label="Documents">
        <FileText size={18} style={{ color: 'var(--text-sub)' }} />
      </button>

      {/* Language Selector Dropdown */}
      <div style={{ position: 'relative' }} ref={menuRef}>
        <button
          className="action-btn"
          onClick={() => setShowLangMenu(!showLangMenu)}
          title={t.header.language}
          style={{
            width: 'auto',
            padding: '0 12px',
            gap: '6px',
            fontSize: '13px',
            fontWeight: 600,
            fontFamily: "'Space Grotesk', sans-serif",
            color: 'var(--text-label)',
          }}
        >
          <span style={{ fontSize: '15px' }}>{language === 'en' ? '🇬🇧' : '🇻🇳'}</span>
          <span>{language === 'en' ? 'EN' : 'VI'}</span>
          <ChevronDown
            size={13}
            style={{
              transform: showLangMenu ? 'rotate(180deg)' : 'rotate(0deg)',
              transition: 'transform 0.2s ease',
              opacity: 0.6,
            }}
          />
        </button>

        {showLangMenu && (
          <div
            style={{
              position: 'absolute',
              top: '48px',
              right: 0,
              background: 'var(--bg-elevated)',
              border: '1px solid var(--border-strong)',
              borderRadius: '12px',
              boxShadow: '0 16px 40px rgba(0, 0, 0, 0.7), 0 0 0 1px rgba(245, 158, 11, 0.15)',
              overflow: 'hidden',
              zIndex: 200,
              width: '160px',
              animation: 'tabFadeIn 0.2s cubic-bezier(0.16, 1, 0.3, 1) forwards',
            }}
          >
            <button
              onClick={() => {
                changeLanguage('en');
                setShowLangMenu(false);
              }}
              style={{
                width: '100%',
                padding: '10px 14px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                background: language === 'en' ? 'var(--accent-dim)' : 'transparent',
                border: 'none',
                cursor: 'pointer',
                color: language === 'en' ? 'var(--accent-light)' : 'var(--text-label)',
                fontWeight: language === 'en' ? 600 : 400,
                fontSize: '13px',
                fontFamily: "'Space Grotesk', sans-serif",
                transition: 'background 0.15s',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{ fontSize: '16px' }}>🇬🇧</span>
                <span>English</span>
              </div>
              {language === 'en' && <Check size={14} />}
            </button>

            <div style={{ height: '1px', background: 'var(--border)' }} />

            <button
              onClick={() => {
                changeLanguage('vi');
                setShowLangMenu(false);
              }}
              style={{
                width: '100%',
                padding: '10px 14px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                background: language === 'vi' ? 'var(--accent-dim)' : 'transparent',
                border: 'none',
                cursor: 'pointer',
                color: language === 'vi' ? 'var(--accent-light)' : 'var(--text-label)',
                fontWeight: language === 'vi' ? 600 : 400,
                fontSize: '13px',
                fontFamily: "'Space Grotesk', sans-serif",
                transition: 'background 0.15s',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{ fontSize: '16px' }}>🇻🇳</span>
                <span>Tiếng Việt</span>
              </div>
              {language === 'vi' && <Check size={14} />}
            </button>
          </div>
        )}
      </div>

      {/* User Profile */}
      <button className="action-btn" title={t.header.profile} aria-label="User Profile">
        <User size={18} style={{ color: 'var(--text-sub)' }} />
      </button>
    </div>
  );
}
