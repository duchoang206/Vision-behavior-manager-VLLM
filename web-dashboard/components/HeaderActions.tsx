'use client';

import React, { useEffect, useRef, useState } from 'react';
import { Check, ChevronDown, Globe2, LogOut, Moon, Sun, UserRound } from 'lucide-react';
import { useLanguage } from './LanguageContext';
import { useAppTheme } from './ThemeContext';
import styles from './DashboardShell.module.css';

export default function HeaderActions() {
  const { language, changeLanguage } = useLanguage();
  const { isDark, toggleTheme } = useAppTheme();
  const [showLangMenu, setShowLangMenu] = useState(false);
  const [showProfileMenu, setShowProfileMenu] = useState(false);
  const [username, setUsername] = useState('');
  const [loggingOut, setLoggingOut] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const firstLanguageRef = useRef<HTMLButtonElement>(null);
  const profileRef = useRef<HTMLDivElement>(null);
  const profileTriggerRef = useRef<HTMLButtonElement>(null);
  const logoutRef = useRef<HTMLButtonElement>(null);
  const vi = language === 'vi';

  useEffect(() => {
    function closeOutside(event: MouseEvent) {
      if (!menuRef.current?.contains(event.target as Node)) setShowLangMenu(false);
      if (!profileRef.current?.contains(event.target as Node)) setShowProfileMenu(false);
    }
    function closeOnEscape(event: KeyboardEvent) {
      if (event.key !== 'Escape') return;
      if (showLangMenu) {
        setShowLangMenu(false);
        triggerRef.current?.focus();
      }
      if (showProfileMenu) {
        setShowProfileMenu(false);
        profileTriggerRef.current?.focus();
      }
    }
    document.addEventListener('mousedown', closeOutside);
    document.addEventListener('keydown', closeOnEscape);
    if (showLangMenu) firstLanguageRef.current?.focus();
    if (showProfileMenu) logoutRef.current?.focus();
    return () => {
      document.removeEventListener('mousedown', closeOutside);
      document.removeEventListener('keydown', closeOnEscape);
    };
  }, [showLangMenu, showProfileMenu]);

  useEffect(() => {
    const controller = new AbortController();
    fetch('/api/auth/session', { cache: 'no-store', signal: controller.signal })
      .then(response => response.ok ? response.json() : null)
      .then(session => { if (session?.username) setUsername(session.username); })
      .catch(() => {});
    return () => controller.abort();
  }, []);

  const logout = async () => {
    setLoggingOut(true);
    try {
      await fetch('/api/auth/logout', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
    } finally {
      window.location.assign('/login');
    }
  };

  return (
    <div className={styles.actions}>
      <button className={styles.themeButton} onClick={toggleTheme} type="button" aria-label="Toggle Theme"
        title={vi ? (isDark ? 'Giao diện sáng' : 'Giao diện tối') : (isDark ? 'Light theme' : 'Dark theme')}>
        {isDark ? <Sun size={15} /> : <Moon size={15} />}<span>{vi ? 'Giao diện' : 'Theme'}</span>
      </button>
      <div className={styles.languagePicker} ref={menuRef}>
        <button ref={triggerRef} type="button" className={styles.languageButton} onClick={() => setShowLangMenu(value => !value)} aria-expanded={showLangMenu} aria-controls="workspace-languages" aria-label="Language">
          <Globe2 size={15} /><span>{language.toUpperCase()}</span><ChevronDown size={12} />
        </button>
        {showLangMenu && <div id="workspace-languages" className={styles.languageMenu}>
          {(['en', 'vi'] as const).map((value, index) => <button ref={index === 0 ? firstLanguageRef : undefined} key={value} type="button"
            onClick={() => { changeLanguage(value); setShowLangMenu(false); triggerRef.current?.focus(); }}>
            <span>{value === 'en' ? 'English' : 'Tiếng Việt'}</span>{language === value && <Check size={14} />}
          </button>)}
        </div>}
      </div>
      <div className={styles.languagePicker} ref={profileRef}>
        <button ref={profileTriggerRef} type="button" className={styles.profile} onClick={() => setShowProfileMenu(value => !value)}
          aria-expanded={showProfileMenu} aria-controls="workspace-profile" aria-label={vi ? 'Tài khoản' : 'Account'}
          title={username ? `${vi ? 'Đăng nhập' : 'Signed in'}: ${username}` : 'RTC Vision Management System'}>
          <span className={styles.avatar}><UserRound size={17} /></span>
          <span className={styles.profileText}><strong>RTC Workspace</strong><small>{username || 'Vision Management'}</small></span>
          <ChevronDown size={12} className={styles.profileChevron} />
        </button>
        {showProfileMenu && <div id="workspace-profile" className={`${styles.languageMenu} ${styles.profileMenu}`}>
          {username && <div className={styles.profileMenuUser}>{vi ? 'Tài khoản' : 'Account'}<strong>{username}</strong></div>}
          <button ref={logoutRef} type="button" onClick={() => void logout()} disabled={loggingOut} className={styles.logoutButton}>
            <span>{loggingOut ? (vi ? 'Đang đăng xuất…' : 'Signing out…') : (vi ? 'Đăng xuất' : 'Sign out')}</span><LogOut size={14} />
          </button>
        </div>}
      </div>
    </div>
  );
}
