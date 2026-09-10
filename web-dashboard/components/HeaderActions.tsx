'use client';

import React, { useEffect, useRef, useState } from 'react';
import { Check, ChevronDown, Globe2, Moon, Sun, UserRound } from 'lucide-react';
import { useLanguage } from './LanguageContext';
import { useAppTheme } from './ThemeContext';
import styles from './DashboardShell.module.css';

export default function HeaderActions() {
  const { language, changeLanguage } = useLanguage();
  const { isDark, toggleTheme } = useAppTheme();
  const [showLangMenu, setShowLangMenu] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const firstLanguageRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    function closeOutside(event: MouseEvent) {
      if (!menuRef.current?.contains(event.target as Node)) setShowLangMenu(false);
    }
    function closeOnEscape(event: KeyboardEvent) {
      if (event.key === 'Escape' && showLangMenu) {
        setShowLangMenu(false);
        triggerRef.current?.focus();
      }
    }
    document.addEventListener('mousedown', closeOutside);
    document.addEventListener('keydown', closeOnEscape);
    if (showLangMenu) firstLanguageRef.current?.focus();
    return () => {
      document.removeEventListener('mousedown', closeOutside);
      document.removeEventListener('keydown', closeOnEscape);
    };
  }, [showLangMenu]);

  return (
    <div className={styles.actions}>
      <button className={styles.themeButton} onClick={toggleTheme} type="button" aria-label="Toggle Theme"
        title={language === 'vi' ? (isDark ? 'Giao diện sáng' : 'Giao diện tối') : (isDark ? 'Light theme' : 'Dark theme')}>
        {isDark ? <Sun size={15} /> : <Moon size={15} />}<span>{language === 'vi' ? 'Giao diện' : 'Theme'}</span>
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
      <div className={styles.profile} title="RTC Vision Management System">
        <span className={styles.avatar}><UserRound size={17} /></span>
        <span className={styles.profileText}><strong>RTC Workspace</strong><small>Vision Management</small></span>
      </div>
    </div>
  );
}
