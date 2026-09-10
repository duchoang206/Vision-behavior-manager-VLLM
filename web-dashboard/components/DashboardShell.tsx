'use client';

import React, { useEffect, useRef, useState } from 'react';
import { ArrowUpRight, Command, Menu, PanelLeftClose, PanelLeftOpen, Search, ShieldCheck, Video, X } from 'lucide-react';
import Navigation, { workspaceTabs } from './Navigation';
import HeaderActions from './HeaderActions';
import ChatbotWidget from './ChatbotWidget';
import { useTab, type TabType } from './TabContext';
import { useLanguage } from './LanguageContext';
import { useCameras } from './CameraContext';
import styles from './DashboardShell.module.css';

const descriptions = {
  monitor: { en: 'Live cameras, tracking and activity, in one place.', vi: 'Camera trực tiếp, bám vết và hoạt động trong cùng không gian.' },
  building: { en: 'Manage cameras, zones and registered labels.', vi: 'Quản lý camera, vùng giám sát và nhãn đã đăng ký.' },
  robot_map: { en: 'Your fleet and workspace, connected in real time.', vi: 'Kết nối đội robot và không gian vận hành theo thời gian thực.' },
  analytics: { en: 'Explore activity, reports and operational insights.', vi: 'Theo dõi hoạt động, báo cáo và dữ liệu vận hành.' },
};

export default function DashboardShell({ children }: { children: React.ReactNode }) {
  const { activeTab, setActiveTab } = useTab();
  const { language, t } = useLanguage();
  const { cameras, loading } = useCameras();
  const [collapsed, setCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const [query, setQuery] = useState('');
  const [searchOpen, setSearchOpen] = useState(false);
  const [selectedResult, setSelectedResult] = useState(0);
  const searchRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const menuRef = useRef<HTMLButtonElement>(null);
  const sidebarRef = useRef<HTMLElement>(null);
  const currentTab = workspaceTabs.find(tab => tab.id === activeTab)!;
  const results = workspaceTabs.filter(tab =>
    `${t.nav[tab.labelKey]} ${tab.keywords}`.toLocaleLowerCase().includes(query.trim().toLocaleLowerCase())
  );
  const onlineCameras = cameras.filter(camera => camera.status === 'online').length;

  useEffect(() => {
    const saved = localStorage.getItem('vms-sidebar-collapsed');
    if (saved === 'true') setCollapsed(true);
  }, []);

  useEffect(() => {
    function handlePointer(event: MouseEvent) {
      if (!searchRef.current?.contains(event.target as Node)) setSearchOpen(false);
    }
    function handleKey(event: KeyboardEvent) {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'k') {
        event.preventDefault();
        inputRef.current?.focus();
        setSearchOpen(true);
      }
      if (event.key === 'Escape') {
        setSearchOpen(false);
        setMobileOpen(false);
      }
    }
    document.addEventListener('mousedown', handlePointer);
    document.addEventListener('keydown', handleKey);
    return () => {
      document.removeEventListener('mousedown', handlePointer);
      document.removeEventListener('keydown', handleKey);
    };
  }, []);

  useEffect(() => {
    if (!mobileOpen) return;
    const sidebar = sidebarRef.current;
    const focusable = sidebar?.querySelectorAll<HTMLButtonElement>('button');
    focusable?.[0]?.focus();
    function trapFocus(event: KeyboardEvent) {
      if (event.key !== 'Tab' || !focusable?.length) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    }
    document.addEventListener('keydown', trapFocus);
    return () => {
      document.removeEventListener('keydown', trapFocus);
      menuRef.current?.focus();
    };
  }, [mobileOpen]);

  function navigate(tab: TabType) {
    setActiveTab(tab);
    setSearchOpen(false);
    setMobileOpen(false);
    setQuery('');
    inputRef.current?.blur();
  }

  function toggleSidebar() {
    setCollapsed(value => {
      try { localStorage.setItem('vms-sidebar-collapsed', String(!value)); } catch {}
      return !value;
    });
  }

  return (
    <div className={`${styles.shell} ${collapsed ? styles.collapsed : ''} ${mobileOpen ? styles.mobileOpen : ''}`}>
      <a className={styles.skipLink} href="#workspace-content">{language === 'vi' ? 'Đến nội dung' : 'Skip to content'}</a>
      {mobileOpen && <button className={styles.backdrop} aria-label={language === 'vi' ? 'Đóng menu' : 'Close menu'} onClick={() => setMobileOpen(false)} />}
      <aside ref={sidebarRef} id="workspace-sidebar" className={styles.sidebar} aria-label={language === 'vi' ? 'Điều hướng workspace' : 'Workspace navigation'}>
        <div className={styles.brand}>
          <span className={styles.brandMark}><ShieldCheck size={21} strokeWidth={1.6} /></span>
          <div className={styles.brandCopy}><span>RTC VISION</span><strong>Control Center</strong></div>
          <button className={`${styles.iconButton} ${styles.collapseButton}`} onClick={toggleSidebar}
            aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'} aria-expanded={!collapsed} aria-controls="workspace-navigation"
            title={language === 'vi' ? 'Thu gọn / mở rộng' : 'Collapse / expand'}>
            {collapsed ? <PanelLeftOpen size={16} /> : <PanelLeftClose size={16} />}
          </button>
          <button className={`${styles.iconButton} ${styles.mobileClose}`} onClick={() => setMobileOpen(false)} aria-label="Close menu"><X size={18} /></button>
        </div>
        <div className={styles.navCaption}>{language === 'vi' ? 'KHÔNG GIAN LÀM VIỆC' : 'WORKSPACE'}</div>
        <Navigation onNavigate={navigate} />
        <div className={styles.sidebarBottom}>
          <div className={styles.cameraSummary} title={`${onlineCameras}/${cameras.length} cameras online`}>
            <div className={styles.summaryHeading}><Video size={14} /><span>{language === 'vi' ? 'CAMERA KẾT NỐI' : 'CAMERA NETWORK'}</span></div>
            <div className={styles.summaryValue}>{loading ? '—' : onlineCameras}<span>/ {cameras.length}</span><i className={onlineCameras ? styles.onlineDot : styles.offlineDot} /></div>
            <p>{language === 'vi' ? 'Nguồn camera đang trực tuyến' : 'Camera sources online'}</p>
          </div>
          <div className={styles.sidebarFooter}><span>RTC Technology</span><span>VMS · 1.0</span></div>
        </div>
      </aside>
      <div className={styles.workspace}>
        <header className={styles.header}>
          <div className={styles.topbar}>
            <button ref={menuRef} className={`${styles.iconButton} ${styles.mobileMenu}`} onClick={() => setMobileOpen(true)} aria-label="Open menu" aria-expanded={mobileOpen} aria-controls="workspace-sidebar"><Menu size={20} /></button>
            <div className={styles.heading}><span>{language === 'vi' ? 'KHÔNG GIAN LÀM VIỆC' : 'WORKSPACE'}</span><h1>{t.nav[currentTab.labelKey]}</h1></div>
            <HeaderActions />
          </div>
          <div ref={searchRef} className={styles.searchArea}>
            <div className={styles.searchField}>
              <Search size={15} aria-hidden="true" />
              <input ref={inputRef} value={query} role="combobox" aria-autocomplete="list" aria-expanded={searchOpen} aria-controls="workspace-search-results"
                aria-activedescendant={searchOpen && results.length ? `workspace-result-${Math.min(selectedResult, results.length - 1)}` : undefined}
                aria-label={language === 'vi' ? 'Tìm màn hình' : 'Find a screen'}
                placeholder={language === 'vi' ? 'Tìm Monitor, Building, bản đồ, phân tích…' : 'Find Monitor, Building, robot map, analytics…'}
                onFocus={() => setSearchOpen(true)} onChange={event => { setQuery(event.target.value); setSelectedResult(0); setSearchOpen(true); }}
                onKeyDown={event => {
                  if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
                    event.preventDefault();
                    setSearchOpen(true);
                    setSelectedResult(value => results.length ? (value + (event.key === 'ArrowDown' ? 1 : -1) + results.length) % results.length : 0);
                  }
                  if (event.key === 'Enter' && results.length && searchOpen) {
                    event.preventDefault();
                    navigate(results[Math.min(selectedResult, results.length - 1)].id);
                  }
                  if (event.key === 'Tab') setSearchOpen(false);
                }} />
              <kbd><Command size={11} /> K</kbd>
            </div>
            {searchOpen && <div id="workspace-search-results" role="listbox" className={styles.searchResults} aria-label={language === 'vi' ? 'Màn hình' : 'Screens'}>
              {results.length ? results.map((tab, index) => <button id={`workspace-result-${index}`} key={tab.id} type="button" role="option" aria-selected={selectedResult === index}
                onMouseDown={event => event.preventDefault()} onMouseEnter={() => setSelectedResult(index)} onClick={() => navigate(tab.id)}>
                <tab.icon size={17} /><span>{t.nav[tab.labelKey]}<small>{descriptions[tab.id][language]}</small></span><ArrowUpRight size={14} />
              </button>) : <p>{language === 'vi' ? 'Không tìm thấy màn hình phù hợp.' : 'No matching screens.'}</p>}
            </div>}
          </div>
          <p className={styles.description}>{descriptions[activeTab][language]}</p>
        </header>
        <main id="workspace-content" tabIndex={-1} className={styles.content}>{children}</main>
      </div>
      <ChatbotWidget />
    </div>
  );
}
