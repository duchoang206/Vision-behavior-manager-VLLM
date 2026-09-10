'use client';

import React from 'react';
import { Video, Building2, Bot, BarChart3 } from 'lucide-react';
import { useLanguage } from './LanguageContext';
import { useTab, type TabType } from './TabContext';
import styles from './DashboardShell.module.css';

export const workspaceTabs = [
  { id: 'monitor', labelKey: 'monitor', icon: Video, keywords: 'camera live giám sát trực tiếp' },
  { id: 'building', labelKey: 'building', icon: Building2, keywords: 'camera config cấu hình label nhãn roi tripwire' },
  { id: 'robot_map', labelKey: 'robotMap', icon: Bot, keywords: 'map robot 3d bản đồ fms' },
  { id: 'analytics', labelKey: 'analytics', icon: BarChart3, keywords: 'analytics report phân tích báo cáo' },
] as const;

export default function Navigation({ onNavigate }: { onNavigate?: (tab: TabType) => void }) {
  const { t } = useLanguage();
  const { activeTab, setActiveTab } = useTab();

  return (
    <nav id="workspace-navigation" className={styles.navigation} aria-label="Main Navigation">
      {workspaceTabs.map(tab => (
        <button key={tab.id} type="button" onClick={() => (onNavigate || setActiveTab)(tab.id)}
          className={`${styles.navItem} ${activeTab === tab.id ? styles.activeItem : ''}`}
          aria-current={activeTab === tab.id ? 'page' : undefined} title={t.nav[tab.labelKey]}>
          <span className={styles.navIcon}><tab.icon size={18} strokeWidth={1.6} /></span>
          <span className={styles.navLabel}>{t.nav[tab.labelKey]}</span>
          {tab.id === 'robot_map' && <span className={styles.navBadge}>3D</span>}
        </button>
      ))}
    </nav>
  );
}
