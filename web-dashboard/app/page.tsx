'use client';

import React from 'react';
import { useTab } from '../components/TabContext';
import MonitorView from '../components/views/MonitorView';
import BuildingView from '../components/views/BuildingView';
import RobotMap3DView from '../components/views/RobotMap3DView';
import AnalyticsView from '../components/views/AnalyticsView';

export default function DashboardRoot() {
  const { activeTab } = useTab();

  return (
    <>
      <div
        className={activeTab === 'monitor' ? 'tab-content-enter' : ''}
        style={{
          display: activeTab === 'monitor' ? 'block' : 'none',
          width: '100%',
          height: '100%',
          minHeight: 'calc(100vh - 74px)',
        }}
      >
        <MonitorView />
      </div>

      <div
        className={activeTab === 'building' ? 'tab-content-enter' : ''}
        style={{
          display: activeTab === 'building' ? 'block' : 'none',
          width: '100%',
          height: '100%',
          minHeight: 'calc(100vh - 74px)',
        }}
      >
        <BuildingView />
      </div>

      <div
        className={activeTab === 'robot_map' ? 'tab-content-enter' : ''}
        style={{
          display: activeTab === 'robot_map' ? 'block' : 'none',
          width: '100%',
          height: '100%',
          minHeight: 'calc(100vh - 74px)',
        }}
      >
        <RobotMap3DView />
      </div>

      <div
        className={activeTab === 'analytics' ? 'tab-content-enter' : ''}
        style={{
          display: activeTab === 'analytics' ? 'block' : 'none',
          width: '100%',
          height: '100%',
          minHeight: 'calc(100vh - 74px)',
        }}
      >
        {activeTab === 'analytics' && <AnalyticsView />}
      </div>
    </>
  );
}
