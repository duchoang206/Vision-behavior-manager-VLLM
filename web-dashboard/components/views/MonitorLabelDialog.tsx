'use client';

import { useEffect, useRef } from 'react';
import { X } from 'lucide-react';
import ModelLabelManager from './ModelLabelManager';
import styles from './MonitorLabelDialog.module.css';

export default function MonitorLabelDialog({ cameraId, onClose }: { cameraId: string; onClose: () => void }) {
  const dialog = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const element = dialog.current;
    element?.showModal();
    return () => element?.close();
  }, []);
  return <dialog ref={dialog} className={styles.dialog} aria-labelledby="monitor-label-title" onCancel={onClose}>
    <header className={styles.header}>
      <div><h2 id="monitor-label-title">Đăng ký đa góc nhìn trên Monitor</h2><p>Chọn cùng Label → lấy frame GPU mới → khoanh vật / chỉnh SAM2 → lưu. Các camera vẫn tiếp tục tracking.</p></div>
      <button type="button" autoFocus onClick={onClose} aria-label="Đóng đăng ký Label"><X size={20} /></button>
    </header>
    <div className={styles.body}><ModelLabelManager active initialCameraId={cameraId} /></div>
  </dialog>;
}
