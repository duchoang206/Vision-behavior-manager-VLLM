'use client';

import { useCallback, useEffect, useState } from 'react';
import { Download, Play, RefreshCw, Trash2, Video, X } from 'lucide-react';
import { useCameras } from '../CameraContext';
import { useTab } from '../TabContext';
import styles from './RecordingJournal.module.css';

type Recording = { id: string; camera_id: string; camera_name: string; path: string; started_at: string; ended_at: string | null; duration_seconds: number; size_bytes: number; codec: string | null; status: string; reason?: string };
type Recorder = { engine: string; segment_seconds: number; retention_hours: number; free_bytes: number; enabled: boolean; cameras: { camera_id: string; writing: boolean; running: boolean; error: string | null }[] };
const statusLabels: Record<string, string> = { recording: 'Đang ghi', ready: 'Hoàn tất', interrupted: 'Đã lưu · luồng bị ngắt', error: 'File lỗi' };
const videoUrl = (record: Recording, download = false) => `/api/backend/recordings/${record.id}/video${download ? '?download=true' : ''}`;
const timeLabel = (value: string) => new Date(value).toLocaleString('vi-VN');

export default function RecordingJournal() {
  const { cameras } = useCameras();
  const { activeTab } = useTab();
  const [cameraId, setCameraId] = useState('');
  const [offset, setOffset] = useState(0);
  const [records, setRecords] = useState<Recording[]>([]);
  const [recorder, setRecorder] = useState<Recorder | null>(null);
  const [hasMore, setHasMore] = useState(false);
  const [error, setError] = useState('');
  const [revision, setRevision] = useState(0);
  const [selected, setSelected] = useState<Recording | null>(null);
  const [deleting, setDeleting] = useState('');
  const [playError, setPlayError] = useState('');

  useEffect(() => {
    if (activeTab !== 'analytics') { setSelected(null); return; }
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    const load = async () => {
      try {
        const params = new URLSearchParams({ limit: '24', offset: String(offset) });
        if (cameraId) params.set('camera_id', cameraId);
        const response = await fetch(`/api/backend/recordings?${params}`, { cache: 'no-store', signal: controller.signal });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || 'Không tải được danh sách MP4.');
        if (!controller.signal.aborted) { setRecords(data.recordings); setRecorder(data.recorder); setHasMore(data.has_more); setError(''); }
      } catch (reason) {
        if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : String(reason));
      } finally {
        if (!controller.signal.aborted) timer = setTimeout(load, 10000);
      }
    };
    setRecords([]); void load();
    return () => { controller.abort(); clearTimeout(timer); };
  }, [activeTab, cameraId, offset, revision]);

  const closeVideo = useCallback(() => { setSelected(null); setPlayError(''); }, []);
  useEffect(() => {
    if (!selected) return;
    const keyDown = (event: KeyboardEvent) => { if (event.key === 'Escape') closeVideo(); };
    window.addEventListener('keydown', keyDown);
    return () => window.removeEventListener('keydown', keyDown);
  }, [selected, closeVideo]);

  const remove = async (record: Recording) => {
    if (!confirm(`Xóa vĩnh viễn MP4 của ${record.camera_name} lúc ${timeLabel(record.started_at)} khỏi SSD? Không thể hoàn tác.`)) return;
    setDeleting(record.id); setError('');
    try {
      const response = await fetch(`/api/backend/recordings/${record.id}`, { method: 'DELETE' });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Không xóa được video.');
      if (selected?.id === record.id) closeVideo();
      setRevision(value => value + 1);
    } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); }
    finally { setDeleting(''); }
  };

  return <div className={styles.journal}>
    <div className={styles.heading}><strong><Video size={16} />MP4 THEO GIỜ · FFmpeg</strong><button aria-label="Làm mới MP4" onClick={() => setRevision(value => value + 1)}><RefreshCw size={14} /></button></div>
    <p className={styles.note}>Mỗi camera ghi trực tiếp luồng gốc, chia theo giờ · Xóa vòng sau {recorder?.retention_hours || 48} giờ. File đầu/cuối hoặc mất mạng có thể ngắn hơn 1 giờ.</p>
    <div className={styles.toolbar}><select aria-label="Lọc camera MP4" value={cameraId} onChange={event => { setCameraId(event.target.value); setOffset(0); }}><option value="">Tất cả camera</option>{cameras.map(camera => <option key={camera.id} value={camera.id}>{camera.name}</option>)}</select>
      <span>{recorder ? `${recorder.cameras.filter(camera => camera.writing).length}/${recorder.cameras.length} camera đang ghi · SSD trống ${(recorder.free_bytes / 1024 ** 3).toFixed(1)} GB` : 'Đang kết nối bộ ghi…'}</span></div>
    {recorder?.cameras.filter(camera => camera.error).map(camera => <p className={styles.warning} key={camera.camera_id}>{cameras.find(item => item.id === camera.camera_id)?.name || camera.camera_id}: {camera.error}</p>)}
    {error && <p role="alert" className={styles.warning}>{error}</p>}
    <div className={styles.list}>{!records.length && !error && <p className={styles.empty}>Chưa có MP4 trong trang này. File đang ghi xuất hiện sau khi camera gửi dữ liệu.</p>}
      {records.map(record => { const playable = ['ready', 'interrupted'].includes(record.status); return <article key={record.id} className={styles.record}>
        <div className={styles.heading}><b>{record.camera_name}</b><span className={record.status === 'recording' ? styles.recording : styles.badge}>{statusLabels[record.status] || record.status}</span></div>
        <div className={styles.note}>{timeLabel(record.started_at)} → {record.ended_at ? timeLabel(record.ended_at) : '…'}</div>
        <div className={styles.filename} title={record.path}>{record.path}</div>
        <div className={styles.toolbar}><span>{Math.floor(record.duration_seconds / 60)} phút {Math.round(record.duration_seconds % 60)} giây · {(record.size_bytes / 1024 ** 2).toFixed(1)} MB · {record.codec?.toUpperCase() || 'Copy stream'}</span><div className={styles.actions}>
          {playable && <><button onClick={() => { setSelected(record); setPlayError(''); }}><Play size={12} />Xem MP4</button><a href={videoUrl(record, true)} aria-label={`Tải MP4 ${record.camera_name}`}><Download size={13} /></a></>}
          <button aria-label={`Xóa MP4 ${record.camera_name}`} disabled={record.status === 'recording' || Boolean(deleting)} onClick={() => void remove(record)}><Trash2 size={13} /></button>
        </div></div>{record.reason && <p className={styles.note}>{record.reason}</p>}
      </article>; })}
    </div>
    <div className={styles.toolbar}><button disabled={!offset} onClick={() => setOffset(value => Math.max(0, value - 24))}>Trang trước</button><span>Trang {offset / 24 + 1}</span><button disabled={!hasMore} onClick={() => setOffset(value => value + 24)}>Trang sau</button></div>
    {selected && <div className={styles.modal} role="dialog" aria-modal="true" aria-label="Xem MP4 camera"><div className={styles.player}>
      <div className={styles.heading}><strong>{selected.camera_name} · {timeLabel(selected.started_at)}</strong><button aria-label="Đóng video" onClick={closeVideo}><X size={18} /></button></div>
      <video key={selected.id} src={videoUrl(selected)} controls autoPlay onError={() => setPlayError('Trình duyệt không phát được codec gốc hoặc file không còn tồn tại. Hãy tải MP4 để xem bằng VLC.')} />
      {playError && <p role="alert" className={styles.warning}>{playError}</p>}
      <div className={styles.toolbar}><span>Video gốc, không chèn mask/AI hay tái mã hóa.</span><a href={videoUrl(selected, true)}><Download size={14} />Tải MP4</a></div>
    </div></div>}
  </div>;
}
