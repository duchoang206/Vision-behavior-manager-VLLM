'use client';

import { CheckCircle2, Sparkles } from 'lucide-react';
import { DeepCalibPreview, DeepCalibStatus } from '../../lib/deepcalib';
import styles from './CalibrationView.module.css';

type Props = {
  snapshot: string; preview: DeepCalibPreview | null; status: DeepCalibStatus | null;
  pairCount: number; lengthCount: number; saved: boolean; canReuse: boolean;
  onRun: (reuse: boolean) => void; disabled: boolean; running: boolean;
  onRefresh: () => void;
};

export default function DeepCalibPanel({ snapshot, preview, status, pairCount, lengthCount,
  saved, canReuse, onRun, disabled, running, onRefresh }: Props) {
  const ready = Boolean(status?.weights_available && status?.legacy_runtime_available);
  return <div className={styles.deepPanel}>
    <div className={styles.guide}>
      <strong><Sparkles size={16} /> DeepCalib → ảnh hiệu chỉnh → tọa độ FMS</strong>
      <p>Chạy một lần trên ảnh tĩnh. Sau đó chấm trên ảnh hiệu chỉnh bên dưới, ghép với FMS hoặc nhập X/Y thực tế theo hệ FMS; thêm số đo cạnh ô gạch / đoạn thẳng rồi Save.</p>
    </div>
    <div className={styles.pipelineSteps}>
      <span className={snapshot ? styles.deepStepDone : ''}>1 · Ảnh gốc</span>
      <span className={preview ? styles.deepStepDone : ''}>2 · DeepCalib / GPU</span>
      <span className={pairCount >= 4 ? styles.deepStepDone : ''}>3 · {pairCount} cặp ảnh ↔ FMS</span>
      <span className={lengthCount ? styles.deepStepDone : ''}>4 · {lengthCount} đoạn đo mét (tùy chọn)</span>
      <span className={saved ? styles.deepStepDone : ''}>5 · {saved ? 'Đã lưu & áp dụng' : 'Save camera'}</span>
    </div>
    <div className={styles.deepGrid}>
      <article className={styles.panel}>
        <div className={styles.heading}><h2>Trước · Ảnh gốc</h2><button disabled={disabled || running} onClick={onRefresh}>Lấy frame mới</button></div>
        <div className={styles.previewStage}>{snapshot ? <img src={snapshot} alt="Ảnh gốc trước DeepCalib" /> : <span>Đang lấy snapshot…</span>}</div>
      </article>
      <article className={styles.panel}>
        <div className={styles.heading}><h2>Sau · Ảnh đã hiệu chỉnh</h2><span className={preview ? styles.live : styles.offline}>{preview ? 'CUDA · sẵn sàng chấm điểm' : 'Chưa chạy DeepCalib'}</span></div>
        <div className={styles.previewStage}>{preview ? <img src={preview.rectified_image} alt="Ảnh sau DeepCalib" /> : <span>{running ? 'Đang suy luận và hiệu chỉnh trên GPU…' : 'Bấm Chạy DeepCalib để tạo ảnh hiệu chỉnh.'}</span>}</div>
      </article>
    </div>
    <div className={styles.toolbar}>
      <div className={styles.actions}>
        <button className={styles.save} disabled={disabled || running || !snapshot || !ready} onClick={() => onRun(false)}><Sparkles size={15} />{running ? 'Đang xử lý…' : 'Chạy DeepCalib'}</button>
        {canReuse && <button disabled={disabled || running || !snapshot} onClick={() => onRun(true)}><CheckCircle2 size={15} />Mở lại với profile đã lưu</button>}
      </div>
      <span className={styles.caption}>{preview ? `f = ${preview.profile.focal_length_px.toFixed(1)} px · ξ = ${preview.profile.distortion_xi.toFixed(3)}` : ready ? 'SingleNet · alexvbogdan/DeepCalib' : 'Model/runtime chưa sẵn sàng; không dùng kết quả giả.'}</span>
    </div>
    {preview && [preview.profile.focal_confidence, preview.profile.distortion_confidence].some(value => value !== undefined && value < .2) &&
      <div className={styles.deepNote}>DeepCalib chưa chắc chắn về intrinsic của ảnh này. Kiểm tra đường thẳng sau hiệu chỉnh và dùng các đoạn đo đã biết để đối chiếu trước khi Save; không xem dự đoán model là bảo đảm độ chính xác.</div>}
    <div className={styles.deepNote}>Chỉ đo trên cùng mặt phẳng sàn. DeepCalib sửa méo ống kính, không tự xác định gốc FMS hay biến ảnh thành nhìn từ trên xuống. Không chấm trên viền đen, nóc robot hoặc mặt kệ. Chỉ số đo mét không đủ xác định vị trí tuyệt đối trên FMS.</div>
  </div>;
}
