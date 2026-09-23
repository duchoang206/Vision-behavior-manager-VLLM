'use client';

import { useEffect, useState } from 'react';
import { CalibrationPair, FmsFrame, Point2, floorToFms, isDistinctPair } from '../../lib/calibration-points';
import styles from './CalibrationView.module.css';
import { LengthConstraint } from '../../lib/deepcalib';

export type { LengthConstraint } from '../../lib/deepcalib';

export default function CalibrationInputs({ pairs, pending, imageSize, frame, disabled, onPair, constraints, lengthPoints, lengthMode, onLengthMode, onLengths }:
  { pairs: CalibrationPair[]; pending: Point2 | null; imageSize: Point2 | null; frame: FmsFrame | null; disabled: boolean;
    onPair: (pair: CalibrationPair, index: number | null) => void; constraints: LengthConstraint[]; lengthPoints: Point2[]; lengthMode: boolean;
    onLengthMode: (value: boolean) => void; onLengths: (value: LengthConstraint[]) => void }) {
  const [values, setValues] = useState(['', '', '', '']);
  const [editing, setEditing] = useState<number | null>(null);
  const [distance, setDistance] = useState('');
  const [error, setError] = useState('');
  useEffect(() => {
    if (pending && imageSize) { setValues(previous => [String(pending[0] * imageSize[0]), String(pending[1] * imageSize[1]), previous[2], previous[3]]); setEditing(null); }
  }, [pending, imageSize]);
  const editPair = (index: number | null) => {
    setEditing(index); setError('');
    if (index === null || !frame || !imageSize) { setValues(['', '', '', '']); return; }
    const pair = pairs[index];
    const coordinate = floorToFms(pair.floor, frame);
    setValues([String(pair.camera[0] * imageSize[0]), String(pair.camera[1] * imageSize[1]), String(coordinate[0]), String(coordinate[1])]);
  };
  return <div className={styles.panels}>
    <form className={`${styles.panel} ${styles.inputPanel}`} onSubmit={event => {
      event.preventDefault(); if (!imageSize || !frame || disabled) return;
      const [horizontal, vertical, worldX, worldY] = values.map(Number);
      if (values.some(value => !value.trim()) || ![horizontal, vertical, worldX, worldY].every(Number.isFinite)
        || horizontal < 0 || vertical < 0 || horizontal > imageSize[0] || vertical > imageSize[1]) { setError('Pixel phải nằm trong ảnh; tọa độ FMS phải là số hữu hạn.'); return; }
      const pair: CalibrationPair = { camera: [horizontal / imageSize[0], vertical / imageSize[1]], floor: [worldX - frame.origin_x, frame.origin_y + frame.layout_depth - worldY] };
      if (!isDistinctPair(pairs.filter((_, index) => index !== editing), pair.camera, pair.floor)) { setError('Điểm camera hoặc FMS bị trùng.'); return; }
      onPair(pair, editing); editPair(null);
    }}>
      <strong>Nhập / sửa cặp điểm</strong><p className={styles.caption}>Chấm ảnh rồi nhập X/Y thực tế theo hệ FMS, hoặc nhập đủ bốn số. Pixel theo ảnh đang chấm {imageSize?.join(' × ') || 'đang tải'}.</p>
      <select aria-label="Cặp điểm cần sửa" disabled={disabled} value={editing ?? ''} onChange={event => editPair(event.target.value === '' ? null : Number(event.target.value))}><option value="">Thêm cặp mới</option>{pairs.map((_, index) => <option key={index} value={index}>Sửa cặp #{index + 1}</option>)}</select>
      <div className={styles.inputGrid}>{['u · pixel', 'v · pixel', 'X FMS · m', 'Y FMS · m'].map((label, index) => <label key={label}>{label}<input aria-label={label} required type="number" step="any" value={values[index]} disabled={disabled} onChange={event => setValues(previous => previous.map((value, valueIndex) => valueIndex === index ? event.target.value : value))} /></label>)}</div>
      <button disabled={disabled || !imageSize || !frame} type="submit">{editing === null ? 'Thêm cặp điểm' : 'Cập nhật cặp điểm'}</button>
      {error && <span role="alert" className={styles.error}>{error}</span>}
    </form>
    <div className={`${styles.panel} ${styles.inputPanel}`}><strong>Chiều dài thực tế trên mặt sàn</strong><p className={styles.caption}>Chọn hai đầu cạnh ô vuông / đường thẳng rồi nhập mét, lặp lại với các cạnh khác. Đặt đoạn đo trong vùng phủ các điểm neo. Cần ít nhất 4 cặp Camera ↔ FMS để xác định gốc, hướng và phối cảnh.</p>
      <button disabled={disabled} aria-pressed={lengthMode} onClick={() => onLengthMode(!lengthMode)}>{lengthMode ? 'Kết thúc chọn đoạn' : 'Chọn hai đầu đoạn trên ảnh'}</button>
      <form className={styles.actions} onSubmit={event => { event.preventDefault(); const meters = Number(distance); if (disabled || lengthPoints.length !== 2 || !Number.isFinite(meters) || meters <= 0) return;
        onLengths([...constraints, { points: lengthPoints, distance_m: meters }]); setDistance(''); onLengthMode(true); }}>
        <label>Chiều dài · m<input aria-label="Chiều dài thực tế · m" type="number" step="any" min="0.001" max="10000" required value={distance} disabled={disabled} onChange={event => setDistance(event.target.value)} /></label><button type="submit" disabled={disabled || lengthPoints.length !== 2}>Thêm đoạn ({lengthPoints.length}/2 điểm)</button>
      </form>
      {constraints.map((item, index) => <div className={styles.actions} key={index}><span>Đoạn #{index + 1} · {item.distance_m} m</span><button disabled={disabled} onClick={() => onLengths(constraints.filter((_, number) => number !== index))}>Xóa đoạn</button></div>)}
    </div>
  </div>;
}
