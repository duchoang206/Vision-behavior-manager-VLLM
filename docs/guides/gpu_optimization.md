# Hướng dẫn tối ưu hóa hiệu năng và tài nguyên GPU

## 1. Lưu ý quan trọng về Pose Estimation
- **Cấu hình `interval=0`**: Với pipeline suy luận khung xương người (YOLO11l-Pose), giữ giá trị `interval=0` trong `config_infer_primary.txt`. Bỏ qua khung hình sẽ giảm nhịp cập nhật keypoint và có thể làm mất độ liên tục của pose.

## 2. Các phương pháp tối ưu hóa hiệu quả cao
1. **Shared Feature Backbone cho SAM 2**:
   - Biến môi trường: `REGISTERED_MASK_SHARED_FEATURES=1`.
   - Tính toán backbone ảnh 1 lần cho tất cả các nhãn robot/kệ trên cùng một khung hình, giảm tới 60% tải suy luận cho worker mask.
2. **Khống chế Target FPS cho Mask Engine**:
   - Biến môi trường: `REGISTERED_MASK_TARGET_FPS=11`.
   - Giúp ổn định GPU, tránh việc worker mask chạy không tải ở mức tối đa làm tăng nhiệt độ card đồ họa.
3. **Điều chỉnh độ phân giải luồng phụ**:
   - Khi mở rộng quy mô hệ thống lên nhiều camera ($\ge 8$ camera), khuyến nghị giảm độ phân giải luồng phụ từ 1080p xuống 720p hoặc 540p trước khi đưa vào DeepStream muxer.

## 3. Model pose nhẹ hơn, giữ nguyên pipeline

Cấu hình mặc định dùng `yolo11l-pose`, TensorRT FP16, batch `2`. Đây là lựa chọn cân bằng giữa tải và độ chính xác, không phải cam kết kết quả giống hệt `yolov8x-pose`. YOLO chỉ phụ trách người/pose; SAM2 vẫn phụ trách mask robot/kệ đã đăng ký, nên giảm tải YOLO không làm công suất toàn GPU giảm cùng tỷ lệ.

Không đổi input `640×640`, confidence `0.35`, NMS IoU `0.45`, `interval=0`, NvDCF/ReID, FPS mục tiêu, tìm lại đối tượng, calibration hoặc luồng nhận FMS. COCO-17 vẫn gồm 17 keypoint. Calibration vẫn chỉ lấy một ảnh mới nhất khi được yêu cầu.

Kiểm tra ngày 12/09/2026:
- 4 ảnh camera không có người: cả model cũ, YOLO11m và YOLO11l đều trả 0 người; đây không phải phép đo recall thực tế.
- 2 ảnh mẫu Ultralytics có người: cả ba model trả 6 người. So với model cũ, YOLO11l có IoU bbox trung bình `0.9806`, sai lệch keypoint đồng thời có confidence ≥ 0.5 trung bình `5.31 px`; YOLO11m tương ứng `0.9804`, `7.06 px`.
- Các số trên là mức độ tương đồng trên mẫu nhỏ, không phải độ chính xác đo bằng nhãn chuẩn hoặc kiểm định tracking khi người di chuyển/che khuất.
- Số tham số sau fuse đo trực tiếp: YOLOv8x `69,491,724`, YOLO11l `26,169,836` (giảm khoảng 62%). Không suy ra mức giảm điện năng từ số tham số.

### Xuất model và kiểm tra tương thích

Không dùng `DeepStream-Yolo/utils/export_yolo11.py` cho pose: exporter đó trả detection, không giữ tensor COCO-17. Dùng helper sau trong môi trường export có `ultralytics`, `onnx` và `trtexec`; không nâng cấp NumPy hoặc CUDA của dịch vụ đang chạy. `YOLO_AUTOINSTALL=false` trong helper chặn tự cài/nâng cấp dependencies.

```bash
docker compose exec backend python3 scripts/export_pose.py \
  --weights /app/models/yolo11l-pose.pt --batch 2 --build-engine
```

ONNX phải có `images` FP32 `[2,3,640,640]` và `output0` FP32 `[2,56,8400]`, không tích hợp NMS. Engine dùng FP16 nội bộ nhưng giữ input/output FP32 để parser hiện tại đọc đúng. Giữ tên hàm `NvDsInferParseYoloV8Pose` để tương thích ABI; tên hàm không giới hạn model ở YOLOv8.

Build TensorRT có thể mất nhiều phút và tạm tăng tải GPU. Chỉ chuyển model sau khi engine đã build và thử inference qua parser thành công. Kiểm tra `/api/debug/pipeline` để xác nhận engine thực sự đang chạy, đủ camera và nhịp tracker; kiểm tra `/api/fms/status` và WebSocket `/ws/fms` để xác nhận FMS tiếp tục cập nhật.

### Quay lại model cũ

Các file `.pt`, `.onnx`, `.engine` của YOLOv8x vẫn nằm trong `backend/models`. Chạy từ thư mục dự án:

```bash
docker compose -f docker-compose.yml -f docker-compose.pose-rollback.yml \
  up -d --no-deps --no-build backend
```

Quay về YOLO11l:

```bash
docker compose up -d --no-deps --no-build backend
```

Cả hai lệnh tạo lại riêng backend; tracking và kết nối FMS gián đoạn ngắn trong lúc khởi động lại rồi tự kết nối lại. Không xóa dữ liệu camera/calibration/registry, không restart frontend, MediaMTX hoặc database. Lệnh `docker compose restart` đơn thuần không áp dụng biến môi trường model mới. Compose vẫn giữ `restart: always` để backend tự chạy lại khi Docker khởi động.
