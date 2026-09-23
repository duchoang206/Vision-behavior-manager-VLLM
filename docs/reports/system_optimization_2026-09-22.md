# Báo cáo tối ưu hệ thống R-SkyView

**Ngày đo:** 22/09/2026 (ICT)

## 1. Trạng thái triển khai

- `yolo_deepstream_backend`, `yolo_frontend`, `yolo_mediamtx` và `yolo_analytics_db` đang chạy.
- Frontend đã build production bằng Next.js 16.2.12 và đã recreate container `yolo_frontend`.
- Detector đang chạy TensorRT FP16 trong DeepStream với tracker NvDCF. SAM2 vẫn chạy CUDA PyTorch FP16; chưa phải TensorRT.
- Pose estimation đang tắt theo yêu cầu trước đó.

## 2. Log và truy vấn

- Log có cấu trúc được ghi song song vào PostgreSQL `audit.system_logs` và SSD dưới dạng JSONL gzip lossless.
- Queue ghi log có giới hạn, ghi theo batch, có index theo thời gian, level, source và camera; dữ liệu nhạy cảm được che trước khi lưu.
- PostgreSQL giữ 7 ngày / tối đa 200.000 bản ghi; archive SSD giữ tối đa 30 ngày hoặc 512 MiB, tự xoay chunk 16 MiB.
- Kiểm tra runtime: queue `0`, dropped `0`, lỗi database `null`, lỗi archive `null`, đã ghi `491` bản ghi.
- MP4 camera vẫn dùng FFmpeg `-c copy` theo file từng giờ và vòng đời 48 giờ; không re-encode nên không làm giảm chất lượng.

## 3. Độ trễ và mask realtime

Trong cửa sổ đo 15 giây trên 5 camera đã deploy:

- WebSocket: `0` lỗi, `0` đóng bất thường, không có packet đi ngược thứ tự.
- Metadata từ probe backend đến receiver trình duyệt: p95 khoảng `8,15–12,17 ms`. Đây là độ trễ metadata, không phải cam-to-photon và không thể là `0 ms` về mặt vật lý.
- Metadata đạt khoảng `19,99–24,92 Hz`; cơ chế latest-only bỏ dữ liệu cũ khi cần để không tích trễ.
- Callback vẽ overlay trên Monitor chỉ khoảng `0,1–0,3 ms` p95; nút thắt chính là bằng chứng mask mới từ SAM2, không phải JavaScript vẽ.
- Bằng chứng SAM2 mới quan sát được khoảng `6,38–7,49 Hz`, tuổi p95 khoảng `209–292 ms`. Các packet trung gian dùng vị trí/trạng thái mới nhất và giữ mask trong TTL ngắn để giảm nhấp nháy, không giữ vô hạn.
- Khi detector confidence thấp, gallery/mask người dùng vẫn được ưu tiên nếu identity, hình học, vùng FMS và trạng thái track hợp lệ. Không dùng một mask cũ vô thời hạn để che lỗi mất dấu thật.

## 4. Kiểm tra FPS Map 3D

- Chrome headless dùng SwiftShader/software renderer chỉ đạt khoảng `6 FPS`; đây là giới hạn môi trường kiểm thử, không phản ánh GPU thật.
- Chrome có cửa sổ thật trên `NVIDIA GeForce RTX 5060 Ti` đạt khoảng `110 FPS`, p95 khoảng `13,7 ms`, không áp trần 30 FPS trong animation loop.
- Telemetry được giữ trong `ref`, mesh đọc trực tiếp mỗi `requestAnimationFrame`; bảng thông tin React được giới hạn tần suất để không kéo tụt render.
- Kết nối Digital Twin có timeout và đóng socket bị treo; packet cũ/out-of-order bị loại; khi tab ẩn thì render tạm dừng.
- Camera đã calibration được vẽ theo vùng coverage thực tế và có quy đổi hai chiều camera ↔ mặt sàn ↔ FMS. Nếu chưa lưu physical pose, biểu tượng camera là tâm vùng calibration, không phải vị trí lắp đặt suy đoán.

## 5. Workflow và nguồn camera

- Workflow Editor có lựa chọn camera, model/TensorRT, chức năng và chính sách hiển thị Monitor; pipeline đã deploy được xem riêng.
- `display.monitor=false` chỉ ẩn overlay của camera đó trên Monitor, không tắt inference, tracking hay event. Deployment trực tiếp hoặc workflow khác yêu cầu hiển thị sẽ được ưu tiên.
- Camera Sources đã có tìm kiếm, lọc trạng thái, đếm online/calibrated, thêm/sửa/xóa và panel chi tiết; preview dùng một GPU snapshot, không mở thêm decoder CPU liên tục.

## 6. MLOps hiện trạng

- Worker Active Learning đang chạy idle-only để không tranh GPU với DeepStream; hiện chưa train.
- MLOps hiện fine-tune detector YOLO, không fine-tune SAM2; polygon người dùng vẫn được giữ làm ground truth.
- Model hiện thiếu checkpoint YOLO `.pt` và dataset gốc/validation cố định tại `base/data.yaml`, nên chưa tự retrain/promote. ONNX/engine không thể dùng trực tiếp làm checkpoint fine-tune.
- Cần upload `.pt` tin cậy và dataset validation trước khi bật retrain; sau đó mới đánh giá mAP/mIoU, class drop và promote model mới.

## 7. API key

Không thể xác định thời hạn của key chỉ từ chuỗi key vì chưa biết nhà cung cấp, tài khoản và chính sách quota. Key đã được xử lý như secret: không lưu vào source/log/report và không kiểm thử bằng cách gọi ra ngoài. Vì key đã được gửi trong cuộc trò chuyện, nên revoke/rotate key đó và lưu key mới trong secret/env backend, không đưa vào frontend hoặc URL.

