# Hướng dẫn tối ưu hóa hiệu năng và tài nguyên GPU

## 1. Lưu ý quan trọng về Pose Estimation
- **Cấu hình `interval=0`**: Với pipeline suy luận khung xương người (YOLOv8x-Pose), bắt buộc giữ giá trị `interval=0` trong `config_infer_primary.txt`. Nếu bật interval bỏ qua khung hình (frame skipping), các điểm keypoint trên cơ thể sẽ bị nhấp nháy liên tục do mất dấu vết bám đuôi.

## 2. Các phương pháp tối ưu hóa hiệu quả cao
1. **Shared Feature Backbone cho SAM 2**:
   - Biến môi trường: `REGISTERED_MASK_SHARED_FEATURES=1`.
   - Tính toán backbone ảnh 1 lần cho tất cả các nhãn robot/kệ trên cùng một khung hình, giảm tới 60% tải suy luận cho worker mask.
2. **Khống chế Target FPS cho Mask Engine**:
   - Biến môi trường: `REGISTERED_MASK_TARGET_FPS=11`.
   - Giúp ổn định GPU, tránh việc worker mask chạy không tải ở mức tối đa làm tăng nhiệt độ card đồ họa.
3. **Điều chỉnh độ phân giải luồng phụ**:
   - Khi mở rộng quy mô hệ thống lên nhiều camera ($\ge 8$ camera), khuyến nghị giảm độ phân giải luồng phụ từ 1080p xuống 720p hoặc 540p trước khi đưa vào DeepStream muxer.
