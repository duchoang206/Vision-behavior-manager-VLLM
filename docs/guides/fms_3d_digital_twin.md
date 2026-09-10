# Hướng dẫn tích hợp Bản đồ số 3D Digital Twin FMS

## 1. Nguyên lý kết nối
Bản đồ 3D Digital Twin kết nối đồng bộ 2 chiều với hệ thống điều hành đội xe FMS:
- **MQTT VDA 5050**: Nhận các gói tin trạng thái robot (`order`, `state`, `instantActions`, `visualization`) với tần số 10Hz.
- **PostgreSQL Database**: Đồng bộ thông tin layout bản đồ, trạm sạc, kệ hàng và danh sách robot.

## 2. Các tính năng chính trong giao diện 3D
- **Nội suy chuyển động (LERP Interpolation)**: Tính toán vị trí mượt mà 60 FPS giữa 2 mốc thời gian nhận dữ liệu telemetry.
- **Chế độ Camera Follow**: Nhấp chọn một robot và kích hoạt chế độ bám đuôi để camera tự động xoay và di chuyển theo robot.
- **Lớp phủ bản đồ quét Laser (SLAM Grid Overlay)**: Nạp trực tiếp file `fms_map.png` và căn chỉnh tỷ lệ mét/pixel theo đúng kích thước thực tế của nhà xưởng.
- **Kiểm tra chéo Vision ⟷ FMS (Cross-Check)**: Cảnh báo khi có sự sai lệch vị trí giữa thị giác máy tính và hệ thống định vị nội tại của robot (Odometry).
