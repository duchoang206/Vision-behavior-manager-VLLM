# Hướng dẫn triển khai môi trường Docker & NVIDIA Container Toolkit

## 1. Yêu cầu tiên quyết
- **NVIDIA GPU Driver**: Phiên bản driver tương thích CUDA 12.x trở lên.
- **NVIDIA Container Toolkit**: Đã cấu hình runtime `nvidia` mặc định cho Docker daemon (`/etc/docker/daemon.json`).

## 2. Các container dịch vụ trong hệ thống
1. `yolo_deepstream_backend`: Container chạy DeepStream 8.0, FastAPI core và AI inference engine.
2. `yolo_mediamtx`: Máy chủ WebRTC WHEP chuyển tiếp luồng video siêu độ trễ thấp tới trình duyệt.
3. `yolo_analytics_db`: Container PostgreSQL lưu trữ nhật ký sự kiện, trạng thái FMS và bản ghi telemetry.
4. `yolo_frontend`: Ứng dụng Next.js giao diện điều hành thời gian thực.

## 3. Khởi động và kiểm tra
```bash
# Khởi động toàn bộ dịch vụ
./run_all.sh

# Kiểm tra trạng thái hoạt động
./run_all.sh status

# Xem log trực tiếp của backend
docker logs -f yolo_deepstream_backend
```
