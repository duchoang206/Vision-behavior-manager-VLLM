# Tổng quan kiến trúc hệ thống (System Architecture Overview)

## 1. Giới thiệu
Hệ thống **RTC Vision & Robot Management System** là sự kết hợp thống nhất giữa:
- **Vision AI VMS**: Giám sát video thông minh đa camera, ứng dụng NVIDIA DeepStream 8.0, YOLOv8x-Pose, SAM 2 và OSNet ReID.
- **FMS 3D Digital Twin**: Bản đồ số 3D kết nối thời gian thực với hệ thống điều hành đội xe AGV/AMR thông qua chuẩn MQTT VDA 5050 và PostgreSQL.

## 2. Luồng dữ liệu thời gian thực (Realtime Data Pipeline)
1. **Camera Ingestion**: Luồng video RTSP từ các camera IP trên mạng LAN được đưa vào MediaMTX và DeepStream pipeline qua NVDEC.
2. **AI Inference Layer**: DeepStream và TensorRT thực hiện phát hiện người, 17 điểm khung xương (pose), bóc tách mask SAM2 và sinh vector đặc trưng ReID.
3. **Fusion & Cross-Check**: Backend FastAPI nhận telemetry từ FMS Server qua MQTT broker (10Hz), tính toán ma trận Homography để đối chiếu chéo vị trí vật thể trên camera và vị trí robot từ odometry.
4. **WebSocket & UI Broadcasting**: Kết quả phân tích, bounding boxes, mask coordinates và telemetry robot được broadcast tới Next.js Web Dashboard qua kênh WebSocket `/ws` và `/ws/fms`.
