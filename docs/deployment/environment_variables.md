# Danh mục các biến môi trường cấu hình (Environment Variables)

## 1. Cấu hình Backend & DeepStream
| Biến môi trường | Giá trị mặc định | Ý nghĩa |
| :--- | :--- | :--- |
| `REGISTERED_MASK_SHARED_FEATURES` | `1` | Bật bộ nhớ đệm chia sẻ backbone SAM 2 cho nhiều nhãn trên cùng frame |
| `REGISTERED_MASK_TARGET_FPS` | `11` | Giới hạn nhịp xử lý tối đa cho worker bóc tách mask SAM 2 |
| `POSE_MODEL_PATH` | `/app/models/yolo11l-pose.pt` | Checkpoint pose cho adapter CUDA |
| `CPU_TRACKER_MODEL_PATH` | `/app/models/yolo11l-pose.pt` | Model của fallback; fallback vẫn tắt trong Compose |
| `DEEPSTREAM_ONNX_FILE` | `/app/models/yolo11l-pose.onnx` | Model COCO-17, input 640×640, output FP32 `[2,56,8400]` |
| `DEEPSTREAM_ENGINE_FILE` | `/app/models/yolo11l-pose_b2_gpu0_fp16.engine` | Engine TensorRT FP16 đã build cho GPU hiện tại |
| `DEEPSTREAM_BATCH_SIZE` | `2` | Batch cố định của model/engine, không đổi theo số camera |
| `FMS_MQTT_BROKER_HOST` | `192.168.5.105` | Địa chỉ IP máy chủ MQTT Broker VDA 5050 của FMS |
| `FMS_MQTT_BROKER_PORT` | `1883` | Cổng kết nối MQTT Broker |
| `FMS_DB_HOST` | `192.168.5.105` | Địa chỉ PostgreSQL máy chủ FMS |
| `MEDIAMTX_WHEP_URL` | `http://localhost:8081` | Địa chỉ dịch vụ phát luồng WebRTC MediaMTX |

## 3. Ghi hình chứng cứ MP4
| Biến môi trường | Giá trị mặc định | Ý nghĩa |
| :--- | :--- | :--- |
| `RECORDINGS_HOST_DIR` | `./recordings` | Thư mục host để lưu MP4; nên trỏ vào SSD chuyên dụng nếu có |
| `RECORDINGS_DIR` | `/var/vms/recordings` | Đường dẫn bên trong container backend |
| `ENABLE_EVIDENCE_RECORDING` | `1` | Bật/tắt ghi hình liên tục |
| `RECORDING_SEGMENT_SECONDS` | `3600` | Mỗi file một giờ; dùng FFmpeg `-c:v copy`, không encode lại |
| `RECORDING_RETENTION_HOURS` | `48` | Tự xóa file đã hoàn tất quá 48 giờ |
| `RECORDING_MIN_FREE_GB` | `2` | Dừng ghi khi SSD còn dưới mức dự phòng này |
| `MEDIAMTX_RTSP_ORIGIN` | `rtsp://127.0.0.1:8554` | Relay nội bộ dùng chung cho recorder và tracker nhãn/SAM2; không mở thêm phiên trực tiếp tới camera |

## 2. Cấu hình Web Dashboard
| Biến môi trường | Giá trị mặc định | Ý nghĩa |
| :--- | :--- | :--- |
| `NEXT_PUBLIC_BACKEND_URL` | `http://localhost:8000` | Địa chỉ REST API backend |
| `NEXT_PUBLIC_WS_URL` | `ws://localhost:8000/ws` | Kênh WebSocket truyền nhận dữ liệu AI |
