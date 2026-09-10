# Danh mục các biến môi trường cấu hình (Environment Variables)

## 1. Cấu hình Backend & DeepStream
| Biến môi trường | Giá trị mặc định | Ý nghĩa |
| :--- | :--- | :--- |
| `REGISTERED_MASK_SHARED_FEATURES` | `1` | Bật bộ nhớ đệm chia sẻ backbone SAM 2 cho nhiều nhãn trên cùng frame |
| `REGISTERED_MASK_TARGET_FPS` | `11` | Giới hạn nhịp xử lý tối đa cho worker bóc tách mask SAM 2 |
| `FMS_MQTT_BROKER_HOST` | `192.168.5.105` | Địa chỉ IP máy chủ MQTT Broker VDA 5050 của FMS |
| `FMS_MQTT_BROKER_PORT` | `1883` | Cổng kết nối MQTT Broker |
| `FMS_DB_HOST` | `192.168.5.105` | Địa chỉ PostgreSQL máy chủ FMS |
| `MEDIAMTX_WHEP_URL` | `http://localhost:8081` | Địa chỉ dịch vụ phát luồng WebRTC MediaMTX |

## 2. Cấu hình Web Dashboard
| Biến môi trường | Giá trị mặc định | Ý nghĩa |
| :--- | :--- | :--- |
| `NEXT_PUBLIC_BACKEND_URL` | `http://localhost:8000` | Địa chỉ REST API backend |
| `NEXT_PUBLIC_WS_URL` | `ws://localhost:8000/ws` | Kênh WebSocket truyền nhận dữ liệu AI |
