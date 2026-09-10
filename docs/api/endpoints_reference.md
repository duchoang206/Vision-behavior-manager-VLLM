# Danh mục REST API & WebSocket Endpoints

## 1. REST APIs (FastAPI Core - Port 8000)

| Endpoint | Method | Mô tả |
| :--- | :--- | :--- |
| `/api/cameras` | `GET` / `POST` | Quản lý danh sách cấu hình camera RTSP và tham số giải mã |
| `/api/rules` | `GET` / `POST` / `DELETE` | Thiết lập các quy tắc an ninh ROI (Polygon, Tripwire, Loitering) |
| `/api/registry/mask` | `GET` / `POST` | Quản lý danh sách các nhãn vật thể đăng ký và mẫu ảnh SAM2 |
| `/api/registry/mask/status` | `GET` | Xem trạng thái bộ nhớ đệm chia sẻ feature SAM2 và FPS xử lý |
| `/api/debug/pipeline` | `GET` | Kiểm tra chi tiết độ trễ, số luồng decoder và frame age từng camera |
| `/api/analytics/events` | `GET` | Lấy danh sách sự cố an ninh và bản ghi video lịch sử |
| `/api/chat` | `POST` | Giao tiếp với RAG AI Chatbot hỗ trợ vận hành |

## 2. WebSocket Channels

- `ws://localhost:8000/ws`: Kênh truyền dữ liệu nhận diện thời gian thực (Bounding Box, Pose Keypoints, Mask Polygons, Security Alerts).
- `ws://localhost:8000/ws/fms`: Kênh truyền dữ liệu Telemetry đội xe AGV/AMR (Tọa độ X/Y/Z, Pin, Tốc độ, Trạng thái FSM, Trạm đích).
