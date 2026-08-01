# RTC Vision & Robot Management System (VMS + FMS Digital Twin 3D)

Hệ thống quản lý tích hợp thống nhất giữa:
1. **Vision AI VMS (DeepStream / YOLO)**: Giám sát đa camera thời gian thực, bám đuôi đối tượng (ReID tracking), thiết lập vùng ROI, phát hiện xâm nhập và WebRTC WHEP streaming.
2. **FMS 3D Robot Digital Twin**: Bản đồ số 3D kết nối thời gian thực với hệ thống điều hành đội xe FMS qua MQTT và PostgreSQL, trực quan hóa chuyển động robot AGV/AMR, theo dõi pin, vận tốc và lộ trình điều phối.

---

## 🏗 Kiến trúc hệ thống

```text
                                       ┌──────────────────────────────────────────────┐
                                       │              FMS Server (LAN)                │
                                       │   192.168.5.105:1883 (MQTT Broker VDA 5050)  │
                                       │   192.168.5.105:5432 (PostgreSQL fms_db_v2)  │
                                       └──────────────────────┬───────────────────────┘
                                                              │
                                                        MQTT & DB Sync
                                                              │
                                                              ▼
┌─────────────────────────────────┐               ┌───────────────────────────────────────────┐
│       Camera Streams (RTSP)     │──────────────▶│       FastAPI Core Backend (Port 8000)    │
└─────────────────────────────────┘               │ - DeepStream / YOLO Multi-Cam Analytics   │
                                                  │ - FMS Realtime Bridge (10Hz telemetry)    │
┌─────────────────────────────────┐               │ - WebSocket Server (/ws, /ws/fms)         │
│   MediaMTX WebRTC (Port 8081)   │◀──────────────│ - Database Manager & RAG AI Chatbot       │
└─────────────────────────────────┘               └─────────────────────┬─────────────────────┘
                                                                        │
                                                             WebSocket & REST APIs
                                                                        │
                                                                        ▼
                                                  ┌───────────────────────────────────────────┐
                                                  │    Next.js Web Dashboard (Port 3000)      │
                                                  │ ├── 1. Monitor (Live Camera AI Grid)      │
                                                  │ ├── 2. Building (Camera & ROI Config)     │
                                                  │ ├── 3. Map Robot 3D (3D Digital Twin FMS) │
                                                  │ └── 4. Analytics (Stats & Event History)  │
                                                  └───────────────────────────────────────────┘
```

---

## 🚀 Hướng dẫn khởi động nhanh

Chỉ cần chạy 1 câu lệnh duy nhất tại thư mục gốc:

```bash
cd "/home/rtcai/Desktop/Vision Manager"
./run_all.sh
```

### Các dịch vụ khả dụng:
- 🌐 **Web Dashboard**: [http://localhost:3000](http://localhost:3000)
  - Tab 1: **Monitor** (Giám sát camera trực tiếp)
  - Tab 2: **Building** (Cấu hình camera, ROI và hiệu chuẩn)
  - Tab 3: **Map Robot 3D** (Bản đồ 3D Digital Twin điều hành đội xe FMS)
  - Tab 4: **Analytics** (Thống kê và lịch sử sự kiện)
- 🔌 **Backend REST API**: [http://localhost:8000/docs](http://localhost:8000/docs)
- 🤖 **FMS Telemetry WebSocket**: `ws://localhost:8000/ws`
- 📹 **MediaMTX WebRTC Streamer**: [http://localhost:8081](http://localhost:8081)

---

## 🛠 Lệnh quản trị hữu ích

- **Xem nhật ký hoạt động (logs)**:
  ```bash
  ./run_all.sh logs
  ```
- **Kiểm tra trạng thái các container**:
  ```bash
  ./run_all.sh status
  ```
- **Dừng toàn bộ hệ thống**:
  ```bash
  ./run_all.sh stop
  ```

## 🎯 Đăng ký mask cho Building/Label

Mask chỉ được tạo và hiển thị cho **nhãn robot/kệ đã đăng ký** trong tab
**Building**. Luồng sử dụng:

1. Chọn camera, loại `robot` hoặc `rack`, nhập tên nhãn và khoanh sát vật.
2. Chọn **Tạo mask** để tạo vùng vật thể; dùng `+ Vật` hoặc `− Nền` để chỉnh
   nếu đường biên chưa chính xác.
3. Chọn **Lưu góc nhìn**. Chọn lại cùng nhãn để lưu thêm các góc nhìn, vị trí
   hoặc điều kiện ánh sáng khác nhau.
4. Mở **Monitor** để xem mask bám theo vật khi di chuyển. Khi vật mất khỏi
   khung hình, mask được ẩn; khi vật xuất hiện lại, hệ thống cố gắng bám lại
   đúng nhãn đã đăng ký.

Hệ thống hiện dùng checkpoint SAM2 dựng sẵn để tạo và theo dõi mask theo bộ
nhớ của từng nhãn; đây **chưa phải là huấn luyện model riêng**. Các ảnh, frame
và mask đã lưu được giữ lại để có thể dùng cho bước huấn luyện riêng sau này.

Đăng ký người vẫn dùng pipeline nhận diện, tracking và pose hiện tại; người
không được đưa vào bộ xử lý mask. Đối tượng robot/kệ chưa đăng ký cũng không
được tự động tô mask.

---

## 🤖 Tab "Map Robot 3D" (FMS Digital Twin)

- **3D Perspective & 2D Top-Down View**: Chuyển đổi linh hoạt giữa góc nhìn không gian 3D và sơ đồ mặt phẳng 2D quét laser (`fms_map.png`).
- **Smooth 60 FPS LERP**: Tự động nội suy chuyển động giữa các gói tin 10Hz từ FMS để robot di chuyển mượt mà.
- **Tương tác 3D**: Click chọn trực tiếp robot trên sàn 3D hoặc từ danh sách để xem chi tiết % Pin, vận tốc, tọa độ X/Y/Z, góc quay $\theta$, trạm đích đến.
- **Chế độ theo dõi (Camera Follow)**: Khóa camera tự động bay theo hành trình của robot được chọn.
- **Simulation Fallback**: Tự động chuyển đổi thông minh giữa tín hiệu **LIVE FMS** và mô phỏng offline nếu máy trạm chưa kết nối mạng FMS.
