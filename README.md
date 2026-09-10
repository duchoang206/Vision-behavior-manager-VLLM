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

### Độ trễ mask

Bộ đọc RTSP cho nhãn dùng FFmpeg với `CAP_PROP_N_THREADS=1` ngay khi mở
camera, kể cả khi kết nối lại. Queue một frame ở Python không loại bỏ được
độ trễ do bộ giải mã nhiều luồng giữ các frame bên trong.
Frame được chọn sau khi worker mask sẵn sàng; không dịch hoặc phóng mask
bằng vận tốc dự đoán. `observed_at` là thời điểm giải mã frame, không phải
thời điểm kết thúc suy luận; kết quả cũ hơn 500 ms không được xuất thành mask.

Trong `/api/debug/pipeline`, mỗi camera có `decoder_threads`,
`last_processing_ms` và `last_frame_age_ms`. Hai số thời gian sau đo xử lý và
tuổi frame **sau giải mã**; không đại diện cho tổng độ trễ camera → trình duyệt.
Cấu hình này không thay đổi bộ đọc/pipeline nhận diện người.

### Giảm tải GPU cho mask

`REGISTERED_MASK_SHARED_FEATURES=1` (mặc định) tính ảnh đầu vào và backbone
SAM2 một lần cho các nhãn trên **cùng frame/camera**. Bộ nhớ nhận dạng của mỗi
nhãn vẫn độc lập. Cache bị xóa sau frame, kể cả khi suy luận lỗi; không dùng
lại feature của frame trước. Model, FP16, độ phân giải và đường biên mask không
bị thay đổi bởi tối ưu này. Camera chỉ có một nhãn hoạt động dùng đường xử lý cũ.

`REGISTERED_MASK_TARGET_FPS=11` giới hạn nhịp xử lý nhãn để không tiêu hết phần
GPU tiết kiệm được vào việc tăng FPS. Mức này cao hơn khoảng 10 FPS/camera đã
đo trước tối ưu trên máy hiện tại; không phải FPS của video hay landmark người.
Frame mới nhất vẫn được lấy sau khi worker sẵn sàng, và thời gian nghỉ nằm ngoài
khóa GPU. Có thể đặt giá trị này thành `24` để ưu tiên tốc độ cập nhật tối đa.

Để đối chiếu/hoàn tác tối ưu, đặt `REGISTERED_MASK_SHARED_FEATURES=0` và
`REGISTERED_MASK_TARGET_FPS=24` trong environment của dịch vụ backend rồi tạo
lại container. Trạng thái chia sẻ feature nằm ở `/api/registry/mask/status`;
FPS và thời gian xử lý riêng từng camera nằm ở `/api/debug/pipeline`.

---

## 🤖 Tab "Map Robot 3D" (FMS Digital Twin)

- **3D Perspective & 2D Top-Down View**: Chuyển đổi linh hoạt giữa góc nhìn không gian 3D và sơ đồ mặt phẳng 2D quét laser (`fms_map.png`).
- **Smooth 60 FPS LERP**: Tự động nội suy chuyển động giữa các gói tin 10Hz từ FMS để robot di chuyển mượt mà.
- **Tương tác 3D**: Click chọn trực tiếp robot trên sàn 3D hoặc từ danh sách để xem chi tiết % Pin, vận tốc, tọa độ X/Y/Z, góc quay $\theta$, trạm đích đến.
- **Chế độ theo dõi (Camera Follow)**: Khóa camera tự động bay theo hành trình của robot được chọn.
- **Simulation Fallback**: Tự động chuyển đổi thông minh giữa tín hiệu **LIVE FMS** và mô phỏng offline nếu máy trạm chưa kết nối mạng FMS.
