# 👁️🤖 RTC Vision & Robot Management System (VMS + FMS Digital Twin 3D)

> **Hệ thống giám sát thị giác máy tính AI (VMS) kết hợp Bản đồ số 3D Digital Twin điều hành đội xe AGV/AMR (FMS)**  
> Tích hợp thời gian thực NVIDIA DeepStream, YOLOv8x-Pose, Segment Anything 2 (SAM 2), Three.js 3D Digital Twin, WebRTC WHEP và MQTT VDA-5050.

---

## 📑 Mục lục
- [1. Kiến trúc hệ thống tổng quan](#-1-kiến-trúc-hệ-thống-tổng-quan)
- [2. Chi tiết các module giao diện](#-2-chi-tiết-các-module-giao-diện)
  - [2.1. Monitor View (Giám sát Multi-Camera AI)](#21-monitor-view---giám-sát-multi-camera-ai-live)
  - [2.2. Map Robot 3D (Digital Twin FMS)](#22-map-robot-3d---digital-twin-fms-threejs)
  - [2.3. Building & Target Registration (Cấu hình ROI & SAM2 Mask)](#23-building--target-registration---cấu-hình-roi--sam2-mask)
  - [2.4. Analytics & Historical Audit (Thống kê & Báo cáo)](#24-analytics--historical-audit---thống-kê--báo-cáo)
- [3. Công nghệ AI & Thị giác máy tính](#-3-công-nghệ-ai--thị-giác-máy-tính)
- [4. Hướng dẫn cài đặt & Khởi động nhanh](#-4-hướng-dẫn-cài-đặt--khởi-động-nhanh)
- [5. Danh mục cổng dịch vụ (Ports)](#-5-danh-mục-cổng-dịch-vụ-ports)
- [6. Lệnh quản trị hữu ích](#-6-lệnh-quản-trị-hữu-ích)

---

## 🏗️ 1. Kiến trúc hệ thống tổng quan

```
                                      ┌──────────────────────────────────────────────┐
                                      │              FMS Server (LAN)                │
                                      │   192.168.5.105:1883 (MQTT Broker VDA 5050)  │
                                      │   192.168.5.105:5432 (PostgreSQL fms_db_v2)  │
                                      └──────────────────────┬───────────────────────┘
                                                             │
                                                       MQTT & DB Sync (10Hz)
                                                             │
                                                             ▼
┌─────────────────────────────────┐               ┌───────────────────────────────────────────┐
│     IP Cameras (RTSP Streams)   │──────────────▶│       FastAPI Core Backend (Port 8000)    │
│  - 192.168.5.201, 241, 242, 243 │               │ - DeepStream 8.0 / TensorRT YOLOv8x-Pose  │
└─────────────────────────────────┘               │ - SAM 2 Realtime Mask Engine              │
                                                  │ - Cross-Camera ReID & Homography Calib    │
┌─────────────────────────────────┐               │ - Fusion Bridge: Vision AI ⟷ FMS Odom     │
│   MediaMTX WebRTC (Port 8081)   │◀──────────────│ - WebSocket Server (/ws, /ws/fms)         │
│     - Ultra Low-latency WHEP    │               │ - ChromaDB Vector Store & RAG AI Chatbot  │
└─────────────────────────────────┘               └─────────────────────┬─────────────────────┘
                                                                        │
                                                              WebSocket & REST APIs
                                                                        │
                                                                        ▼
                                                  ┌───────────────────────────────────────────┐
                                                  │    Next.js Web Dashboard (Port 3000)      │
                                                  │ ├── 🖥️ 1. Monitor (Live AI Grid)          │
                                                  │ ├── 🗺️ 2. Map Robot 3D (Three.js Twin)    │
                                                  │ ├── 📐 3. Building (ROI, Homography, SAM2)│
                                                  │ └── 📊 4. Analytics (BI, Heatmap & Logs)  │
                                                  └───────────────────────────────────────────┘
```

---

## 🖥️ 2. Chi tiết các module giao diện

### 2.1. Monitor View - Giám sát Multi-Camera AI Live

Giao diện trung tâm giám sát thời gian thực toàn bộ mạng lưới camera với độ trễ cực thấp (<200ms qua WebRTC WHEP), tích hợp toàn bộ các lớp trực quan hóa AI:

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 👁️ RTC VISION MONITOR                                             [Layout: 2x2 ▼] [🔴 LIVE] [Theme 🌗] │
├──────────────────────────────────────────┬─────────────────────────────────────────────────────────────┤
│ 📹 CAMERA 01 - Cửa Kho Chính (RTSP-01)    │ 📹 CAMERA 02 - Khu vực AGV A1 (RTSP-02)                     │
│ ┌──────────────────────────────────────┐ │ ┌─────────────────────────────────────────────────────────┐ │
│ │ [Global_ID: 104] 🏃 (Standing)       │ │ │ 🤖 [AGV-01] ⚡ 87% | 1.2 m/s [SYNCED] [Carrying: RACK-03]│ │
│ │  ├── 17 YOLO-Pose Keypoint Skeleton  │ │ │  └── 🟩 SAM2 Realtime Mask Tracking                     │ │
│ │  └── Floor Pos: (X: 12.4m, Y: 5.1m)  │ │ │                                                         │ │
│ │                                      │ │ │ 🛑 [ROI WARNING]: Human In AGV Path!                    │ │
│ │ 🚨 [ALERT: ROI INTRUSION]            │ │ │ [Global_ID: 108] 🚶 (Walking)                           │ │
│ └──────────────────────────────────────┘ │ └─────────────────────────────────────────────────────────┘ │
├──────────────────────────────────────────┴─────────────────────────────────────────────────────────────┤
│ 🔔 REAL-TIME EVENT STREAM:                                                                             │
│ [16:45:10] 🚨 ROI Violation: ID#104 entered Forbidden Racking Zone (Cam 01)                            │
│ [16:45:12] ⚠️ Fall Incident Detected: ID#112 posture changed to 'LYING' (Cam 03)                       │
│ [16:45:15] 🔄 Fusion Cross-Check: AGV-02 vision pos verified with FMS odometry (Δ = 0.08m)            │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

* **YOLOv8x-Pose Skeleton Overlay**: Nhận diện 17 điểm khớp xương người theo thời gian thực, phát hiện dáng điệu (`Standing`, `Sitting`, `Lying Down / Fall Alert`).
* **SAM 2 Instance Mask**: Hiển thị lớp phủ màu bán trong suốt bám sát đường biên thực của robot/kệ hàng đã đăng ký.
* **Định vị mặt sàn (Homography Floor Mapping)**: Chuyển đổi tọa độ pixel chân đối tượng sang tọa độ mặt sàn thế giới thực $(X, Y)$ mét.
* **Fusion Cross-Check Badge**: Đối chiếu chéo vị trí nhận diện từ Camera với tọa độ Odometry từ FMS Server (`SYNCED`, `DEVIATED`, `FMS_ONLY`, `VISION_ONLY`).
* **Đăng ký vật thể trực tiếp**: Cho phép nhấp chọn trực tiếp đối tượng trên khung hình để tạo mask và lưu vào bộ nhớ nhận dạng.

---

### 2.2. Map Robot 3D - Digital Twin FMS (Three.js)

Bản đồ số 3D tái hiện toàn diện không gian nhà xưởng, vị trí đội xe AGV/AMR, con người và hàng hóa:

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 🗺️ FMS 3D DIGITAL TWIN                           [3D Perspective | 2D SLAM] [🎯 Follow: AGV-01] [⚙️]    │
├───────────────────────────────────────────────────────────────────────┬────────────────────────────────┤
│                                                                       │ 🤖 THÔNG TIN AGV-01            │
│                 ┌───────────────────────────┐                         ├────────────────────────────────┤
│                 │  RACKING STORAGE ZONE B   │                         │ • Model: AMR-LIFT-1000         │
│                 └───────────────────────────┘                         │ • Trạng thái: ACTIVE / RUNNING │
│                                                                       │ • Vận tốc: 1.25 m/s            │
│             🚶 (Human #104)                                           │ • Pin: [████████░░] 84%        │
│                [Vision 3D]          🤖 [AGV-01] ───► Destination      │ • Tọa độ X,Y,Z: 14.2, 8.5, 0.0 │
│                                     (Carrying RACK-03)                │ • Hướng quay θ: 125.4°         │
│                                                                       │ • Đích đến: NODE_STATION_12    │
│                        ⚡ [Charging Bay 01]                           │ • Kệ đang nâng: RACK-03        │
│                                                                       │ • Cross-Check: ✅ SYNCED (0.05m)│
│    ┌────────────┐                                                     ├────────────────────────────────┤
│    │ CAMERA 01  │▒▒▒▒▒▒ (Camera FOV Frustum)                          │ 🎛️ LAYER FILTER                │
│    └────────────┘                                                     │ [x] AGV/AMR Robots             │
│                                                                       │ [x] Human 3D Spatial Position  │
│  [Grid: 1.0m/cell]  [Laser SLAM Map Overlay: fms_map.png]             │ [x] Camera Frustums            │
│                                                                       │ [x] Safety/Forbidden Zones     │
└───────────────────────────────────────────────────────────────────────┴────────────────────────────────┘
```

* **Công nghệ 3D Three.js & OrbitControls**: Cho phép xoay, zoom, pan tự do hoặc khóa camera theo dõi sát đuôi robot (`Camera Follow Mode`).
* **Nội suy chuyển động 60 FPS LERP**: Tự động mượt mà hóa quỹ đạo di chuyển giữa các gói tin Telemetry 10Hz từ FMS Server.
* **Laser SLAM 2D/3D Hybrid Overlay**: Hiển thị ảnh quét laser SLAM thực tế (`fms_map.png`) kết hợp các mô hình khối 3D (kệ hàng, trạm sạc, tường ngăn, cột mốc).
* **Tích hợp vị trí con người từ Vision AI**: Chiếu tọa độ 3D của công nhân từ hệ thống camera trực tiếp lên sàn 3D, giúp cảnh báo va chạm giữa người và robot.
* **Tự động chuyển đổi Chế độ mô phỏng (Simulation Fallback)**: Nếu mất kết nối với FMS LAN, hệ thống tự động chạy engine giả lập chuyển động để duy trì demo/thử nghiệm.

---

### 2.3. Building & Target Registration - Cấu hình ROI & SAM2 Mask

Không gian thiết lập các vùng an ninh, hiệu chuẩn ma trận Homography và đăng ký học mẫu vật thể:

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 📐 CẤU HÌNH CAMERA & ĐĂNG KÝ VẬT THỂ (BUILDING)                   [Camera: CAM_01 ▼] [Lưu cấu hình 💾] │
├────────────────────────────────────────┬───────────────────────────────────────────────────────────────┤
│ 🖼️ KHUNG HÌNH HIỆU CHUẨN & MASK        │ 🛠️ DANH SÁCH VÙNG & NHÃN ĐĂNG KÝ                             │
│ ┌────────────────────────────────────┐ │ 📌 Vùng quy tắc (Rules):                                      │
│ │    [ROI_01: Khu vực cấm xâm nhập]   │ │  ├── 🛑 Khu vực cấm di chuyển (Forbidden Zone) - Polygon      │
│ │    ┌───────────────────────────┐   │ │  ├── ⚠️ Cảnh báo lảng vảng (Loitering Zone > 10s)            │
│ │    │ ⚠️ CẤM NGƯỜI ĐI LẠI       │   │ │  └── 🚧 Vạch ranh giới (Tripwire Cross-line)                 │
│ │    └───────────────────────────┘   │ ├───────────────────────────────────────────────────────────────┤
│ │                                    │ │ 🎯 Bộ nhớ nhãn SAM 2 (Registered Targets):                   │
│ │      [+] Điểm vật  [-] Điểm nền    │ │  ├── 🤖 AGV_ECO_01 (Category: robot, 4 samples view)         │
│ │      ┌───────────────┐             │ │  ├── 📦 RACK_HEAVY_B2 (Category: rack, 2 samples view)       │
│ │      │ 🟩 MASK SAM 2 │             │ │  └── ➕ [Thêm nhãn mới bằng cách khoanh vùng]                 │
│ │      └───────────────┘             │ ├───────────────────────────────────────────────────────────────┤
│ └────────────────────────────────────┘ │ 📐 Ma trận Homography (Pixel ➔ Mét sàn):                      │
│                                        │  4 Điểm mốc: P1(x,y)➔(0,0) | P2➔(10,0) | P3➔(10,8) | P4➔(0,8)  │
└────────────────────────────────────────┴───────────────────────────────────────────────────────────────┘
```

* **Vẽ vùng an ninh đa giác (Polygon ROI)**: Thiết lập không giới hạn vùng cấm, vùng cảnh báo an toàn cho xe nâng / AGV.
* **Zero-Shot SAM 2 Interactive Prompter**: Dùng nhấp chuột trái (`+ Điểm vật`) và nhấp chuột phải (`- Điểm nền`) để lấy đường biên hoàn hảo cho bất kỳ vật thể nào không cần train lại model.
* **Lưu đa góc nhìn (Multi-View Angle Memory)**: Lưu trữ các góc nhìn khác nhau của cùng một đối tượng để tăng độ chính xác khi tracking dưới nhiều điều kiện ánh sáng.

---

### 2.4. Analytics & Historical Audit - Thống kê & Báo cáo

Trung tâm phân tích dữ liệu thị giác máy tính và hiệu suất vận hành:

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 📊 BÁO CÁO THỐNG KÊ & PHÂN TÍCH HOẠT ĐỘNG (ANALYTICS)                      [Hôm nay ▼] [Xuất PDF/CSV 📥]│
├────────────────────────┬────────────────────────┬────────────────────────┬─────────────────────────────┤
│ 👥 Lượt người phát hiện │ 🤖 Hiệu suất đội AGV   │ 🚨 Sự cố an ninh       │ ⏱️ Tốc độ phản hồi AI       │
│      **1,420**         │      **94.8%**         │      **18 lần**        │      **14.2 ms/frame**      │
├────────────────────────┴────────────────────────┴────────────────────────┴─────────────────────────────┤
│ 📈 BIỂU ĐỒ LƯU LƯỢNG NGƯỜI & AGV THEO GIỜ (HOURLY DETECTIONS)                                         │
│ 200 │                                            ██                                                    │
│ 150 │                        ██                  ██    ██                                              │
│ 100 │            ██    ██    ██    ██            ██    ██    ██                                        │
│  50 │      ██    ██    ██    ██    ██    ██      ██    ██    ██    ██                                  │
│   0 └──────┴─────┴─────┴─────┴─────┴─────┴───────┴─────┴─────┴─────┴───────────────────────────────────│
│     08:00 09:00 10:00 11:00 12:00 13:00 14:00 15:00 16:00 17:00 18:00                                 │
├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 📋 NHẬT KÝ SỰ CỐ & VIDEO REPLAY:                                                                       │
│ [16:30:15] | Cam 01 | ID: #104 | Xâm nhập vùng cấm | Thời lượng: 12s | [▶️ Xem lại Clip] [⬇️ Tải về] │
│ [15:10:02] | Cam 03 | ID: #089 | Té ngã lao động   | Cấp độ: CAO     | [▶️ Xem lại Clip] [⬇️ Tải về] │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

* **Biểu đồ động tương tác (Recharts)**: Thống kê mật độ lưu lượng theo giờ, biểu đồ phân bố trạng thái di chuyển của đội xe FMS.
* **Ma trận chuyển vùng liên Camera (Cross-Camera Transitions)**: Theo dõi hành trình di chuyển của đối tượng qua nhiều góc camera trong tòa nhà.
* **Kho lưu trữ sự cố kèm Video Playback**: Tự động cắt và lưu video ngắn 15s-30s khi có sự cố cảnh báo (ROI violation / Fall detection) để tra soát lại.

---

## 🧠 3. Công nghệ AI & Thị giác máy tính

| Thành phần AI | Công nghệ / Framework | Vai trò & Tính năng |
| :--- | :--- | :--- |
| **DeepStream Engine** | NVIDIA DeepStream 8.0 + TensorRT | Xử lý song song nhiều luồng RTSP camera ở mức phần cứng GPU, tối ưu hóa NVDEC & NVEGL. |
| **Human Pose & Detection** | YOLOv8x-Pose TensorRT Engine | Phát hiện con người & bóc tách 17 khớp xương (mũi, mắt, tai, vai, khuỷu, cổ tay, hông, gối, cổ chân) để phân tích hành vi ngã/ngồi/đứng. |
| **Instance Segmentation** | Segment Anything 2 (SAM 2) | Bám vết và vẽ mask chính xác theo thời gian thực cho robot AGV và kệ hàng đã đăng ký. |
| **Shared Feature Backbone** | SAM2 Image Feature Caching | Trích xuất đặc trưng ảnh một lần cho nhiều nhãn trên cùng camera, giảm 60% tải suy luận GPU. |
| **Re-Identification (ReID)** | OSNet / ResNet50 ReID Embeddings | Định danh người đồng nhất giữa các camera khác nhau (Global ID Tracking). |
| **Vector Search Engine** | ChromaDB / Cosine Similarity | Lưu trữ và đối sánh vector đặc trưng khuôn mặt / dáng người / nhãn vật thể với tốc độ mili-giây. |
| **RAG Assistant** | LLM + Semantic Vector Index | Trợ lý AI tích hợp sẵn, giải đáp thắc mắc về sự cố, vị trí robot và nhật ký hệ thống bằng ngôn ngữ tự nhiên. |

---

## 🚀 4. Hướng dẫn cài đặt & Khởi động nhanh

### Yêu cầu phần cứng & môi trường:
* **GPU**: NVIDIA RTX (khuyên dùng RTX 3060 / 4060 Ti / 5060 Ti trở lên, VRAM $\ge$ 8GB).
* **Hệ điều hành**: Ubuntu 22.04 LTS / 24.04 LTS.
* **Phần mềm**: Docker & NVIDIA Container Toolkit, Node.js 18+, Python 3.10+.

### Khởi động toàn bộ hệ thống bằng 1 lệnh:
```bash
cd "/home/rtcai/Desktop/Vision Manager"
./run_all.sh
```

---

## 🔌 5. Danh mục cổng dịch vụ (Ports)

| Dịch vụ | Địa chỉ URL / Port | Mô tả |
| :--- | :--- | :--- |
| 🌐 **Next.js Web Dashboard** | [http://localhost:3000](http://localhost:3000) | Giao diện điều hành chính (Monitor, 3D Map, Building, Analytics) |
| 🔌 **FastAPI Backend REST** | [http://localhost:8000/docs](http://localhost:8000/docs) | Swagger API documentation & Core controller |
| 🤖 **FMS Telemetry WebSocket** | `ws://localhost:8000/ws` | Kênh WebSocket truyền dữ liệu robot & bounding box thời gian thực |
| 📹 **MediaMTX WebRTC Streamer** | [http://localhost:8081](http://localhost:8081) | Máy chủ phân phối luồng WHEP WebRTC siêu độ trễ thấp |
| 🗄️ **PostgreSQL FMS DB** | `localhost:5432` | Cơ sở dữ liệu lưu trữ lịch sử sự kiện & layout bản đồ |

---

## 🛠️ 6. Lệnh quản trị hữu ích

* **Xem nhật ký hoạt động trực tiếp (Logs)**:
  ```bash
  ./run_all.sh logs
  ```
* **Kiểm tra trạng thái các container**:
  ```bash
  ./run_all.sh status
  ```
* **Dừng an toàn toàn bộ hệ thống**:
  ```bash
  ./run_all.sh stop
  ```
