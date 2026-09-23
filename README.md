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
- [7. Tối ưu tài nguyên](#7-tối-ưu-tài-nguyên)

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

![Monitor AI Live Grid](docs/images/monitor_ai_view.jpg)

* **YOLOv8x-Pose Skeleton Overlay**: Nhận diện 17 điểm khớp xương người theo thời gian thực, phát hiện dáng điệu (`Standing`, `Sitting`, `Lying Down / Fall Alert`).
* **SAM 2 Instance Mask**: Hiển thị lớp phủ màu bán trong suốt bám sát đường biên thực của robot/kệ hàng đã đăng ký.
* **Định vị mặt sàn (Homography Floor Mapping)**: Chuyển đổi tọa độ pixel chân đối tượng sang tọa độ mặt sàn thế giới thực $(X, Y)$ mét.
* **Fusion Cross-Check Badge**: Đối chiếu chéo vị trí nhận diện từ Camera với tọa độ Odometry từ FMS Server (`SYNCED`, `DEVIATED`, `FMS_ONLY`, `VISION_ONLY`).
* **Đăng ký vật thể trực tiếp**: Cho phép nhấp chọn trực tiếp đối tượng trên khung hình để tạo mask và lưu vào bộ nhớ nhận dạng.

---

### 2.2. Map Robot 3D - Digital Twin FMS (Three.js)

Bản đồ số 3D tái hiện toàn diện không gian nhà xưởng, vị trí đội xe AGV/AMR, con người và hàng hóa:

![FMS 3D Digital Twin Map](docs/images/robot_map_3d.jpg)

* **Công nghệ 3D Three.js & OrbitControls**: Cho phép xoay, zoom, pan tự do hoặc khóa camera theo dõi sát đuôi robot (`Camera Follow Mode`).
* **Nội suy chuyển động 60 FPS LERP**: Tự động mượt mà hóa quỹ đạo di chuyển giữa các gói tin Telemetry 10Hz từ FMS Server.
* **Laser SLAM 2D/3D Hybrid Overlay**: Hiển thị ảnh quét laser SLAM thực tế (`fms_map.png`) kết hợp các mô hình khối 3D (kệ hàng, trạm sạc, tường ngăn, cột mốc).
* **Tích hợp vị trí con người từ Vision AI**: Chiếu tọa độ 3D của công nhân từ hệ thống camera trực tiếp lên sàn 3D, giúp cảnh báo va chạm giữa người và robot.
* **Tự động chuyển đổi Chế độ mô phỏng (Simulation Fallback)**: Nếu mất kết nối với FMS LAN, hệ thống tự động chạy engine giả lập chuyển động để duy trì demo/thử nghiệm.

---

### 2.3. Building & Target Registration - Cấu hình ROI & SAM2 Mask

Không gian thiết lập các vùng an ninh, hiệu chuẩn ma trận Homography và đăng ký học mẫu vật thể:

![Building ROI and SAM 2 Registration](docs/images/building_roi_sam2.jpg)

* **Vẽ vùng an ninh đa giác (Polygon ROI)**: Thiết lập không giới hạn vùng cấm, vùng cảnh báo an toàn cho xe nâng / AGV.
* **Zero-Shot SAM 2 Interactive Prompter**: Dùng nhấp chuột trái (`+ Điểm vật`) và nhấp chuột phải (`- Điểm nền`) để lấy đường biên hoàn hảo cho bất kỳ vật thể nào không cần train lại model.
* **Lưu đa góc nhìn (Multi-View Angle Memory)**: Lưu trữ các góc nhìn khác nhau của cùng một đối tượng để tăng độ chính xác khi tracking dưới nhiều điều kiện ánh sáng.

---

### 2.4. Analytics & Historical Audit - Thống kê & Báo cáo

Trung tâm phân tích dữ liệu thị giác máy tính và hiệu suất vận hành:

![Analytics BI Dashboard](docs/images/analytics_dashboard.jpg)

* **Biểu đồ động tương tác (Recharts)**: Thống kê mật độ lưu lượng theo giờ, biểu đồ phân bố trạng thái di chuyển của đội xe FMS.
* **Ma trận chuyển vùng liên Camera (Cross-Camera Transitions)**: Theo dõi hành trình di chuyển của đối tượng qua nhiều góc camera trong tòa nhà.
* **Kho lưu trữ sự cố kèm Video Playback**: Tự động cắt và lưu video ngắn 15s-30s khi có sự cố cảnh báo (ROI violation / Fall detection) để tra soát lại.

---

### 2.5. Workflow Editor - Tùy chọn tính năng Vision

Mở **Workflow Editor** ở sidebar hoặc đường dẫn `/?tab=workflow_editor`.

1. Chọn **Camera Source** trên canvas rồi chọn camera đã đăng ký trong Building.
2. Thêm block bằng nút **+** hoặc kéo từ thư viện; nối cổng bên phải (output) vào cổng bên trái (input). Có thể phân nhánh và gộp kết quả; không cho phép vòng lặp.
3. Cấu hình từng block ở Inspector. ROI Filter / Dwell Time dùng polygon và Line Crossing dùng tripwire có thật của camera trong Building. Sửa hình học ROI cần triển khai lại để cập nhật.
4. **Lưu Workflow** chỉ lưu bản nháp trên server. **Triển khai Pipeline** kiểm tra toàn bộ đồ thị rồi áp dụng lên metadata Vision trực tiếp; kết quả, số lượng và thời điểm cập nhật hiển thị dưới canvas.
5. **Dừng** chỉ dừng workflow, không dừng camera, detector, Calibration hay kết nối FMS. Workflow tiếp tục xử lý trên backend khi chuyển tab hoặc đóng trình duyệt.

- Hỗ trợ kéo node, zoom/fit, xóa kết nối, nhân bản, Undo/Redo, Import/Export JSON; `Ctrl+S` lưu và `Ctrl+Z` hoàn tác. Bản nháp chưa lưu có bản sao phục hồi trên trình duyệt.
- Object Detector lọc confidence của nhận diện hiện có; Class Filter lọc lớp đối tượng; Object Matcher lọc nhãn do tracker đã gán. Các block này không nạp model AI mới.
- ROI và cắt vạch dùng điểm tiếp xúc đáy bounding box trong tọa độ camera normalized `0..1`. Dwell Time đo thời gian hiện diện liên tục; mất track hoặc gián đoạn quá 2 giây sẽ bắt đầu lại. Object Counter đếm trên frame, không phải tổng lượt tích lũy.
- Display Output hiển thị tại editor và API runtime, chưa thay thế overlay Monitor / Map. **Alert / Webhook chưa hỗ trợ thực thi**, được vô hiệu hóa trong thư viện.
- Mỗi lần triển khai chạy một workflow với một camera, có thể có nhiều nhánh/output. Tắt block xử lý là bỏ qua bước đó và truyền dữ liệu tiếp; không tắt Camera Source hoặc Display Output.
- API: `GET /api/workflow`, `POST /api/workflow/draft`, `POST /api/workflow/deploy`, `POST /api/workflow/stop`, `GET /api/workflow/runtime`. Ghi dữ liệu cần `revision` hiện tại để tránh ghi đè phiên khác.
- Lưu nguyên tử vào `backend/data/workflows.json` (hoặc `WORKFLOW_STORE_PATH`). Bản triển khai được khôi phục khi backend khởi động lại. Dùng một backend worker sở hữu camera và engine; chưa hỗ trợ đồng bộ workflow giữa nhiều worker/process.
- Editor lấy trạng thái và danh sách robot từ FMS mỗi 2 giây, ngay cả khi chuyển tab. Chỉ báo LIVE khi MQTT đang kết nối và gói tin gần nhất chưa quá 10 giây; dữ liệu workflow quá 5 giây được đánh dấu cũ.

Kiểm thử phần workflow: `docker exec yolo_deepstream_backend sh -c 'cd /app && PYTHONPATH=/app:/app/tests python3 -m unittest discover -s tests -p "test_workflow*.py" -v'`.

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
Trên máy mới, tạo `web-dashboard/.env.local` theo `web-dashboard/.env.example` và điền `RSKYVIEW_SESSION_SECRET` riêng (tối thiểu 32 ký tự). Frontend production được build ở lần chạy đầu nếu chưa có image; không cài npm hay chạy hot-reload mỗi lần khởi động.

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

### Truy cập video qua IP LAN

- Mở `http://192.168.0.84:3000` (mạng Wi-Fi) hoặc `http://192.168.5.212:3000` (mạng LAN dây) trên máy trong cùng mạng; giao diện và camera dùng chung dữ liệu với `localhost:3000`.
- Dashboard chuyển tiếp WHEP qua `/api/stream/<camera-id>/whep` trên cổng 3000, không yêu cầu trình duyệt gọi trực tiếp cổng 8081. Có thể cấu hình `MEDIAMTX_WHEP_ORIGIN` trên frontend nếu gateway nằm ở máy khác.
- Video WebRTC vẫn truyền trực tiếp qua cổng **18189 UDP/TCP**; cho phép cổng này giữa máy xem và server. Metadata tracking dùng cổng **8000 TCP**. Chỉ forward cổng 3000 ra Internet không đủ cho video WebRTC.
- Khi đổi IP server, cập nhật `webrtcAdditionalHosts` trong `services/mediamtx/mediamtx.yml` và `allowedDevOrigins` trong `web-dashboard/next.config.ts` nếu chạy chế độ dev.
- Backend tự đồng bộ đường dẫn camera mỗi 10 giây để khôi phục stream sau khi MediaMTX khởi động lại. Player tự kết nối lại khi stream tạm ngắt.

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

## 7. Tối ưu tài nguyên

- **Calibration chỉ lấy một ảnh JPEG mới nhất** khi mở tab, đổi camera hoặc nhấn **Frame mới nhất**. Không mở WebRTC hay tự tải lại ảnh định kỳ. Backend ưu tiên dùng frame đã giải mã gần nhất (không quá 500 ms), không lấy mất frame của tracker; nếu chưa có thì chụp RTSP một lần và đóng decoder. Các yêu cầu trùng camera được gộp, tối đa hai lần chụp/encode đồng thời. Giữ nguyên độ phân giải và JPEG quality 85.
- **Tab không hiển thị ngừng tác vụ giao diện**: đóng WebRTC/WebSocket, hủy reconnect/timer/render loop, thu hồi texture/geometry/WebGL của Map 3D. Building, Calibration và Workflow giữ bản nháp trong phiên trình duyệt khi chuyển tab. Khi mở lại Monitor, video cần kết nối WebRTC lại; pipeline tracking backend không bị dừng.
- **Không đổi thuật toán tracking**: giữ model, target FPS, ngưỡng, ID, khả năng tìm lại đối tượng và nội suy khi tab đang mở. FMS vẫn được backend nhận liên tục, độc lập với tab và đăng nhập. Decoder template chỉ chạy khi camera có target đăng ký; chỉ giải phóng SAM2 khi không còn camera dùng nó.
- **Metadata nhẹ hơn**: không gửi vector Re-ID nội bộ trong WebSocket dashboard; server vẫn giữ vector cho định danh. Bbox, mask, pose, timestamp và FMS không bị giảm tần suất hay lược bỏ.
- **Chatbot RAG tải theo nhu cầu**: ChromaDB/PDF được khởi tạo khi hỏi chatbot lần đầu và xử lý ngoài event loop. Lần hỏi đầu có thể lâu hơn; không preload tài nguyên chatbot khi chỉ giám sát camera.
- **Frontend production** chạy image standalone, không chạy `next dev`/npm install trong lúc vận hành. `.env.local` chỉ nạp lúc chạy, không đưa secret vào image.

Sau khi sửa frontend, cập nhật riêng dashboard, không restart tracking/FMS:

```bash
./run_all.sh build-frontend
docker compose up -d --no-deps frontend
```

Thay đổi Python trong backend có hiệu lực sau `docker compose restart backend`; chọn thời điểm phù hợp vì restart làm gián đoạn tạm thời tracking/FMS và khởi tạo lại model. Khi phát triển giao diện, dùng `npm ci && npm run dev` trong `web-dashboard` (dừng frontend container trước nếu cùng dùng cổng 3000).
