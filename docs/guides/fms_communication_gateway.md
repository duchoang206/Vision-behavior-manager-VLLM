# Cổng truyền thông FMS WCS & điều khiển thiết bị ngoại vi (System Config)

Tab **System Config** quản lý toàn bộ kênh truyền thông của Vision Manager: gửi trạng thái ô hàng cho FMS WCS,
đẩy sự kiện sang WMS/ERP và điều khiển thiết bị ngoại vi (PLC, đèn tháp, cửa, thang máy…). Mọi cấu hình lưu
PostgreSQL (`comm_channels`) và áp dụng ngay khi bấm **Lưu**, không cần khởi động lại backend.

## 1. Giao thức FMS WCS (thiết bị `CAMERA_AI`)

Phân tích binary `rtcserver_wcs` (`WsCameraClient`) cho thấy FMS WCS là **WebSocket client**:

| FMS → Call Box → Add | Giá trị |
| :--- | :--- |
| Device Type | `CAMERA_AI` |
| Protocol | `WebSocket` (không dùng `TCP/IP`) |
| Mode | `client` |
| Device Ip / Device Port | IP máy Vision Manager / cổng của kênh (mặc định `8000`) |
| Device Configuration | `{"path": "/ws/wcs_camera"}` (FMS mặc định `/` nếu bỏ trống) |

FMS tự kết nối lại mỗi 3 giây. Với mỗi gói nhận được, FMS:
1. lưu nguyên gói vào Redis `CameraStateWCS` (dùng để chặn giao hàng vào ô đầy);
2. duyệt mảng `slots`, đọc `slot_id` bằng `std::stoi` (**Slot ID phải là số nguyên**);
3. coi `state == "Car Full"` là có hàng, mọi giá trị khác là trống; chỉ phản ứng khi trạng thái ô thay đổi
   (kích hoạt task `CALL_CARD` theo **Config** của thiết bị CAMERA_AI cho từng Slot ID).

Vì FMS ghi đè Redis bằng gói mới nhất, **mọi gói đều chứa snapshot đầy đủ**:

```json
{"slot_id": "5", "state": "Car Full", "slots": [{"slot_id": "2", "state": "Empty"}, {"slot_id": "5", "state": "Car Full"}]}
```

- `slot_id`/`state` ở cấp ngoài: ô vừa đổi trạng thái (để theo dõi bằng `wscat`).
- Khi client vừa kết nối: gửi ngay snapshot `{"slots": [...]}` (resync).
- Heartbeat: gửi lại snapshot theo chu kỳ cấu hình (mặc định 2 giây).
- Ô chưa được AI xác nhận trạng thái (vừa khởi động) không được đưa vào snapshot.
- Một Slot ID được nhiều camera theo dõi: `Car Full` nếu bất kỳ camera nào thấy có hàng.

Kênh mặc định `fms_wcs` được tạo tự động lần đầu khởi động: WebSocket server dùng chung cổng API 8000 tại
`/ws/wcs_camera`. Kênh WebSocket server đặt ở cổng khác sẽ mở listener riêng và chấp nhận mọi path.

## 2. Gán Slot ID cho ô hàng

Building → Phân tích Hành vi → loại **Ô chứa hàng** → nhập **Mã ô FMS (Slot ID)**, chọn **Kênh truyền thông**
(mặc định: tất cả kênh đăng ký sự kiện) và **Bật đồng bộ FMS**. Bộ lọc AI giữ nguyên: 2 frame liên tiếp → `Car Full`,
5 frame trống liên tiếp → `Empty`; trạng thái giữ nguyên thì không gửi lặp.

## 3. Kênh khác & thiết bị ngoại vi

| Giao thức | Chế độ | Ghi chú |
| :--- | :--- | :--- |
| WebSocket | Server / Client | Client tự kết nối lại (backoff 1→30 s), hỗ trợ Bearer token, TLS |
| TCP/IP | Server / Client | Ký tự kết thúc gói `\n`/`\r\n`/`\r`/không; payload text, JSON hoặc HEX |
| MQTT | Client | Publish lên Topic, QoS 0–2, retain, user/pass |
| HTTP Webhook | Client | POST/PUT/PATCH JSON, timeout cấu hình (mặc định 3 s) |
| Modbus TCP | Client | FC05 ghi coil, FC06 ghi thanh ghi, FC01/FC03 đọc; poll định kỳ |

- **Payload template**: chèn biến `{slot_id}`, `{state}`, `{is_occupied}`, `{camera_name}`, `{occupant_label}`,
  `{timestamp_iso}`… Biến trong chuỗi JSON được escape; biến ngoài chuỗi xuất kiểu JSON gốc (`true`, số).
  Live Preview cập nhật khi gõ; template sai cú pháp bị từ chối khi lưu, backend không bị ảnh hưởng.
- **Lệnh điều khiển**: định nghĩa lệnh (gửi payload hoặc ghi coil/thanh ghi), chạy tay ở nút **Lệnh** hoặc gán tự
  động theo sự kiện `SLOT_CARFULL`, `SLOT_EMPTY`, `ROI_ALERT` (xâm nhập / lảng vãng / mật độ).
- **Gửi thử nghiệm**: gửi một gói mẫu ngay, báo thành công (số client nhận, HTTP status) hoặc lỗi cụ thể
  (`Connection Refused`, `Connection Timeout`, `Host Unreachable`, Modbus exception). Với kênh FMS, gói thử được FMS
  xử lý như dữ liệu thật; heartbeat kế tiếp khôi phục trạng thái thực.

## 4. Hiệu năng & an toàn

- Luồng AI chỉ chuyển sự kiện vào event loop bằng `call_soon_threadsafe` (dưới 1 ms); mỗi kênh có hàng đợi
  riêng (1000 gói, bỏ gói cũ nhất khi đầy) và mỗi client WebSocket/TCP có hàng đợi riêng, nên thiết bị chậm hoặc
  mất mạng không làm chậm video.
- API `/api/comm/*` yêu cầu phiên đăng nhập dashboard (có thể điều khiển PLC). WebSocket cho FMS không cần đăng nhập.

## 5. API

| Phương thức | Đường dẫn | Mô tả |
| :--- | :--- | :--- |
| GET/POST | `/api/comm/channels` | Danh sách / tạo kênh |
| GET/PUT/DELETE | `/api/comm/channels/{id}` | Xem / sửa (áp dụng ngay) / xóa |
| POST | `/api/comm/channels/{id}/restart` | Khởi động lại driver |
| POST | `/api/comm/test_dispatch` | `{"channel_id","slot_id","state","command_id?","channel?"}` gửi thử (cả bản nháp chưa lưu) |
| POST | `/api/comm/preview` | Render template với dữ liệu mẫu |
| POST | `/api/comm/channels/{id}/commands/{cmd}` | Chạy lệnh thiết bị |
| POST | `/api/comm/channels/{id}/modbus` | Đọc/ghi Modbus trực tiếp |
| GET | `/api/comm/status`, `/api/comm/slots`, `/api/comm/logs`, `/api/comm/meta` | Trạng thái, ô hàng, nhật ký, danh mục |
| WS | `/ws/wcs_camera`, `/ws/comm/{channel_id}` | Endpoint cho FMS WCS / kênh WebSocket server cổng 8000 |

## 6. Kiểm thử

```bash
cd backend && python3 -m pytest tests/test_comm_gateway.py -q
```

`test_comm_gateway.py` giả lập FMS đúng cách `WsCameraClient` đọc gói và kiểm tra: resync khi kết nối, độ trễ
transition < 50 ms, không gửi lặp, template động (boolean gốc), lỗi Send Test cụ thể, luồng 25 FPS không bị chặn
khi webhook treo, điều khiển Modbus theo sự kiện. Đặt `COMM_TEST_DB_URL` tới một PostgreSQL dùng-một-lần để chạy
thêm kiểm thử lưu trữ qua khởi động lại.
