# Workflow Editor

Workflow Editor triển khai các quy tắc Vision trên metadata tracking đang có. Module không thay model, mask SAM2, calibration, luồng Monitor hoặc cấu hình FMS; không mở thêm luồng giải mã video cho mỗi workflow.

## Hai tab

- **Pipeline đã deploy**: danh sách bản nháp và deployment, revision, camera, pause/resume/stop, xóa pipeline đã dừng, kết quả từng khối và nhật ký có phân trang.
- **Chọn cách deploy**: chọn mẫu, chọn camera, thêm/kéo/nối/xóa khối, cấu hình từng khối, kiểm tra, lưu nháp, deploy chạy ngay hoặc deploy tạm dừng trên máy biên hiện tại.

Chọn khối nguồn trên graph, bấm **Nối khối**, rồi bấm khối đích. Có thể tách nhiều nhánh; nhiều đầu vào được hợp theo OR và khử trùng ID trong cùng camera, không phải điều kiện AND. Xóa kết nối ở bảng cấu hình. ROI và vạch được chấm trên một ảnh tĩnh camera bằng **Lấy 1 frame mới**; không phát video trong editor. Xóa hình mẫu trước khi chấm vùng thật.

Những thay đổi editor được lưu vào backend khi bấm **Lưu nháp** hoặc **Deploy pipeline**. Deploy luôn validate và lưu các thay đổi hiện tại trước, không lặng lẽ dùng bản nháp cũ. Bản deploy có snapshot riêng; sửa nháp không đổi bản đang chạy. Dừng deployment hiện tại trước khi deploy revision mới. Khi backend khởi động lại, deployment đang running sẽ tự phục hồi; paused/stopped không tự chạy.

## Khối đang thực thi

| Khối | Xử lý |
|---|---|
| Camera source | Nhận metadata của 1–32 camera đã đăng ký; camera mới được liệt kê nhưng cần chủ động chọn vào workflow |
| Object Detector / Classifier | Lọc lớp/confidence của metadata model hiện tại, không nạp model phân loại mới |
| Object Matcher | Lọc nhãn định danh đã đăng ký; không huấn luyện lại ReID |
| ROI Filter | Kiểm tra điểm chân bbox thuộc polygon chuẩn hóa 0..1 |
| Line Crossing | Kiểm tra giao đoạn hữu hạn, hướng theo tích có hướng, hysteresis chống rung, tổng lần qua vạch trong phiên runtime |
| Dwell Time | Thời gian liên tục; reset khi đối tượng mất khỏi nhánh hoặc metadata gián đoạn |
| Object Counter | Số lượng hiện tại với `gte/lte/eq`, có hỗ trợ không có hàng (=0) trên metadata hợp lệ |
| Safety Distance | Khoảng cách người–robot theo tọa độ Vision chiếu về mặt phẳng FMS, chỉ bên trong vùng đã hiệu chuẩn |
| Display Output | Hiển thị số lượng/trạng thái từng khối ở chi tiết deployment; không đổi mask trên Monitor |
| Event / MP4 | Lưu sự kiện vào Analytics; chứng cứ là đoạn MP4 liên tục do FFmpeg ghi và liên kết khi đoạn hoàn tất |
| Alert / Webhook | Gửi JSON tới connector backend cho phép, có nhật ký và retry |

ROI/Line chỉ deploy cho **một camera** vì mỗi camera có hình học khác nhau. Tác vụ trên nhiều camera xử lý độc lập theo từng camera; không dùng workflow để hợp nhất vị trí nhiều camera lần nữa.

**Không giả lập tính năng chưa có:** PPE/té ngã, anomaly, attendance/5S và SMTP hiện bị khóa vì chưa có model/adapter nghiệp vụ tương ứng được tích hợp. Classifier không thể nhận diện lớp mới mà upstream chưa phát hiện. Không tự bật pose đang tắt. Các mẫu an toàn chỉ là cảnh báo hỗ trợ vận hành, không thay thế hệ thống an toàn được chứng nhận. Không có remote cluster hay cam kết 1000 camera/latency <30ms chỉ từ module này.

## Backend và lưu trữ

- `backend/core/workflow_definition.py`: schema, thư viện khối, validation DAG/camera/nhãn/hình học/điều kiện triển khai.
- `backend/core/workflow_runtime.py`: graph evaluator, latest-only mailbox theo camera, worker metadata, worker lưu sự kiện và worker connector độc lập.
- `backend/core/workflow_store.py`: PostgreSQL, transaction, revision conflict và deployment snapshot.
- `backend/routers/workflows.py`: API có xác thực session đăng nhập.
- `web-dashboard/components/views/WorkflowView.tsx`: hai tab quản lý và editor.
- `web-dashboard/components/views/WorkflowGeometry.tsx`: chấm hình học trên ảnh tĩnh.

PostgreSQL dùng schema **workflow**, tách thành `pipelines`, `deployments`, `logs`, `deliveries`. MP4 nằm tại thư mục SSD do `RECORDINGS_HOST_DIR` mount vào `/var/vms/recordings`, metadata ở `archive.recordings`, không nhét video vào PostgreSQL. Retention MP4 vẫn do FFmpeg recorder quản lý (mặc định 48h).

Frontend không tự thực thi workflow. Runtime xử lý metadata không phụ thuộc người dùng có mở tab hay không. Không có pipeline thì không copy đối tượng; có pipeline thì chỉ copy trường metadata cần thiết, không copy mask/ảnh/vector. Queue chờ theo camera chỉ giữ frame mới nhất, tác vụ lưu DB/webhook chạy worker riêng, có giới hạn hàng đợi và số sự kiện bỏ qua được hiển thị. Không bảo đảm xử lý đủ từng frame nếu máy bị quá tải; ưu tiên không tích trễ. Không biến metadata mất kết nối thành "không có hàng". Trạng thái chờ metadata hiển thị riêng, không giữ kết quả live cũ vô hạn.

Giới hạn hiện tại: 64 khối, 128 cạnh, 32 camera nguồn/pipeline, tối đa 32 pipeline hoạt động trên máy biên. Bộ đếm frame/crossing/dwell trong RAM reset sau restart hoặc resume; lịch sử sự kiện và deployment giữ trong DB. UI tải lại kết quả mỗi 5 giây, không phải FPS inference. API không đo/cam kết latency camera end-to-end.

## API

Tất cả đường dẫn bắt đầu bằng `/api/workflows`, yêu cầu cookie phiên đăng nhập `rskyview_session`. Frontend gọi cùng origin qua `/api/backend/workflows`. Backend và frontend cần cùng `RSKYVIEW_SESSION_SECRET`; compose chia sẻ `.env.local`.

| Method | Path | Ý nghĩa |
|---|---|---|
| GET | `/catalog` | Khối, camera, nhãn, connector khả dụng |
| GET | `` | Bản nháp, deployment và trạng thái runtime |
| POST | `/validate` | Kiểm tra definition, không chạy tác vụ |
| POST | `` | Tạo nháp `{definition}` |
| PUT | `/{pipeline_id}` | Lưu `{definition, revision}` |
| DELETE | `/{pipeline_id}?revision=N` | Xóa pipeline đã dừng, giữ nhật ký |
| POST | `/{pipeline_id}/deploy` | `{revision, start_paused}` |
| POST | `/deployments/{deployment_id}/action` | `{action: "pause" | "resume" | "stop"}` |
| GET | `/deployments/{deployment_id}/logs?before=N` | 100 log trước ID N và kết quả runtime |

Revision sai, deploy trùng, camera/nhãn đã xóa, hình học lỗi, graph có chu trình, model/connector thiếu sẽ bị backend từ chối. Lệnh trả thành công khi thay đổi control plane đã được commit; worker áp dụng ngay sau đó, không quá lần refresh bình thường (2s nếu không lỗi DB). Pause/stop không chặn camera hoặc recorder dùng chung.

## Connector FMS / WMS / ERP

Quản trị viên khai báo `WORKFLOW_CONNECTORS_JSON` trong môi trường **backend**, không nhập URL/token tùy ý từ frontend:

```json
{"fms_events":{"url":"https://fms-gateway.example/vision/events","token":"replace-with-server-secret"}}
```

Endpoint là adapter nhận sự kiện Vision, **không** phải API điều khiển robot tự động. FMS/WMS/ERP thực tế có contract riêng cần adapter chuyển đổi; module không tự phát lệnh chạy/dừng robot. `require_fms` yêu cầu đối tượng có tọa độ Vision chuyển về FMS hợp lệ, không khẳng định robot telemetry còn online.

Webhook gửi event ID, deployment/revision/camera/node, thời điểm, số lượng, danh sách đối tượng và vị trí FMS nếu có. `Idempotency-Key` giữ nguyên khi retry; bên nhận cần khử trùng. Timeout kết nối/đọc 2s/3s, không follow redirect, không dùng proxy environment, retry tối đa 5 lần. Sự kiện quá 60 giây không gửi để tránh đưa vị trí cũ vào hệ thống bên ngoài. Pause/stop hủy các lần gửi đang chờ, không phát lại vị trí cũ khi resume. Request đã gửi ra mạng trước khi pause/stop có thể hoàn tất và không thể thu hồi.

## Kiểm thử

```bash
docker exec yolo_deepstream_backend python3 -m unittest discover -s tests -p test_workflows.py -v
docker compose build frontend
```

Ba kiểm thử PostgreSQL chỉ chạy khi có `WORKFLOW_TEST_DATABASE_URL` trỏ tới **DB test độc lập** có tên chứa `workflow_test_`; không dùng database vận hành.
