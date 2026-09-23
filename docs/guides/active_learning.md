# Active Learning cho R-SkyView

## Phạm vi đang hỗ trợ

Luồng hiện tại là **YOLO detection TensorRT → NvDCF → SAM2 CUDA**. Module
Active Learning học lại **YOLO detector** từ các frame đã được người dùng duyệt,
không huấn luyện lại SAM2. Polygon chuẩn được giữ trên SSD; bbox huấn luyện
YOLO được tính từ polygon. Chỉ số kiểm định hiện tại là **bbox mAP50–95,
mAP50 và mAP từng lớp**, không phải mIoU segmentation. Không tự đổi model
detection sang YOLO segmentation vì parser/runtime hiện tại không hỗ trợ
output đó.

Gallery Label đa góc nhìn và triplet hiện có vẫn giữ nguyên. Không tự lấy các
ảnh Label chỉ gán một vật làm nhãn huấn luyện toàn frame, vì sẽ làm những vật
chưa gán bị hiểu nhầm là background. Lưu một correction không đổi weights hay
ghi đè ngay kết quả tracking trên Monitor.

## 1. Thu thập và sửa nhãn

1. Deploy model Ready lên camera như trước.
2. Trong **Monitor**, bấm **Đúng / Sửa nhãn** trên camera.
3. Bấm **Lấy frame GPU**. Ảnh được lấy từ DeepStream worker, không chụp canvas
   có overlay hoặc tin frame/embedding do trình duyệt gửi lên.
4. Kiểm tra mọi vật thuộc các lớp model: chọn lớp đúng, kéo đỉnh để sửa mask,
   vẽ polygon/thêm vùng, thêm vật bị bỏ sót hoặc loại vật nhận nhầm. Có zoom.
   Bbox nét đứt vàng chỉ là gợi ý, không được coi là mask ground truth.
5. Xác nhận đã kiểm tra **tất cả vật trong frame** rồi bấm **Đúng** hoặc
   **Lưu nhãn đã sửa**. **Loại frame** giữ trạng thái rejected và không đưa
   frame vào train. Frame background rỗng phải được xác nhận chủ động.

Mỗi ảnh gắn `camera_id`, `source_id`, `frame_id`, PTS, thời điểm lấy frame,
thế hệ worker, model/version và SHA-256. Chỉ gợi ý polygon SAM2 có cùng
`frame_id` với ảnh; không ghép mask cũ vào ảnh mới. Đây là đồng bộ ảnh
annotation với inference, **không** phải cam kết video WHEP trên trình duyệt
đã đồng bộ exposure/frame-ID với metadata live.

Xem, tiếp tục duyệt và xóa mẫu tại **Building → Active Learning → Dataset &
phản hồi**. Mẫu đã duyệt là bất biến; muốn thay phải xóa/lấy ảnh mới. Xóa mẫu
chặn promote/hủy job còn đang dùng mẫu đó; không tự đảo ngược ảnh hưởng của
một model đã deploy. Xóa không làm mất Label/gallery riêng của tính năng cũ.
Ảnh nháp hết hạn sau 1 giờ, tối đa 32 ảnh nháp toàn hệ thống. Mẫu đã duyệt
không tự hết hạn; dung lượng SSD vẫn là giới hạn thực tế.

## 2. Chuẩn bị checkpoint và dataset gốc

Trong **Weights & lịch học**, upload `best.pt` đáng tin cậy đã dùng để xuất
ONNX của model hiện tại. ONNX/engine không thay thế checkpoint huấn luyện.
`.pt` có thể chứa mã Python: chỉ upload file do chính bạn tạo hoặc đã kiểm tra,
không dùng file từ nguồn không tin cậy. API chỉ lưu theo chunk và checksum;
việc load checkpoint diễn ra trong tiến trình training riêng.

Đặt dataset YOLO detection gốc vào thư mục sau trên SSD máy chủ:

```text
backend/data/active_learning/<model-id>/base/
├── data.yaml
├── images/train/*.jpg
├── images/val/*.jpg
├── labels/train/*.txt
└── labels/val/*.txt
```

Ví dụ `data.yaml` (thứ tự `names` phải khớp chính xác model của bạn):

```yaml
path: .
train: images/train
val: images/val
names: [Rack, Robot_1, Robot_2001, Robot_28100, Robot_6868]
```

Mỗi dòng label là `class_id cx cy width height`, tọa độ chuẩn hóa 0–1.
Ảnh background cần file `.txt` rỗng; thiếu file label là lỗi. JPG/JPEG/PNG
được hỗ trợ. Cần ít nhất 2 ảnh train và 2 ảnh validation, tập validation có đủ
các lớp, không trùng nội dung với train/correction. Đây chỉ là kiểm tra tối
thiểu; hãy dùng validation đủ lớn, đa camera/góc nhìn và tách theo video/
phiên để hạn chế rò rỉ giữa các frame gần nhau.

YAML chỉ tham chiếu đường dẫn cục bộ bên trong `base/`, không có script
download. Dataset được đặt qua filesystem; chưa có uploader ZIP. UI hiển thị
đường dẫn container `/app/data/active_learning/<model-id>/base/`, tương ứng
với thư mục `backend/data/active_learning/<model-id>/base/` trên máy chủ.

## 3. Lịch học và ưu tiên realtime

- Mặc định **tắt** tự retrain và tự promote. Bật lịch sau khi chuẩn bị dữ liệu.
- Mặc định xếp hàng khi có 500 frame mới đã duyệt, hoặc sau 00:00 theo múi giờ
  `Asia/Bangkok` nếu có dữ liệu mới. Scheduler kiểm tra mỗi phút, không train
  lại liên tục cùng một batch dữ liệu sau khi một job lỗi.
- Có thể tạo/hủy job thủ công trong **Job & kiểm định**.
- Worker chỉ chạy train/export/build/validation khi **không có deployment
  Monitor/workflow** và GPU đủ trống. Không tự dừng camera/deployment để học.
  Nếu người dùng deploy live trong khi train, tiến trình train nhường GPU.
- Đóng tab Monitor không phải dừng deployment. `waiting_resources` là trạng
  thái có chủ đích, không phải treo. Muốn học liên tục mà không nhường GPU live
  cần GPU/máy training riêng; bản hiện tại chưa có remote training worker.
- `ACTIVE_LEARNING_MIN_GPU_FREE_MB` mặc định 6000 MiB là điều kiện tối thiểu,
  không bảo đảm đủ cho mọi model/batch. Không fallback training sang CPU.
- Trạng thái worker `running=true, state=idle` nghĩa là scheduler đang sống
  nhưng chưa có job; `job_active=true` mới là lúc đang train/build/promote.

Job trộn dataset gốc với correction, đóng băng manifest/checksum, fine-tune
checkpoint hiện tại với số epoch nhỏ và learning rate thấp. Sau đó export
ONNX input tĩnh batch 1, build TensorRT FP16 trong version riêng. Mẫu bị xóa,
baseline thay đổi, GPU bận, thiếu SSD hoặc job hủy đều chặn áp dụng model.

## 4. Kiểm định và thay engine

Baseline và candidate được kiểm định bằng **hai TensorRT engine thực tế**
trên cùng tập validation. Candidate chỉ thành Ready nếu mAP50–95 tăng hơn
ngưỡng cấu hình và không làm mAP50/mAP từng lớp giảm quá ngưỡng. Loss theo
epoch, metric và log được lưu theo job; job không đạt vẫn giữ kết quả để xem.

Khi bấm áp dụng, backend kiểm tra lại dataset, engine checksum và baseline;
cập nhật thuộc tính `model-engine-file` của `nvinfer`, chờ tín hiệu
`model-updated` thành công rồi mới lưu active version trong PostgreSQL.
Không chỉ đổi symlink rồi coi như runtime đã nạp. Kiến trúc, input/output,
kích thước ảnh và thứ tự lớp phải tương thích. Lỗi sẽ thử trở về engine cũ;
nếu runtime không xác nhận, supervisor khởi động lại worker theo version
đã lưu. Nếu model chưa chạy, version mới được dùng ở lần deploy tiếp theo.

Không cam kết zero-downtime hay 30 FPS trên phần cứng thực tế trước khi chạy
kiểm định với checkpoint/dataset thật. Retrain cũng không bảo đảm nhận diện
đúng tuyệt đối; metric detector tốt hơn không tự chứng minh mask SAM2 tốt hơn.

## Lưu trữ và API

PostgreSQL dùng schema riêng `active_learning` với bảng `samples`, `settings`,
`weights`, `jobs`; không xóa hoặc truncate schema/dữ liệu cũ.

```text
backend/data/active_learning/<model-id>/
├── images/<sample-id>.jpg
├── masks/<sample-id>.json
├── metadata/<sample-id>.json
├── metadata_log.csv
├── weights/<upload-id>.pt
├── base/
└── jobs/<job-id>/
    ├── dataset/{data.yaml,manifest.jsonl,images,labels,masks}
    ├── training/
    └── train.log

backend/models/uploads/<model-id>/versions/<job-id>/
└── best.pt, model.onnx, detector.engine, labels.txt, nvinfer.txt, contract.json
```

`metadata/*.json` là log chi tiết từng mẫu: đường dẫn ảnh/mask tương đối, nguồn,
frame/version, người duyệt, feedback và hash. `metadata_log.csv` là log append-only
nhẹ cho collector, gồm sự kiện capture/review, camera/source/frame, số prediction,
đường dẫn ảnh/mask và hash nhãn. `manifest.jsonl` là log dataset
đóng băng của mỗi job. `ACTIVE_LEARNING_DIR` có thể đổi gốc lưu trữ nhưng
phải trỏ tới SSD đã mount bền vững. Các file dữ liệu mới được Git bỏ qua.
Backup cả PostgreSQL và các thư mục SSD để phục hồi đủ dữ liệu.

Mọi API sau yêu cầu phiên dashboard, dưới prefix
`/api/active-learning/models/{model_id}`:

| Method / suffix | Chức năng |
|---|---|
| `GET /` | Cấu hình, prerequisites, weights, jobs, trạng thái worker |
| `PUT /settings` | Lưu cấu hình backend kiểm tra |
| `POST /snapshots` | Lấy đúng frame từ worker camera |
| `GET /samples`, `POST /samples/{id}/review` | Duyệt dataset, lưu ground truth |
| `GET /samples/{id}/image`, `DELETE /samples/{id}` | Xem/xóa dữ liệu mẫu |
| `POST /weights`, `PUT /weights/{id}/file`, `POST /weights/{id}/complete` | Upload checkpoint, chunk 2 MiB, tối đa 512 MiB |
| `POST /jobs`, `POST /jobs/{id}/cancel`, `POST /jobs/{id}/promote` | Điều khiển job |
| `GET /jobs/{id}/log` | Đọc phần cuối log, không tải log không giới hạn |

## Kiểm tra kỹ thuật

`tests/test_active_learning.py` kiểm tra annotation, auth, chống giả frame,
chunk upload, chất lượng, ưu tiên GPU live và rollback. Integration test tạo
database tạm `al_test_<uuid>` rồi xóa database tạm, không truncate database
sản xuất. Chạy trong backend container với `CUDA_VISIBLE_DEVICES=`;
`ACTIVE_LEARNING_TEST_DSN` cần tài khoản có quyền tạo database tạm.

Chưa có checkpoint YOLO `.pt` và dataset gốc tương ứng thì chỉ sử dụng luồng
thu thập/duyệt dữ liệu; không tự tải model khác hoặc giả lập một lần retrain
thành công.
