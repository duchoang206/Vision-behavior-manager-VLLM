# Model TensorRT và Calibration

## Upload model

Vào **Building → Model / TensorRT**, chọn file `best.onnx`, nhập tên và chỉ nhập
labels dự phòng khi ONNX không có metadata `names`. Frontend upload theo từng
chunk 2 MiB; backend giữ tiến độ trong PostgreSQL nên không cần gửi cả file vào
một request.

Backend hiện nhận hợp đồng an toàn cho model phát hiện **YOLOv8/YOLO11 raw
detection**:

- một input FP32 tĩnh `[1,3,H,W]`, `H/W` từ 128 đến 1920 và chia hết cho 32;
- một output FP32 dạng `[1,4+num_classes,anchors]`, không kèm NMS;
- từ 1 đến 100 class, ID class liên tục bắt đầu từ 0;
- file ONNX tự chứa, không nhận external data, pose, segmentation hoặc
  classification trong luồng custom detector hiện tại.

Model hợp lệ chuyển qua `validating → building → ready`. Build chạy
`trtexec` FP16 trên GPU, sau đó tạo engine và cấu hình `nvinfer`. File model,
engine và log nằm trong `backend/models/uploads/<model-id>/`; metadata, trạng
thái, labels và lỗi nằm trong bảng `vision.models` của PostgreSQL.

Khi model ở trạng thái **Ready**, vào **Workflow Editor → Chọn cách deploy**,
thêm `Object Detector`, chọn model và các class cần giữ lại rồi deploy. Worker
custom dùng DeepStream `nvurisrcbin → nvstreammux → nvinfer → NvDCF`, chỉ giữ
frame mới nhất để không tích trễ. Nhiều pipeline dùng cùng một model sẽ dùng
chung worker; không deploy đồng thời hai model custom khác nhau trên GPU.

Model custom sinh detection/tracking mới và không tự biến mọi class thành mask
SAM2. Mask của vật đã đăng ký vẫn đi theo luồng SAM2/identity hiện tại; không
gán `Robot_2001` chỉ vì một detection generic trùng vị trí.

## Calibration

Trong **Calibration → Camera ↔ FMS**, có thể:

1. click điểm trên snapshot camera rồi nhập cặp tọa độ `X/Y` FMS;
2. nhập trực tiếp cả `u/v` pixel và `X/Y` FMS;
3. bật thước đo, click hai đầu đoạn trên ảnh rồi nhập chiều dài thực tế mét.

Các điểm được lưu theo tọa độ ảnh chuẩn hóa và tọa độ mặt sàn FMS. Đo chiều dài
chỉ là ràng buộc tinh chỉnh; camera vẫn cần ít nhất 4 cặp điểm neo hợp lệ để
xác định gốc, hướng và phối cảnh. Khi bấm **Lưu**, backend kiểm tra sai số,
ghi cấu hình vào PostgreSQL và kích hoạt cấu hình đó làm calibration chính của
camera. Từ đó detect, tracking, workflow proximity và Map 3D dùng cùng phép
đổi tọa độ; vị trí robot vẫn được cập nhật realtime từ FMS.

## Xác nhận vận hành

- `GET /api/models` hiển thị trạng thái build và tuổi metadata từng camera.
- Runtime custom chỉ ở trạng thái `live` khi frame mới thực sự về; lỗi nguồn
  hoặc quá hạn sẽ được báo, không giữ vị trí cũ vô hạn.
- Pipeline người dùng đang chạy không bị thay đổi khi chỉ upload hoặc build
  model. Chỉ khi chọn model Ready trong Workflow và deploy thì custom detector
  mới được khởi động.
