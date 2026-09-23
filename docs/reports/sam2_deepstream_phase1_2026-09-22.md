# SAM2 / DeepStream Phase 1 — 2026-09-22

## Đã triển khai

- Bỏ override ép `NvDCF enableReAssoc: 0`; các worker mới dùng lại cấu hình
  `enableReAssoc: 1` và các ngưỡng Re-ID hiện có.
- Thêm đường native DeepStream `MaskTracker` dùng TensorRT SAM2 memory tracker
  của NVIDIA dưới cờ `DEEPSTREAM_NATIVE_MASKTRACKER`.
- Tự sinh config native từ các engine/ONNX đã có tại
  `/app/models/sam2_tracker`, không rebuild engine khi khởi động.
- Đọc `NvDsObjectMeta.mask_params`, chuyển mask về polygon tương đối bbox để
  giữ nguyên WebSocket contract của Monitor.
- Mask native chỉ là nguồn hình học fallback; khi Label/triplet đã xác thực,
  mask user/SAM2 hiện có vẫn được ưu tiên. Vì vậy không làm mất logic Label.
- Giảm cửa sổ chờ mailbox metadata từ 100 ms xuống 5 ms; không thay đổi thứ tự
  frame và vẫn loại payload quá cũ.

## Cách bật thử nghiệm

Mặc định vẫn giữ `DEEPSTREAM_NATIVE_MASKTRACKER=0` để không làm gián đoạn
deployment hiện tại. Sau khi dừng deployment, bật biến này trong service
backend rồi khởi động lại container. Native tracker sẽ dùng:

```text
/app/models/sam2_tracker/config_tracker_MaskTracker.yml
/app/models/sam2_tracker/config_tracker_module_Segmenter.yml
```

Nếu native tracker không khởi tạo được, tắt biến về `0` để quay lại Python
SAM2; không xóa gallery, calibration hay engine hiện tại.

## Giới hạn đã biết

Đây là bước chuyển tiếp an toàn, chưa phải hoàn tất Giai đoạn 3 của plan:
encoder/decoder native đã chạy trong `nvtracker`, nhưng identity Label vẫn đi
qua lớp xác thực hiện hữu để giữ đúng yêu cầu “user label là authority”. Việc
đưa descriptor/triplet vào custom `nvinferserver` cần tensor contract và probe
metadata cụ thể, không được thay bằng skeleton chưa build.

## Kiểm tra

- Python compile các module DeepStream/SAM2: đạt.
- Sinh config native trong container: đạt.
- Test polygon `NvDsObjectMeta.mask_params`: đạt.
- `native-build.log` xác nhận NVIDIA `NvMultiObjectTracker` đã build engine
  SAM2 từ các ONNX project-local trước đó.

## Smoke test native thực tế

Ngày 22/09/2026, chạy `deepstream-app` với file video mẫu và cấu hình native
project-local. Kết quả:

- `image_encoder.engine`: Loaded Complete.
- `mask_decoder.engine`: Loaded Complete.
- `memory_attention.engine`: Loaded Complete.
- `memory_encoder.engine`: Loaded Complete.
- `NvMultiObjectTracker`: Initialized → pipeline running → De-initialized sạch.
- Throughput quan sát được: khoảng 81–83 FPS trên video mẫu.
- Không có lỗi TensorRT/SAM2; cảnh báo audio của file mẫu không ảnh hưởng video.

Smoke test này xác nhận engine và memory pipeline native khởi tạo được. Việc
đối chiếu polygon native với từng Label trong camera thật vẫn nên thực hiện
sau khi bật cờ opt-in trên một deployment thử nghiệm, vì worker hiện vẫn giữ
Python Label verifier để không làm thay đổi logic identity đã ổn định.
