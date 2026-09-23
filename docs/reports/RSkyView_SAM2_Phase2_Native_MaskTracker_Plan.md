# SAM2 / DEEPSTREAM — PHASE 2: CHUẨN HÓA NATIVE MASKTRACKER
## Dự án: R-SkyView — AGV Team (RTC Technology)
### Kế thừa: Phase 0 (đo baseline) + Phase 1 (native MaskTracker opt-in, báo cáo 2026-09-22)

---

## 0. BỐI CẢNH — VÌ SAO CÓ FILE NÀY

Phase 1 đã phát hiện một điều quan trọng làm thay đổi hướng kỹ thuật so với
kế hoạch gốc: DeepStream có sẵn tracker tham chiếu tên **`MaskTracker`**,
nằm cùng nhóm với IOU, NvSORT, NvDCF, NvDeepSORT trong thư viện
`NvMultiObjectTracker` — tức đây là **giải pháp chính thức của NVIDIA**,
không phải hướng "tự viết `IInferCustomProcessor` cho SAM2" như file kế
hoạch Phase 0 đề xuất ban đầu. Tài liệu chính thức xác nhận:

> "*...including the reference implementations provided by the
> NvMultiObjectTracker library: IOU, NvSORT, NvDCF, **MaskTracker** and
> NvDeepSORT trackers.*"
> — Gst-nvtracker, NVIDIA DeepStream Developer Guide

**Quyết định kiến trúc của Phase 2:** ưu tiên tuyệt đối cho việc chuẩn hóa
và kiểm chứng **native `MaskTracker`** thay vì tiếp tục đầu tư vào việc tự
viết `IInferCustomProcessor` từ đầu. Hướng `nvinferserver` custom processor
(Giai đoạn 3 trong file Phase 0) được **hạ xuống thành phương án dự phòng**,
chỉ dùng nếu native `MaskTracker` không đáp ứng được về độ chính xác mask
hoặc không đủ linh hoạt cho yêu cầu nghiệp vụ (Label/triplet identity) của
hệ thống.

**Việc CHƯA làm xong ở Phase 1 (bắt buộc phải hoàn tất trước khi bật mặc
định), theo đúng cổng quyết định đã đặt ra trước đó:**

1. Chưa đối chiếu tần suất frame rớt do queue "latest-only" với thời điểm
   khựng ~200ms quan sát được — đây là việc bắt buộc phải có KẾT QUẢ, không
   phải chỉ "đã lên kế hoạch làm".
2. Giảm mailbox timeout 100ms→5ms là một thay đổi hành vi, chưa được đo
   tách biệt trước/sau, và chưa xác nhận có dùng chung cơ chế queue với
   nghi phạm ở mục (1) hay không.
3. Smoke test native `MaskTracker` mới chạy 1 video mẫu đơn luồng
   (81–83 FPS), CHƯA benchmark trên tải thật (6 camera đồng thời).
4. Polygon native CHƯA được đối chiếu định lượng (IoU) với Label đã xác
   thực trên dữ liệu thật.
5. 3 nghi phạm cũ của Phase 0 (GIL, copy có vòng qua host, tranh chấp CUDA
   stream) vẫn chưa có kết luận cuối cùng, dù đã hạ độ ưu tiên xuống sau
   nghi phạm queue.

Phase 2 tổ chức lại toàn bộ các việc còn thiếu này thành một kế hoạch có
thứ tự, có cổng quyết định, và có tiêu chí nghiệm thu rõ ràng — không cho
phép nhảy sang "bật mặc định" khi thiếu bất kỳ số liệu nào ở trên.

---

## 1. KIẾN TRÚC ĐÍCH CỦA PHASE 2

```
Camera (RTSP/GigE/USB)
   → nvurisrcbin → nvstreammux
   → nvinfer [YOLO/Pose, TensorRT — PHẢI ĐỨNG TRƯỚC tracker để sinh bbox,
              đây là điểm agent đã tự sửa đúng trong Phase 1, giữ nguyên]
   → nvtracker [ll-lib chọn NvMultiObjectTracker, cấu hình low-level
                dùng MaskTracker (thay vì/kèm NvDCF), enableReAssoc=1]
        - Image Encoder (TensorRT engine, .engine đã build sẵn)
        - Mask Decoder (TensorRT engine)
        - Memory Attention (TensorRT engine)
        - Memory Encoder (TensorRT engine)
        → tất cả chạy NATIVE bên trong low-level tracker library,
          không qua Python, không qua thread side-car, không copy
          GPU thủ công giữa các tiến trình/thread khác nhau
   → NvDsObjectMeta.mask_params (polygon mask gắn thẳng vào object metadata
     chuẩn của DeepStream, không cần lớp chuyển đổi contract thủ công)
   → [Lớp xác thực Label/triplet hiện tại — GIỮ NGUYÊN, vẫn là authority
      cho identity, không đổi logic nghiệp vụ]
   → nvdsosd / metadata broadcaster → Monitor (WebSocket contract giữ nguyên)
```

Đây MỚI là "chuẩn pipeline DeepStream" đúng nghĩa cho phần segmentation:
toàn bộ 4 engine SAM2 chạy như một phần của `nvtracker`, cùng 1 CUDA
context/stream, cùng batch xử lý nhiều stream như NvDCF hiện tại — không
còn khái niệm "worker Python side-car" nữa, dù là dạng tiến trình riêng
hay thread riêng.

---

## 2. GIAI ĐOẠN 2A — HOÀN TẤT ĐO LƯỜNG CÒN THIẾU CỦA PHASE 0/1 (3–5 ngày)

**Không được bỏ qua bước này để "cho nhanh".** Đây là điều kiện bắt buộc
của cổng quyết định đã đặt ra — nếu không có số liệu, không biết native
MaskTracker có thực sự giải quyết đúng nguyên nhân hay không, hay chỉ tình
cờ "chạy mượt hơn" vì lý do khác (ví dụ chỉ vì đổi mailbox timeout).

### 2A.1. Đối chiếu queue latest-only với thời điểm khựng hình
- Dùng lại trace đã thu ở Phase 0 (13.610 frame tracker / 7.821 object,
  cửa sổ đo 90–100s) — không thu lại từ đầu nếu dữ liệu cũ còn dùng được.
- Tính tương quan thời gian giữa (a) các event frame bị rớt do queue
  latest-only và (b) các lần khựng ~200ms đã ghi nhận qua log GPU/CPU.
- Xuất kết quả dạng số cụ thể: % số lần khựng trùng với frame rớt, không
  chỉ kết luận định tính "có vẻ liên quan".

### 2A.2. Đo tách biệt tác động của thay đổi mailbox timeout 100ms→5ms
- Dựng lại 2 kịch bản A/B: timeout=100ms vs timeout=5ms, cùng workload,
  cùng 6 camera, đo tỷ lệ rớt frame + độ dài khựng hình quan sát được ở
  mỗi kịch bản.
- Xác nhận rõ: mailbox này có phải cùng cơ chế với "queue latest-only" ở
  mục 2A.1 hay là 2 hàng đợi khác nhau trong pipeline.

### 2A.3. Kết luận nghi phạm chính thức
- Dựa trên 2A.1 và 2A.2, xếp hạng lại 4 nghi phạm (queue latest-only, GIL,
  copy-host, tranh chấp CUDA stream) theo mức độ đóng góp thực tế vào hiện
  tượng khựng 200ms — có số liệu kèm theo, không suy đoán.
- Đây là căn cứ để quyết định: nếu native MaskTracker giải quyết được,
  liệu có phải vì nó loại bỏ đúng nghi phạm chính hay chỉ là hiệu ứng phụ.

---

## 3. GIAI ĐOẠN 2B — CHUẨN HÓA CẤU HÌNH NATIVE MASKTRACKER (1–2 tuần)

### 3.1. Chuyển từ "tự sinh config từ engine có sẵn" sang cấu hình chuẩn theo tài liệu

Phase 1 đang tự sinh config native tại `deepstream_masktracker.py` dựa
trên engine/ONNX đã có sẵn (`/app/models/sam2_tracker`). Cần đối chiếu lại
với đúng schema cấu hình chính thức của `nvtracker` cho low-level library,
đảm bảo không có tham số bị bỏ sót hoặc gán sai do suy luận thủ công từ
engine có sẵn thay vì đọc đúng tài liệu.

- Đọc kỹ mục cấu hình `[tracker]` trong Gst-nvtracker: `ll-lib-file`,
  `ll-config-file`, `tracker-width`, `tracker-height`.
- Đối chiếu `config_tracker_MaskTracker.yml` và
  `config_tracker_module_Segmenter.yml` hiện có với đúng format tham chiếu
  của NVIDIA (không phải chỉ dựa vào việc "engine load thành công" để coi
  là cấu hình đúng — load thành công không đồng nghĩa tham số tối ưu).

### 3.2. Đảm bảo `enableReAssoc=1` áp dụng nhất quán

Phase 1 đã sửa `custom_detector.py:208`, nhưng cần rà soát toàn bộ
codebase (bao gồm cả nhánh native mới thêm) để đảm bảo không còn nơi nào
khác override `enableReAssoc=0`, đặc biệt trong config native mới sinh ra
— tránh lặp lại đúng loại lỗi "override ẩn" mà Phase 1 vừa phát hiện.

### 3.3. Đối chiếu `NvDsObjectMeta.mask_params` với contract Monitor hiện tại

- Đảm bảo việc chuyển mask native → polygon tương đối bbox
  (`deepstream_native_mask.py`) không làm mất độ chính xác biên dạng so
  với mask gốc dạng raw mask từ decoder — so sánh trực quan + IoU giữa
  polygon sau chuyển đổi và mask gốc trước khi bàn tới việc so với Label.

### Tài liệu tham khảo
- Gst-nvtracker (tổng quan, liệt kê MaskTracker là tracker tham chiếu
  chính thức trong NvMultiObjectTracker): `https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvtracker.html`
- Gst-nvtracker – Tracker Accuracy Tuning: `https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvtracker.html#tracker-accuracy-tuning`
- Gst-nvtracker – Miscellaneous Data Output (Shadow Tracking, Terminated
  Track List — cần cho việc dọn state khi object mất track): `https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvtracker.html#miscellaneous-data-output`
- MetaData in the DeepStream SDK (NvDsObjectMeta, mask_params, user meta): `https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_metadata.html`
- Gst-nvinfer (đảm bảo YOLO/Pose vẫn đứng trước tracker đúng chuẩn, sinh
  bbox làm input cho MaskTracker): `https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvinfer.html`
- NvDsTracker API for Low-Level Tracker Library (chi tiết API
  `NvMOT_Process`, `NvMOT_RetrieveMiscData` mà MaskTracker triển khai): `https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvtracker.html` (mục "NvDsTracker API for Low-Level Tracker Library" trong cùng trang)

---

## 4. GIAI ĐOẠN 2C — KIỂM CHỨNG ĐỊNH LƯỢNG TRÊN TẢI THẬT (1–2 tuần)

**Đây là cổng quyết định quan trọng nhất của Phase 2. Không bật mặc định
nếu chưa đạt đủ các tiêu chí dưới đây.**

### 4.1. Benchmark tải thật (không dùng video mẫu đơn luồng)

| Hạng mục | Cách đo | Ngưỡng đạt |
|---|---|---|
| Throughput 6 camera đồng thời | FPS thực tế mỗi camera khi native bật | Không thấp hơn đáng kể so với baseline Python hiện tại |
| Latency end-to-end | T0 (tracker nhận frame) → T_mask_ready (polygon sẵn sàng gửi Monitor) | Cải thiện rõ rệt so với baseline Phase 0, tiệm cận <30ms |
| Tỷ lệ khựng hình quan sát | Số lần khựng ≥200ms trong cửa sổ đo 90–100s giống Phase 0 | Giảm so với baseline, đối chiếu với kết luận nghi phạm ở mục 2A.3 |
| Ổn định qua occlusion | Robot bị che khuất 1–2s rồi xuất hiện lại | Track ID + mask không reset sai, không nhảy object (đối chiếu Shadow Tracking/Terminated Track List) |

### 4.2. Đối chiếu độ chính xác mask với Label đã xác thực

- Lấy tối thiểu vài trăm frame có robot di chuyển/xoay/bị che khuất một
  phần, tính IoU giữa polygon native và Label/triplet đã xác thực.
- Báo cáo IoU trung bình + phân bố (không chỉ 1 con số trung bình, cần
  thấy rõ trường hợp xấu nhất — worst-case, không chỉ average-case).
- Nếu IoU thấp ở nhóm case cụ thể (ví dụ robot xoay góc lớn, hoặc bị che
  khuất >50% diện tích), ghi rõ thành hạn chế đã biết, không che giấu
  trong số trung bình.

### 4.3. Tiêu chí để chuyển `DEEPSTREAM_NATIVE_MASKTRACKER=1` làm mặc định

Chỉ được đề xuất bật mặc định khi ĐỦ CẢ 4 điều kiện:
1. Đã có kết luận nghi phạm chính thức từ mục 2A.3 (biết rõ vì sao khựng
   giảm, không phải "có vẻ tốt hơn").
2. Benchmark 6 camera thật đạt các ngưỡng ở bảng 4.1.
3. IoU trung bình với Label ≥ 0.9, và không có nhóm case nào IoU < 0.7 mà
   chưa được ghi nhận + có phương án xử lý.
4. Đã chạy ổn định tối thiểu 24–48 giờ liên tục trên môi trường thử
   nghiệm (staging), không có crash/leak bộ nhớ GPU.

---

## 5. GIAI ĐOẠN 2D — DỌN DẸP KIẾN TRÚC SONG SONG (SAU KHI ĐÃ BẬT MẶC ĐỊNH)

Chỉ thực hiện sau khi Giai đoạn 2C đạt đủ tiêu chí và đã chạy production
ổn định một thời gian (đề xuất tối thiểu 2 tuần không sự cố):

1. Gỡ bỏ dần đường Python SAM2 side-car (không xóa ngay — chuyển thành
   "cold fallback" có thể bật lại bằng cấu hình, không phải xóa code).
2. Dọn các lớp chuyển đổi contract trung gian không còn cần thiết nếu
   `NvDsObjectMeta.mask_params` đã đủ để Monitor tiêu thụ trực tiếp.
3. Cập nhật lại tài liệu kiến trúc hệ thống (README, sơ đồ pipeline) để
   phản ánh đúng pipeline native, tránh tài liệu cũ gây hiểu nhầm cho
   người sau.
4. Đưa cấu hình `[tracker]`/MaskTracker vào quy trình quản lý phiên bản
   model chính thức (không còn là "config tự sinh trong workspace" mà là
   artifact được version hóa, review như mọi model production khác).

---

## 6. AN TOÀN VẬN HÀNH (ĐỘC LẬP, SONG SONG VỚI CÁC GIAI ĐOẠN TRÊN)

Kế thừa từ phát hiện sự cố `DELETE /api/models/deployment` không khóa
trong lúc đo baseline Phase 0:

1. Thêm xác nhận/khóa quyền cho API dừng deployment khi đang có phiên đo
   hoặc production active.
2. Ghi log đầy đủ ai/phiên nào gọi API này.
3. Áp dụng nguyên tắc tương tự cho việc bật/tắt
   `DEEPSTREAM_NATIVE_MASKTRACKER`: đây là thay đổi có ảnh hưởng lớn tới
   toàn bộ 6 camera, nên có xác nhận rõ ràng trước khi áp dụng trên
   production, không chỉ đổi biến môi trường rồi restart container.

---

## 7. RỦI RO & PHƯƠNG ÁN DỰ PHÒNG

| Rủi ro | Phương án |
|---|---|
| Native MaskTracker cho throughput tốt nhưng IoU với Label thấp hơn Python SAM2 | Giữ Python SAM2 làm fallback lâu dài cho các case đặc thù (ví dụ object hình dạng phức tạp), dùng native cho case phổ biến — kiến trúc hybrid có kiểm soát, không phải all-or-nothing |
| Cấu hình native tự sinh từ engine có sẵn che giấu lỗi tham số (engine load được nhưng tham số suy luận sai) | Bắt buộc đối chiếu thủ công với schema tài liệu chính thức (mục 3.1), không tin tưởng hoàn toàn vào "load thành công = đúng" |
| Việc đối chiếu IoU chỉ dùng số trung bình che giấu case xấu | Bắt buộc báo cáo phân bố + worst-case, không chỉ trung bình (mục 4.2) |
| Bật mặc định quá sớm khi chưa đủ 24–48h chạy ổn định | Tiêu chí cứng ở mục 4.3, không có ngoại lệ "gấp thì bật trước" |
| Mất khả năng rollback nếu dọn code Python quá sớm | Giai đoạn 2D giữ Python làm "cold fallback" có thể bật lại bằng cấu hình, không xóa code ngay |

---

## 8. TOÀN BỘ LINK THAM KHẢO (TỔNG HỢP PHASE 2)

1. Gst-nvtracker (tổng quan, xác nhận MaskTracker là tracker tham chiếu
   chính thức): `https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvtracker.html`
2. Gst-nvtracker – Tracker Accuracy Tuning: `https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvtracker.html#tracker-accuracy-tuning`
3. Gst-nvtracker – Miscellaneous Data Output (Shadow Tracking, Terminated
   Track List): `https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvtracker.html#miscellaneous-data-output`
4. MetaData in the DeepStream SDK: `https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_metadata.html`
5. Gst-nvinfer: `https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvinfer.html`
6. Using a Custom Model with DeepStream (tham khảo nếu cần fallback sang
   hướng nvinferserver dự phòng): `https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_using_custom_model.html`
7. Gst-nvinferserver + IInferCustomProcessor (PHƯƠNG ÁN DỰ PHÒNG, chỉ
   dùng nếu native MaskTracker không đạt tiêu chí Giai đoạn 2C): `https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvinferserver.html#custom-process-interface-iinfercustomprocessor-for-extra-input-lstm-loop-output-tensor-postprocess`
8. Performance (đo lường & benchmark): `https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Performance.html`
9. NTP Timestamp trong DeepStream: `https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_NTP_Timestamp.html`
10. Gst-nvstreammux New – tuning low-latency: `https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvstreammux2.html#optimizing-nvstreammux-config-for-low-latency-vs-compute`
11. Troubleshooting Guide: `https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_troubleshooting.html`
12. On the Fly Model Update: `https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_on_the_fly_model.html`

---

## 9. MỐC THỜI GIAN TỔNG THỂ PHASE 2

```
Tuần 1        Giai đoạn 2A — hoàn tất đo lường còn thiếu (queue latest-only,
              mailbox timeout A/B), có kết luận nghi phạm chính thức
Tuần 2-3      Giai đoạn 2B — chuẩn hóa cấu hình native MaskTracker theo
              đúng tài liệu, rà soát enableReAssoc, đối chiếu mask contract
Tuần 3-4      Giai đoạn 2C — benchmark tải thật 6 camera + đối chiếu IoU
              với Label, áp đủ 4 tiêu chí trước khi đề xuất bật mặc định
Tuần 5+       (Chỉ nếu đạt tiêu chí) Giai đoạn 2D — dọn dẹp kiến trúc
              song song, version hóa config, cập nhật tài liệu hệ thống
Song song     An toàn vận hành — khóa API deployment, log truy vết
```

---

## 10. PROMPT MẪU ĐỂ ĐƯA CHO CODING AGENT

```
Bạn là kỹ sư triển khai DeepStream/NVIDIA cho dự án R-SkyView. Nhiệm vụ:
thực hiện GIAI ĐOẠN 2A và 2B của file
RSkyView_SAM2_Phase2_Native_MaskTracker_Plan.md đã đính kèm.

BẮT BUỘC ĐỌC TRƯỚC KHI VIẾT BẤT KỲ CODE NÀO:
1. https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvtracker.html
   (đọc TOÀN BỘ trang, không chỉ phần MaskTracker — cần hiểu đúng NvMOT_Process,
   NvMOT_RetrieveMiscData, cấu hình [tracker] section, ll-lib-file, ll-config-file)
2. https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvtracker.html#tracker-accuracy-tuning
3. https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvtracker.html#miscellaneous-data-output
4. https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_metadata.html
   (hiểu đúng NvDsObjectMeta, mask_params, cách gắn/đọc user meta chuẩn)
5. https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvinfer.html
   (xác nhận lại thứ tự nvinfer PHẢI đứng trước nvtracker để sinh bbox
   đầu vào cho MaskTracker, đúng như đã tự phát hiện và sửa ở Phase 1)

SAU KHI ĐỌC XONG, TÓM TẮT LẠI CHO TÔI (trước khi code):
- Đúng cấu trúc file cấu hình [tracker] section theo tài liệu, so sánh
  với config_tracker_MaskTracker.yml và config_tracker_module_Segmenter.yml
  hiện có trong /app/models/sam2_tracker -- liệt kê rõ điểm nào khớp,
  điểm nào lệch so với tài liệu chính thức.
- Cách đúng để đọc Shadow Tracking Target Data và Terminated Track List
  từ NvMOT_RetrieveMiscData, áp dụng vào việc dọn state khi object mất
  track (tránh memory bank "rác" như đã cảnh báo ở Phase 0).

NHIỆM VỤ CỤ THỂ (Giai đoạn 2A -- đo lường, KHÔNG sửa code pipeline chính):
1. Dùng lại trace Phase 0 (13.610 frame tracker / 7.821 object), tính
   tương quan thời gian giữa event frame rớt do queue "latest-only" và
   các lần khựng ~200ms đã ghi nhận. Xuất số liệu % trùng khớp cụ thể,
   không kết luận định tính.
2. Dựng 2 kịch bản A/B (mailbox timeout 100ms vs 5ms), cùng workload 6
   camera, đo tỷ lệ rớt frame + độ dài khựng ở mỗi kịch bản. Xác nhận rõ
   đây có phải cùng cơ chế queue với mục (1) không.
3. Viết báo cáo kết luận nghi phạm chính thức, xếp hạng theo mức đóng góp
   thực tế (có số liệu), lưu vào docs/reports/.

NHIỆM VỤ CỤ THỂ (Giai đoạn 2B -- chuẩn hóa cấu hình, SAU khi có kết quả 2A):
1. Đối chiếu deepstream_masktracker.py và 2 file config native hiện có
   với đúng schema tài liệu chính thức (không chỉ dựa vào "engine load
   thành công"). Sửa mọi tham số lệch, ghi rõ từng thay đổi và lý do dựa
   trên đoạn tài liệu nào.
2. Rà soát TOÀN BỘ codebase (kể cả nhánh native mới) tìm mọi nơi có thể
   override enableReAssoc=0, không chỉ sửa 1 chỗ đã biết ở
   custom_detector.py:208.
3. Đối chiếu polygon sau chuyển đổi (deepstream_native_mask.py) với mask
   gốc raw từ decoder bằng IoU, TRƯỚC KHI so với Label -- đảm bảo bước
   chuyển đổi contract không tự làm mất độ chính xác.
4. KHÔNG bật DEEPSTREAM_NATIVE_MASKTRACKER=1 làm mặc định ở bước này.
   KHÔNG xóa hoặc tắt đường Python SAM2 hiện tại.

RÀNG BUỘC CHUNG (giữ nguyên từ các phase trước):
- Giữ nguyên logic Label/triplet là authority cho identity, native mask
  chỉ là nguồn hình học.
- Giữ WebSocket contract của Monitor không đổi.
- Mọi thay đổi phải có khả năng rollback về Python SAM2 bằng cấu hình,
  không xóa code cũ.
- Báo cáo kết quả bằng số liệu cụ thể (bảng, %, ms), không dùng mô tả
  định tính như "nhanh hơn", "mượt hơn" mà không kèm con số.
- Nếu phát hiện tài liệu NVIDIA mô tả khác với những gì đã triển khai ở
  Phase 1 (ví dụ cách cấu hình MaskTracker), báo cáo rõ điểm lệch trước
  khi tự ý sửa, để tôi xác nhận hướng xử lý.

Xong Giai đoạn 2A và 2B, dừng lại và báo cáo đầy đủ trước khi làm tiếp
Giai đoạn 2C (benchmark tải thật + đối chiếu IoU với Label) -- đây là cổng
quyết định quan trọng nhất, cần tôi xác nhận trước khi tiếp tục.
```
