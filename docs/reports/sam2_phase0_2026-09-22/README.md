# Phase 0 — baseline SAM2 / DeepStream chạy inference thật

Ngày đo: **22/09/2026**, múi giờ **Asia/Bangkok (UTC+07)**. Đây là báo cáo đo, không phải bản tối ưu hay chứng nhận hết khựng.

## 1. Kết luận để quyết định bước tiếp theo

- **Đã bật SAM2 thật**, xác nhận bằng CUDA events encoder/decoder, video-memory propagation và mask đầu ra; không sử dụng kết quả idle làm baseline AI.
- Kiến trúc vẫn là **SAM2 Python bất đồng bộ ngoài graph infer GStreamer + biến đổi polygon theo bbox**, nhưng **không có một tiến trình SAM riêng nhận/trả frame qua IPC**. Hai thread SAM nằm trong chính subprocess chạy GStreamer. Chỉ JSON metadata đi qua stdout IPC về FastAPI.
- **Đã tái hiện >200 ms**: sau loại 15 giây khởi động, 476/4.853 mask có đủ T0–T3 vượt 200 ms. Loại thêm khoảng đổi tab và 2 giây sau quay lại vẫn còn **447/4.605** mask vượt 200 ms.
- Có ba thành phần: chờ worker; tính encoder + tuần tự nhiều object và xác thực; **mask đã sẵn sàng nhưng phải chờ frame tiếp theo của camera để attach**. Không thể quy toàn bộ cho encoder hoặc IPC.
- Ví dụ ngoài khoảng đổi tab: Cam góc rộng frame 991 có **T0→T3 = 341,32 ms**, trong đó **T2→T3 = 223,81 ms**; sau cache còn chờ **216,10 ms mới có tracker frame tiếp theo**. Cam 4 frame 1264 có **T1→T2 = 233,31 ms** dù decoder riêng object đó chỉ 16,00 ms.
- Baseline không upload: stdout IPC trung bình **0,19–0,84 ms**, P95 **0,49–1,44 ms**. **Không phải IPC 200 ms trong lượt này.** Kịch bản mở trang upload có một đỉnh đường truyền riêng, được tách rõ ở mục 7.
- Encoder CUDA trung bình chỉ **8,61–9,36 ms/frame có encoder**. TensorRT hóa encoder có thể là một bước sau, nhưng số đo **không chứng minh chỉ bước này sẽ hết khựng**. Chưa triển khai Giai đoạn 1/2/3/4.

## 2. Phạm vi và điều kiện tái hiện

Đọc toàn bộ tài liệu người dùng `RSkyView_SAM2_DeepStream_Optimization_Plan.md`; thực hiện yêu cầu Giai đoạn 0. Audit đường chạy được deploy và các module liên quan, không chỉ suy đoán theo tên file hoặc mô tả cũ. Không tuyên bố kiểm định mọi chức năng không liên quan trong repository.

| Điều kiện | Giá trị |
|---|---|
| HEAD | `4cf867bfd10bc76d9559bae1eb494fc3538ecba3`; workspace có nhiều thay đổi sẵn, không reset |
| GPU | GeForce RTX 5060 Ti, 16.311 MiB; driver 595.84 |
| Runtime đã kiểm tra | PyTorch 2.14.0; Ultralytics 8.4.153; CuPy 13.4.1; trtexec TensorRT 10.9 |
| Detector | `best(1).onnx`, ID `479ea92c0b6044febba5b31a5e8e5e08`, engine TensorRT FP16 |
| ONNX SHA256 | `a9f19caf82d0ceb396531ca74a54c8e3cdcf8162f78d73c43782ade840f51e0b` |
| ONNX contract | `[1,3,640,640]` → `[1,9,8400]`, detection, không phải segmentation |
| Lớp model | Rack, Robot_1, Robot_2001, Robot_28100, Robot_6868 |
| SAM | `sam2.1_t.pt`, CUDA FP16, imgsz 640; 2 worker thread, stream CUDA riêng |
| Gallery | revision 170, 168 views; không thêm/xóa/sửa label, threshold hoặc calibration |
| Camera AI | 6 camera, ID và tên trong `evidence/cameras.json` |
| Browser đo | Chrome headless, 1600×1000, chạy cùng máy server; Monitor có 7 video |

Camera thứ 7 `b5e8f409` (Cam trong khu làm việc) chỉ có video, không thuộc deployment AI thử nghiệm. Không so trực tiếp thành tích browser lượt này với benchmark idle 6 camera trước đó.

Các lượt:

1. Lượt thăm dò thực: khoảng 111 giây, bị một phiên dashboard khác dừng deployment lúc **17:07:03,477**; không dùng làm bảng baseline chính. Browser của lượt đó bắt đầu sau khi đã dừng nên **không được coi là đo AI**.
2. **Baseline chính:** deploy lúc **17:14:39,515**, subprocess bắt đầu **17:14:40,162**, dừng đo/khôi phục idle **17:16:19,770**. Browser **17:14:34,289–17:16:28,530**; đổi Monitor→Building **17:15:31,084**, quay lại **17:15:33,593**. Các bảng backend loại 15 giây đầu subprocess, còn khoảng 85 giây; bảng browser loại cả thời gian đổi tab và 2 giây sau quay lại.
3. **Upload/build:** deploy **17:30:09,443**, subprocess **17:30:09,795**; bấm upload **17:30:36,420**, UI xác nhận **17:30:37,273**; dừng deployment thử nghiệm **17:32:39,548**. Build hoàn thành `ready` **17:34:02,275**. Trace AI/GPU/CPU bao phủ lúc bấm upload và khoảng 123 giây đầu sau đó, **không bao phủ toàn bộ 205 giây build**.

Bốn lần restart trong bộ đếm supervisor là lỗi thiết lập tracer trước các lượt hợp lệ, không tính là sự cố của baseline. Mỗi lượt chính/upload ghi nhận đúng một subprocess inference, không restart giữa lượt.

## 3. Audit kiến trúc đang thực thi

```text
Camera → MediaMTX ┬→ WHEP → browser video
                 └→ nvurisrcbin → queue leaky(max=1)
                    → nvstreammux(batch=1,1280×720,10ms)
                    → nvinfer(YOLO TensorRT) → nvtracker(NvDCF) [T0]
                    → nvvideoconvert RGBA GPU → capsfilter probe → fakesink
                                                        │
                      CuPy GPU RGB copy + synchronize → latest-only/camera
                                                        │
                                    SAM worker thread dequeue [T1]
                                      PyTorch encoder chung/frame
                                      live decoder từng object [T2]
                                      verify + polygon + memory encode
                                      cache kết quả từng object
                                                        │
             probe của frame tiếp theo → attach polygon theo bbox [T3]
                       → mailbox → stdout JSON → FastAPI → WebSocket
                                                        → canvas overlay
```

Đối chiếu mã nguồn (đường dẫn từ `/home/rtcai/Desktop/Vision Manager`):

- `backend/core/custom_deepstream_worker.py:116`: cấu hình mux, YOLO, NvDCF, RGBA, fakesink. Probe submit/attach tại RGBA, không phải SAM GStreamer plugin.
- `backend/core/custom_deepstream_worker.py:152`: đầu vào thực là MediaMTX RTSP localhost, latency 80 ms, drop-on-latency và leaky queue. Không phát video đã vẽ mask từ pipeline này.
- `backend/core/model_sam2.py:104`: pending mới nhất theo camera, camera gắn worker, không FIFO frame vô hạn.
- `backend/core/model_sam2.py:241`: hai thread inference/CUDA stream. Việc chạy song song đã có, không phải hoàn toàn đơn luồng.
- `backend/core/model_sam2.py:320`: SAM2 PyTorch FP16, không TensorRT encoder. GPU surface → CuPy RGB copy → DLPack; không có serialize full frame sang tiến trình SAM riêng.
- `backend/core/model_sam2.py:435` và `backend/core/sam2_live_predictor.py`: encoder chung/frame, decoder/memory theo từng object; xác thực bằng gallery/triplet trước khi commit memory. Reference views được encode/cache, không phải học lại YOLO weights.
- `backend/core/model_sam2.py:117`: publish từng object vào cache ngay, không cần chờ tất cả object xong. **Nhưng publish cache chưa tự phát mask về Monitor**.
- `backend/core/model_track_masks.py:83`: attach tại probe kế tiếp; polygon scale/translate theo bbox, TTL cache 700 ms, còn chịu identity/position gates. Không suy luận lại mask 30 FPS cho từng object.
- `backend/core/custom_detector.py:130`: stdout reader tách publisher; mailbox ưu tiên mới, publisher tick 5 ms. Callback tiếp tục qua FMS/workflow/metadata broadcaster.
- `web-dashboard/components/views/MonitorView.tsx` và `web-dashboard/lib/segmentation-overlay.ts`: canvas overlay trên video WHEP độc lập, không nvdsosd GPU vẽ trực tiếp vào encoded stream.
- `backend/core/model_registry.py:215`: upload/build chạy thread và subprocess riêng. Upload ONNX tạo model mới, **không tự deploy/swap engine đang dùng**. Hot-reload đã có đường điều khiển và callback `model-updated` ở worker; lượt này không swap engine production.

Vì vậy mô tả “Python worker ngoài GStreamer + bbox interpolation” **đúng về graph suy luận và cách propagate polygon**, nhưng **sai nếu hiểu frame đang đi qua IPC sang một SAM process khác rồi quay về**. Cũng chưa phải toàn bộ xử lý GPU: polygon boundary loops và các bước `.tolist()`/đọc score có CPU/synchronization.

## 4. Định nghĩa đo và cách đọc CSV

| Mốc | Định nghĩa chính xác |
|---|---|
| T0 | `perf_counter_ns` tại `nvtracker.src`, kèm camera, source frame_num, PTS, số NvDs objects |
| T1 | worker lấy frame khỏi pending, trước segment; gồm thời gian vào worker chứ không chỉ copy |
| T2 | CUDA event kết thúc `infer_live` decoder của **từng object**, sau encoder frame và các object trước nó |
| T3 | lần đầu output `ModelSAM2.attach` có mask đúng camera/source-frame/object đó; trước stdout và WebSocket |

**T1→T2 không phải “pure forward của một object”**: còn encoder chung, reference setup, các object trước đó, xác thực/host scheduling giữa chúng. Vì thế CSV có thêm `encoder_cuda_ms`, `decoder_cuda_ms`, memory encode, identity verify, binary/polygon postprocess và `segment_wall_ms`.

**T2→T3 không phải chỉ IPC trả kết quả**: gồm verify/postprocess, cập nhật video memory, cache và chờ probe kế tiếp/đủ điều kiện attach. T2 là hoàn thành prediction, **memory commit ở sau**, có span riêng. T3 là metadata Python của pipeline, không chèn segmentation tensor vào NvDsUserMeta hay video đã encode.

- `frames.csv`: một dòng mỗi tracker frame; frame không được chọn inference vẫn hiện, không bịa T1/T2/T3.
- `objects_t0_t3.csv`: một dòng mỗi decoder object; T3 trống có thể do rejected/không attach/bị thay thế/kết thúc lượt, không tự động kết luận mất kết nối.
- `completed_stage_summary.csv`: các stage có **cùng tập object hoàn tất T3**, nên cộng mean ba stage bằng mean tổng. Không cộng các P95/P99 với nhau.
- `by_camera.csv`: frame stages dùng số frame, object stages dùng số decoder object; có thể khác tập mẫu giữa từng metric.
- `by_camera_objects.csv`: chia camera × `tracker_objects` × `targets` × `decoder_objects` × metric; `targets` gồm prompt do Label, khác số bbox NvDCF.
- `decoder_objects=0` nghĩa không chạy decoder lượt đó (ví dụ rejection cache), **không dùng các dòng này để quảng cáo tốc độ SAM forward**.
- Percentile nearest-rank; loại 15 giây đầu subprocess. Mask không đến T3 không nằm trong percentile T0→T3: đây là latency **có điều kiện kết quả được attach**, không phải tỷ lệ nhận dạng thành công.

## 5. Baseline từng camera

Các ô stage là **mean / P95 / P99**, đơn vị **ms**, cùng tập mẫu hoàn tất sau warmup.

| Camera | n | T0→T1 | T1→T2 | T2→T3 | T0→T3 |
|---|---:|---:|---:|---:|---:|
| Cam 4 — b1269e28 | 1.936 | 37,11 / 87,99 / 145,34 | 65,73 / 120,16 / 139,68 | 42,29 / 91,08 / 145,76 | **145,13 / 238,16 / 289,79** |
| Góc rộng — 06e9e5b9 | 1.092 | 32,53 / 109,19 / 164,74 | 51,09 / 89,67 / 105,68 | 44,52 / 147,08 / 232,07 | **128,14 / 283,23 / 293,30** |
| Cam mới — c7e22c1f | 833 | 25,21 / 61,14 / 82,12 | 27,97 / 43,25 / 50,35 | 33,83 / 63,75 / 98,88 | **87,01 / 139,10 / 180,78** |
| Hik 1 — d1f37ed1 | 260 | 26,85 / 66,45 / 133,37 | 24,84 / 35,30 / 40,31 | 47,97 / 193,82 / 242,29 | **99,66 / 274,65 / 352,42** |
| Hướng cửa — e0606b55 | 731 | 25,48 / 66,01 / 99,66 | 30,51 / 51,06 / 64,27 | 42,92 / 152,34 / 221,72 | **98,92 / 270,64 / 287,28** |
| Dahua — 1ac2ffac | **1** | 40,59 / 40,59 / 40,59 | 15,05 / 15,05 / 15,05 | 47,04 / 47,04 / 47,04 | **102,68 / 102,68 / 102,68** |

**Dahua n=1, không có ý nghĩa P95/P99 ổn định**; có 209 decoder sau warmup, 134 kết quả reject, chỉ 1 mask tới attach. Không thể gọi camera này “mượt/đúng” chỉ dựa vào latency thấp. Identity correctness cần bài test ground truth riêng, ngoài phạm vi Phase 0.

Chi phí frame, vẫn mean / P95 / P99 (ms):

| Camera | GPU-copy wall | Pending wait | Encoder CUDA | Segment wall tất cả target |
|---|---:|---:|---:|---:|
| Cam 4 | 0,50 / 1,53 / 2,47 | 36,16 / 86,05 / 144,59 | 9,36 / 13,05 / 14,23 | 123,13 / 160,00 / 179,47 |
| Góc rộng | 0,52 / 1,54 / 2,05 | 31,54 / 106,90 / 163,68 | 9,27 / 13,09 / 14,83 | 88,00 / 115,09 / 130,01 |
| Cam mới | xem CSV | 24,20 / 59,44 / 81,67 | 8,70 / 12,30 / 13,96 | 39,73 / 64,53 / 77,10 |
| Hik 1 | xem CSV | 25,66 / 62,78 / 97,46 | 8,69 / 12,15 / 13,69 | 25,49 / 61,26 / 81,93 |
| Hướng cửa | xem CSV | 25,28 / 65,53 / 99,13 | 8,74 / 12,34 / 13,96 | 38,99 / 59,88 / 75,37 |
| Dahua | xem CSV | 24,25 / 65,57 / 96,21 | 8,61 / 11,86 / 13,69 | 17,11 / 35,31 / 44,04 |

Trong 85 giây sau warmup, Cam 4 có 423 lượt segment (~4,97 lượt/s), góc rộng 373 (~4,39 lượt/s), Cam mới 850 (~10 lượt/s). Đây là tốc độ **lượt xử lý frame**; không phải mỗi object đều có mask mới ở tần số này. Video 20–25 FPS và polygon biến đổi theo bbox không có nghĩa SAM thực sự inference mọi frame.

Phân bố mẫu theo số object (trích `by_camera_objects.csv`, `segment_wall_ms`):

| Camera | NvDCF objects | Targets gồm Label | Decoder thật | n frame | Mean | P95 | P99 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cam 4 | 4 | 8 | 4 | 69 | 97,83 | 124,12 | 178,56 |
| Cam 4 | 4 | 8 | 5 | 99 | 119,39 | 143,34 | 148,77 |
| Cam 4 | 4 | 8 | 6 | 202 | 132,10 | 164,63 | 192,03 |
| Góc rộng | 4 | 6 | 3 | 106 | 77,11 | 97,20 | 106,33 |
| Góc rộng | 4 | 6 | 4 | 147 | 91,24 | 114,25 | 130,01 |
| Hik 1 | 1 | 1 | 1 | 119 | 21,94 | 29,67 | 30,84 |
| Hik 1 | 1 | 2 | 1 | 114 | 30,18 | 42,59 | 45,89 |

Đây là phân nhóm các quan sát thực, **không phải thí nghiệm giữ cảnh cố định rồi tăng N**; tải camera khác/reference cache có thể khác giữa nhóm.

## 6. Khựng nằm ở khâu nào?

| Ví dụ baseline | T0→T1 | T1→T2 | T2→T3 | Tổng | Diễn giải |
|---|---:|---:|---:|---:|---|
| Cam 4, frame 675 | 179,52 | 41,62 | 92,77 | 313,91 | Chờ worker chiếm phần lớn |
| Cam 4, frame 1264, object 5410850743960 | 73,75 | 233,31 | 48,30 | 355,36 | Trong worker, 8 targets/6 decoders; decoder object riêng 16,00 ms, không phải encoder 233 ms |
| Góc rộng, frame 991, object 190452056632062 | 67,64 | 49,87 | 223,81 | 341,32 | Cache→attach 217,13 ms; tracker kế tiếp đến sau cache 216,10 ms |
| Hik 1, frame 1144, object 62396681866199 | 43,95 | 39,95 | 290,29 | 374,18 | Cache→attach 282,84 ms; chờ tracker đầu tiên 193,23 ms, còn chờ thêm 2 tracker frame mới đủ điều kiện attach; xảy ra trong khoảng đổi tab |

Ví dụ Hik cực đại nằm trong đổi tab nên không dùng một mình để kết luận steady-state. Ba ví dụ trước nằm ngoài khoảng đổi tab; chỉ riêng steady-state vẫn có hàng trăm mask >200 ms.

**Bằng chứng return-path stall:** 74 mask baseline có cache→attach ≥180 ms; **72/74** tới attach ở tracker frame đầu tiên sau khi cache hoàn tất. Lượt upload **85/85** tương tự. Hai trường hợp baseline còn lại có thêm gating/chờ các frame sau. Xem `slow_attach_context.csv`.

**Khoảng trống trước T0 cũng có thật:** P99 khoảng cách tracker frame lần lượt ~280 ms (góc rộng), 276 ms (Hik), 277 ms (hướng cửa). Ví dụ góc rộng frame 997→998: wall gap **254,53 ms**, PTS chỉ tăng **39,99 ms**; metadata frame 997 đã emit sau T0 **1,05 ms**, trong gap có **23 tracker frame của camera khác**. Hướng cửa 286→287: wall gap **293,90 ms**, PTS **40,00 ms**, 34 frame camera khác tiếp tục đi qua.

Điều này **không giống toàn server/GPU ngừng 200 ms**, và không phải previous probe xử lý 250 ms. Dữ liệu chậm tới đầu ra tracker theo camera, rồi cơ chế “chỉ attach khi frame tới” kéo dài T2→T3. **Chưa đủ probe trước decoder/mux/nvinfer/tracker để phân biệt chính xác RTSP jitter, decoder, mux scheduling hay NvDCF nội bộ**; PTS đều không đồng nghĩa gói RTSP tới đều. `source_gap_pts.csv` lưu bằng chứng; không gán lỗi chắc chắn cho camera/network/GIL khi chưa đo sâu hơn.

Tổng số toàn lượt baseline: 13.610 tracker frames, 9.884 enqueue, 6.054 pending replacements, 3.825 dequeues, 7.821 decoder objects, 5.416 mask lần đầu attach. Sau warmup: 6.940 decoders, 4.853 attached; 1.728 `appearance_mismatch`, 9 `object_not_present`, 2 `category_conflict`; 142 kết quả có mask nhưng không quan sát T3. Pending replacement là cơ chế chọn frame mới đã có, **không đồng nghĩa mất kết nối**. Không thay đổi các gate trong Phase 0.

## 7. Upload/build và tài nguyên

Upload qua **nút giao diện thật** ở Building → Model / TensorRT, trên trang riêng; trang Monitor đo vẫn mở. File là bản sao ONNX hiện tại 37.934.551 bytes, tên `PHASE0 diagnostic upload 2026-09-22`, ID `676fc8b7e4714613b318fe9c9925afc7`. Không thay engine đang chạy. Có log mọi chunk/HTTP response và timestamp bấm nút trong evidence.

- Bấm **17:30:36,420**; UI nhận xong **17:30:37,273** (~853 ms).
- `trtexec --fp16 --skipInference --memPoolSize=workspace:1024 --builderOptimizationLevel=2 --maxAuxStreams=0` chạy thật; log `PASSED`, model `ready` lúc **17:34:02,275**. Không giả lập build.
- Trong cửa sổ GPU 1 giây có AI: trước click SM trung bình **71,17%** (12 mẫu); giai đoạn sau click/build trung bình **80,02%**, P95 **97%**, max **99%** (117 mẫu). VRAM tăng khoảng **3.641 → 4.508 MiB**. Có contention/tải GPU cao, **không thấy kịch bản GPU idle trong cả khoảng đo**.
- Baseline không build: SM mean **87,20%**, P95 **94%**, max95%; power mean115,78 W, max124 W. Không so trung bình hai lượt như A/B nhân quả: cảnh/số decoder khác nhau.
- CPU process DeepStream/SAM baseline mean **173%**, P95 **181%**; upload mean **187,30%**, P95 **198%**, max203% (100% = một lõi). Uvicorn lần lượt mean37,38% và40,63%. Một thread có thể gần một lõi (max94–97%), **không chứng minh bị GIL hoặc blocking hoàn toàn**. pidstat lọc Python/uvicorn, không thống kê CPU `trtexec` đầy đủ.

T0→T3 trong 5 giây trước/sau click, cùng camera:

| Camera | Trước n | Mean / P95 / P99 | Sau n | Mean / P95 / P99 |
|---|---:|---:|---:|---:|
| Cam 4 | 140 | 128,43 / 197,44 / 217,29 | 131 | 126,46 / 205,46 / 272,72 |
| Góc rộng | 90 | 117,67 / 284,41 / 286,01 | 69 | 118,11 / 277,20 / 281,82 |
| Cam mới | 68 | 95,82 / 151,38 / 192,98 | 54 | 104,92 / 156,96 / 202,25 |
| Hướng cửa | 29 | 108,54 / 284,65 / 286,83 | 30 | 104,68 / 278,25 / 291,49 |

Hik không có mask hoàn tất trong 5 giây trước click; Dahua không có mask attach sau warmup trong lượt upload. Không tạo số so sánh giả cho các ô không dữ liệu. Toàn bộ nhóm trong `upload/upload_windows.csv`.

**Phát hiện thêm ở client/transport:** lượt mở trang upload có đỉnh emit→Node-WebSocket **193,10 ms**, nhưng lúc **17:30:36,118 — trước click 302 ms**. Một số camera cùng lúc có IPC read delay 96–131 ms; riêng gói Cam 4 có publish→client 179,57 ms. Đây là khi automation mở trang Building/chuẩn bị file, **trước khi upload/build bắt đầu**; không được gọi là trtexec làm đứng backend. Browser automation và WS recorder cùng Node process nên Node event-loop bị trì hoãn cũng có thể làm receipt timestamp lớn; chưa có profiler để tách phần này khỏi server callback.

Trong 5 giây **sau click**, max IPC chỉ **3,10 ms**, max emit→client **68,43 ms**; sau 5 giây tới cuối, max lần lượt **6,13 ms** và **40,02 ms**. T0→T3 vẫn có nhiều đỉnh >200 ms. Đã tái hiện độ trễ mask và có tải build cao, nhưng **chưa xác nhận một cú đóng băng 200 ms chỉ do chính thao tác upload hoặc hot-reload**. Không hot-swap production trong phép đo này.

## 8. Frontend có vẽ chậm không?

- Không sửa frontend. Đo callback `requestVideoFrameCallback`, WebRTC `getStats`, browser long tasks và WS receiver độc lập.
- Baseline steady, Cam 4 callback mean **0,085 ms**, P95 **0,30 ms**, P99 **0,40 ms**, max **0,60 ms**; các camera khác max **≤0,70 ms**. Đây là thời gian JavaScript callback, **không bao gồm toàn bộ GPU compositor/scanout**.
- Chỉ một long task **54 ms lúc khởi tạo trang**, không thấy long task ≥50 ms trong steady-state đã lọc. Lượt upload có một long task62 ms lúc khởi tạo Monitor.
- Browser decode trung bình đúng khoảng **20 FPS Cam 4**, **25 FPS các camera AI còn lại**; số này không chứng minh mask mới 20–25 FPS.
- Jitter-buffer WebRTC Cam 4 mean **84,45 ms** (P95 89,31 ms), camera AI khác khoảng20,55–23,39 ms. Đây là metric khác T0→T3, không cộng thẳng các percentile hoặc khẳng định end-to-end 35–86 ms khi AI live.
- Có các rVFC gap >120 ms, gồm 32 gap ở camera thứ 7 không chạy AI trong khoảng steady đã lọc. Callback có thể bị browser không gọi đủ, video ở ngoài viewport hoặc decode/presentation scheduling; **không coi mọi gap là frame bị mất hoặc AI làm đông video**. Xem RTC counters và `video_gaps.csv` thay vì chỉ đếm callback.
- WS emit→client baseline P95 **8,15–10,86 ms**, max20,04 ms. Trong điều kiện local/headless đã đo, **không có bằng chứng canvas JavaScript mất 200 ms**; delay lớn của mask chủ yếu đã có trước khi emit. Không suy rộng sang máy browser từ xa/mạng khác.

## 9. Giới hạn và tính tin cậy

1. Tracer đo bằng monotonic clock + CUDA events, không đo GPU forward bằng riêng thời gian Python enqueue. Có một clock anchor synchronize lúc khởi động (~0,215 ms baseline), không device-wide synchronize mỗi frame. GPU timestamp được ánh xạ gần đúng sang host; CUDA duration gồm scheduling/preemption/host gaps trong event interval, **không phải tổng thời gian kernel thuần**.
2. Thêm recorder thread/JSONL/CUDA events luôn có overhead. Không có A/B profiler-disabled cùng cảnh để định lượng overhead. Tracer có bounded queue100k; bộ đếm dropped không được xuất cuối lượt nên không chứng minh tuyệt đối zero trace loss. Khi kết thúc subprocess, phần đuôi chưa flush (<0,5s theo chu kỳ writer, và CUDA events chưa hoàn tất) có thể thiếu; giữ mẫu thiếu T2/T3 trống, không tự gán 0.
3. T3 là attach đủ điều kiện theo identity của object, không luôn là frame đầu tiên sau decoder. Frame có prompt rejection/cached rejection không thể gộp vào mẫu successful mà bỏ qua n. Mean stage dùng cùng tập trong `completed_stage_summary.csv`; các tổng hợp khác ghi rõ weighting.
4. Chưa đo camera exposure→screen photon, IoU, ID switches, chất lượng bám khi robot đổi góc, hay ground truth FMS. Không hứa 30 FPS segmentation/object hoặc 0 ms latency từ các bảng này.
5. GPU/CPU sampling1s không đủ xác định nguyên nhân kernel/GIL cho riêng mỗi spike200ms. Đã khoanh vùng stage, nhưng nguyên nhân sâu của gap trướcT0 và đỉnh Node/IPC trước upload cần probe/profile bổ sung nếu người dùng muốn.
6. Không đo hết cuối build sau dừng inference, không benchmark hot-reload model khác, không thay detector/model SAM/label threshold/cache TTL. Giai đoạn0 này không tự động cho phép chuyển sang tối ưu.

## 10. File bàn giao, tái lập và trạng thái cuối

Thư mục gốc: `/home/rtcai/Desktop/Vision Manager/docs/reports/sam2_phase0_2026-09-22/`.

- `README.md`: báo cáo này.
- `completed_stage_summary.csv`, `by_camera.csv`, `by_camera_objects.csv`: mean/P95/P99 camera/stage/số object.
- `frames.csv`, `objects_t0_t3.csv`, `slow_masks.csv`: dữ liệu từng frame/object và đỉnh latency.
- `tracker_gaps.csv`, `slow_attach_context.csv`, `source_gap_pts.csv`: đối chiếu chờ attach với tracker/PTS/camera khác.
- `metadata_transport.csv`, `transport_summary.csv`: IPC, parent queue, emit→client.
- `video_stats.csv`, `video_gaps.csv`, `browser_callbacks.csv`, `browser_longtasks.csv`, `steady_browser_tracker_summary.csv`: frontend/video evidence.
- `gpu_samples.csv`, `gpu_summary.csv`, `cpu_process_summary.csv`: tải máy.
- `upload/`: bộ bảng tương tự cho lượt upload/build, thêm `upload_windows.csv`.
- `evidence/`: trace/client/action JSONL gzip, GPU/CPU logs, build.log, metadata model thử nghiệm, camera map, hash code/model, trạng thái cuối. Không chứa cookie hoặc RTSP password trong các artifact do phép đo này xuất.
- `tools/analyze.py`, `tools/correlate.py`, `tools/source_gaps.py`: dựng lại bảng; `tools/capture.mjs` là recorder Playwright của máy đo, cần cookie riêng và đường dẫn Node/Chrome cục bộ.

Để dựng lại CSV offline, giải nén `controlled_trace.jsonl.gz` vào một thư mục dưới tên `events-1297.jsonl`, `controlled_client.jsonl.gz` thành `client.jsonl`. Chạy `tools/analyze.py --trace <thư_mục_trace> --client <client.jsonl> --output <thư_mục_kết_quả>`. `--cameras` mặc định dùng camera map kèm báo cáo. `tools/correlate.py <thư_mục_kết_quả> <thư_mục_client_actions_gpu_cpu>` dùng `actions.jsonl`, `gpu.log`, `cpu.log`; `tools/source_gaps.py <thư_mục_báo_cáo>` đọc hai trace gzip và tạo bảng PTS.

**Thay đổi code chỉ phục vụ đo:** thêm `backend/core/phase0_baseline.py`, các hook opt-in trong `custom_deepstream_worker.py`, `model_sam2.py`, `custom_detector.py`. Không sửa thuật toán, scheduling/interval, buffer, UI, model hoặc calibration. Khi `control.json` disabled và subprocess mới khởi động, không cài wrapper inference/probe tracer.

**Trạng thái cuối:** đã đặt `backend/data/phase0_baseline/control.json` thành `enabled:false`; dừng loggers/browser thử nghiệm; deployment trở lại **null / idle như trước đo**. Đã xóa đúng row/files của model diagnostic mới sau build, giữ nguyên ba model có sẵn, label và calibration. Backend/frontend dịch vụ vẫn chạy. Muốn tiếp tục xem AI phải Deploy lại model có sẵn; không tự bật một deployment mới sau khi hoàn tất benchmark.

Kiểm tra sau khi tắt tracer: `test_model_sam2_live`, `test_label_monitor_continuity`, `test_metadata_broadcaster`: **24 tests pass**; `py_compile` các file đo/runtime được chạm: pass. Đây không phải kiểm thử chất lượng tracking/IoU.

**Đề nghị quyết định:** không có căn cứ coi encoder TensorRT là giải pháp duy nhất. Nếu cho phép đo tiếp, ưu tiên khoanh sâu gap theo camera trước tracker và phần tuần tự nhiều object/chờ attach; nếu chọn Giai đoạn2, giữ bộ baseline này để so sánh lại cùng camera, số target/decoder và reference cache. **Chưa thực hiện tối ưu nào trong các hướng đó.**
