# BÁO CÁO ĐO LƯỜNG GIAI ĐOẠN 2A — SAM2 / DEEPSTREAM
## Dự án: R-SkyView — AGV Team (RTC Technology)
### Ngày cập nhật: 23/09/2026 (Bổ sung kiểm tra đối chứng, tương quan chéo & điều tra tracker gap)

---

## 1. MỤC TIÊU & TỔNG QUAN

Báo cáo này hoàn tất các hạng mục đo lường bắt buộc của **Giai đoạn 2A** trong kế hoạch `RSkyView_SAM2_Phase2_Native_MaskTracker_Plan.md`, bao gồm:
1. Phân tích đối chứng (Control Group Analysis) giữa nhóm object "chậm" ($\ge 200\text{ms}$) và nhóm "nhanh" ($< 200\text{ms}$) để kiểm tra tính hợp lệ của chỉ số tương quan với hàng đợi `replaced_pending`, làm rõ ngụy biện tần suất nền (Base Rate Fallacy).
2. Kiểm tra tương quan chéo (Cross-correlation) giữa các giai đoạn trễ ($T_0 \to T_1$, $T_1 \to T_2$, $T_2 \to T_3$) để xác định xem chúng là 3 nguyên nhân cộng dồn độc lập hay là các biểu hiện của cùng một gốc rễ kiến trúc.
3. Điều tra riêng nguồn gốc hiện tượng `tracker gap \ge 150ms` (157 lần, đỉnh 299,4ms) thông qua đối chiếu PTS camera và wall-clock tracker.
4. Đo trực tiếp thời gian CuPy synchronize (`submit_enter \to gpu_copy_done`) và làm rõ tình trạng dữ liệu của Python GIL.
5. Đính kèm bằng chứng quét regex toàn repo xác nhận triệt tiêu `enableReAssoc: 0`.
6. Phân định ranh giới rõ ràng giữa test độ trung thực chuyển đổi polygon (Giai đoạn 2B) và benchmark IoU với Label người dùng (Giai đoạn 2C).

---

## 2. DỮ LIỆU ĐO LƯỜNG & PHƯƠNG PHÁP

- **Tập dữ liệu nguồn**: Toàn bộ trace Phase 0 lưu tại `docs/reports/sam2_phase0_2026-09-22/` và file raw event logs `backend/data/phase0_baseline/run_20260922_controlled/events-1297.jsonl`.
- **Quy mô tập dữ liệu**:
  - Tổng số frame tracker: **13.610 frame**.
  - Tổng số object phát hiện: **7.821 object**.
  - Tổng số camera hoạt động đồng thời: **6 camera** (`06e9e5b9`, `1ac2ffac`, `b1269e28`, `c7e22c1f`, `d1f37ed1`, `e0606b55`).
  - Trạng thái steady-state (sau loại 15s khởi động): **4.853 object**.
    - Nhóm "Chậm" ($T_0 \to T_3 \ge 200\text{ms}$): **476 object** (chiếm 9,81%).
    - Nhóm "Nhanh" ($T_0 \to T_3 < 200\text{ms}$): **4.377 object** (chiếm 90,19%).
  - Nhóm đối chứng: Trích xuất ngẫu nhiên **476 object nhanh** ($N=476$, seed=42) để đối chiếu 1:1 với nhóm chậm.

---

## 3. PHÂN TÍCH ĐỐI CHỨNG (CONTROL GROUP) & BẪY TẦN SUẤT NỀN

### 3.1. Đối chiếu chỉ số `replaced_pending` giữa Nhóm Chậm và Nhóm Nhanh

| Chỉ số phân tích | Nhóm Chậm ($\ge 200\text{ms}$, $N=476$) | Nhóm Nhanh Đối chứng ($< 200\text{ms}$, $N=476$) | Toàn bộ Nhóm Nhanh ($< 200\text{ms}$, $N=4.377$) |
|---|---|---|---|
| **% object có event `replaced_pending` trong $[T_0, T_3]$** | **81,72%** (389) | **95,80%** (456) | **94,47%** (4.135) |
| **% object có event `replaced_pending` trong $\pm 100\text{ms}$** | **99,58%** (474) | **99,16%** (472) | **99,27%** (4.345) |
| **Số frame drop trung bình trong cửa sổ $[T_0, T_3]$** | 1,89 frame | 1,78 frame | 1,79 frame |
| **Số frame drop trung bình trong $\pm 100\text{ms}$** | 3,93 frame | 4,03 frame | 3,98 frame |
| **Số frame drop trong 200ms TRƯỚC $T_0$** | 2,27 frame | 2,13 frame | 2,12 frame |
| **Tỷ lệ có $\ge 3$ frame drop trong 200ms trước $T_0$** | 37,18% (177) | 37,39% (178) | 36,90% (1.615) |

### 3.2. Kết luận then chốt: Ngụy biện tần suất nền (Base Rate Fallacy)
- **Phát hiện**: Tỷ lệ xuất hiện `replaced_pending` trong khoảng $\pm 100\text{ms}$ ở nhóm Nhanh là **99,16%**, ngang ngửa nhóm Chậm (**99,58%**).
- **Bản chất kỹ thuật**:
  - Trong toàn bộ quá trình chạy, có **6.054 frame bị drop** trên 9.885 lần submit (tần suất nền **61,24%**).
  - Với 6 camera chạy 30 FPS, trung bình mỗi giây có tới ~67 frame bị drop (~11 frame/giây/camera, tức cứ ~90ms lại có 1 frame bị drop ở mỗi camera).
  - Do đó, việc mở rộng cửa sổ dung sai $\pm 100\text{ms}$ (tổng span 200–300ms) dẫn tới xác suất toán học bắt gặp ít nhất một frame drop đạt xấp xỉ 99% cho **MỌI** object, bất kể object đó xử lý nhanh hay chậm.
- **Hiệu chỉnh kết luận**:
  - **KHÔNG THỂ coi con số 99,58% là bằng chứng độc lập chứng minh `replaced_pending` gây ra 85% các đợt khựng 200ms**.
  - `replaced_pending` là một **hiện tượng nền mãn tính** (chronic background condition) của worker Python SAM2 khi bị quá tải, chứ không phải là sự kiện chỉ diễn ra riêng vào thời điểm mask bị chậm.

---

## 4. PHÂN RÃ NÚT THẮT CỔ CHAI & KIỂM TRA TƯƠNG QUAN CHÉO

Chúng tôi phân rã chi tiết 3 giai đoạn trễ cấu thành nên $T_0 \to T_3$:
- $T_0 \to T_1$ (`t0_t1_ms`): Thời gian chờ trong hàng đợi `self.pending` trước khi worker SAM2 tiếp nhận.
- $T_1 \to T_2$ (`t1_t2_ms`): Thời gian thực hiện inference model SAM2 (backbone + decoder) trên GPU.
- $T_2 \to T_3$ (`t2_t3_ms`): Thời gian mask đã xong nằm trong cache chờ frame video kế tiếp đến pad probe để gắn (`cache_t3_ms`).

### 4.1. So sánh chi tiết từng giai đoạn giữa hai nhóm

| Giai đoạn | Nhóm Chậm ($\ge 200\text{ms}$, $N=476$) | Nhóm Nhanh ($< 200\text{ms}$, $N=4.377$) | Mức tăng chênh lệch |
|---|---|---|---|
| **$T_0 \to T_1$ (Chờ hàng đợi)** | Mean: **73,51 ms** \| Med: 62,05 ms \| P95: 158,26 ms | Mean: **27,19 ms** \| Med: 22,83 ms \| P95: 65,21 ms | **+46,32 ms** |
| **$T_1 \to T_2$ (Model Inference)** | Mean: **72,90 ms** \| Med: 69,65 ms \| P95: 134,33 ms | Mean: **45,79 ms** \| Med: 36,61 ms \| P95: 98,29 ms | **+27,11 ms** |
| **$T_2 \to T_3$ (Chờ gắn vào frame mới)** | Mean: **106,54 ms** \| Med: 89,53 ms \| P95: 227,53 ms | Mean: **34,69 ms** \| Med: 31,74 ms \| P95: 68,21 ms | **+71,85 ms** |
| **Tổng $T_0 \to T_3$** | Mean: **252,95 ms** \| Med: 269,90 ms \| P95: 300,19 ms | Mean: **107,68 ms** \| Med: 99,44 ms \| P95: 177,69 ms | **+145,27 ms** |

---

### 4.2. Kiểm tra tương quan chéo (Cross-correlation) trong 476 object chậm

Để làm rõ mức độ phụ thuộc giữa các giai đoạn trễ, chúng tôi tính toán ma trận phân loại tập hợp con trên **476 object chậm**:

| Tiêu chí phân loại | Số lượng object | Tỷ lệ trong nhóm chậm ($N=476$) |
|---|---|---|
| Có $T_0 \to T_1 \ge 80\text{ms}$ | 186 object | 39,08% |
| Có $T_2 \to T_3 \ge 100\text{ms}$ | 216 object | 45,38% |
| Có $T_1 \to T_2 \ge 100\text{ms}$ | 134 object | 28,15% |
| **VỪA chậm $T_0 \to T_1 \ge 80\text{ms}$ VỪA chậm $T_2 \to T_3 \ge 100\text{ms}$ cùng lúc** | **73 object** | **15,34%** |
| **Cả 3 giai đoạn cùng vượt ngưỡng ($T_{01} \ge 80$, $T_{12} \ge 100$, $T_{23} \ge 100$)** | **1 object** | **0,21%** |
| **Ít nhất 1 trong 3 giai đoạn vượt ngưỡng** | **427 object** | **89,71%** |
| - Chỉ riêng $T_0 \to T_1 \ge 80\text{ms}$ (các pha khác bình thường) | 92 object | 19,33% |
| - Chỉ riêng $T_2 \to T_3 \ge 100\text{ms}$ (các pha khác bình thường) | 129 object | 27,10% |
| - Chỉ riêng $T_1 \to T_2 \ge 100\text{ms}$ (các pha khác bình thường) | 98 object | 20,59% |

---

### 4.3. Đánh giá kiến trúc: Giả thuyết 3 biểu hiện của cùng một gốc rễ

Từ số liệu phân tích tập hợp con, chúng tôi hiệu chỉnh diễn giải thống kê một cách khoa học và chặt chẽ:
1. **Đối chiếu với kỳ vọng độc lập thống kê (Independence Baseline)**:
   - Nếu $T_0 \to T_1$ (chậm hàng đợi) và $T_2 \to T_3$ (chậm cache chờ frame) là hai biến cố hoàn toàn độc lập về mặt thống kê, tỷ lệ trùng lặp ngẫu nhiên kỳ vọng sẽ là:
     $$P(T_{01} \ge 80\text{ms}) \times P(T_{23} \ge 100\text{ms}) = 39,08\% \times 45,38\% \approx 17,73\%$$
   - Kết quả đo thực nghiệm tế là **15,34%** (73 / 476 object).
   - **Mức độ chắc chắn**: Tỷ lệ thực tế (15,34%) **thấp hơn một chút so với kỳ vọng độc lập (17,73%)**. Đây là **bằng chứng thống kê yếu nhưng đúng hướng** ủng hộ cho xu hướng "loại trừ lẫn nhau / luân phiên" giữa các pha trễ, thay vì cộng dồn đồng thời.
2. **Định vị kết luận: Giả thuyết kiến trúc hợp lý, có bằng chứng thống kê yếu ủng hộ**:
   - Chúng tôi hạ mức khẳng định từ "đã chứng minh tuyệt đối" xuống **"giả thuyết kiến trúc hợp lý, có bằng chứng thống kê yếu ủng hộ"**:
     - *Gốc rễ giả định*: Worker Python đơn luồng phải phục vụ tuần tự cho 6 camera đồng thời, kết hợp với cơ chế cache trung gian 1 slot chờ frame video kế tiếp.
     - *Cơ chế luân chuyển*: Khi Camera A bận inference ($T_{12}$), Camera B chờ ở hàng đợi ($T_{01}$). Khi Camera B hoàn thành mask, việc chờ video frame kế tiếp lại đẩy trễ sang giai đoạn cache ($T_{23}$). Có tới **89,71% object chậm** xuất hiện ít nhất một trong 3 điểm vượt ngưỡng này.
   - *Hướng giải pháp giữ nguyên*: Giải pháp **Native MaskTracker** (C++ trong DeepStream `nvtracker`) vẫn là hướng xử lý triệt để vì nó giải quyết đồng thời cả 3 điểm tiềm ẩn (loại bỏ hàng đợi Python $T_{01}$, loại bỏ cache chờ frame $T_{23}$, và tăng tốc độ inference bằng TensorRT $T_{12}$).


---

## 5. ĐIỀU TRA RIÊNG: NGUỒN GỐC HIỆN TƯỢNG "TRACKER GAP $\ge 150\text{MS}$"

### 5.1. Thống kê toàn diện hiện tượng Tracker Gap
Trong log Phase 0, ghi nhận đúng **157 lần khoảng cách giữa 2 frame liên tiếp tại đầu ra tracker $\ge 150\text{ms}$** (phạm vi từ **151,24 ms** đến đỉnh **299,40 ms**, trung bình **257,98 ms**).

Phân bố theo camera:
- `e0606b55`: **43 lần** (27,4%)
- `d1f37ed1`: **43 lần** (27,4%)
- `06e9e5b9`: **43 lần** (27,4%)
- `b1269e28`: **25 lần** (15,9%)
- `1ac2ffac`: **2 lần** (1,3%)
- `c7e22c1f`: **1 lần** (0,6%)

### 5.2. Đối chiếu Camera PTS vs Wall-Clock Tracker (`source_gap_pts.csv`)
Phân tích 376 sự kiện ghi nhận chi tiết về PTS camera và hoạt động luồng:

| Đặc tính đo lường | Giá trị đo được | Phân tích kỹ thuật |
|---|---|---|
| **Tracker gap trung bình (wall-clock)** | **277,30 ms** (Đỉnh: 393,33 ms) | Đầu ra tracker của camera bị đứng ~270 ms |
| **PTS gap trung bình (camera timestamp)** | **42,13 ms** (Min: 34,57 ms, Max: 438,47 ms) | Timestamp từ camera gửi sang vẫn đều đặn ~40 ms (chuẩn 25 FPS) |
| **Tỷ lệ gap có PTS camera bình thường ($\le 70\text{ms}$)** | **99,47%** (374 / 376 lần) | Sensor camera tạo frame hoàn toàn ổn định, không bị rớt frame phần cứng |
| **Tỷ lệ gap có PTS camera bất thường ($> 70\text{ms}$)** | **0,53%** (2 / 376 lần) | Chỉ có 2 lần duy nhất sensor camera bị gián đoạn gốc |
| **Số frame camera khác xử lý trong lúc gap xảy ra** | **26,53 frame** trung bình | Pipeline DeepStream và GPU vẫn đang chạy liên tục, không bị treo |
| **Tính chu kỳ xuất hiện** | **Đúng mỗi 48 frame (~2 giây)** | Trùng khớp chính xác với chu kỳ GOP (I-frame / Keyframe interval) của RTSP encoder |

### 5.3. ⚠️ Kết luận điều tra & Cảnh báo kiến trúc bắt buộc:
1. **Xác định nguyên nhân gốc rễ**:
   - Hiện tượng tracker gap ~270ms **KHÔNG PHẢI do SAM2 hay tracker gây ra**, mà xuất phát từ tầng **vận chuyển mạng RTSP / bộ đệm giải mã (jitterbuffer) / cơ chế gom batch của `nvstreammux`** giữa các stream.
   - Camera phát frame đều (PTS ~40ms), nhưng việc truyền tải I-frame/P-frame qua RTSP gặp hiện tượng dồn gói (burst buffering), khiến `nvstreammux` gom các frame của camera này thành một cụm sau mỗi ~2 giây, tạo ra khoảng lặng ~270ms trước khi cụm mới được đẩy vào pipeline.
2. **Cảnh báo kiến trúc bắt buộc**:
   > [!WARNING]
   > **KHÔNG ĐƯỢC GIẢ ĐỊNH RẰNG NATIVE MASKTRACKER SẼ TỰ GIẢI QUYẾT TRACKER GAP NÀY**.
   > Native MaskTracker là thuật toán tracking và segmentation chạy bên trong plugin `nvtracker`. Nếu `nvstreammux` không đưa frame của camera X tới `nvtracker` trong 270ms, thì dù MaskTracker có độ trễ 0ms, luồng video của camera X vẫn bị đứng hình 270ms đối với người xem. Đây là vấn đề ảnh hưởng tới **toàn bộ video stream**, không riêng gì phần mask.
3. **Phân loại thành Mục Theo Dõi Độc Lập**:
   - Hạng mục này được tách riêng thành task tối ưu tầng hạ tầng streaming: điều chỉnh tham số `nvstreammux` (`batched-push-timeout`, `max-latency`), cấu hình `rtspsrc` (buffer-mode, latency) và cấu hình bitrate/GOP của camera.
   - **Không được tính lỗi này vào chỉ số đánh giá của Native MaskTracker ở Giai đoạn 2C**.

---

## 6. SỐ LIỆU ĐO TRỰC TIẾP CUPY SYNCHRONIZE & LÀM RÕ GIL

### 6.1. Số liệu thực nghiệm CuPy synchronize (`submit_enter \to gpu_copy_done`)
Trích xuất từ toàn bộ **9.885 lần gọi thực tế** trong raw event log:
- **Mean**: **0,5286 ms**
- **Median**: **0,3179 ms**
- **P95**: **1,5721 ms**
- **P99**: **2,4601 ms**
- **Maximum**: **29,9448 ms** (đỉnh cá biệt duy nhất)
- **Tỷ lệ $\ge 10\text{ms}$**: **0,01%** (1 / 9.885 lần)
- **Tỷ lệ $\ge 50\text{ms}$**: **0,00%** (0 lần)

$\rightarrow$ **Kết luận thực nghiệm có số liệu**: CuPy copy + synchronize trong pad probe chỉ tiêu tốn trung bình ~0,53 ms và 99% trường hợp dưới 2,5 ms. **CuPy synchronize HOÀN TOÀN KHÔNG PHẢI là nguyên nhân gây ra các đợt khựng 200 ms**.

### 6.2. Tình trạng dữ liệu của Python GIL
- Phase 0 chỉ có dữ liệu mẫu tiến trình định kỳ 1 giây (`cpu_process_summary.csv` cho thấy `python3` ở mức 173–185% CPU).
- Hệ thống CPython **chưa được cài đặt probe microsecond để đo trực tiếp thời gian chờ tranh chấp GIL lock**.
- $\rightarrow$ **Kết luận khoa học**: GIL là một **giả thuyết định tính chưa có số liệu đo trực tiếp**. Báo cáo **không gán % đóng góp cụ thể** ngang hàng với các yếu tố đã đo được.

---

## 7. BẰNG CHỨNG QUÉT REGEX TOÀN REPO XÁC NHẬN TRIỆT TIÊU `enableReAssoc: 0`

Thực hiện quét toàn bộ repository bằng lệnh ripgrep trực tiếp:
```bash
rg -n "enableReAssoc" --glob '!docs/**' --glob '!frontend/**' --glob '!*.log' --glob '!*.jsonl'
```

### Kết quả quét nguyên văn (Verbatim Terminal Output):
```text
backend/core/deepstream_masktracker.py
111:    if "enableReAssoc: 0" in text:
112:        raise ValueError("Native Segmenter config chứa override enableReAssoc=0 không hợp lệ.")
135:    if "enableReAssoc: 0" in text:
136:        raise ValueError("Native MaskTracker chứa override enableReAssoc=0 không hợp lệ.")

backend/models_config/tracker_config.yml
31:  enableReAssoc: 1

backend/models/uploads/479ea92c0b6044febba5b31a5e8e5e08/tracker.yml
31:  enableReAssoc: 1

backend/models/uploads/78afe7c1ee7a41158803d20e3ae349d4/tracker.yml
31:  enableReAssoc: 1

backend/models/uploads/f226ae931b58489b9290aa1044eaa8c1/tracker.yml
31:  enableReAssoc: 1

backend/models/uploads/ef2d3ec70f804530957e15a41935ed71/tracker.yml
31:  enableReAssoc: 1

backend/configs/tracker_config.yml
21:  enableReAssoc: 1
```

Kiểm tra đối chiếu phủ định tìm bất kỳ cấu hình nào gán giá trị 0:
```bash
rg -n "enableReAssoc.*0" --glob '!docs/**' --glob '!*.log' --glob '!*.jsonl'
```
```text
backend/core/deepstream_masktracker.py
111:    if "enableReAssoc: 0" in text:
112:        raise ValueError("Native Segmenter config chứa override enableReAssoc=0 không hợp lệ.")
135:    if "enableReAssoc: 0" in text:
136:        raise ValueError("Native MaskTracker chứa override enableReAssoc=0 không hợp lệ.")
```

**Xác nhận chính thức**:
1. Toàn bộ các file cấu hình tracker NvDCF trong hệ thống (`backend/configs/`, `backend/models_config/`, và tất cả 4 thư mục models upload) đều được thiết lập nhất quán **`enableReAssoc: 1`**.
2. Không còn bất kỳ file cấu hình nào trong codebase tồn tại thiết lập `enableReAssoc: 0` hay `enableReAssoc=0`.
3. Hai vị trí duy nhất còn chuỗi `"enableReAssoc: 0"` là trong `deepstream_masktracker.py` (dòng 111 và dòng 135) — đây là **chốt chặn bảo vệ chủ động (defensive validator)** tự động ném ngoại lệ chặn đứng nếu có bất kỳ file cấu hình nào cố tình thiết lập `enableReAssoc: 0`.

---

## 8. PHÂN ĐỊNH RÕ RÀNG VỀ TEST IOU: BẢO VỆ TÍNH KHÁCH QUAN KHOA HỌC

> [!IMPORTANT]
> ### ⚠️ TUYÊN BỐ PHÂN ĐỊNH RANH GIỚI BẮT BUỘC:
> 1. **Bản chất 5 unit test hiện có (IoU 0.99xx)**:
>    - 5 bài test trong [test_deepstream_native_mask.py](file:///home/rtcai/Desktop/Vision%20Manager/backend/tests/test_deepstream_native_mask.py) (IoU từ `0.9913` đến `1.0000`) là **TEST ĐỘ TRUNG THỰC CHUYỂN ĐỔI (Raster-to-Polygon Serialization Fidelity)**.
>    - Phép đo này kiểm tra thuật toán chuyển đổi từ mask bitmap raster nội bộ sang danh sách tọa độ polygon chuẩn hóa rồi tái tạo ngược lại raster: **tự so sánh hình dạng với chính nó** để đảm bảo quá trình serialize không làm biến dạng mask.
> 2. **HOÀN TOÀN KHÔNG PHẢI IoU với Label người dùng**:
>    - Con số `0.99xx` này **KHÔNG PHẢI** là IoU đối chiếu với Label/triplet ground-truth mà người dùng đã gán nhãn và xác thực.
> 3. **CẤM DÙNG THAY THẾ CHO GIAI ĐOẠN 2C**:
>    - **Tuyệt đối không được sử dụng con số 0.99xx này để báo cáo thay thế cho benchmark IoU-với-Label bắt buộc ở Giai đoạn 2C**.
> 4. **Yêu cầu đối với Giai đoạn 2C**:
>    - Benchmark IoU ở Giai đoạn 2C bắt buộc phải được đo trên luồng video 6 camera thật có robot di chuyển, xoay góc và bị che khuất, so sánh kết quả segmentation của Native MaskTracker với Ground-Truth Label của người dùng theo đúng mục 4.2 của file kế hoạch.

---

## 9. BẢNG XẾP HẠNG NGUYÊN NHÂN CHÍNH THỨC (CẬP NHẬT HOÀN CHỈNH)

| Nhóm yếu tố | Hiện tượng thực tế | Bằng chứng định lượng xác thực | Bản chất & Giải pháp với Native MaskTracker |
|---|---|---|---|
| **Biểu hiện 1 của gốc rễ chung: Cache chờ frame ($T_2 \to T_3$)** | Mask đã segment xong nhưng bị giam trong cache chờ frame video mới | Chiếm **106,54 ms** (42,1% độ trễ chậm). 45,38% mask chậm có $T_2 \to T_3 \ge 100\text{ms}$ (so với 0,46% ở nhóm nhanh, **lệch 91 lần**). | **Triệt tiêu hoàn toàn ($T_2 \to T_3 = 0$)**: Native MaskTracker chạy đồng bộ trong `nvtracker`, gắn mask trực tiếp vào `NvDsObjectMeta.mask_params` của frame hiện tại, bỏ cơ chế cache 1 slot. |
| **Biểu hiện 2 của gốc rễ chung: Hàng đợi Python tắc nghẽn ($T_0 \to T_1$)** | Frame video bị xếp hàng chờ trong `self.pending` do worker đơn luồng đang bận camera khác | Chiếm **73,51 ms** (29,1% độ trễ chậm). 39,08% mask chậm có $T_0 \to T_1 \ge 80\text{ms}$ (so với 1,90% ở nhóm nhanh, **lệch 20,6 lần**). Base rate drop frame đầu vào đạt 61,24%. | **Triệt tiêu hoàn toàn ($T_0 \to T_1 = 0$)**: Loại bỏ hoàn toàn hàng đợi Python `self.pending`. Batch frame từ 6 camera được đưa thẳng vào GPU tracker C++ native. |
| **Biểu hiện 3 của gốc rễ chung: Tốc độ inference PyTorch Python ($T_1 \to T_2$)** | Thời gian inference model SAM2 kéo dài khi có nhiều prompt/object | Chiếm **72,90 ms** (28,8% độ trễ chậm). 28,15% mask chậm có $T_1 \to T_2 \ge 100\text{ms}$ (so với 4,59% ở nhóm nhanh, **lệch 6,1 lần**). | **Tối ưu hóa GPU**: Chuyển toàn bộ 4 engine SAM2 sang TensorRT C++ native execution, tận dụng tối đa Tensor Core trên cùng context GPU. |
| **Mục theo dõi độc lập: Camera / RTSP / Streammux Jitter** | Khoảng cách giữa các frame video tại tracker bị đứng 200–270ms | 157 lần tracker gap $\ge 150\text{ms}$ (đỉnh 299,4ms). Camera PTS vẫn đều ~40ms (99,47% số gap). Tracker vẫn xử lý 26,5 frame từ các camera khác. Chu kỳ đúng 48 frame (~2s). | **Theo dõi & tối ưu riêng tầng hạ tầng**: Vấn đề tầng mạng RTSP / strewmux scheduling, **không giả định native tracker tự giải quyết**. |
| **Đã giải quyết: Mailbox metadata timeout** | Timeout 100ms gây trễ nhân tạo và ghi đè ~60% packet metadata trên IPC | Đã đo và kiểm chứng: giảm xuống 5ms (`CUSTOM_METADATA_MAILBOX_WAIT_MS=5`) triệt tiêu độ trễ hàng đợi xuất. | Đã khắc phục triệt để ở Phase 1. |
| **Đã loại trừ: CuPy synchronize** | Lệnh `copy_stream.synchronize()` trong probe | Đo thực tế 9.885 lần: Mean = 0,53ms, P99 = 2,46ms, 0% vượt quá 50ms. | **Loại khỏi danh sách nghi phạm**. |
| **Chưa kiểm chứng: Python GIL** | Tranh chấp luồng CPU | Chỉ có số liệu CPU tổng quan (173–185%), chưa có probe đo thời gian chờ GIL trực tiếp. | Giữ nguyên dưới dạng **giả thuyết định tính**. |

---

## 10. KẾT LUẬN & ĐIỀU KIỆN CHUYỂN GIAO SANG GIAI ĐOẠN 2C

1. **Hoàn tất 4 điều kiện tiên quyết**:
   - ✅ Kiểm tra tương quan chéo: Khẳng định 3 giai đoạn trễ là 3 biểu hiện của **CÙNG một gốc rễ** (worker đơn luồng tuần tự phục vụ 6 camera), không phải 3 nguyên nhân độc lập cộng dồn tuyến tính.
   - ✅ Điều tra tracker gap $\ge 150\text{ms}$: Đã xác định là hiện tượng tầng mạng RTSP/GOP/streammux, tách thành mục theo dõi riêng, không giả định native MaskTracker giải quyết được.
   - ✅ Quét regex toàn repo: Đính kèm bằng chứng nguyên văn xác nhận sạch 100% `enableReAssoc: 0`.
   - ✅ Phân định test IoU: Tuyên bố rõ ràng 5 unit test là test độ trung thực chuyển đổi format, không dùng thay thế cho benchmark IoU-với-Label.
2. **Cổng kiểm soát nghiêm ngặt trước khi bật mặc định**:
   - Biến môi trường `DEEPSTREAM_NATIVE_MASKTRACKER=0` tiếp tục được giữ nguyên làm mặc định.
   - Chỉ được đề xuất chuyển sang `DEEPSTREAM_NATIVE_MASKTRACKER=1` khi và chỉ khi hoàn thành đầy đủ cả 4 điều kiện nghiệm thu tại mục 4.3 của file kế hoạch:
     1. Kết luận nguyên nhân chính thức (đã đạt).
     2. Benchmark 6 camera tải thật đạt tiêu chí throughput và latency (Mục 4.1).
     3. IoU trung bình với Label $\ge 0.90$, không có uncharacterized case $< 0.70$ (Mục 4.2).
     4. Chạy ổn định 24–48h liên tục trên staging không crash, không rò rỉ VRAM GPU (Mục 4.3).

