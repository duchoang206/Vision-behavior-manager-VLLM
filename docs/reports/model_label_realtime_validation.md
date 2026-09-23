# Model / Label Realtime Validation

**Thời gian đo:** 22/09/2026 01:35:26 ICT (21/09/2026 18:35:26 UTC); cửa sổ 15,007 giây, đồng thời mở Monitor bằng Chrome headless.

## Phạm vi

- Model đang deploy: `best(1).onnx`, ID `479ea92c0b6044febba5b31a5e8e5e08`.
- ONNX contract: input `[1,3,640,640]`, output `[1,9,8400]`; runtime đã build và đang chạy TensorRT FP16.
- SHA256 ONNX: `a9f19caf82d0ceb396531ca74a54c8e3cdcf8162f78d73c43782ade840f51e0b`. Kiểm thử chạy engine thực; không tính `onnx.checker` là đạt vì container không có package `onnx`.
- Gallery đã lưu: 75 góc nhìn, gồm `Robot_6868` (44), `Robot_2001` (24), `Robot_28100` (3), `Robot_1` (3), `Rack` (1). Gallery revision 76 được worker xác nhận đã áp dụng, với 10 prompt hình học hiện hành theo camera/nhãn.
- Đã kiểm tra 5 camera được deploy; camera thứ sáu không nằm trong deployment nên không đưa vào kết luận inference.

## Kết quả backend

Trong cửa sổ 15 giây sau khi backend khởi động lại:

| Camera | Metadata | Nhãn có mask / packets | Bằng chứng SAM2 mới | Tuổi bằng chứng p95 |
|---|---:|---:|---:|---:|
| `86c5119c` | 23.66 Hz | `Robot_28100` 100%; `Robot_2001` 96.34%; `Robot_6868` 96.34% | 7.49 Hz | 252 ms |
| `b1269e28` | 19.99 Hz | `Robot_1` 100% | 7.00 Hz | 213 ms |
| `d1f37ed1` | 22.66 Hz | `Robot_1` 91.47% | 6.69 Hz | 211 ms |
| `06e9e5b9` | 22.46 Hz | `Robot_28100` 97.92% | 6.43 Hz | 317 ms |
| `1ac2ffac` | 24.92 Hz | Không có object được xác nhận | — | — |

- Không có lỗi WebSocket, không đóng kết nối bất thường, và nhận đủ 15 heartbeat.
- Độ trễ từ timestamp metadata tại probe tới client WebSocket p95 là 7,79–12,09 ms. Đây không phải độ trễ từ lúc camera chụp tới màn hình.
- Không có frame metadata đi ngược thứ tự. Có bỏ qua một số frame nguồn do cơ chế latest-only; đây là chủ ý để ưu tiên dữ liệu mới và tránh tích trễ.
- Tất cả object được xuất trong cửa sổ đo đều có polygon mask. Tỷ lệ ở trên thấp hơn 100% tại một số camera vì có packet không có object/đang chờ bằng chứng mới, không phải vì frontend biến polygon thành bounding box.
- `Robot_2001` và `Robot_6868` trên `86c5119c` đã được xác nhận bởi gallery và xuất mask từ `sam2_live`; không còn phụ thuộc việc YOLO phải tự phát hiện đúng class ở từng frame đầu tiên.

## Kết quả Monitor

Smoke test bằng Chrome headless trên `/?tab=monitor`:

- 6/6 card camera hiển thị video với `readyState=4`, không có lỗi console/page.
- Callback vẽ theo frame video đạt khoảng 14.7–24.5 Hz tùy nguồn; thời gian callback p95 là 0.1–0.3 ms.
- Thời gian callback chỉ đo JavaScript đồng bộ, không bao gồm toàn bộ GPU raster/compositor. Các số liệu này cho thấy việc tạo mask mới chậm hơn việc gửi metadata/vẽ callback, nhưng không chứng minh toàn bộ độ trễ video nằm ở backend.
- Canvas overlay có pixel mask/label trên các card đang có object; các card không có object xác nhận để canvas trong suốt.
- Đây là tốc độ hiển thị/propagation theo frame video, không phải tốc độ suy luận SAM2 mới.

## Giới hạn cần ghi nhận

- SAM2 hiện chạy CUDA FP16/PyTorch, còn detector/tracker chạy DeepStream TensorRT/NvDCF. Trong tải hiện tại, client nhận các frame mask mới phân biệt khoảng 6.4–7.5 Hz. Đây là tốc độ bằng chứng SAM2 quan sát được ở client, không phải tổng số suy luận backend; các kết quả bị thay thế có thể không tới client. Track detector dùng vị trí NvDCF giữa các lần SAM2; track chỉ từ Label giữ kết quả SAM2 mới nhất trong giới hạn tuổi bằng chứng, không có vị trí NvDCF độc lập. Vì vậy không thể ghi là SAM2 suy luận mới 30 FPS.
- FMS đang báo các robot `2001`, `6868`, `28100` ở trạng thái `OFFLINE`. Cơ chế an toàn vẫn trả `fms_not_synchronized`, nên phép kiểm tra nhận dạng robot theo vị trí FMS chưa thể coi là đã xác nhận trên robot đang di chuyển. Không dùng lại vị trí offline để ép nhận dạng.
- Chưa có video ground-truth robot di chuyển và nhãn đúng/sai để tính precision/recall hoặc chứng minh bám 100% khi che khuất. Kết quả trên chứng minh luồng runtime, đồng bộ gallery và hiển thị mask thực tế; không thay thế một bài đánh giá accuracy có ground-truth.
- `Rack` có mẫu đã lưu nhưng không xuất hiện như object được xác nhận trong cửa sổ đo; chưa kết luận accuracy của nhãn này. Sau khi mất dấu, prompt thử lại tại tọa độ đã lưu, chưa bảo đảm tìm lại robot đã đi xa vị trí đó.

## Kiểm thử hồi quy

- `py_compile` thành công cho các module đồng bộ Label, SAM2, mask cache, worker và router đã sửa.
- 75 test thành công: `test_model_labels.py` (27), `test_model_sam2_live.py` (3), `test_model_track_masks.py` (10), `test_robot_spatial_identity.py` (12), `test_registered_mask_recovery.py` (14), `test_model_label_prompts.py` (9).
- Test mới kiểm tra tọa độ không tự tạo mask giả, nhầm identity bị loại, mask hết hạn không được gia hạn timestamp, xóa nhãn vô hiệu hóa prompt, raw YOLO không chặn prompt và reload xác nhận trước khi background training hoàn tất.
- Phép thử reload dùng lại giá trị `require_labels=false` hiện có; chỉ tăng revision lên 76, không thêm/xóa mẫu, không đổi trọng số ONNX/TensorRT, không deploy thêm camera và không bật pose.

## Kết luận

Tọa độ/bbox/mask của các Label đã lưu được đưa vào gallery, reload vào worker GPU ngay sau khi lưu và được dùng làm prompt trên frame DeepStream hiện tại. Monitor nhận metadata liên tục và vẽ mask trên canvas theo video. Luồng hiện không bị mất kết nối; giới hạn chính còn lại là tốc độ bằng chứng SAM2 mới và việc FMS chưa có robot online để kiểm tra spatial identity động.
