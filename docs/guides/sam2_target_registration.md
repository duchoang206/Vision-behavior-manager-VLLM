# Hướng dẫn đăng ký nhãn vật thể với SAM 2 (Target Registration Guide)

## 1. Cơ chế hoạt động
Hệ thống sử dụng checkpoint **Segment Anything 2 (SAM 2)** dựng sẵn kết hợp bộ nhớ đặc trưng (Visual Memory Bank) để theo dõi các đối tượng robot hoặc kệ hàng đặc thù mà không cần huấn luyện lại mô hình (Zero-shot Fine-grained Tracking).

## 2. Các bước thực hiện trên giao diện
1. Mở tab **Building** trên Web Dashboard.
2. Chọn Camera tương ứng trong danh sách.
3. Chọn danh mục `robot` hoặc `rack`, nhập tên định danh cho nhãn (VD: `AGV_ECO_01`).
4. Dùng chuột kéo khung bao quanh vật thể.
5. Nhấn **Tạo mask** (Generate Mask).
6. Sử dụng công cụ tương tác:
   - **[+] Điểm vật (Foreground)**: Nhấp chuột trái vào các vị trí thuộc vật thể cần lấy thêm.
   - **[-] Điểm nền (Background)**: Nhấp chuột phải vào các vùng nền bị nhận diện nhầm để loại bỏ.
7. Nhấn **Lưu góc nhìn (Save View)**.
8. Nên chụp và lưu thêm 2-4 góc nhìn khác nhau khi vật thể di chuyển để nâng cao độ tin cậy của thuật toán bám vết.

## 3. Bộ nhớ live và tự tìm lại vật thể

- Mẫu đăng ký được lưu riêng với bộ nhớ bám vết. Mất dấu không xóa nhãn hoặc các góc nhìn đã lưu.
- Sau khi xác minh danh tính và hình học mask, hệ thống cập nhật bộ nhớ SAM2 từ frame live. Bộ nhớ có giới hạn để tránh tăng VRAM theo thời gian; không tự thêm frame dự đoán vào tập mẫu đăng ký.
- Sau nhiều frame không có mask, bộ nhớ bám vết được khởi tạo lại. Hệ thống thử vị trí gần nhất rồi tìm ứng viên bằng đặc trưng SAM2 trên GPU. Ứng viên phải vượt kiểm tra danh tính trước khi được hiển thị.
- Mỗi camera chỉ thử một nhãn đang mất dấu trong một lượt, có thời gian nghỉ giữa các lần thử. Các nhãn đang bám được vẫn tiếp tục xử lý.
- Mask được truyền chuyển động giữa các lần suy luận trong thời gian giới hạn. Khi xác nhận sai danh tính hoặc mất dấu, mask cũ bị loại bỏ, không giữ bóng vật thể vô hạn.

Các tham số backend: `REGISTERED_MASK_LIVE_MEMORIES` (mặc định 3), `REGISTERED_MASK_RESET_MISSES` (3), `REGISTERED_MASK_RECOVERY_INTERVAL` (0.6 giây).

Xem `/api/debug/pipeline` → `template_identity_state` → camera → `mask_tracking.camera.targets`: `state`, `memory_updates`, `memory_count`, `recoveries`, `recovery_attempts`, `last_rejection`. `last_observed_labels` chỉ đếm nhãn thực sự trả về mask. `searching` nghĩa là vẫn giữ đăng ký và đang tìm lại, không phải nhãn bị xóa. Không thể bảo đảm mask khi vật ra khỏi hình, bị che hoàn toàn hoặc chưa đủ đặc trưng để phân biệt với vật khác.

## 4. Ràng buộc robot với FMS

- Nhãn dạng `Robot_2001` được quy đổi cố định thành robot FMS `2001`; không chọn robot gần nhất để thay thế định danh này.
- Với camera có calibration lưu ở tọa độ `fms_floor_metric`, backend chiếu điểm đáy giữa của bbox qua homography rồi so với pose FMS đúng timestamp của frame. Pose được nội suy từ lịch sử FMS để giảm lệch thời gian.
- Nếu vị trí không khớp, robot cạnh tranh gần hơn, calibration chưa hợp lệ hoặc FMS không đồng bộ, mask/metadata bị từ chối hoặc đánh dấu `unchecked`; backend không dùng pose cũ để giả nhận robot khác.
- Tham số mặc định: `REGISTERED_FMS_MAX_DISTANCE_M=1.5`, `REGISTERED_FMS_RIVAL_MARGIN_M=0.35`, `REGISTERED_FMS_MAX_SKEW_SEC=0.5`. Có thể chỉnh qua environment theo sai số lắp camera và kích thước robot.

## 5. Triplet loss cho nhãn đã đăng ký

- Mọi mẫu có vector `reid_512` hoặc `clip_512` đều vào metric triplet riêng theo encoder; không trộn hai không gian vector khác nhau.
- Mẫu SAM2 dùng metric mask riêng trong worker SAM2, cũng sử dụng batch-hard triplet loss để kéo các góc nhìn cùng nhãn lại gần và đẩy nhãn khác ra xa.
- Một nhãn cần ít nhất hai góc nhìn và phải có nhãn khác làm negative thì mới chuyển sang trạng thái `trained`. Khi chưa đủ mẫu, hệ thống không tự nhận là đã học và vẫn dùng kiểm tra gallery/điều kiện an toàn hiện có.
- Theo dõi trạng thái tại `/api/registry/mask/status`: `identity` là metric SAM2; `identity_metrics` là các encoder Re-ID/CLIP. `positive_labels`, `waiting_for_more_views`, `revision` và `last_error` cho biết chất lượng học hiện tại.
