# Camera mới và nhãn đa camera

- Camera được thêm/cập nhật qua Building được đăng ký path MediaMTX theo `cam_id`. Thay đổi nguồn dùng PATCH; path không đổi không bị tạo lại/ngắt video.
- Tracker nhãn chỉ bắt đầu khi camera có nhãn hợp lệ, đọc frame mới nhất từ `MEDIAMTX_RTSP_ORIGIN/<cam_id>` và tự kết nối lại khi mất luồng. Khởi động backend sẽ khôi phục các mẫu đã lưu của từng camera.
- Một nhãn chính xác (ví dụ `Robot_2001`, `Rack_A`, `Person_A`) biểu thị một thực thể và một `global_id` chung. Không dùng cùng nhãn cho hai vật/người khác nhau.
- Mỗi camera giữ mẫu ảnh, bbox và mask riêng. Gán cùng nhãn ở camera khác không sao chép tọa độ hay mask từ camera cũ. Xóa mẫu trên một camera không xóa các góc nhìn ở camera khác.
- Nhãn người chỉ được gắn khi mẫu đã bám được người và trùng với phát hiện người hiện tại; vẫn dùng pose/track người của detector, không tạo thêm người chỉ từ nhãn đã lưu.
- Digital Twin ưu tiên nhãn đã xác nhận, không để kết quả so khớp lại ghi đè nhãn đó. Tọa độ bản đồ vẫn phụ thuộc hiệu chỉnh của từng camera; FMS tiếp tục cung cấp telemetry trực tiếp.
- Chuyển Monitor sang Building giữ WebRTC/WebSocket. Canvas và HUD Monitor ngừng vẽ khi ẩn; metadata mới vẫn được nhận đầy đủ và khung vẽ dùng vị trí mới nhất khi mở lại.

## Kiểm tra

`GET /api/debug/pipeline`: camera đã có nhãn phải có `running: true`, `stream_source: mediamtx_relay`, `processed_frames` tăng; nhãn có mask thành công sẽ có `mask_source: sam2_memory`.

`GET /api/registry/targets`: các bản ghi cùng nhãn có `global_id` giống nhau và `key`/`cam_id` riêng. Có cùng ID không đảm bảo nhận dạng tuyệt đối trong mọi góc nhìn/che khuất; cần đăng ký mẫu đúng cho từng camera.
