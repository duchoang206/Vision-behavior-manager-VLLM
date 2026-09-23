# Monitor: giảm khựng video và tự phục hồi WHEP

Ngày: 22/09/2026, Asia/Bangkok. Phạm vi thay đổi: dashboard; không đổi model, Label, calibration, camera, database hoặc cấu hình MediaMTX.

## Nguyên nhân xác nhận

- Khi POST WHEP lỗi, component cũ chuyển sang iframe nhưng vẫn thử kết nối trực tiếp. Khi retry thành công không thoát iframe: có thể tồn tại hai đường phát của cùng camera. Phép thử chỉ làm hỏng một POST trong trình duyệt thử nghiệm xác nhận iframe vẫn tồn tại sau khi direct peer đã kết nối lại.
- Timer reconnect cũ không bị hủy khi ICE tự hồi phục. Request SDP không có deadline/cancellation theo từng attempt, handler cũ có thể tác động luồng mới, và không DELETE resource WHEP khi tháo kết nối.
- Video ngừng cập nhật nhưng ICE còn connected không được tự nối lại; watchdog cũ chỉ xóa canvas.
- Monitor nhận cả audio dù phát muted. Bộ đệm WebRTC của bốn camera có audio tích lũy đáng kể trong phép đo trước sửa. Bản mới chỉ nhận video, đặt jitter-buffer target nhỏ khi trình duyệt hỗ trợ; đây là gợi ý cho trình duyệt, không phải giới hạn độ trễ tuyệt đối.
- Callback canvas chỉ tốn khoảng 0,1–0,2 ms ở phép đo trước sửa, không ghi nhận long task trong các cửa sổ đo. Con số này không bao gồm toàn bộ compositor/GPU và không chứng minh backend SAM2 đang nhanh.

## Thay đổi

- `web-dashboard/lib/realtime-video.ts`: một peer/attempt, một đường video trực tiếp; không fallback iframe song song. Gửi SDP sau bước gom ICE candidate có giới hạn thời gian, không phụ thuộc public STUN cho mạng LAN đang dùng.
- Deadline kết nối 6,5 giây; retry tăng dần 250 ms đến 4 giây; hủy request/handler/timer của attempt cũ; dừng track và DELETE session. Response cũ không được áp SDP hoặc gắn stream vào video mới.
- Watchdog 500 ms dùng cả frame thực sự trình bày và `framesDecoded`. Khi card đang xem ngừng tiến triển hơn 1,5 giây, nối lại thay vì chờ F5. Không coi việc trình duyệt ngừng render card/tab ẩn là lỗi; không tháo kết nối khi chuyển Monitor ↔ Building.
- Video-only, jitter-buffer target 40 ms nếu được hỗ trợ. Trình duyệt vẫn có thể tăng đệm vì jitter nguồn. Thử nghiệm riêng target 0 ms làm tăng số freeze ở hai nguồn 720p nên không dùng cấu hình này.
- Camera card được memo hóa, không render lại chỉ vì state tổng hợp của camera khác đổi. Canvas vẫn theo callback video và dữ liệu live mới nhất; không thêm hàng đợi, làm chậm tọa độ, nâng tuổi mask hay giữ mask lịch sử vô hạn.

## Số liệu trước/sau

Chrome headless, 6 camera cùng lúc, warm-up 6 giây rồi đo 15 giây; chuyển Building 2,5 giây, quay Monitor, chờ 1,5 giây và đo tiếp 15 giây. Độ phân giải và FPS nguồn không đổi: hai nguồn 1280×720, bốn nguồn 640×360; một camera 20 FPS, năm camera 25 FPS.

Trung bình thời gian lưu trong jitter buffer ở cửa sổ **sau khi quay lại Monitor**:

| Camera | Trước (ms) | Sau bản cuối (ms) |
| --- | ---: | ---: |
| b1269e28 | 256,99 | 86,09 |
| 1ac2ffac | 298,13 | 34,74 |
| d1f37ed1 | 160,91 | 21,97 |
| 06e9e5b9 | 185,22 | 22,22 |
| e0606b55 | 11,75 | 21,78 |
| c7e22c1f | 8,47 | 22,14 |

Hai nguồn không audio vốn có đệm thấp; target 40 ms tăng nhẹ đệm của chúng để tránh ép về 0. Không tuyên bố mọi camera đều giảm độ trễ. Giá trị bảng là delta `jitterBufferDelay / jitterBufferEmittedCount`, **không phải camera-chụp → màn hình hoặc độ trễ SAM2**.

- Bản cuối giải mã 19,99–25,06 FPS tùy nguồn; không mất packet hay dropped-frame trong bộ đếm inbound RTP ở hai cửa sổ đo. Không đổi FPS cấu hình để đạt kết quả.
- Cửa sổ quay lại Monitor: 6 peer mở, không peer mới, không iframe, không freeze trong bộ đếm RTP; không lỗi JavaScript hoặc long task.
- Cửa sổ đầu của bản cuối vẫn ghi nhận một freeze khoảng 216 ms ở Cam 4 và một freeze khoảng 229 ms ở Dahua. **Chưa thể khẳng định đã hết mọi khựng nguồn/mạng/trình duyệt.**
- Callback JavaScript có thể bỏ qua lần gọi trong khi compositor vẫn nhận frame. Vì vậy số callback/giây không được dùng làm FPS trình bày: dùng thêm delta `presentedFrames` và `framesDecoded`. Video nằm ngoài viewport cũng có thể bị trình duyệt bỏ vẽ.

## Kiểm thử hồi phục và hồi quy

- Lỗi giả lập một POST WHEP: trước sửa còn một iframe và một peer trực tiếp cùng camera; sau sửa có 7 attempt tổng cộng nhưng chỉ 6 peer mở, 0 iframe, cả 6 video ready. Không gây lỗi cho session người dùng khác.
- Dừng video track của một peer thử nghiệm, không tắt server: tự tạo kết nối mới sau 2.144 ms, có frame phát lại sau 3.993 ms. Tổng thời gian này bao gồm phát hiện đứng, nối lại và chờ frame giải mã được, không phải độ trễ ổn định của video. Session cũ bị xóa, số session thử nghiệm vẫn là 6.
- Sau đóng trình duyệt thử nghiệm, API MediaMTX không còn session localhost của bài thử. Không xóa session của người dùng.
- 14 test Node qua: negotiation timeout, retry/backoff, ICE hồi phục, stale handler/response, teardown, hidden tab/card, decode/presentation stall, trình duyệt không hỗ trợ frame callback.
- 18 test backend hồi quy qua: `test_metadata_broadcaster`, `test_label_monitor_continuity`.
- Docker production build và TypeScript qua; ESLint hai file runtime không lỗi, còn các cảnh báo unused có sẵn của Monitor. Đã rebuild/recreate riêng frontend với bản cuối. Kiểm tra TypeScript trực tiếp workspace bị ảnh hưởng bởi validator `.next` cũ; production build sạch không gặp các lỗi đó.

Chạy lại test frontend với Node 24 và dependencies dashboard đã cài:

```sh
node --test web-dashboard/tests/realtime-video.test.cjs
```

Chi tiết các phép đo tại máy: `/tmp/rsky-video-before.json`, `/tmp/rsky-video-before-recovery-controlled.json`, `/tmp/rsky-video-after-final.json`, `/tmp/rsky-video-after-recovery.json`, `/tmp/rsky-video-after-stall-recovery.json`. Script đo: `/tmp/rsky_video_health.mjs`.

## Giới hạn xác minh

Trong suốt lượt đo, API trả `deployment=null`, runtime `idle`. Không tự deploy lại model trái với cấu hình hiện tại. Kết quả trên xác minh đường video và hồi phục giao diện, **không xác minh realtime SAM2 nhiều camera, độ chính xác bám vật hoặc sự trùng khớp mask/video dưới tải AI**. Cần chạy lại phép đo metadata/mask khi người dùng deploy model. Label và cơ chế hết hạn bằng chứng giữ nguyên.

Trình duyệt đang mở bundle cũ cần tải lại một lần sau triển khai; đó là nạp bản sửa, không phải cách khắc phục đứng hình khi vận hành.
