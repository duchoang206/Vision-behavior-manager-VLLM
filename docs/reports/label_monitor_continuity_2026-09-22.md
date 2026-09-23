# Label → SAM2 → Monitor: ưu tiên người dùng và giảm nhấp nháy

Ngày kiểm tra: 22/09/2026, múi giờ Asia/Bangkok.

## Logic đã sửa

- Lưu Label khởi tạo bám vật từ ảnh/mask gốc do người dùng xác nhận; không đợi YOLO phát hiện đúng lớp. Mẫu đã lưu được decode trên GPU, nạp vào SAM2 video memory. Bộ nhận dạng vẫn dùng toàn bộ embedding tương thích, không giới hạn 40–50 mẫu.
- Mask xác nhận trên frame còn mới có thể được gửi trực tiếp. Nếu người dùng thao tác trên ảnh đã cũ, backend chuyển mask từ ảnh đó sang frame live trước khi gửi; không thay timestamp để giả thành mask realtime.
- Thêm góc nhìn giữ ID và memory đang bám. Thêm negative/xóa mẫu vẫn buộc kiểm tra lại. Label được ưu tiên hơn dự đoán model/triplet, nhưng không bỏ kiểm tra nhầm vật, negative hay vật đã rời cảnh.
- Hai worker CUDA xử lý camera song song, mỗi camera có memory riêng; hàng chờ chỉ giữ frame mới nhất. Mask từng vật được đưa vào cache đầu ra ngay khi hoàn thành, không đợi mọi vật trong frame.
- Mask theo vị trí NvDCF đã gắn đúng track giữa các lần SAM2; không tự gắn lại vào một ID thay thế ở vị trí cũ.
- Monitor không xóa mask chỉ vì thiếu object trong một packet. Thiếu mask chỉ được nối từ mask trước khi identity vẫn xác nhận, vị trí mới và hình học liên tục. Giới hạn tuổi hình dạng 700 ms, tuổi vị trí 350 ms; packet lặp không gia hạn hai mốc này.
- Xóa Label, thay generation/model, ẩn đầu ra hoặc backend thu hồi track đều ngừng hiển thị. Kết quả suy luận cũ không được ghi đè/thu hồi sửa nhãn mới.

## Chẩn đoán live trước lượt tối ưu phát từng object

Đo 15,006 giây, bắt đầu 13:06:08 ngày 22/09/2026; 5 camera deploy, Chrome headless mở Monitor:

| Chỉ số | Kết quả |
| --- | --- |
| Metadata nhận được | 19,93–24,66 Hz/camera |
| Tuổi metadata probe → client p95 | 10,38–13,21 ms |
| Callback vẽ canvas p95 | 0,1–0,3 ms |
| Lỗi WebSocket / đóng bất thường | 0 / 0 |
| Frame mask mới thường thấy | khoảng 4–5,2 Hz cho các track xuất hiện thường xuyên |
| Tuổi bằng chứng mask p95 | khoảng 313–451 ms tùy track/camera |

Các số này cho thấy việc tạo/cấp lịch mask và quyết định giữ track ở backend là phần chậm chính so với JavaScript vẽ overlay. Một số camera/track vẫn thiếu mask; đây không phải chứng nhận hết nhấp nháy hay accuracy 100%. Callback không đo toàn bộ GPU raster/compositor và không phải độ trễ camera-chụp → màn hình. Tỷ lệ xuất hiện object không đồng nghĩa độ chính xác nhận dạng.

Đã sửa thêm việc phải chờ hết các object rồi mới công bố mask sau lượt đo trên. **Chưa có số đo live nhiều camera hợp lệ cho thay đổi cuối này:** lúc đo lại 13:48:45, API cho biết `deployment=null`, runtime `idle`. Log ghi nhận lệnh `DELETE /api/models/deployment` và thay đổi camera trong quá trình kiểm tra. Giữ nguyên lựa chọn hiện tại, không tự deploy lại. Video không đồng nghĩa AI đang chạy; các lỗi 404 của nguồn camera thay đổi ở lượt cuối không được tính là kết quả AI.

## Kiểm chứng phiên bản cuối

- 76 test backend qua: Label API/session, gallery/negative, reference memory FP16, tracker bridge, thêm/xóa góc nhìn, publish từng mask, kết quả sai thứ tự và expiry.
- 12 assertion frontend qua: nối mask theo vị trí, giữ tuổi bằng chứng, hết hạn, đổi ID, thu hồi, mất vật, nhảy vị trí và packet cũ.
- Frontend Docker/TypeScript production build thành công; service frontend đã được recreate. Backend đã restart, compile thành công. Các sửa worker cuối nằm trong source bind mount và được nạp khi deployment chạy lại.
- Bộ test model rộng vẫn có 5 lỗi setup do container thiếu module `onnx`; không ghi nhận là bộ test này đã qua và không tự đổi môi trường GPU đang dùng.

Phép thử GPU cô lập, chỉ đọc ảnh/mask đã lưu của `Robot_2001` (gallery revision 170, 168 mẫu):

| Bước | Dịch ảnh ngang | IoU với mask mẫu dịch tương ứng | Thời gian tới mask đầu |
| --- | --- | --- | --- |
| Nạp lần đầu | 0 px | 0,9762 | 517,52 ms |
| Đã warm-up | 0 px | 0,9718 | 13,90 ms |
| 2 | 4 px | 0,9653 | 13,82 ms |
| 3 | 8 px | 0,9632 | 13,72 ms |
| 4 | 12 px | 0,9570 | 13,71 ms |
| 5 | 16 px | 0,9586 | 13,65 ms |

Giữ đúng `Robot_2001` ở cả 6 bước, 2 reference memory, không lỗi encode. Đây là dịch toàn ảnh tổng hợp trên GPU khi deployment đã dừng, **không phải** phép đo robot di chuyển độc lập, nhiều camera đồng thời hay FPS production.

## Vận hành và giới hạn

1. Deploy lại model cho camera muốn dùng tại Building → Model/TensorRT khi sẵn sàng.
2. Label → lấy frame → sửa mask → Save. Thông báo phân biệt mẫu đã lưu, worker đã nạp, mask khởi tạo và đang đưa sang frame live.
3. Đo lại trên robot thực đang đi/quay/che khuất; cần video ground truth để kết luận sai/đúng danh tính và chất lượng mask.

Không đổi ONNX/TensorRT, không xóa Label/calibration và không bật pose. SAM2 vẫn CUDA FP16/PyTorch; không tuyên bố đã cài SAM3 hoặc có độ trễ 0 ms. Không giữ một mask tại tọa độ lịch sử vô hạn khi không còn bằng chứng vật đang ở đó.
