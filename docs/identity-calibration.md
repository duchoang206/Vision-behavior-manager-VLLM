# Đăng ký đa góc nhìn và Calibration

## Đăng ký nhãn

Trong Building → Đăng ký nhãn, chọn nhãn đã có bằng **+ Góc nhìn**, lấy frame mới, khoanh sát cùng vật, xác nhận mask rồi bấm **Lưu góc nhìn**. Có thể lặp lại không giới hạn số lần, trong giới hạn dung lượng SSD. Hai vật khác nhau phải dùng hai nhãn khác nhau. Cùng nhãn trên nhiều camera là cùng danh tính.

- Tất cả lần đăng ký mới được lưu dưới `backend/data/registered_samples/<camera-label-hash>/`. Registry JSON giữ tham chiếu; không giữ toàn bộ frame trong RAM.
- Dữ liệu cũ vẫn đọc được; không xóa ảnh/mask cũ. SAM khởi tạo tối đa 8 góc nhìn trong video memory (cấu hình `REGISTERED_MASK_MEMORY_VIEWS`), bộ tracker giữ tối đa 16 frame seed và gallery xác minh giữ 64 đặc trưng đa dạng mỗi nhãn. Giới hạn này chỉ áp dụng cho realtime, không phải kho mẫu.
- Descriptor 512 chiều lấy mean/std của đặc trưng SAM2 **bên trong mask**, có xử lý phần padding ảnh. Dùng lại image backbone GPU của frame live, không gọi thêm một image encoder cho mỗi vật.
- Projection metric được học nền trên CUDA bằng batch-hard triplet loss với cosine distance và phạt thêm các negative còn quá gần. Cần ít nhất hai ảnh khác nhau của một vật và ảnh của ít nhất một vật khác. Ảnh giống hệt không được tính là hai góc nhìn mới cho metric.
- Chỉ dùng mẫu người vận hành xác nhận; tuyệt đối không tự học từ mask dự đoán. Backbone SAM2 không được fine-tune. Các batch học nhỏ lấy mẫu từ kho lưu trữ, không nạp vô hạn ảnh lên GPU.
- Checkpoint `backend/data/identity_metric/projection.pt` được lưu nguyên tử và dùng trực tiếp khi xác minh frame live. Không đổi checkpoint nếu loss huấn luyện xấu hơn hoặc dữ liệu bị xóa trong lúc học. Loss huấn luyện thấp không phải là cam kết độ chính xác trên dữ liệu chưa thấy.
- Mỗi mask phải khớp gallery của chính nhãn và hơn nhãn cạnh tranh một khoảng an toàn. Nếu danh tính không đạt, backend xóa flow/hold của nhãn và gửi `identity_rejected_labels` để Monitor bỏ mask cũ ngay. SAM không phát hiện được vật thì vẫn dùng chính sách hết hạn ngắn hiện có.

Trạng thái học: `GET /api/registry/mask/status` hoặc `GET /api/registry/targets`. Chẩn đoán mỗi nhãn tại `GET /api/debug/pipeline`: `raw_score`, `score`, `rival_score`, `revision` và `last_rejection`.

`REGISTERED_IDENTITY_MIN_SIMILARITY` mặc định `0.86`, `REGISTERED_IDENTITY_RIVAL_MARGIN` mặc định `0.035`. Không hạ các ngưỡng chỉ để mask xuất hiện nếu đang nhận nhầm vật. Ưu tiên thêm ảnh thật của cùng vật và ảnh những vật dễ nhầm, với nhãn đúng và mask loại nền.

Đường robot/kệ hiện tại dùng SAM2 CUDA và lớp motion hiện có; không chuyển toàn bộ pipeline sang DeepStream/TensorRT trong thay đổi này. Pose người giữ nguyên trạng thái tắt. Không có bảo đảm rằng vật bị che hoàn toàn hoặc robot khác giống hệt sẽ luôn nhận diện được.

## Calibration

Mở tab **Calibration** trên thanh bên. Có hai tab nhỏ: **Hiệu chuẩn Robot / FMS** giữ luồng hiệu chuẩn hiện có và **Thước đo**. Trong tab hiệu chuẩn: chọn camera → chấm một điểm cố định trên mặt sàn trong ảnh → chấm điểm tương ứng trên bản đồ FMS TT → lặp lại → **Save · Áp dụng**.

- Không cần nhập số. Có thể thêm bao nhiêu cặp tùy ý, tối thiểu 4 cặp không trùng/không thẳng hàng. Phải phân bố điểm trên cùng mặt phẳng, tránh gom vào một góc; không chấm trên nóc robot/kệ.
- Camera chỉ lấy một ảnh tĩnh khi mở/chuyển camera hoặc khi bấm lấy frame mới. Video Monitor không bị tháo khỏi trang khi chuyển tab.
- Dùng ảnh SLAM FMS thật với `slam_map.rect`. Chuyển tọa độ click bằng ma trận SVG nghịch đảo để đúng khi có letterbox, zoom hoặc resize. Cuộn để zoom, Shift+kéo để pan. Robot cập nhật qua WebSocket, không làm dịch khung bản đồ.
- Mỗi cặp gồm camera `(u,v)` chuẩn hóa và sàn `(x,z)` tính bằng mét. Hiển thị tọa độ FMS `(x,y)` bằng `x = floor_x + origin_x`, `y = origin_y + layout_depth - floor_z`. Backend từ chối lưu nếu map/hệ tọa độ FMS đã thay đổi.
- Backend kiểm tra toàn bộ cặp, tính homography và sai số reprojection, lưu vào PostgreSQL (`cameras.calibration_points`, `cameras.homography_matrix`) rồi mới kích hoạt. Lỗi lưu không thay ma trận đang chạy.
- Hiệu chuẩn vừa lưu là nguồn chính của camera đó (`manual_camera_fms_click`); dừng session auto-calibration cũ của camera. Các camera khác không thay đổi. Các đường tracking/detection và map 3D dùng chung `camera_calibrator`.
- Sai số khớp được hiển thị theo mét. Bốn điểm thường khớp gần như tuyệt đối dù người dùng chấm sai; số này không thay thế kiểm chứng vị trí thực.

Các kho dữ liệu nằm dưới `backend/data` đang bind-mount vào container. Cần sao lưu cả thư mục này và PostgreSQL. Không commit mẫu ảnh/đặc trưng/checkpoint vào Git.

### Thước đo

Chọn camera đã hiệu chuẩn → **Thước đo** → chấm ít nhất hai điểm trên ảnh. Các điểm nối liên tiếp thành đoạn thẳng; bảng bên cạnh hiển thị từng đoạn, tổng chiều dài và khoảng cách thẳng từ đầu đến cuối. Có thể đổi đơn vị m/cm/mm, hoàn tác điểm hoặc bắt đầu đường đo mới (tối đa 256 điểm mỗi đường).

- Cuộn hoặc bấm +/− để zoom ảnh tới 16×, Shift+kéo hoặc bật bàn tay để pan; nút toàn ảnh đưa về khung ban đầu. Điểm luôn quy về tọa độ ảnh gốc, không phụ thuộc mức zoom/kích thước cửa sổ.
- Dùng cùng ảnh tĩnh của Calibration, không chạy video hay suy luận AI riêng. **Lấy frame mới** xóa đường đo cũ sau khi xác nhận. Chuyển camera bắt đầu đường mới; đổi tab nhỏ giữ bản nháp hiệu chuẩn và đường đo.
- Mỗi lần thêm/xóa điểm gửi `POST /api/camera/{id}/calibration/measure` với `points: [[u,v], ...]` trong 0..1. Backend lấy một bản sao cấu hình hiện hành cho cả đường, chiếu homography sang mặt sàn và tính độ dài mét. Request cũ bị hủy khi điểm/camera/tab đổi; không hiển thị kết quả cũ cho điểm mới. Nút **Đo lại** đọc lại hiệu chuẩn đang lưu.
- API chỉ đọc, không ghi PostgreSQL hay sửa tracking; không dùng các tọa độ dự phòng của camera chưa calib. Camera chưa hiệu chuẩn trả 409, điểm sai hoặc đi qua chân trời trả 400 (payload sai cấu trúc trả 422). Có cảnh báo khi đo ngoài vùng điểm hiệu chuẩn.
- Thước chỉ đo **trên mặt phẳng sàn đã hiệu chuẩn**, không đo chiều cao 3D/nóc robot/kệ. Số chữ số hiển thị và đơn vị mm không bảo đảm độ chính xác mm; độ chính xác phụ thuộc calib và vị trí chấm. Sai số khớp calib không thay thế kiểm chứng bằng thước thật.
