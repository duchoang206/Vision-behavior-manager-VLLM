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
