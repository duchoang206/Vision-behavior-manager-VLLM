# Cơ chế xử lý quy tắc an ninh (Security Rules & ROI Engine)

## 1. Các loại quy tắc an ninh được hỗ trợ
1. **Khu vực cấm xâm nhập (Forbidden Zone - Polygon ROI)**:
   - Sử dụng thuật toán Ray Casting để kiểm tra điểm chân $(x, y)$ của đối tượng có nằm trong đa giác hay không.
2. **Vạch ranh giới định hướng (Tripwire Line Crossing)**:
   - Theo dõi vector chuyển động của đối tượng qua 2 khung hình liên tiếp $(P_{t-1} \to P_t)$ và kiểm tra giao điểm với đoạn thẳng quy định.
3. **Cảnh báo lảng vảng (Loitering Detection)**:
   - Bắt đầu đếm thời gian khi một đối tượng ở liên tục trong một vùng cụ thể; nếu $t_{stay} > T_{threshold}$ ($10$ giây), hệ thống phát cảnh báo `LOITERING_WARNING`.
