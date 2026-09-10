# Kiến trúc định danh đối tượng đa Camera (Cross-Camera ReID Tracking)

## 1. Trích xuất đặc trưng ReID
- Sử dụng mô hình mạng nơ-ron sâu trích xuất vector đặc trưng embedding $512$-chiều cho mỗi bounding box người.
- Vector được chuẩn hóa $L_2$ để tính toán độ tương đồng cosine nhanh chóng trên GPU.

## 2. Quản lý ID toàn cục (Global ID Management)
- Khi một người rời khỏi tầm nhìn của Camera A và xuất hiện tại Camera B:
  1. Trích xuất vector embedding của người mới tại Camera B.
  2. Truy vấn top-1 độ tương đồng gần nhất trong bộ nhớ ChromaDB.
  3. Nếu $\text{similarity} \ge \text{Threshold}$ ($0.75$), hệ thống tự động gán lại cùng `global_id`.
  4. Cập nhật ma trận chuyển vùng camera (Transition Matrix) trong bảng Analytics.
