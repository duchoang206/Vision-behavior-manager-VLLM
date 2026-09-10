# Chi tiết cơ chế dung hợp dữ liệu (Vision & FMS Fusion Pipeline)

## 1. Cơ chế ma trận Homography
- Ma trận Homography bậc 3 ($3 \times 3$) được thiết lập từ 4 điểm mốc sàn $(x_i, y_i) \leftrightarrow (X_i, Y_i)$ mét.
- Vị trí chân của vật thể trong khung hình camera được tính toán:
  $$\begin{bmatrix} X \\ Y \\ 1 \end{bmatrix} = \mathbf{H} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}$$

## 2. Quy tắc đối chiếu chéo (Cross-Check Verification)
- Khi khoảng cách lệch $\Delta d = \sqrt{(X_{vision} - X_{fms})^2 + (Y_{vision} - Y_{fms})^2} \le 0.5\text{m}$, trạng thái được gán là `SYNCED`.
- Khi $\Delta d > 0.5\text{m}$, cảnh báo `DEVIATED` được kích hoạt để thông báo cho người vận hành kiểm tra camera hoặc bánh xe robot.
