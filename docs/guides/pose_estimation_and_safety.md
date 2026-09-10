# Hướng dẫn nhận diện dáng người và cảnh báo ngã lao động (Pose & Safety Rules)

## 1. 17 Điểm khớp xương (COCO Pose Keypoints)
Mô hình YOLOv8x-Pose phát hiện 17 điểm mốc:
- 0: Mũi, 1-2: Mắt (trái/phải), 3-4: Tai (trái/phải)
- 5-6: Vai (trái/phải), 7-8: Khuỷu tay (trái/phải), 9-10: Cổ tay (trái/phải)
- 11-12: Khớp hông (trái/phải), 13-14: Đầu gối (trái/phải), 15-16: Mắt cá chân (trái/phải)

## 2. Thuật toán phân tích dáng điệu & phát hiện té ngã
1. **Tỷ lệ khung hình Bounding Box**: $R = \frac{W}{H}$. Nếu $R > 1.3$, cơ thể đang có xu hướng nằm ngang.
2. **Góc nghiêng cột sống**: Tính góc nghiêng giữa trung điểm vai và trung điểm hông so với phương thẳng đứng.
3. **Độ cao trung bình khớp hông**: Phát hiện chuyển động rơi nhanh của khớp hông trong khoảng thời gian $< 0.5$ giây.
4. Kích hoạt sự kiện `FALL_INCIDENT_ALERT` khi người duy trì trạng thái nằm trong hơn 3 giây.
