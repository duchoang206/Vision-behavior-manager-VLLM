# Hướng dẫn xử lý sự cố đồng bộ dữ liệu FMS MQTT VDA 5050

## 1. Kiểm tra kết nối MQTT Broker FMS
Địa chỉ mặc định của FMS MQTT Broker là `192.168.5.105:1883`.
Kiểm tra bằng lệnh `mosquitto_sub`:
```bash
mosquitto_sub -h 192.168.5.105 -p 1883 -t "uagv/v2/+/state" -v
```

## 2. Kiểm tra cơ chế tự động chuyển sang mô phỏng (Simulation Fallback)
- Khi backend không thể kết nối tới IP `192.168.5.105` sau 5 giây, hệ thống sẽ tự động chuyển trạng thái sang `SIMULATION_MODE`.
- Trên giao diện 3D Map, trạng thái sẽ hiển thị `[SIMULATION FALLBACK]` kèm theo danh sách robot mẫu di chuyển theo lộ trình mô phỏng định sẵn để đảm bảo không gián đoạn việc demo.

## 3. Khôi phục kết nối FMS thực tế
- Đảm bảo cáp mạng nối trực tiếp vào mạng nội bộ nhà xưởng FMS.
- Kiểm tra quyền truy cập PostgreSQL tại cổng `5432` trên máy chủ FMS.
