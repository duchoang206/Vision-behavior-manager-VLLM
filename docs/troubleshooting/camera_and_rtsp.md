# Hướng dẫn khắc phục sự cố kết nối Camera IP và Luồng RTSP

## 1. Kiểm tra kết nối mạng và cổng RTSP
Để quét tìm các IP camera đang hoạt động trong dải mạng nội bộ (LAN):
```bash
sudo nmap -p 554 --open 192.168.5.0/24
```
Các IP camera thông thường được thiết lập tại dải IP: `192.168.5.201`, `192.168.5.241`, `192.168.5.242`, `192.168.5.243`.

## 2. Kiểm tra trực tiếp luồng RTSP bằng FFplay / VLC
```bash
ffplay -rtsp_transport tcp "rtsp://admin:password@192.168.5.201:554/Streaming/Channels/101"
```

## 3. Khắc phục lỗi trễ hình hoặc đọng khung hình (Frame Lag)
- Kiểm tra cấu hình `CAP_PROP_N_THREADS=1` trong `rtsp_reader.py`.
- Đảm bảo tham số `last_frame_age_ms` trong `/api/debug/pipeline` luôn nhỏ hơn 200ms.
- Sử dụng giao thức TCP thay vì UDP cho các luồng RTSP khi mạng có độ suy hao gói tin cao.
