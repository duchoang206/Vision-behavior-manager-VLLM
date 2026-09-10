# Hướng dẫn đóng góp mã nguồn (Contributing Guidelines)

## 1. Quy chuẩn cấu trúc mã nguồn
- **Frontend**: Next.js 14+ với TypeScript, Tailwind CSS / Vanilla CSS Module, Lucide Icons, Three.js 3D.
- **Backend**: FastAPI, DeepStream 8.0 Python Bindings (pyds), PyTorch / TensorRT.

## 2. Quy trình kiểm thử trước khi commit
1. Chạy kiểm thử đơn vị backend:
   ```bash
   pytest backend/tests/
   ```
2. Kiểm tra định kiểu TypeScript:
   ```bash
   cd web-dashboard && npm run build
   ```

## 3. Tiêu chuẩn thông điệp commit
- Sử dụng thông điệp rõ ràng, nêu bật module và tính năng được chỉnh sửa.
- Không commit các file mô hình có dung lượng lớn hơn 100MB lên GitHub trực tiếp; thay vào đó sử dụng Git LFS hoặc script tải tự động.
