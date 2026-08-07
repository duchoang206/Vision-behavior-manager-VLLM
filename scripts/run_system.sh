#!/usr/bin/env bash
# ==============================================================================
# Vision Manager: 3D Digital Twin, DeepStream AI, Re-ID & Multi-Entity Tracker
# Unified Launch Script for All 4 Sessions
# ==============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "======================================================================"
echo "🚀 Khởi chạy Hệ thống Vision Manager 3D Digital Twin & AI Perception"
echo "======================================================================"

# 1. Check Docker & GPU
if ! command -v docker &> /dev/null; then
    echo "❌ Lỗi: Docker chưa được cài đặt!"
    exit 1
fi

echo "🔍 Kiểm tra container Docker đang chạy..."
docker compose ps

# 2. Start Services
echo "🚀 Đang khởi động MediaMTX, PostgreSQL, Backend FastAPI và Frontend Next.js..."
# Code, model engines and registry data are bind-mounted from the host.
# Rebuild only with: docker compose build backend
docker compose up -d --no-build --remove-orphans

echo ""
echo "✅ Hệ thống đã sẵn sàng hoạt động:"
echo "  - 🌐 Frontend 3D Dashboard: http://192.168.5.104:3000 (hoặc http://localhost:3000)"
echo "  - ⚡ Backend FastAPI:        http://192.168.5.104:8000"
echo "  - 📹 WebRTC WHEP MediaMTX:  http://192.168.5.104:8081"
echo "  - 📡 3D Telemetry WS:       ws://192.168.5.104:8000/ws/digital_twin"
echo "======================================================================"
