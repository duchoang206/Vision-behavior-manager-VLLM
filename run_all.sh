#!/bin/bash

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

case "$1" in
  down|stop)
    echo "Stopping Unified Vision & Robot Management System..."
    sudo docker compose down
    ;;
  logs|log)
    sudo docker compose logs -f
    ;;
  restart)
    echo "Restarting Unified System..."
    sudo docker compose restart
    ;;
  build)
    echo "Building backend image explicitly (engines and registry stay on host volumes)..."
    sudo docker compose build backend
    ;;
  status|ps)
    sudo docker compose ps
    ;;
  *)
    echo "================================================================="
    echo "  Starting Unified Vision & FMS 3D Robot Management Platform     "
    echo "================================================================="
    echo ""
    echo "[1/2] Launching containers (Postgres, MediaMTX, Backend + FMS Bridge, Web Dashboard)..."
    # Source/config changes are visible through bind mounts.  Do not rebuild
    # the DeepStream image on every restart; use './run_all.sh build' only
    # after changing Dockerfile or Python dependencies.
    sudo docker compose up -d --no-build --remove-orphans

    if [ $? -ne 0 ]; then
      echo "[Error] Docker compose failed to start."
      exit 1
    fi

    echo -e "\n[2/2] Checking container statuses..."
    sleep 3
    sudo docker compose ps

    echo -e "\n================================================================="
    echo "  🚀 All services started successfully!"
    echo "  - 🌐 Web Dashboard (Monitor, Building, Map Robot 3D, Analytics):"
    echo "       http://localhost:3000"
    echo "  - 🔌 Backend API & Documentation:"
    echo "       http://localhost:8000/docs"
    echo "  - 🤖 FMS Realtime Robot Telemetry WebSocket:"
    echo "       ws://localhost:8000/ws"
    echo "  - 📹 WebRTC Live Stream Server:"
    echo "       http://localhost:8081"
    echo ""
    echo "  Commands:"
    echo "  - View live logs : ./run_all.sh logs"
    echo "  - Check status   : ./run_all.sh status"
    echo "  - Stop system    : ./run_all.sh stop"
    echo "================================================================="
    ;;
esac
