#!/bin/bash
# =============================================================================
# Ultra-Low-Latency Realtime FFplay Script for All Cameras
# RTC VMS - Vision Manager
# =============================================================================

export DISPLAY="${DISPLAY:-:1}"

# Common low-latency flags for ffplay
FFPLAY_FLAGS="-rtsp_transport tcp -fflags nobuffer -flags low_delay -framedrop -strict experimental -vf setpts=0"

CAM1_URL="rtsp://admin:rtc%402025@192.168.5.243:554/Streaming/Channels/102"
CAM2_URL="rtsp://admin:rtc%402025@192.168.5.243:554/Streaming/Channels/202"
CAM3_URL="rtsp://admin:rtc%402025@192.168.5.201:554/cam/realmonitor?channel=1&subtype=0"
CAM4_URL="rtsp://admin:rtc%402025@192.168.5.201:554/cam/realmonitor?channel=2&subtype=0"
CAM5_URL="rtsp://admin:rtc%401234@192.168.5.241:554/Streaming/Channels/102"
CAM6_URL="rtsp://admin:MinhIT%4097@192.168.5.240:554/Streaming/Channels/102"

case "$1" in
  1)
    echo "[FFPLAY] Opening Camera 1 (Hik 1 - 192.168.5.243 Ch102)..."
    ffplay $FFPLAY_FLAGS -window_title "Cam 1 - Hik 1" "$CAM1_URL"
    ;;
  2)
    echo "[FFPLAY] Opening Camera 2 (Góc rộng - 192.168.5.243 Ch202)..."
    ffplay $FFPLAY_FLAGS -window_title "Cam 2 - Góc Rộng" "$CAM2_URL"
    ;;
  3)
    echo "[FFPLAY] Opening Camera 3 (Dahua Ch1 - 192.168.5.201)..."
    ffplay $FFPLAY_FLAGS -window_title "Cam 3 - Dahua Ch1" "$CAM3_URL"
    ;;
  4)
    echo "[FFPLAY] Opening Camera 4 (Dahua Ch2 - 192.168.5.201)..."
    ffplay $FFPLAY_FLAGS -window_title "Cam 4 - Dahua Ch2" "$CAM4_URL"
    ;;
  5)
    echo "[FFPLAY] Opening Camera 5 (Hướng ra cửa - 192.168.5.241 Ch102)..."
    ffplay $FFPLAY_FLAGS -window_title "Cam 5 - Hướng Cửa" "$CAM5_URL"
    ;;
  6)
    echo "[FFPLAY] Opening Camera 6 (Cam mới - 192.168.5.240 Ch102)..."
    ffplay $FFPLAY_FLAGS -window_title "Cam 6 - Cam Mới" "$CAM6_URL"
    ;;
  all)
    echo "[FFPLAY] Launching all 6 cameras in realtime..."
    ffplay $FFPLAY_FLAGS -x 640 -y 360 -window_title "Cam 1 - Hik 1" "$CAM1_URL" >/dev/null 2>&1 &
    ffplay $FFPLAY_FLAGS -x 640 -y 360 -window_title "Cam 2 - Góc Rộng" "$CAM2_URL" >/dev/null 2>&1 &
    ffplay $FFPLAY_FLAGS -x 640 -y 360 -window_title "Cam 3 - Dahua Ch1" "$CAM3_URL" >/dev/null 2>&1 &
    ffplay $FFPLAY_FLAGS -x 640 -y 360 -window_title "Cam 4 - Dahua Ch2" "$CAM4_URL" >/dev/null 2>&1 &
    ffplay $FFPLAY_FLAGS -x 640 -y 360 -window_title "Cam 5 - Hướng Cửa" "$CAM5_URL" >/dev/null 2>&1 &
    ffplay $FFPLAY_FLAGS -x 640 -y 360 -window_title "Cam 6 - Cam Mới" "$CAM6_URL" >/dev/null 2>&1 &
    echo "All 6 camera windows opened on DISPLAY=$DISPLAY."
    echo "Run './scripts/ffplay_cameras.sh stop' to close all windows."
    ;;
  stop)
    echo "Closing all ffplay windows..."
    pkill -f ffplay || true
    echo "Done."
    ;;
  *)
    echo "========================================================"
    echo "  Realtime Low-Latency Camera Player (FFplay)"
    echo "========================================================"
    echo "Usage: $0 [1|2|3|4|5|6|all|stop]"
    echo ""
    echo "Options:"
    echo "  1     : Cam 1 - Hik 1 (192.168.5.243: Ch102)"
    echo "  2     : Cam 2 - Góc rộng (192.168.5.243: Ch202)"
    echo "  3     : Cam 3 - Dahua Ch1 (192.168.5.201)"
    echo "  4     : Cam 4 - Dahua Ch2 (192.168.5.201)"
    echo "  5     : Cam 5 - Hướng ra cửa (192.168.5.241: Ch102)"
    echo "  6     : Cam 6 - Cam mới (192.168.5.240: Ch102)"
    echo "  all   : Open all 6 cameras in separate realtime windows"
    echo "  stop  : Close all ffplay windows"
    echo "========================================================"
    ;;
esac
