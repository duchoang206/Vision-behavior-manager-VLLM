import asyncio
import socket
from urllib.parse import urlparse

async def is_rtsp_valid_async(url: str, timeout: float = 2.0) -> bool:
    """
    Ultra-safe, non-blocking RTSP availability probe.
    Uses socket connect to avoid multi-threaded C-level pointer collisions in OpenCV FFmpeg.
    """
    try:
        if url.startswith("file://") or url.startswith("/"):
            return True
        
        parsed = urlparse(url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 554
        
        loop = asyncio.get_running_loop()
        
        def probe_tcp():
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(timeout)
            try:
                s.connect((host, port))
                s.close()
                return True
            except Exception:
                return False
                
        return await loop.run_in_executor(None, probe_tcp)
    except Exception as e:
        print(f"[check_rtsp] Error checking {url}: {e}", flush=True)
        return False
