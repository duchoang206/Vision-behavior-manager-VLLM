import asyncio
import json

async def is_rtsp_valid_async(url: str, timeout: float = 2.0) -> bool:
    """
    Verify that an RTSP endpoint can deliver a decodable video stream.

    A TCP connect to port 554 is not sufficient: cameras commonly accept the
    socket while authentication, the RTSP path, or the video source is broken.
    ffprobe runs out-of-process so a damaged stream cannot destabilize OpenCV
    or the backend process.
    """
    if url.startswith("file://") or url.startswith("/"):
        return True

    process = None
    try:
        timeout_us = max(500_000, int(float(timeout) * 1_000_000))
        process = await asyncio.create_subprocess_exec(
            "ffprobe", "-v", "error", "-rtsp_transport", "tcp",
            "-rw_timeout", str(timeout_us), "-analyzeduration", "3000000",
            "-probesize", "2000000", "-select_streams", "v:0",
            "-show_entries", "stream=codec_type,width,height", "-of", "json", url,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=max(1.0, float(timeout) + 1.0))
        if process.returncode != 0:
            return False
        payload = json.loads(stdout.decode("utf-8", errors="replace") or "{}")
        return any(
            stream.get("codec_type") == "video"
            and int(stream.get("width") or 0) > 0
            and int(stream.get("height") or 0) > 0
            for stream in payload.get("streams", [])
        )
    except (asyncio.TimeoutError, FileNotFoundError, OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    finally:
        if process is not None and process.returncode is None:
            process.kill()
            try:
                await process.wait()
            except (ProcessLookupError, ChildProcessError):
                pass
