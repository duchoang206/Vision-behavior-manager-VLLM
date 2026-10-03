"""Publish a controllable synthetic warehouse camera as a real H.264 RTSP stream.

End-to-end tests of the inspection stations need a camera whose scene is
known to the millimetre.  This tool renders ``inspection_scenes`` frames and
pushes them through ffmpeg/libx264 to an RTSP server (MediaMTX), so the
backend consumes them exactly like a factory camera: RTSP, H.264
compression, decode, ROI.

The scene is read from a JSON control file and re-rendered whenever it
changes, e.g.::

    {"version": 3, "scene": {"article": {"cx": 565, "angle": 0}, "gain": 1.0}}

Usage::

    python3 tests/inspection_rtsp_streamer.py --url rtsp://127.0.0.1:18554/insp_cam \
        --control /tmp/scene.json --fps 10
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from inspection_scenes import FRAME_H, FRAME_W, Article, Scene, render  # noqa: E402

VARIANTS = 6


def build_scene(spec: dict, seed: int) -> Scene:
    article = spec.get("article")
    shadow = spec.get("shadow")
    glare = spec.get("glare")
    return Scene(
        article=Article(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in article.items()}) if article is not None else None,
        debris=[(kind, tuple(params)) for kind, params in spec.get("debris", [])],
        gain=float(spec.get("gain", 1.0)), noise_sigma=float(spec.get("noise_sigma", 2.0)),
        shadow=np.asarray(shadow, np.float64) if shadow else None, shadow_factor=float(spec.get("shadow_factor", 0.45)),
        cell=tuple(spec.get("cell") or (1000.0, 1000.0)),
        glare=(tuple(glare[0]), tuple(glare[1]), float(glare[2])) if glare else None,
        blur_sigma=float(spec.get("blur_sigma", 0.0)), seed=seed,
        light=np.asarray(spec["light"], np.float64) if spec.get("light") else None,
        light_gain=tuple(spec.get("light_gain") or (0.7, 1.0, 1.5)))


def load_control(path: str):
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        return data.get("version"), data.get("scene") or {}
    except (OSError, ValueError):
        return None, None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("--fps", type=float, default=10.0)
    args = parser.parse_args()

    ffmpeg = subprocess.Popen([
        "ffmpeg", "-hide_banner", "-loglevel", "warning", "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{FRAME_W}x{FRAME_H}", "-r", str(args.fps), "-i", "-", "-c:v", "libx264", "-preset", "ultrafast",
        "-tune", "zerolatency", "-crf", "18", "-g", str(int(args.fps)), "-pix_fmt", "yuv420p",
        "-f", "rtsp", "-rtsp_transport", "tcp", args.url], stdin=subprocess.PIPE)
    version, frames, index = object(), [], 0
    period = 1.0 / args.fps
    try:
        while ffmpeg.poll() is None:
            started = time.monotonic()
            current, spec = load_control(args.control)
            if spec is not None and current != version:
                frames = [render(build_scene(spec, 7000 + n)).tobytes() for n in range(VARIANTS)]
                version = current
                print(f"[streamer] scene version {current}", flush=True)
            if frames:
                ffmpeg.stdin.write(frames[index % len(frames)])
                index += 1
            time.sleep(max(0.0, period - (time.monotonic() - started)))
    except (BrokenPipeError, KeyboardInterrupt):
        pass
    finally:
        try:
            ffmpeg.stdin.close()
        except OSError:
            pass
        ffmpeg.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
