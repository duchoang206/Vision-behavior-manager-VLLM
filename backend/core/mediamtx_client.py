import os
from urllib.parse import quote

import requests


def camera_relay_url(cam_id: str) -> str:
    origin = os.getenv("MEDIAMTX_RTSP_ORIGIN", "rtsp://127.0.0.1:8554").rstrip("/")
    return f"{origin}/{quote(cam_id, safe='')}"


class MediaMTXClient:
    def __init__(self, api=None):
        self.api = (api or os.getenv("MEDIAMTX_API", "http://127.0.0.1:9997/v3/config/paths")).rstrip("/")

    def ensure_path(self, cam_id: str, source: str):
        path = quote(cam_id, safe="")
        config = {"source": source, "sourceOnDemand": False, "rtspTransport": "tcp"}
        current = requests.get(f"{self.api}/get/{path}", timeout=3)
        if current.status_code == 200:
            if all(current.json().get(key) == value for key, value in config.items()):
                return
            response = requests.patch(f"{self.api}/patch/{path}", json=config, timeout=3)
        elif current.status_code == 404:
            response = requests.post(f"{self.api}/add/{path}", json=config, timeout=3)
            if response.status_code == 409:
                response = requests.patch(f"{self.api}/patch/{path}", json=config, timeout=3)
        else:
            current.raise_for_status()
            return
        response.raise_for_status()

    def delete_path(self, cam_id: str):
        response = requests.delete(f"{self.api}/delete/{quote(cam_id, safe='')}", timeout=3)
        if response.status_code != 404:
            response.raise_for_status()


mediamtx_client = MediaMTXClient()
