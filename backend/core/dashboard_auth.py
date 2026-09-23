import base64
import hashlib
import hmac
import json
import os
import re
import time
from urllib.parse import urlparse

from fastapi import HTTPException, Request


DEFAULT_PASSWORD_HASH = "5b46b622d7bef1357804f3040dfdb64235e4152a99b4660add351425e9d2388f73eace07357868ee6fb1f9c6714af8dd33e52560b34740577da3f5fe011f3bcf"


def verify_dashboard_session(token, now=None):
    secret = os.getenv("RSKYVIEW_SESSION_SECRET", "")
    if len(secret) < 32 or not token or len(token) > 1024:
        return None
    try:
        payload_text, signature_text = token.split(".")
        if not all(re.fullmatch(r"[A-Za-z0-9_-]+", part) for part in (payload_text, signature_text)):
            return None
        username = os.getenv("RSKYVIEW_ADMIN_USERNAME", "admin")
        salt = os.getenv("RSKYVIEW_PASSWORD_SALT", "r-skyview-admin-v1")
        password_hash = os.getenv("RSKYVIEW_ADMIN_PASSWORD_HASH", DEFAULT_PASSWORD_HASH)
        key = hmac.new(secret.encode(), f"{username}:{salt}:{password_hash}".encode(), hashlib.sha256).digest()
        expected = hmac.new(key, payload_text.encode(), hashlib.sha256).digest()
        received = base64.urlsafe_b64decode(signature_text + "=" * (-len(signature_text) % 4))
        if not hmac.compare_digest(expected, received):
            return None
        payload = json.loads(base64.urlsafe_b64decode(payload_text + "=" * (-len(payload_text) % 4)))
        current = time.time() if now is None else now
        if (payload.get("username") != username or type(payload.get("issuedAt")) is not int
                or type(payload.get("expiresAt")) is not int or payload["issuedAt"] > current + 30
                or payload["expiresAt"] <= current or payload["expiresAt"] - payload["issuedAt"] != 28800
                or not re.fullmatch(r"[a-f0-9]{32}", payload.get("nonce", ""))):
            return None
        return payload
    except (ValueError, TypeError, AttributeError):
        return None


def require_dashboard_user(request: Request):
    session = verify_dashboard_session(request.cookies.get("rskyview_session"))
    if not session:
        raise HTTPException(401, "Vui lòng đăng nhập để quản lý hệ thống.")
    if request.method not in {"GET", "HEAD", "OPTIONS"} and request.headers.get("origin"):
        origin_host = urlparse(request.headers["origin"]).hostname
        forwarded_host = request.headers.get("x-forwarded-host", request.headers.get("host", ""))
        destination_host = urlparse(f"http://{forwarded_host}").hostname
        aliases = {"localhost": "127.0.0.1", "::1": "127.0.0.1"}
        if aliases.get(origin_host, origin_host) != aliases.get(destination_host, destination_host):
            raise HTTPException(403, "Nguồn yêu cầu không hợp lệ.")
    return session["username"]
