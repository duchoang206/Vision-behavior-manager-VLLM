"""HTTP webhook driver (WMS / ERP REST endpoints) on ``httpx.AsyncClient``."""

import logging
from typing import Any, Dict, Optional

import httpx

from .base_driver import BaseDriver, SendResult, describe_error

logger = logging.getLogger("CommGateway")


class WebhookDriver(BaseDriver):
    protocol = "HTTP_WEBHOOK"

    def __init__(self, channel: Dict[str, Any], hooks=None):
        super().__init__(channel, hooks)
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def target(self) -> str:
        path = self.channel.get("endpoint_path") or "/"
        if path.startswith(("http://", "https://")):
            return path
        scheme = "https" if self.config.get("use_tls") else "http"
        return f"{scheme}://{self.channel.get('host')}:{self.channel.get('port')}{path if path.startswith('/') else '/' + path}"

    def _headers(self) -> Dict[str, str]:
        auth = self.channel.get("auth_config") or {}
        content_type = "application/json" if self.config.get("payload_type", "json") == "json" else "text/plain"
        headers = {"Content-Type": f"{content_type}; charset=utf-8"}
        headers.update({str(k): str(v) for k, v in (auth.get("headers") or {}).items()})
        if auth.get("bearer_token"):
            headers["Authorization"] = f"Bearer {auth['bearer_token']}"
        return headers

    async def _open(self) -> None:
        timeout = float(self.config.get("timeout_sec") or 3.0)
        auth = self.channel.get("auth_config") or {}
        basic = (auth["username"], auth.get("password") or "") if auth.get("username") else None
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(timeout), auth=basic,
                                         verify=not self.config.get("insecure_tls"))
        self.state = "online"

    async def _close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def send(self, payload: str) -> SendResult:
        if self._client is None:
            await self._open()
        method = str(self.config.get("http_method") or "POST").upper()
        try:
            response = await self._client.request(method, self.target, content=payload.encode("utf-8"),
                                                  headers=self._headers())
        except httpx.TimeoutException:
            self.state = "error"
            return SendResult(False, f"Connection Timeout: {self.target} không phản hồi sau "
                                     f"{float(self.config.get('timeout_sec') or 3.0):g}s", self.target)
        except httpx.HTTPError as exc:
            self.state = "error"
            cause = exc.__cause__ or exc.__context__ or exc
            if "refused" in str(exc).lower() or isinstance(cause, ConnectionRefusedError):
                return SendResult(False, f"Connection Refused: {self.target} đang đóng hoặc dịch vụ chưa chạy", self.target)
            return SendResult(False, describe_error(cause if isinstance(cause, OSError) else exc, self.target), self.target)
        body = response.text[:300]
        detail = f"HTTP {response.status_code} {response.reason_phrase} — {self.target}"
        ok = 200 <= response.status_code < 300
        self.state = "online" if ok else "error"
        return SendResult(ok, detail, self.target, 1 if ok else 0, response={"status": response.status_code, "body": body})
