"""REST API of the communication gateway (System Config → Devices & Channels)."""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from core.comm_gateway.channel_store import ChannelValidationError
from core.comm_gateway.constants import (COMMAND_KINDS, DEVICE_TYPES, EVENTS, MODES, PAYLOAD_FORMATS, PAYLOAD_TYPES,
                                         PROTOCOLS, TEMPLATE_PRESETS, TEMPLATE_VARIABLES)
from core.dashboard_auth import require_dashboard_user


class ChannelPayload(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    device_type: Optional[str] = None
    protocol: Optional[str] = None
    mode: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = None
    endpoint_path: Optional[str] = None
    auth_config: Optional[Dict[str, Any]] = None
    payload_template: Optional[str] = None
    trigger_events: Optional[List[str]] = None
    is_enabled: Optional[bool] = None
    priority: Optional[int] = None
    config: Optional[Dict[str, Any]] = None

    def data(self) -> Dict[str, Any]:
        return self.model_dump(exclude_none=True)


class TestDispatchRequest(BaseModel):
    channel_id: Optional[str] = None
    slot_id: Optional[str] = "5"
    state: Optional[str] = "Car Full"
    event: Optional[str] = None
    command_id: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)
    channel: Optional[ChannelPayload] = None


class PreviewRequest(BaseModel):
    payload_template: str = ""
    payload_type: str = "json"
    payload_format: str = "template"
    channel_id: str = ""
    context: Dict[str, Any] = Field(default_factory=dict)


class CommandRequest(BaseModel):
    context: Dict[str, Any] = Field(default_factory=dict)


class ModbusOperation(BaseModel):
    kind: str
    address: int = Field(0, ge=0, le=65535)
    value: Any = None
    count: int = Field(1, ge=1, le=125)


def _validation_error(exc: ChannelValidationError) -> HTTPException:
    return HTTPException(422, {"message": "Cấu hình không hợp lệ", "errors": exc.errors})


def create_comm_gateway_router(gateway):
    router = APIRouter(prefix="/api/comm", tags=["Communication gateway"], dependencies=[Depends(require_dashboard_user)])

    @router.get("/meta")
    def meta():
        return {"protocols": PROTOCOLS, "modes": MODES, "device_types": DEVICE_TYPES, "events": EVENTS,
                "payload_formats": PAYLOAD_FORMATS, "payload_types": PAYLOAD_TYPES, "command_kinds": COMMAND_KINDS,
                "variables": TEMPLATE_VARIABLES, "presets": TEMPLATE_PRESETS, "app_port": gateway.app_port}

    @router.get("/status")
    def status():
        return gateway.status()

    @router.get("/channels")
    def list_channels():
        return {"status": "success", "channels": gateway.list_channels()}

    @router.get("/channels/{channel_id}")
    def get_channel(channel_id: str):
        if channel_id not in gateway.channels:
            raise HTTPException(404, "Không tìm thấy kênh")
        return gateway.describe_channel(channel_id)

    @router.post("/channels", status_code=201)
    async def create_channel(body: ChannelPayload):
        try:
            return await gateway.save_channel(body.data())
        except ChannelValidationError as exc:
            raise _validation_error(exc)

    @router.put("/channels/{channel_id}")
    async def update_channel(channel_id: str, body: ChannelPayload):
        try:
            return await gateway.save_channel(body.data(), channel_id)
        except KeyError:
            raise HTTPException(404, "Không tìm thấy kênh")
        except ChannelValidationError as exc:
            raise _validation_error(exc)

    @router.delete("/channels/{channel_id}")
    async def delete_channel(channel_id: str):
        try:
            await gateway.delete_channel(channel_id)
        except KeyError:
            raise HTTPException(404, "Không tìm thấy kênh")
        return {"status": "success"}

    @router.post("/channels/{channel_id}/restart")
    async def restart_channel(channel_id: str):
        if channel_id not in gateway.channels:
            raise HTTPException(404, "Không tìm thấy kênh")
        return await gateway.restart_channel(channel_id)

    @router.post("/reload")
    async def reload_channels():
        await gateway.reload_channels()
        return {"status": "success", "channels": len(gateway.channels)}

    @router.post("/test_dispatch")
    async def test_dispatch(body: TestDispatchRequest):
        overrides = {**body.context, "slot_id": body.slot_id, "state": body.state, "event": body.event}
        draft = body.channel.data() if body.channel else None
        return await gateway.test_dispatch(body.channel_id, overrides, draft=draft, command_id=body.command_id)

    @router.post("/preview")
    async def preview(body: PreviewRequest):
        return await gateway.preview(body.payload_template, body.payload_type, body.context,
                                     body.payload_format, body.channel_id)

    @router.post("/channels/{channel_id}/commands/{command_id}")
    async def run_command(channel_id: str, command_id: str, body: CommandRequest):
        try:
            return await gateway.execute_command(channel_id, command_id, body.context)
        except KeyError:
            raise HTTPException(404, "Không tìm thấy kênh hoặc lệnh")

    @router.post("/channels/{channel_id}/modbus")
    async def modbus(channel_id: str, body: ModbusOperation):
        if body.kind not in ("modbus_coil", "modbus_register", "modbus_read_coils", "modbus_read_registers"):
            raise HTTPException(422, "kind phải là modbus_coil / modbus_register / modbus_read_coils / modbus_read_registers")
        try:
            return await gateway.modbus_operation(channel_id, body.model_dump())
        except KeyError:
            raise HTTPException(404, "Không tìm thấy kênh")

    @router.get("/slots")
    def slots():
        return {"status": "success", "slots": gateway.registry.entries()}

    @router.get("/logs")
    def logs(after: int = Query(0, ge=0), channel_id: Optional[str] = None, limit: int = Query(200, ge=1, le=300)):
        return {"status": "success", "logs": gateway.logs(after, channel_id, limit)}

    return router
