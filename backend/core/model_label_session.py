import base64
import queue
import threading
import time
import uuid


class ModelLabelSession:
    def __init__(self, reply, clock=time.monotonic):
        self.reply = reply
        self.clock = clock
        self.lock = threading.Lock()
        self.commands = queue.Queue(maxsize=4)
        self.waiting = {}
        self.snapshots = {}
        self.previews = {}

    def request(self, request):
        request = dict(request, deadline=self.clock() + 10)
        try:
            if request.get("action") == "snapshot":
                with self.lock:
                    if len(self.waiting) >= 4:
                        raise ValueError("Đang lấy frame; hãy chờ.")
                    self.waiting[request["id"]] = request
            elif request.get("action") in {"preview", "sample"}:
                self.commands.put_nowait(request)
            else:
                raise ValueError("Lệnh Label không hợp lệ.")
        except (ValueError, queue.Full) as error:
            self.reply(dict(id=request["id"], error=str(error) or "Hàng đợi Label đã đầy."))

    def needs_frame(self, camera_id):
        with self.lock:
            return any(item["camera_id"] == camera_id and item["deadline"] > self.clock() for item in self.waiting.values())

    def capture(self, camera_id, image, frame_id, captured_at, objects=None, frame_info=None):
        with self.lock:
            requests = [self.waiting.pop(identifier) for identifier, item in list(self.waiting.items())
                        if item["camera_id"] == camera_id and item["deadline"] > self.clock()]
        if not requests:
            return
        try:
            import torch
            from torchvision.io import encode_jpeg

            tensor = torch.from_dlpack(image)
            encoded = encode_jpeg(tensor.permute(2, 0, 1).contiguous(), quality=90)
            jpeg = base64.b64encode(encoded.cpu().numpy().tobytes()).decode("ascii")
            for request in requests:
                identifier = uuid.uuid4().hex
                with self.lock:
                    if len(self.snapshots) >= 4:
                        self.snapshots.pop(next(iter(self.snapshots)))
                    self.snapshots[identifier] = dict(image=image, jpeg=jpeg, camera_id=camera_id,
                        frame_id=frame_id, captured_at=captured_at, frame_info=dict(frame_info or {}), expires=self.clock() + 180)
                self.reply(dict(id=request["id"], result=dict(snapshot_id=identifier, camera_id=camera_id,
                    image=jpeg, width=int(tensor.shape[1]), height=int(tensor.shape[0]),
                    frame_id=frame_id, captured_at=captured_at, expires_in=180,
                    objects=objects or [], **(frame_info or {}))))
        except Exception as error:
            for request in requests:
                self.reply(dict(id=request["id"], error=f"Không lấy được ảnh JPEG bằng GPU: {error}"))

    def _get(self, records, identifier, camera_id):
        with self.lock:
            sample = records.get(identifier)
        if not sample or sample["expires"] <= self.clock() or sample["camera_id"] != camera_id:
            raise ValueError("Frame/preview đã hết hạn hoặc khác camera. Lấy frame mới.")
        return sample

    def service(self, runtime):
        now = self.clock()
        with self.lock:
            expired = [self.waiting.pop(identifier) for identifier, item in list(self.waiting.items()) if item["deadline"] <= now]
        for request in expired:
            self.reply(dict(id=request["id"], error="Camera không gửi frame mới. Kiểm tra kết nối camera."))
        with self.lock:
            for records in (self.snapshots, self.previews):
                for identifier in list(records):
                    if records[identifier]["expires"] <= now:
                        records.pop(identifier)
        try:
            request = self.commands.get_nowait()
        except queue.Empty:
            return
        try:
            if request["deadline"] <= now:
                raise ValueError("Lệnh đã hết hạn, hãy thử lại.")
            if request["action"] == "preview":
                snapshot = self._get(self.snapshots, request["snapshot_id"], request["camera_id"])
                mask, descriptor = runtime.preview(snapshot["image"], request["bbox"], request.get("points", []), request.get("point_labels", []))
                identifier = uuid.uuid4().hex
                preview = dict(camera_id=request["camera_id"], expires=snapshot["expires"],
                    sample=dict(camera_id=request["camera_id"], frame_id=snapshot["frame_id"],
                        captured_at=snapshot.get('captured_at', 0), frame_info=snapshot.get('frame_info', {}),
                        preview_id=identifier, image=snapshot["jpeg"], mask=mask, bbox=request["bbox"],
                        signature=runtime.signature, vector=descriptor.tolist()))
                with self.lock:
                    if len(self.previews) >= 8:
                        self.previews.pop(next(iter(self.previews)))
                    self.previews[identifier] = preview
                result = dict(preview_id=identifier, mask=mask, frame_id=snapshot["frame_id"])
            else:
                result = self._get(self.previews, request["preview_id"], request["camera_id"])["sample"]
            self.reply(dict(id=request["id"], result=result))
        except Exception as error:
            self.reply(dict(id=request["id"], error=str(error)))
