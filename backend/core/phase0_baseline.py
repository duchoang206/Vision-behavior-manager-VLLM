import functools
import json
import os
from pathlib import Path
import queue
import threading
import time

CONTROL = Path(__file__).resolve().parents[1] / 'data/phase0_baseline/control.json'
RECORDER = None


class BaselineRecorder:
    def __init__(self, control):
        self.control = control
        self.directory = Path(control['directory'])
        self.directory.mkdir(parents=True, exist_ok=True)
        self.events = queue.Queue(maxsize=100000)
        self.frames = {}
        self.local = threading.local()
        self.clock_lock = threading.Lock()
        self.anchor = None
        self.anchor_ns = None
        self.dropped = 0
        self.alive = True
        self.phase = control.get('phase', 'startup')
        self.checked_at = 0
        self.thread = threading.Thread(target=self._write, name='phase0-recorder', daemon=True)
        self.thread.start()
        self.emit('process', epoch_ns=time.time_ns(), mono_ns=time.perf_counter_ns())

    def enabled(self):
        now = time.monotonic()
        if now - self.checked_at > 1:
            self.checked_at = now
            try:
                self.control = json.loads(CONTROL.read_text())
                self.phase = self.control.get('phase', 'unspecified')
            except (OSError, ValueError):
                self.control = {}
        return bool(self.control.get('enabled'))

    def emit(self, event, **values):
        row = dict(event=event, ns=time.perf_counter_ns(), phase=self.phase, pid=os.getpid(), **values)
        try:
            self.events.put_nowait(row)
        except queue.Full:
            self.dropped += 1

    def _write(self):
        pending = []
        last_flush = time.monotonic()
        with (self.directory / f'events-{os.getpid()}.jsonl').open('a', buffering=1024 * 1024) as output:
            while self.alive or not self.events.empty() or pending:
                try:
                    pending.append(self.events.get(timeout=.05))
                except queue.Empty:
                    pass
                retained = []
                for row in pending:
                    pair = row.pop('_cuda', None)
                    if pair:
                        start, end = pair
                        if not end.query():
                            row['_cuda'] = pair
                            retained.append(row)
                            continue
                        row['cuda_ms'] = start.elapsed_time(end)
                        row['gpu_start_ns'] = self.anchor_ns + round(self.anchor.elapsed_time(start) * 1000000)
                        row['gpu_end_ns'] = self.anchor_ns + round(self.anchor.elapsed_time(end) * 1000000)
                    output.write(json.dumps(row, separators=(',', ':')) + '\n')
                pending = retained
                if time.monotonic() - last_flush > .5:
                    output.flush()
                    last_flush = time.monotonic()

    def tracker(self, camera, frame, objects, pts):
        if not self.enabled():
            return
        now = time.perf_counter_ns()
        self.frames[(camera, frame)] = dict(camera=camera, frame=frame, tracker_objects=objects,
                                          t0_ns=now, phase=self.phase, attached=set())
        self.emit('tracker', camera=camera, frame=frame, tracker_objects=objects, t0_ns=now, pts=pts)
        if len(self.frames) > 12000:
            cutoff = now - 30000000000
            self.frames = {key: value for key, value in self.frames.items() if value['t0_ns'] > cutoff}

    def mark(self, camera, frame, event, **values):
        row = self.frames.get((camera, frame))
        if not row or not self.enabled():
            return
        self.emit(event, camera=camera, frame=frame, **values)

    def gpu_clock(self):
        if self.anchor is not None:
            return
        import torch
        with self.clock_lock:
            if self.anchor is None:
                anchor = torch.cuda.Event(enable_timing=True)
                started = time.perf_counter_ns()
                anchor.record()
                anchor.synchronize()
                self.anchor_ns = time.perf_counter_ns()
                self.anchor = anchor
                self.emit('cuda_clock', anchor_ns=self.anchor_ns, sync_wall_ms=(self.anchor_ns-started)/1000000)

    def measure(self, function, stage, cuda=True):
        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            context = getattr(self.local, 'context', None)
            if not context or not self.enabled():
                return function(*args, **kwargs)
            if cuda:
                import torch
                self.gpu_clock()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
            entered = time.perf_counter_ns()
            try:
                return function(*args, **kwargs)
            finally:
                if cuda:
                    end.record()
                row = dict(context, stage=stage, start_ns=entered, end_ns=time.perf_counter_ns())
                if cuda:
                    row['_cuda'] = (start, end)
                self.emit('span', **row)
        return wrapped

    def install(self):
        from core.model_sam2 import ModelSAM2, CudaMaskRuntime
        original_runtime_init = CudaMaskRuntime.__init__
        original_take = ModelSAM2._take_pending
        original_segment = CudaMaskRuntime.segment
        original_image = CudaMaskRuntime._image
        original_predictor = CudaMaskRuntime._predictor
        original_publish = ModelSAM2._publish_result
        original_attach = ModelSAM2.attach

        @functools.wraps(original_runtime_init)
        def runtime_init(runtime, *args, **kwargs):
            started = time.perf_counter_ns()
            original_runtime_init(runtime, *args, **kwargs)
            runtime.references._encode = self.measure(runtime.references._encode, 'reference_encode')
            if not getattr(runtime.metric, '_phase0_wrapped', False):
                runtime.metric.verify = self.measure(runtime.metric.verify, 'identity_verify')
                runtime.metric._phase0_wrapped = True
            self.emit('runtime_initialized', start_ns=started)

        @functools.wraps(original_take)
        def take(owner, worker_id=None):
            self.local.context = None
            camera, item = original_take(owner, worker_id)
            if item is not None and self.enabled():
                frame = item[1]
                self.local.context = dict(camera=camera, frame=frame, worker=worker_id, object_id=None)
                self.mark(camera, frame, 'worker_enter', worker=worker_id, targets_before_prompts=len(item[3]))
            return camera, item

        @functools.wraps(original_segment)
        def segment(runtime, camera, image, objects, captured_at, on_result=None):
            context = getattr(self.local, 'context', None)
            if not context or not self.enabled():
                return original_segment(runtime, camera, image, objects, captured_at, on_result=on_result)
            context.update(targets=len(objects), label_prompts=sum(bool(obj.get('label_prompt_id')) for obj in objects))
            self.emit('segment_enter', **context)
            started = time.perf_counter_ns()
            try:
                result = original_segment(runtime, camera, image, objects, captured_at, on_result=on_result)
                self.emit('segment_done', **context, start_ns=started,
                          masks=sum(bool(mask) for _, mask, _ in result), **runtime.last_segment_stats)
                return result
            except Exception as error:
                self.emit('segment_error', **context, error=str(error))
                raise

        @functools.wraps(original_image)
        def image(runtime, value):
            if not getattr(runtime.base.model, '_phase0_wrapped', False):
                runtime.base.model.forward_image = self.measure(runtime.base.model.forward_image, 'encoder')
                runtime.base.model._phase0_wrapped = True
            return original_image(runtime, value)

        @functools.wraps(original_predictor)
        def predictor(runtime, backbone, shape):
            result = original_predictor(runtime, backbone, shape)
            result.infer_live = infer_object(result.infer_live)
            result.commit_live_memory = self.measure(result.commit_live_memory, 'memory_encode')
            return result

        def infer_object(function):
            @functools.wraps(function)
            def wrapped(*args, **kwargs):
                context = getattr(self.local, 'context', None)
                if context:
                    import sys
                    caller = sys._getframe(1)
                    obj = caller.f_locals.get('obj', {})
                    context['object_id'] = obj.get('id')
                    context['label'] = obj.get('label_prompt_expected') or obj.get('label') or obj.get('class')
                    context['is_label'] = bool(obj.get('label_prompt_id'))
                return self.measure(function, 'decoder')(*args, **kwargs)
            return wrapped

        @functools.wraps(original_publish)
        def publish(owner, camera, frame, captured_at, detections, result):
            output = original_publish(owner, camera, frame, captured_at, detections, result)
            obj, mask, identity = result
            self.mark(camera, frame, 'cache_publish', object_id=obj['id'], has_mask=bool(mask),
                      accepted=bool(not identity or identity.get('accepted')), reason=(identity or {}).get('reason'))
            return output

        @functools.wraps(original_attach)
        def attach(owner, camera, objects, frame, now):
            output = original_attach(owner, camera, objects, frame, now)
            for obj in output:
                mask = obj.get('mask')
                if not mask:
                    continue
                record = self.frames.get((camera, mask.get('frame_id')))
                if record and obj['id'] not in record['attached']:
                    record['attached'].add(obj['id'])
                    self.mark(camera, mask['frame_id'], 'mask_attach', object_id=obj['id'], output_frame=frame,
                              label=obj.get('label'), mask_source=mask.get('source'))
            return output

        ModelSAM2._take_pending = take
        ModelSAM2._publish_result = publish
        ModelSAM2.attach = attach
        CudaMaskRuntime.segment = segment
        CudaMaskRuntime.__init__ = runtime_init
        CudaMaskRuntime._image = self.measure(image, 'image_preprocess_encoder')
        CudaMaskRuntime._predictor = predictor
        CudaMaskRuntime._binary = self.measure(CudaMaskRuntime._binary, 'binary_postprocess')
        CudaMaskRuntime._polygons = self.measure(CudaMaskRuntime._polygons, 'polygon_postprocess', cuda=False)
        CudaMaskRuntime._descriptor_from_features = self.measure(CudaMaskRuntime._descriptor_from_features, 'descriptor')


def initialize():
    global RECORDER
    try:
        control = json.loads(CONTROL.read_text())
    except (OSError, ValueError):
        return None
    if not control.get('enabled'):
        return None
    RECORDER = BaselineRecorder(control)
    RECORDER.install()
    return RECORDER


def mark(camera, frame, event, **values):
    if RECORDER is not None:
        RECORDER.mark(camera, frame, event, **values)
