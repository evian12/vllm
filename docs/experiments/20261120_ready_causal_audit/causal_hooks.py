"""Read-only causal event stream and exclusive host-thread CPU accounting.

No tensor inspection/synchronization, return-value edits, or scheduler mutation.
Observer's existing background writer drains rows; foreground does no file I/O.
"""
import functools
import threading
import time

EVENTS = {'ready', 'p2_handoff', 'p2_boundary', 'p2_verified',
          'draft_wait_enqueued', 'draft_wait_admitted', 'draft_wait_removed',
          'target_ready_select', 'p2_window_begin', 'p2_target_begin',
          'p2_draft_step', 'p2_idle_step', 'p2_reset', 'p2_reuse_plan'}
TLS = threading.local()


def normalize(event, data):
    result = dict(data)
    for name in ('tokens', 'drafts', 'generated'):
        if name in result:
            result[name + '_count'] = len(result.pop(name))
    return dict(kind='causal_event', event=event, **result)


def install(observer):
    def append(row):
        observer.start()
        with observer.LOCK:
            observer.ROWS.append(dict(audit_ns=time.monotonic_ns(),
                                      epoch=observer.EPOCH, **row))

    def augment(name, patch):
        old = observer.PATCHES.get(name)
        def patched(module):
            if old is not None:
                old(module)
            patch(module)
        observer.PATCHES[name] = patched

    def timer(cls, method, label):
        original = getattr(cls, method)
        @functools.wraps(original)
        def observed(self, *args, **kwargs):
            stack = getattr(TLS, 'stack', None)
            if stack is None:
                stack = TLS.stack = []
            frame = [time.thread_time_ns(), time.monotonic_ns(), 0, 0]
            stack.append(frame)
            try:
                return original(self, *args, **kwargs)
            finally:
                cpu = time.thread_time_ns() - frame[0]
                wall = time.monotonic_ns() - frame[1]
                assert stack.pop() is frame
                if stack:
                    stack[-1][2] += cpu
                    stack[-1][3] += wall
                append(dict(kind='host_stage', stage=label,
                            cpu_ns=cpu, wall_ns=wall,
                            cpu_exclusive_ns=cpu-frame[2],
                            wall_exclusive_ns=wall-frame[3]))
        setattr(cls, method, observed)

    def refill(module):
        old = module.emit
        @functools.wraps(old)
        def emitted(event, **data):
            if event in EVENTS:
                append(normalize(event, data))
            return old(event, **data)
        module.emit = emitted
        timer(module.RefillSession, '_forward', 'draft_forward')
        timer(module.RefillSession, '_decode_graph_forward', 'draft_graph_forward')

    def persistent(module):
        # This module imports refill.emit; import interception installs the
        # refill observer first, so every event is recorded exactly once.
        for name in ['refresh_draft_wait', '_draft_wait_reason', 'plans',
                     'advance', 'before_target', 'reconcile', 'finish_window']:
            if hasattr(module.PersistentDraftQueue, name):
                timer(module.PersistentDraftQueue, name, 'queue_' + name)

    def reuse(module):
        for name in ['step', 'prepare']:
            timer(module.ReuseDraftSession, name, 'draft_' + name)
        for name in ['plans', 'advance', 'before_target', 'reconcile',
                     'refresh_draft_wait', '_draft_wait_reason', 'finish_window']:
            if name in module.ReuseDraftQueue.__dict__:
                timer(module.ReuseDraftQueue, name, 'reuse_' + name)

    def runner(module):
        for name in ['execute_model', '_model_forward', 'sample_tokens',
                     '_prepare_inputs', '_prepare_input_ids', '_update_states',
                     '_build_attention_metadata', '_calc_spec_decode_metadata']:
            timer(module.GPUModelRunner, name, 'target_' + name)

    def scheduler(module):
        timer(module.Scheduler, 'schedule', 'target_schedule')
        original = module.Scheduler.schedule
        @functools.wraps(original)
        def scheduled(self, *args, **kwargs):
            output = original(self, *args, **kwargs)
            if output.num_scheduled_tokens:
                rows = {}
                for rid, n in output.num_scheduled_tokens.items():
                    req = self.requests.get(rid)
                    rows[rid] = dict(n=n,
                        output_before=req.num_output_tokens,
                        prompt_tokens=len(req.prompt_token_ids),
                        max_tokens=req.sampling_params.max_tokens,
                        draft_count=len(output.scheduled_spec_decode_tokens.get(rid, [])))
                p2 = getattr(self, '_cdb_p2', None)
                append(dict(kind='target_batch', rows=rows,
                    running=len(self.running), waiting=len(self.waiting),
                    draft_wait=len(getattr(p2, 'draft_wait', ())),
                    published_ready=len(getattr(p2, 'ready', ())),
                    leases=len(getattr(p2, 'leases', ())),
                    window=getattr(p2, 'window', None)))
            return output
        module.Scheduler.schedule = scheduled

    augment('vllm.v1.spec_decode.cdb_av_refill', refill)
    augment('vllm.v1.spec_decode.cdb_av_persistent', persistent)
    augment('vllm.v1.spec_decode.cdb_av_kv_reuse', reuse)
    augment('vllm.v1.worker.gpu_model_runner', runner)
    augment('vllm.v1.core.sched.scheduler', scheduler)
