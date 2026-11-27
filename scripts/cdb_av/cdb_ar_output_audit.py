"""Count tokens emitted by a non-Verify Target step after initial prefill."""

from functools import wraps


def attach(observer, *, strict=False):
    target = 'vllm.v1.core.sched.scheduler'
    patch_scheduler = observer.PATCHES[target]

    def record_tokens(key, value=1):
        # manual_observer.record requires pre-registered keys. Output counters
        # include reason-specific names, so register and add under its lock.
        observer.start()
        with observer.LOCK:
            observer.STATS[key] = observer.STATS.get(key, 0) + value

    def patch(module):
        patch_scheduler(module)
        install(module.Scheduler, record_tokens, strict=strict)

    observer.PATCHES[target] = patch


def install(scheduler_cls, record, *, strict=False):
    original = scheduler_cls.update_from_output

    @wraps(original)
    def audited(scheduler, scheduled, model_output):
        p2 = getattr(scheduler, '_cdb_p2', None)
        if p2 is None:
            return original(scheduler, scheduled, model_output)
        verify = scheduled.scheduled_spec_decode_tokens
        prior = {rid: scheduler.requests[rid].num_output_tokens
                 for rid in scheduled.num_scheduled_tokens if rid in scheduler.requests}
        reasons = {rid: p2._draft_wait_fallbacks.get(rid, ('unclassified',))[0]
                   for rid in prior if prior[rid] and rid not in verify}
        outputs = original(scheduler, scheduled, model_output)
        for group in outputs.values():
            for output in group.outputs:
                rid = output.request_id
                tokens = len(output.new_token_ids)
                if not tokens:
                    continue
                if rid not in prior:
                    raise RuntimeError('CDB output audit observed tokens outside the scheduled batch')
                record('cdb_output_tokens_audited', tokens)
                if rid in verify:
                    record('cdb_output_verify_tokens', tokens)
                    continue
                if prior[rid] == 0:
                    record('cdb_output_first_tokens', 1)
                    tokens -= 1
                if tokens:
                    reason = reasons.get(rid, 'unclassified')
                    record('cdb_output_ar_fallback_tokens', tokens)
                    record(f'cdb_output_ar_fallback_{reason}_tokens', tokens)
                    if strict:
                        raise RuntimeError(f'CDB emitted {tokens} AR fallback tokens ({reason})')
        return outputs

    scheduler_cls.update_from_output = audited
