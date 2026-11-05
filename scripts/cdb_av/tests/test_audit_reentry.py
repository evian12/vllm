import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location('audit_reentry', Path(__file__).parents[1] / 'audit_reentry.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def event(**kw):
    return '[CDB_REFILL] ' + json.dumps(kw)


def test_acceptance_and_later_formal_growth():
    lines = []
    for accepted in (0, 2, 4):
        rid = str(accepted)
        lines += [event(event='p2_verified', req_id=rid, context_tokens=10,
                        generated=list(range(accepted + 1))),
                  event(event='reuse_pending_prefill', req_id=rid,
                        context_tokens=14 + accepted, reused=10 + accepted,
                        pending_tokens=4)]
    result = module.audit(lines)
    assert result['formal_growth_after_verify'] == {3: 3}
    assert result['counts']['query_gt2_with_growth'] == 3


def test_empty_fresh_and_missing_evidence():
    assert module.audit([])['counts'] == {}
    result = module.audit([
        event(event='reuse_pending_prefill', req_id='fresh', context_tokens=20,
              reused=0, pending_tokens=20),
        event(event='reuse_pending_prefill', req_id='unknown', context_tokens=20,
              reused=19, pending_tokens=1)])
    assert result['counts']['uncached_or_fallback'] == 1
    assert result['counts']['cached_without_prior_verify_evidence'] == 1
    assert result['formal_growth_after_verify'] == {}


def test_no_growth_and_interleaved_request_identity():
    result = module.audit([
        event(event='p2_verified', req_id='a', context_tokens=10, generated=[1]),
        event(event='p2_verified', req_id='b', context_tokens=100, generated=[2]),
        event(event='reuse_pending_prefill', req_id='a', context_tokens=11,
              reused=10, pending_tokens=1)])
    assert result['formal_growth_after_verify'] == {0: 1}


def test_malformed_negative_and_shape_inconsistency():
    result = module.audit([
        '[CDB_REFILL] {',
        event(event='p2_verified', req_id='a', context_tokens=20, generated=[1]),
        event(event='reuse_pending_prefill', req_id='a', context_tokens=19,
              reused=18, pending_tokens=1),
        event(event='reuse_pending_prefill', req_id='a', context_tokens=20,
              reused=18, pending_tokens=1),
        event(event='mixed_batch', query_lengths=[1, 3], total_tokens=4),
        event(event='mixed_batch', query_lengths=[1, 3], total_tokens=3)])
    assert result['counts']['malformed_events'] == 1
    assert result['counts']['negative_context_growth'] == 1
    assert result['counts']['invalid_preparation_records'] == 1
    assert result['counts']['invalid_mixed_batch_records'] == 1
    assert result['mixed_forward_query_classes'] == {'q3plus': 1}
