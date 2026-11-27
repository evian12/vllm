"""Check the reserved shared-KV capacity boundary on the real queue class."""

from types import SimpleNamespace as State

import torch

from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.spec_decode.cdb_av_kv_reuse import ReuseDraftQueue


def test_kv_progress_reservation_boundaries():
    req_a = State(request_id='a', num_prompt_tokens=1024, max_tokens=1536)
    req_b = State(request_id='b', num_prompt_tokens=256, max_tokens=1792)
    spec = FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=32, dtype=torch.bfloat16)
    physical = {'a': [None] * 96}
    free = 0
    manager = State(enable_caching=False, block_pool=State(get_num_free_blocks=lambda: free),
                    kv_cache_config=State(kv_cache_groups=[State(kv_cache_spec=spec)]),
                    get_blocks=lambda rid: State(blocks=(physical.get(rid, ()),)))
    scheduler = State(kv_cache_manager=manager, connector=None, running=[req_a],
                      max_model_len=4096, num_lookahead_tokens=8, max_num_running_reqs=4)
    queue = ReuseDraftQueue.__new__(ReuseDraftQueue)
    queue.s, queue.block_size, queue.max_k = scheduler, 16, 8
    queue.leases, queue.cached_leases = {'a':[None] * 90}, {}
    required = 8 + (161-96) + (161-90) + 129 + 129
    free = required - 1
    assert not queue.target_admission_fits_kv(req_b)
    free = required
    assert queue.target_admission_fits_kv(req_b)
    queue.cached_leases = queue.leases
    queue.leases = {}
    assert queue.target_admission_fits_kv(req_b)
    scheduler.running = []
    free = 8+129+129
    assert queue.target_admission_fits_kv(req_b)
    free -= 1
    try:
        queue.target_admission_fits_kv(req_b)
    except RuntimeError as exc:
        assert 'cannot fit one request' in str(exc)
    else:
        raise AssertionError('A request entered an insufficient empty KV pool')
    manager.enable_caching = True
    try:
        queue.target_admission_fits_kv(req_b)
    except RuntimeError as exc:
        assert 'full-attention KV pool' in str(exc)
    else:
        raise AssertionError('An unsupported KV allocation profile was accepted')
    print('KV reservation accepts the exact free-page boundary and waits one page below it')
