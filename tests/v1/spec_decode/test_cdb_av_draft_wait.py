from types import SimpleNamespace

import pytest

from vllm.v1.request import RequestStatus
from vllm.v1.spec_decode.cdb_av_persistent import PersistentDraftQueue


class Request:
    def __init__(self, rid):
        self.request_id = rid
        self.prompt_token_ids = [1, 2]
        self.output_token_ids = [3]
        self.spec_token_ids = []
        self.num_computed_tokens = 2
        self.status = RequestStatus.RUNNING
        self.sampling_params = SimpleNamespace(temperature=0, prompt_logprobs=None, extra_args=None)
        self.prompt_embeds = self.lora_request = self.structured_output_request = None
        self.mm_features = []
        self.has_encoder_inputs = False
        self.max_tokens = 128
        self.finished = False

    @property
    def num_tokens(self):
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    @property
    def num_output_tokens(self):
        return len(self.output_token_ids)

    def is_finished(self):
        return self.finished


class Pool:
    def __init__(self):
        self.free = 100
        self.allocations = []

    def get_num_free_blocks(self):
        return self.free

    def get_new_blocks(self, n):
        blocks = [SimpleNamespace(block_id=len(self.allocations)+i+1, ref_cnt=1, is_null=False) for i in range(n)]
        self.free -= n
        self.allocations.extend(blocks)
        return blocks


@pytest.fixture
def queue(monkeypatch):
    monkeypatch.setenv('CDB_AV_OUTPUT_BUDGET_CAP', '0')
    q = PersistentDraftQueue.__new__(PersistentDraftQueue)
    q.draft_wait_enabled = True
    q.wait_prefix_enabled = True
    q.draft_wait = {}
    q._draft_wait_fallbacks = {}
    q.leases, q.contexts, q.carry, q.ready, q.inflight = {}, {}, {}, {}, {}
    q.acceptance_replay = q.request_average = None
    q._cdb_admit_limit = q._cdb_admit_token_limit = None
    q._uses_mixed_prefill = False
    q._target_selected_ready = set()
    q._p3_window_open = False
    q.window = 1
    q.block_size = 16
    q.max_k = 8
    q.s = SimpleNamespace(running=[], requests={}, max_num_running_reqs=4,
        max_num_scheduled_tokens=1024, max_model_len=1024,
        kv_cache_manager=SimpleNamespace(block_pool=Pool()))
    return q


def add(q, *ids):
    for rid in ids:
        req = Request(rid)
        q.s.running.append(req)
        q.s.requests[rid] = req


def test_prefill_completion_enrolls_without_allocating_draft_pages(queue):
    add(queue, 'a')
    req = queue.s.requests['a']
    req.output_token_ids.clear()
    queue.refresh_draft_wait()
    assert queue.draft_wait == {}
    req.output_token_ids.append(3)
    queue.refresh_draft_wait()
    assert list(queue.draft_wait) == ['a']
    assert queue.frozen == {'a'}
    assert not queue.s.kv_cache_manager.block_pool.allocations


def test_full_draft_capacity_keeps_fifo_waiters_frozen(queue):
    add(queue, 'a', 'b', 'c')
    queue.refresh_draft_wait()
    # Running order changes must not reorder already-enqueued requests.
    queue.s.running.reverse()
    queue._cdb_admit_limit = 0
    assert queue.plans() == []
    assert list(queue.draft_wait) == ['a', 'b', 'c']
    queue._cdb_admit_limit = 1
    assert [p['req_id'] for p in queue.plans()] == ['a']
    assert list(queue.draft_wait) == ['b', 'c']
    assert set(queue.leases) == {'a'}
    assert queue.frozen == {'a', 'b', 'c'}


def test_waiters_do_not_reserve_target_slots_ahead_of_ready(queue):
    add(queue, 'a', 'b', 'c', 'r')
    queue.ready['r'] = dict(tokens=[7], window=1)
    queue.begin_target_schedule()
    assert queue._target_selected_ready == {'r'}
    assert queue.frozen == {'a', 'b', 'c'}
    assert queue.preempt_protected == {'r'}


@pytest.mark.parametrize('change', ['token', 'computed'])
def test_wait_context_cannot_advance_silently(queue, change):
    add(queue, 'a')
    queue.refresh_draft_wait()
    req = queue.s.requests['a']
    if change == 'token':
        req.output_token_ids[0] = 99
    else:
        req.num_computed_tokens += 1
    with pytest.raises(RuntimeError, match='advanced a Draft-wait context'):
        queue.refresh_draft_wait()


@pytest.mark.parametrize('state', ['cancel', 'preempt', 'missing', 'replace'])
def test_invalid_waiter_does_not_hold_a_queue_slot_or_draft_pages(queue, state):
    add(queue, 'a')
    queue.refresh_draft_wait()
    req = queue.s.requests['a']
    if state == 'cancel':
        req.finished = True
    elif state == 'preempt':
        req.status = RequestStatus.PREEMPTED
    elif state == 'missing':
        del queue.s.requests['a']
        queue.s.running.clear()
    else:
        queue.s.requests['a'] = Request('a')
        queue.s.running = [queue.s.requests['a']]
    queue.refresh_draft_wait(enroll=False)
    assert not queue.draft_wait
    assert not queue.leases
    assert queue.s.kv_cache_manager.block_pool.free == 100


@pytest.mark.parametrize('reason', ['kv_resource', 'model_length', 'non_greedy_sampling', 'target_verify_budget'])
def test_only_explicit_resource_or_support_limits_release_to_native(queue, reason):
    add(queue, 'a')
    queue.refresh_draft_wait()
    if reason == 'kv_resource':
        queue.s.kv_cache_manager.block_pool.free = 0
    elif reason == 'model_length':
        queue.s.max_model_len = 8
    elif reason == 'non_greedy_sampling':
        queue.s.requests['a'].sampling_params.temperature = .7
    else:
        queue.s.max_num_scheduled_tokens = 4
    queue.refresh_draft_wait()
    assert not queue.draft_wait
    assert queue._draft_wait_fallbacks['a'][0] == reason
    assert 'a' not in queue.frozen


def test_validation_after_target_selection_does_not_enroll_new_ar_rows(queue):
    add(queue, 'a')
    queue.refresh_draft_wait(enroll=False)
    assert not queue.draft_wait


def test_legacy_off_keeps_running_scan_and_ar_eligibility(queue):
    queue.draft_wait_enabled = False
    add(queue, 'a')
    queue.refresh_draft_wait()
    assert not queue.frozen
    assert [p['req_id'] for p in queue.plans()] == ['a']


def test_all_waiting_can_open_draft_window_without_target_forward(queue):
    add(queue, 'a')
    queue.horizon = 8
    queue.target_epoch = 1
    queue.refresh_draft_wait()
    queue.reconcile = lambda: None
    queue.rpc = lambda op, payload: (dict(opened=True, window=payload['window'], horizon=8, capacity=4, max_k=8)
        if op == 'open' else dict(prepared=[p['req_id'] for p in payload['plans']]))
    assert queue.begin_window(model_executed=False)
    assert queue._p3_window_open
    assert 'a' in queue.leases
    assert not queue.draft_wait


def test_reset_clears_waiters_only_after_worker_acknowledges(queue):
    add(queue, 'a')
    queue.refresh_draft_wait()
    queue.target_epoch = 1
    queue._worker_boundary_ready_ids = set()
    queue.rpc = lambda op, payload: dict(reset=False)
    with pytest.raises(RuntimeError, match='Worker reset failed'):
        queue.reset()
    assert 'a' in queue.draft_wait
    queue.rpc = lambda op, payload: dict(reset=True, dropped=[])
    queue.reset()
    assert not queue.draft_wait
    assert queue.s.kv_cache_manager.block_pool.free == 100


def test_waiting_and_owned_partitions_cannot_overlap(queue):
    add(queue, 'a')
    queue.refresh_draft_wait()
    queue.leases['a'] = []
    with pytest.raises(RuntimeError, match='overlaps an owned candidate'):
        queue.refresh_draft_wait()


@pytest.mark.parametrize('cached', [False, True])
def test_kv_reuse_planner_performs_the_same_fifo_wait_to_lease_transfer(queue, cached):
    from vllm.v1.spec_decode.cdb_av_kv_reuse import ReuseDraftQueue
    reuse = ReuseDraftQueue.__new__(ReuseDraftQueue)
    reuse.__dict__.update(queue.__dict__)
    reuse.cached_leases, reuse.audit_leases, reuse.commits = {}, {}, {}
    reuse.history_pages = set()
    reuse.pending_allocations = []
    reuse._uses_mixed_prefill = True
    add(reuse, 'a', 'b')
    if cached:
        reuse.cached_leases['a'] = reuse.s.kv_cache_manager.block_pool.get_new_blocks(1)
    reuse.refresh_draft_wait()
    reuse.s.running.reverse()
    reuse._cdb_admit_limit = 1
    plans = reuse.plans()
    assert [p['req_id'] for p in plans] == ['a']
    assert plans[0]['reuse'] == cached
    assert list(reuse.draft_wait) == ['b']
    assert set(reuse.leases) == {'a'}
    reuse.refresh_draft_wait(enroll=False)
    assert reuse.frozen == {'a', 'b'}


def test_cached_pages_reduce_required_new_kv_resource(queue):
    add(queue, 'a')
    queue.cached_leases = {'a': [object()]}
    queue.s.kv_cache_manager.block_pool.free = 8
    queue.refresh_draft_wait()
    assert 'a' in queue.draft_wait


@pytest.mark.parametrize('kind', ['bool', 'float', 'subclass', 'prompt', 'append', 'truncate'])
def test_typed_prefix_rejects_same_value_types_and_all_mutation_kinds(queue, kind):
    add(queue, 'a')
    req = queue.s.requests['a']
    req.output_token_ids[0] = 1
    queue.refresh_draft_wait()
    class Int(int): pass
    if kind == 'bool': req.output_token_ids[0] = True
    elif kind == 'float': req.output_token_ids[0] = 1.0
    elif kind == 'subclass': req.output_token_ids[0] = Int(1)
    elif kind == 'prompt': req.prompt_token_ids[0] = 9
    elif kind == 'append': req.output_token_ids.append(2)
    else: req.output_token_ids.clear()
    with pytest.raises(RuntimeError, match='advanced a Draft-wait context'):
        queue.refresh_draft_wait(enroll=False)


def test_typed_wait_validation_does_not_repeat_digest_lookup(queue, monkeypatch):
    from vllm.v1.spec_decode.cdb_av_refill_queue import RefillQueue
    add(queue, 'a')
    queue.refresh_draft_wait()
    assert queue.draft_wait['a']['typed_prefix'] == ((1, 2), (3,))
    def unexpected(req): raise AssertionError('repeated context hash lookup')
    monkeypatch.setattr(RefillQueue, 'context', unexpected)
    queue.refresh_draft_wait(enroll=False)


def test_prefix_snapshot_bound_and_special_type_fallback():
    from vllm.v1.spec_decode.cdb_av_wait_prefix import snapshot
    r = Request('a')
    r.prompt_token_ids = [1]*1024
    assert snapshot(r) is None
    r.prompt_token_ids = [True]
    assert snapshot(r) is None


def test_old_wait_flag_still_uses_digest_guard(queue):
    queue.wait_prefix_enabled = False
    add(queue, 'a')
    queue.refresh_draft_wait()
    assert 'typed_prefix' not in queue.draft_wait['a']
    queue.s.requests['a'].output_token_ids[0] = 99
    with pytest.raises(RuntimeError, match='advanced a Draft-wait context'):
        queue.refresh_draft_wait(enroll=False)
