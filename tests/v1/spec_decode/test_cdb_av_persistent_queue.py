# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from vllm.v1.spec_decode.cdb_av_persistent import (
    PersistentDraftQueue,
    PersistentDraftSession,
)


def test_persistent_session_advances_idle_ticks_until_fixed_h():
    session = PersistentDraftSession.__new__(PersistentDraftSession)
    session.window = 1
    session.done = False
    session.horizon = 8
    session.steps_completed = 3
    session.active = []
    session.pending_prefill = []
    session.capacity = 4
    session.members = {
        "r1": {"ready": True, "progress": 2},
        "r2": {"ready": True, "progress": 4},
    }

    status = session.step()

    assert session.steps_completed == 4
    assert status["steps_completed"] == 4
    assert status["active_req_ids"] == []
    assert status["ready_req_ids"] == ["r1", "r2"]
    assert status["slots"] == 4
    assert status["remaining_steps"] == 4


def test_persistent_session_rejects_step_after_h():
    session = PersistentDraftSession.__new__(PersistentDraftSession)
    session.window = 1
    session.done = False
    session.horizon = 8
    session.steps_completed = 8
    session.active = []
    session.pending_prefill = []
    session.capacity = 4
    session.members = {}

    with pytest.raises(RuntimeError, match="already reached H"):
        session.step()


def test_ready_requests_are_frozen_except_selected_target_subset():
    queue = PersistentDraftQueue.__new__(PersistentDraftQueue)
    queue.leases = {"carry"}
    queue.ready = {"ready0": {}, "ready1": {}, "ready2": {}}
    queue.inflight = {"verify": {}}
    queue._target_selected_ready = {"ready1"}

    assert queue.frozen == {"carry", "ready0", "ready2"}
    assert queue.preempt_protected == {
        "carry", "ready0", "ready1", "ready2", "verify"
    }


def test_finish_window_rejects_ready_early_before_h():
    queue = PersistentDraftQueue.__new__(PersistentDraftQueue)
    queue._p3_window_open = True
    queue._p3_steps = 2
    queue.horizon = 8

    with pytest.raises(RuntimeError, match="exactly H=8"):
        queue.finish_window()


def test_advance_runs_one_draft_step_per_engine_iteration():
    queue = PersistentDraftQueue.__new__(PersistentDraftQueue)
    queue._p3_window_open = True
    queue._p3_steps = 0
    queue.horizon = 3
    calls = {"step": 0, "finish": 0}

    def one_step():
        calls["step"] += 1
        queue._p3_steps += 1
        return True

    def finish():
        calls["finish"] += 1

    queue.advance_one_step = one_step
    queue.finish_window = finish

    queue.advance(model_executed=False)
    assert calls == {"step": 1, "finish": 0}

    queue.advance(model_executed=False)
    assert calls == {"step": 2, "finish": 0}

    queue.advance(model_executed=False)
    assert calls == {"step": 3, "finish": 1}


def _fake_req(req_id):
    from vllm.v1.request import RequestStatus

    return SimpleNamespace(
        request_id=req_id,
        status=RequestStatus.RUNNING,
        is_finished=lambda: False,
    )


def test_ready_requests_remain_frozen_during_open_window():
    queue = PersistentDraftQueue.__new__(PersistentDraftQueue)
    queue.leases = {"carry"}
    queue.ready = {"r1": {}, "r2": {}}
    queue.inflight = {}
    queue._p3_window_open = True
    queue._target_selected_ready = {"r1"}

    queue.begin_target_schedule()

    assert queue._target_selected_ready == set()
    assert queue.frozen == {"carry", "r1", "r2"}


def test_target_ready_selection_respects_remaining_capacity():
    queue = PersistentDraftQueue.__new__(PersistentDraftQueue)
    queue.leases = {"carry"}
    queue.ready = {f"r{i}": {} for i in range(8)}
    queue.inflight = {}
    queue._p3_window_open = False
    queue._target_selected_ready = set()
    queue.s = SimpleNamespace(
        max_num_running_reqs=4,
        running=[
            _fake_req("carry"),
            _fake_req("ar0"),
            _fake_req("r0"),
            _fake_req("r1"),
            _fake_req("r2"),
            _fake_req("r3"),
            _fake_req("r4"),
            _fake_req("r5"),
            _fake_req("r6"),
            _fake_req("r7"),
        ],
    )

    queue.begin_target_schedule()

    assert queue._target_selected_ready == {"r0", "r1", "r2"}
    assert queue.frozen == {"carry", "r3", "r4", "r5", "r6", "r7"}


def test_ready_verification_is_all_or_nothing():
    queue = PersistentDraftQueue.__new__(PersistentDraftQueue)
    queue.ready = {
        "r1": {"tokens": [11, 12, 13], "window": 2},
    }
    request = SimpleNamespace(
        request_id="r1",
        spec_token_ids=[11, 12, 13],
        num_computed_tokens=9,
        num_tokens=10,
        num_output_placeholders=0,
    )

    assert not queue.can_schedule_verify(request, num_new_tokens=3)
    assert queue.can_schedule_verify(request, num_new_tokens=4)


def test_ready_verification_rejects_mutated_candidates():
    queue = PersistentDraftQueue.__new__(PersistentDraftQueue)
    queue.ready = {
        "r1": {"tokens": [11, 12, 13], "window": 2},
    }
    request = SimpleNamespace(
        request_id="r1",
        spec_token_ids=[11, 99, 13],
        num_computed_tokens=9,
        num_tokens=10,
        num_output_placeholders=0,
    )

    with pytest.raises(RuntimeError, match="candidate list changed"):
        queue.can_schedule_verify(request, num_new_tokens=4)


def test_intra_window_refill_is_not_suppressed_by_ready(monkeypatch):
    from vllm.v1.spec_decode import cdb_av_config

    monkeypatch.setattr(
        cdb_av_config, "is_enabled", lambda name, default=False: True
    )
    queue = PersistentDraftQueue.__new__(PersistentDraftQueue)
    queue._p3_window_open = True
    queue._p3_steps = 0
    queue.horizon = 8
    queue._p3_ready_seen = False
    queue._p3_done = False
    queue._cdb_admit_limit = None
    queue.carry = {}
    calls = []

    def rpc(operation, payload=None):
        calls.append((operation, payload))
        if operation == "step":
            return {
                "steps_completed": 1,
                "progress": {"r1": 1},
                "ready_req_ids": ["r1"],
                "active_req_ids": [],
                "slots": 4,
                "remaining_steps": 7,
            }
        if operation == "prepare":
            return {"prepared": ["r2"]}
        raise AssertionError(operation)

    queue.rpc = rpc
    queue.plans = lambda: [{"req_id": "r2"}]

    assert queue.advance_one_step()
    assert queue._p3_ready_seen
    assert queue._p3_steps == 1
    assert [name for name, _ in calls] == ["step", "prepare"]


def test_fixed_h_waits_for_all_eight_step_ticks_even_if_ready():
    queue = PersistentDraftQueue.__new__(PersistentDraftQueue)
    queue._p3_window_open = True
    queue._p3_steps = 0
    queue.horizon = 8
    calls = {"step": 0, "finish": 0}

    def one_step():
        calls["step"] += 1
        queue._p3_steps += 1
        queue._p3_ready_seen = True
        return True

    def finish():
        calls["finish"] += 1

    queue.advance_one_step = one_step
    queue.finish_window = finish

    for _ in range(7):
        queue.advance(model_executed=False)
    assert calls == {"step": 7, "finish": 0}

    queue.advance(model_executed=False)
    assert calls == {"step": 8, "finish": 1}


def test_target_selection_shrinks_to_actually_scheduled_ready():
    queue = PersistentDraftQueue.__new__(PersistentDraftQueue)
    queue.leases = set()
    queue.ready = {"r0": {}, "r1": {}, "r2": {}}
    queue.inflight = {}
    queue._p3_window_open = False
    queue._target_selected_ready = {"r0", "r1"}

    queue.update_target_selection({"r0"})

    assert queue._target_selected_ready == {"r0"}
    assert queue.frozen == {"r1", "r2"}


def test_target_selection_cannot_admit_ready_before_h():
    queue = PersistentDraftQueue.__new__(PersistentDraftQueue)
    queue.leases = set()
    queue.ready = {"r0": {}}
    queue.inflight = {}
    queue._p3_window_open = True
    queue._target_selected_ready = {"r0"}

    with pytest.raises(RuntimeError, match="before the H boundary"):
        queue.update_target_selection({"r0"})


def test_unsupported_request_is_left_for_target_only_generation():
    queue = PersistentDraftQueue.__new__(PersistentDraftQueue)
    request = SimpleNamespace(
        sampling_params=SimpleNamespace(temperature=0, prompt_logprobs=None),
        has_encoder_inputs=False,
        lora_request=None,
        structured_output_request=None,
    )

    assert queue.unsupported_reason(request) is None

    request.sampling_params.temperature = 0.7
    assert queue.unsupported_reason(request) == "non_greedy_sampling"

    request.sampling_params.temperature = 0
    request.structured_output_request = object()
    assert queue.unsupported_reason(request) == "structured_output"


def test_plan_rollback_drops_worker_before_freeing_lease():
    class FakeBlock:
        def __init__(self, block_id):
            self.block_id = block_id
            self.ref_cnt = 1
            self.is_null = False

    class FakePool:
        def __init__(self):
            self.freed = []

        def free_blocks(self, blocks):
            self.freed.extend(blocks)
            for block in blocks:
                block.ref_cnt = 0

    block = FakeBlock(7)
    pool = FakePool()
    queue = PersistentDraftQueue.__new__(PersistentDraftQueue)
    queue.window = 2
    queue.leases = {"r1": [block]}
    queue.contexts = {"r1": {"context": {"context_tokens": 3}, "computed": 2}}
    queue.carry = {}
    queue.ready = {}
    queue.inflight = {}
    queue.s = SimpleNamespace(
        kv_cache_manager=SimpleNamespace(block_pool=pool)
    )
    calls = []

    def rpc(operation, payload=None):
        calls.append((operation, payload))
        return {"dropped": payload["ids"]}

    queue.rpc = rpc
    queue.rollback_plans(["r1"], "test_rollback")

    assert calls == [("drop", {"ids": ["r1"]})]
    assert queue.leases == {}
    assert queue.contexts == {}
    assert pool.freed == [block]
    assert block.ref_cnt == 0


def test_plan_rollback_ack_failure_preserves_allocator_ownership():
    class FakeBlock:
        block_id = 9
        ref_cnt = 1
        is_null = False

    block = FakeBlock()
    queue = PersistentDraftQueue.__new__(PersistentDraftQueue)
    queue.leases = {"r1": [block]}
    queue.contexts = {"r1": {"context": {}, "computed": 0}}
    queue.carry = {}
    queue.s = SimpleNamespace(
        kv_cache_manager=SimpleNamespace(
            block_pool=SimpleNamespace(free_blocks=lambda blocks: None)
        )
    )
    queue.rpc = lambda operation, payload=None: {"dropped": []}

    with pytest.raises(RuntimeError, match="rollback acknowledgement mismatch"):
        queue.rollback_plans(["r1"], "test_rollback")

    assert queue.leases == {"r1": [block]}
    assert block.ref_cnt == 1
