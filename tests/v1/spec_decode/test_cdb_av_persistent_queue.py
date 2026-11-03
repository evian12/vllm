# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from vllm.v1.request import RequestStatus
from vllm.v1.spec_decode.cdb_av_refill_queue import RefillQueue

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



def test_target_ready_queue_reserves_prefill_and_keeps_overflow_queued(monkeypatch):
    queue = PersistentDraftQueue.__new__(PersistentDraftQueue)
    queue.window = 1
    queue.target_epoch = 0
    queue.leases = {"carry"}
    queue.inflight = {"inflight": {}}
    ready_ids = [f"ready{i}" for i in range(8)]
    ready_tokens = {
        rid: [100 + i, 200 + i]
        for i, rid in enumerate(ready_ids)
    }
    requests = {}

    def make_request(rid, arrival_time, tokens):
        return SimpleNamespace(
            request_id=rid,
            status=RequestStatus.RUNNING,
            spec_token_ids=list(tokens),
            arrival_time=arrival_time,
            num_output_tokens=1,
            num_computed_tokens=5,
            num_tokens=6,
            is_finished=lambda: False,
        )

    carry = make_request("carry", 0.0, [])
    requests["carry"] = carry
    running = [carry]
    for i, rid in enumerate(ready_ids):
        req = make_request(rid, float(i + 1), ready_tokens[rid])
        requests[rid] = req
        running.append(req)
    for i in range(4):
        rid = f"ar{i}"
        req = make_request(rid, float(20 + i), [])
        requests[rid] = req
        running.append(req)

    queue.ready = {
        rid: {"tokens": ready_tokens[rid], "ready_at": float(i)}
        for i, rid in enumerate(ready_ids)
    }
    queue._target_selected_ready = set()
    queue._target_yield = set()
    queue.s = SimpleNamespace(
        max_num_running_reqs=4,
        running=running,
        requests=requests,
        waiting=[object()],
    )

    selected = queue.prepare_target()

    # One target slot is reserved for a fresh Prefill/first-token request.
    assert selected == ready_ids[:3]
    assert queue._target_yield == {f"ar{i}" for i in range(4)}
    assert queue.frozen == (
        {"carry"}
        | set(ready_ids[3:])
        | {f"ar{i}" for i in range(4)}
    )
    assert queue.preempt_protected == {"carry", "inflight"} | set(ready_ids)

    # Without fresh waiting work, all four Target slots can verify Ready work.
    queue.s.waiting = []
    selected = queue.prepare_target()
    assert selected == ready_ids[:4]
    assert queue._target_yield == {f"ar{i}" for i in range(4)}



def test_target_ready_admission_does_not_yield_in_progress_prefill():
    queue = PersistentDraftQueue.__new__(PersistentDraftQueue)
    queue.window = 2
    queue.target_epoch = 1
    queue.leases = set()
    queue.inflight = {}
    ready_ids = [f"ready{i}" for i in range(8)]
    requests = {}

    def make_request(rid, arrival_time, tokens, output_tokens, computed, total):
        return SimpleNamespace(
            request_id=rid,
            status=RequestStatus.RUNNING,
            spec_token_ids=list(tokens),
            arrival_time=arrival_time,
            num_output_tokens=output_tokens,
            num_computed_tokens=computed,
            num_tokens=total,
            is_finished=lambda: False,
        )

    running = []
    for i, rid in enumerate(ready_ids):
        tokens = [300 + i, 400 + i]
        req = make_request(rid, float(i), tokens, 1, 5, 6)
        requests[rid] = req
        running.append(req)
    prefill = make_request("prefill", 30.0, [], 0, 5, 10)
    requests["prefill"] = prefill
    running.append(prefill)
    for i in range(4):
        rid = f"ar{i}"
        req = make_request(rid, float(40 + i), [], 1, 5, 6)
        requests[rid] = req
        running.append(req)

    queue.ready = {
        rid: {
            "tokens": [300 + i, 400 + i],
            "ready_at": float(i),
        }
        for i, rid in enumerate(ready_ids)
    }
    queue._target_selected_ready = set()
    queue._target_yield = set()
    queue.s = SimpleNamespace(
        max_num_running_reqs=4,
        running=running,
        requests=requests,
        waiting=[object()],
    )

    selected = queue.prepare_target()

    # Keep the in-progress Prefill and reserve one new Prefill slot.
    assert selected == ready_ids[:2]
    assert queue._target_yield == {f"ar{i}" for i in range(4)}
    assert "prefill" not in queue.frozen
    assert "prefill" not in queue.preempt_protected



def test_before_target_moves_only_selected_ready_requests_to_inflight():
    queue = PersistentDraftQueue.__new__(PersistentDraftQueue)
    queue.window = 3
    queue.target_epoch = 2
    queue.leases = {}
    queue.ready = {}
    queue.inflight = {}
    queue._target_selected_ready = {"r0"}
    queue._target_yield = set()

    requests = {}
    for rid, prompt, output, drafts in (
        ("r0", [1, 2], [3], [11, 12]),
        ("r1", [4, 5], [6], [21, 22]),
    ):
        req = SimpleNamespace(
            request_id=rid,
            prompt_token_ids=prompt,
            output_token_ids=output,
            spec_token_ids=list(drafts),
            status=RequestStatus.RUNNING,
            is_finished=lambda: False,
        )
        requests[rid] = req
        ctx = RefillQueue.context(req)
        queue.ready[rid] = {
            "tokens": list(drafts),
            "context_tokens": ctx["context_tokens"],
            "fingerprint": ctx["fingerprint"],
            "ready_at": 1.0,
            "window": 2,
        }

    queue.s = SimpleNamespace(
        running=list(requests.values()),
        requests=requests,
        kv_cache_manager=SimpleNamespace(get_block_ids=lambda _rid: []),
    )
    queue.reconcile = lambda: None
    output = SimpleNamespace(
        num_scheduled_tokens={"r0": 3},
        total_num_scheduled_tokens=3,
        scheduled_spec_decode_tokens={"r0": [11, 12]},
    )

    queue.before_target(output)

    assert set(queue.inflight) == {"r0"}
    assert set(queue.ready) == {"r1"}
    assert queue._target_selected_ready == set()
    assert queue._target_yield == set()
    assert queue.frozen == {"r1"}


def test_before_target_rejects_unselected_ready_verification():
    queue = PersistentDraftQueue.__new__(PersistentDraftQueue)
    queue.window = 1
    queue.target_epoch = 0
    queue.leases = {}
    queue.inflight = {}
    queue.ready = {"r0": {}, "r1": {}}
    queue._target_selected_ready = {"r0"}
    queue._target_yield = set()
    queue.s = SimpleNamespace(
        running=[],
        requests={},
        kv_cache_manager=SimpleNamespace(get_block_ids=lambda _rid: []),
    )
    queue.reconcile = lambda: None
    output = SimpleNamespace(
        num_scheduled_tokens={"r1": 3},
        total_num_scheduled_tokens=3,
        scheduled_spec_decode_tokens={"r1": [1, 2]},
    )

    with pytest.raises(RuntimeError, match="scheduled frozen Carry"):
        queue.before_target(output)
