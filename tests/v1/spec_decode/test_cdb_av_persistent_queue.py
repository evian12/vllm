# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from vllm.v1.request import RequestStatus
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



def _fake_request(req_id, draft_tokens=(), base_tokens=20):
    tokens = list(draft_tokens)
    return SimpleNamespace(
        request_id=req_id,
        status=RequestStatus.RUNNING,
        spec_token_ids=tokens,
        num_tokens_with_spec=base_tokens + len(tokens) + 1,
        num_output_placeholders=0,
        num_computed_tokens=base_tokens,
        is_finished=lambda: False,
    )


def test_target_admission_selects_only_available_running_slots():
    carry = _fake_request("carry")
    other0 = _fake_request("other0")
    other1 = _fake_request("other1")
    ready = {
        rid: _fake_request(rid, [index, index + 1])
        for index, rid in enumerate(("ready0", "ready1", "ready2"))
    }

    queue = PersistentDraftQueue.__new__(PersistentDraftQueue)
    queue.s = SimpleNamespace(
        max_num_running_reqs=4,
        max_num_scheduled_tokens=8192,
        running=[carry, other0, other1, *ready.values()],
        requests={"carry": carry, "other0": other0, "other1": other1, **ready},
    )
    queue.leases = {"carry"}
    queue.ready = {
        rid: {"tokens": list(req.spec_token_ids)}
        for rid, req in ready.items()
    }
    queue.inflight = {}
    queue._target_selected_ready = set()
    queue.window = 2

    selected = queue.select_target()

    assert selected == ["ready0", "ready1"]
    assert queue.frozen == {"carry", "ready2"}
    assert queue.preempt_protected == {
        "carry", "ready0", "ready1", "ready2"
    }


def test_target_verify_never_accepts_a_partial_candidate_budget():
    req = _fake_request("ready0", [10, 11, 12])
    queue = PersistentDraftQueue.__new__(PersistentDraftQueue)
    queue.ready = {"ready0": {"tokens": [10, 11, 12]}}
    queue._target_selected_ready = {"ready0"}

    assert queue.can_schedule_verify(req, 4)
    assert not queue.can_schedule_verify(req, 3)


def test_intra_window_refill_continues_after_a_request_becomes_ready(
    monkeypatch,
):
    monkeypatch.setenv("CDB_AV_INTRA_WINDOW", "1")
    queue = PersistentDraftQueue.__new__(PersistentDraftQueue)
    queue.window = 1
    queue.horizon = 8
    queue._p3_window_open = True
    queue._p3_done = False
    queue._p3_steps = 0
    queue._p3_ready_seen = False
    queue._cdb_admit_limit = None
    queue.carry = {}
    calls = []
    plan = {"req_id": "new-request"}

    def rpc(operation, payload=None):
        calls.append((operation, payload))
        if operation == "step":
            return {
                "progress": {"ready-request": 2},
                "ready_req_ids": ["ready-request"],
                "active_req_ids": [],
                "slots": 4,
            }
        if operation == "prepare":
            return {"prepared": [item["req_id"] for item in payload["plans"]]}
        raise AssertionError(operation)

    queue.rpc = rpc
    queue.plans = lambda: [plan]

    assert queue.advance_one_step()
    assert queue._p3_ready_seen is True
    assert queue._p3_steps == 1
    assert any(operation == "prepare" for operation, _ in calls)
    assert queue._p3_done is False



def test_target_admission_leaves_ready_overflow_queued():
    ready = {
        f"ready{index}": _fake_request(f"ready{index}", [index, index + 1])
        for index in range(8)
    }
    queue = PersistentDraftQueue.__new__(PersistentDraftQueue)
    queue.s = SimpleNamespace(
        max_num_running_reqs=4,
        max_num_scheduled_tokens=8192,
        running=list(ready.values()),
        requests=ready,
    )
    queue.leases = {}
    queue.ready = {
        rid: {"tokens": list(req.spec_token_ids)}
        for rid, req in ready.items()
    }
    queue.inflight = {}
    queue._target_selected_ready = set()
    queue.window = 3

    selected = queue.select_target()

    assert selected == ["ready0", "ready1", "ready2", "ready3"]
    assert queue.frozen == {"ready4", "ready5", "ready6", "ready7"}
    assert queue.preempt_protected == set(ready)


def test_target_admission_fails_fast_if_full_verify_cannot_fit():
    req = _fake_request("ready0", [1, 2, 3])
    queue = PersistentDraftQueue.__new__(PersistentDraftQueue)
    queue.s = SimpleNamespace(
        max_num_running_reqs=1,
        max_num_scheduled_tokens=3,
        running=[req],
        requests={"ready0": req},
    )
    queue.leases = {}
    queue.ready = {"ready0": {"tokens": [1, 2, 3]}}
    queue.inflight = {}
    queue._target_selected_ready = set()
    queue.window = 1

    with pytest.raises(RuntimeError, match="exceeding max_num_scheduled_tokens"):
        queue.select_target()
