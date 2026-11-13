# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace as NS

import pytest

from vllm.v1.request import RequestStatus
from vllm.v1.spec_decode.cdb_av_kv_reuse import ReuseDraftQueue
from vllm.v1.spec_decode.cdb_av_persistent import PersistentDraftQueue


def queue(fast):
    q = object.__new__(ReuseDraftQueue)
    q._reuse_parent_ownership_check = fast
    block = lambda i: NS(block_id=i, ref_cnt=1, is_null=False)
    q.leases = {"active": [block(1)]}
    q.cached_leases = {"cached": [block(2)]}
    q.audit_leases = {"audit": [block(3)]}
    q.replay_blocks = [block(4)]
    q.pending_allocations = [[block(5)]]
    q.contexts = {}
    q.ready = {}
    q.inflight = {}
    q.carry = {}
    q.commits = {}
    q.window = q.target_epoch = 0
    q._target_selected_ready = set()
    q.formal_ids = [100]
    q.block_reads = 0
    def get_blocks(rid):
        q.block_reads += 1
        assert rid == "formal"
        return [q.formal_ids]
    req = lambda: NS(status=RequestStatus.RUNNING, is_finished=lambda: False)
    q.s = NS(requests={"cached": req()},
             running=[NS(request_id="formal")],
             kv_cache_manager=NS(get_block_ids=get_blocks))
    return q


def output(work):
    return NS(scheduled_spec_decode_tokens={},
              num_scheduled_tokens={"formal": 1} if work else {},
              total_num_scheduled_tokens=int(work))


@pytest.mark.parametrize("fast", [False, True])
@pytest.mark.parametrize("work", [False, True])
def test_real_parent_validation_and_only_duplicate_scan_removed(fast, work):
    q = queue(fast)
    q.before_target(output(work))
    assert q.block_reads == (1 if fast else 2)
    assert q.target_epoch == int(work)
    assert q.leases["active"][0].block_id == 1
    assert q.cached_leases["cached"][0].block_id == 2
    assert not q.ready and not q.inflight


@pytest.mark.parametrize("fast", [False, True])
@pytest.mark.parametrize("bad", [
    "active_cached_duplicate", "formal_overlap", "audit_null",
    "replay_shared", "pending_duplicate", "active_zero_ref",
    "cached_null", "audit_replay_duplicate",
])
def test_strong_all_owner_checks_remain(fast, bad):
    q = queue(fast)
    if bad == "active_cached_duplicate":
        q.cached_leases["cached"][0].block_id = 1
    elif bad == "formal_overlap":
        q.formal_ids[:] = [2]
    elif bad == "audit_null":
        q.audit_leases["audit"][0].is_null = True
    elif bad == "replay_shared":
        q.replay_blocks[0].ref_cnt = 2
    elif bad == "pending_duplicate":
        q.pending_allocations[0][0].block_id = 1
    elif bad == "active_zero_ref":
        q.leases["active"][0].ref_cnt = 0
    elif bad == "cached_null":
        q.cached_leases["cached"][0].is_null = True
    elif bad == "audit_replay_duplicate":
        q.replay_blocks[0].block_id = 3
    with pytest.raises(RuntimeError, match="overlap or lost exclusive"):
        q.before_target(output(True))
    assert q.target_epoch == 0
    assert not q.ready and not q.inflight
    assert "active" in q.leases and "cached" in q.cached_leases


@pytest.mark.parametrize("fast", [False, True])
def test_no_validation_cache_across_target_calls(fast):
    q = queue(fast)
    q.before_target(output(True))
    q.cached_leases["cached"][0].ref_cnt = 2
    with pytest.raises(RuntimeError, match="exclusive"):
        q.before_target(output(True))
    assert q.target_epoch == 1


@pytest.mark.parametrize("fast", [False, True])
def test_parent_failure_is_never_bypassed(fast, monkeypatch):
    q = queue(fast)
    def fail(*args):
        raise RuntimeError("parent failed")
    monkeypatch.setattr(PersistentDraftQueue, "before_target", fail)
    with pytest.raises(RuntimeError, match="parent failed"):
        q.before_target(output(True))


@pytest.mark.parametrize("value", ["", "2", "true"])
def test_invalid_flag_before_engine_state_access(value, monkeypatch):
    monkeypatch.setenv("CDB_AV_KV_CHECK_REUSE", value)
    with pytest.raises(ValueError, match="CDB_AV_KV_CHECK_REUSE"):
        ReuseDraftQueue(object())
