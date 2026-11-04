# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Failure-path tests for CDB-AV KV lease ownership cleanup."""

from types import SimpleNamespace

import pytest

from vllm.v1.spec_decode.cdb_av_persistent import PersistentDraftQueue
from vllm.v1.spec_decode.cdb_av_kv_reuse import ReuseDraftQueue


class FakeBlock:
    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_cnt = 1
        self.is_null = False


class FakeBlockPool:
    def __init__(self, fail_on_call=None):
        self.fail_on_call = fail_on_call
        self.calls = 0
        self.freed = []

    def free_blocks(self, blocks):
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("simulated free failure")
        blocks = list(blocks)
        for block in blocks:
            block.ref_cnt -= 1
        self.freed.extend(block.block_id for block in blocks)


def make_queue(pool):
    queue = ReuseDraftQueue.__new__(ReuseDraftQueue)
    queue.s = SimpleNamespace(
        kv_cache_manager=SimpleNamespace(block_pool=pool)
    )
    queue.audit_leases = {}
    queue.cached_leases = {}
    queue.leases = {}
    queue.replay_blocks = []
    queue.pending_allocations = []
    queue.commits = {}
    queue.ready = {}
    queue.inflight = {}
    queue.history_pages = set()
    queue.replayed = set()
    queue._audit_cleanup_safe = True
    return queue


def test_owned_mapping_cleanup_keeps_only_failed_and_unattempted_owners():
    pool = FakeBlockPool(fail_on_call=2)
    queue = make_queue(pool)
    blocks = [FakeBlock(i) for i in range(3)]
    queue.audit_leases = {
        "r0": [blocks[0]],
        "r1": [blocks[1]],
        "r2": [blocks[2]],
    }

    with pytest.raises(RuntimeError, match="simulated free failure"):
        queue._release_audit_leases()

    assert queue.audit_leases == {
        "r1": [blocks[1]],
        "r2": [blocks[2]],
    }
    assert blocks[0].ref_cnt == 0
    assert blocks[1].ref_cnt == 1
    assert blocks[2].ref_cnt == 1
    assert pool.freed == [0]

    pool.fail_on_call = None
    queue._release_audit_leases()

    assert queue.audit_leases == {}
    assert [block.ref_cnt for block in blocks] == [0, 0, 0]
    assert pool.freed == [0, 1, 2]


def test_owned_mapping_cleanup_rejects_invalid_ownership_without_mutation():
    pool = FakeBlockPool()
    queue = make_queue(pool)
    block = FakeBlock(7)
    block.ref_cnt = 2
    queue.audit_leases = {"r7": [block]}

    with pytest.raises(RuntimeError, match="Cleanup lost audit KV ownership"):
        queue._release_audit_leases()

    assert queue.audit_leases == {"r7": [block]}
    assert block.ref_cnt == 2
    assert pool.freed == []


def test_rpc_retries_audit_cleanup_before_next_worker_operation(monkeypatch):
    pool = FakeBlockPool(fail_on_call=2)
    queue = make_queue(pool)
    blocks = [FakeBlock(10), FakeBlock(11)]
    queue.audit_leases = {
        "r10": [blocks[0]],
        "r11": [blocks[1]],
    }
    queue._audit_cleanup_safe = False
    worker_calls = []

    def base_rpc(self, operation, payload=None):
        worker_calls.append(operation)
        return {"prepared": []} if operation == "prepare" else {"stepped": True}

    monkeypatch.setattr(PersistentDraftQueue, "rpc", base_rpc)
    monkeypatch.setenv("CDB_AV_REUSE_REPLAY", "0")

    with pytest.raises(RuntimeError, match="simulated free failure"):
        queue.rpc("prepare", {"plans": []})

    assert worker_calls == ["prepare"]
    assert list(queue.audit_leases) == ["r11"]

    pool.fail_on_call = None
    result = queue.rpc("step")

    assert result == {"stepped": True}
    assert worker_calls == ["prepare", "step"]
    assert queue.audit_leases == {}
    assert [block.ref_cnt for block in blocks] == [0, 0]


def test_failed_prepare_drops_worker_before_releasing_audit_pages(monkeypatch):
    pool = FakeBlockPool()
    queue = make_queue(pool)
    block = FakeBlock(15)
    queue.audit_leases = {"r15": [block]}
    queue._audit_cleanup_safe = False
    worker_calls = []

    def base_rpc(self, operation, payload=None):
        worker_calls.append(operation)
        if operation == "prepare":
            raise RuntimeError("simulated prepare failure")
        return {"dropped": payload["ids"]}

    monkeypatch.setattr(PersistentDraftQueue, "rpc", base_rpc)

    with pytest.raises(RuntimeError, match="simulated prepare failure"):
        queue.rpc("prepare", {"plans": []})

    assert worker_calls == ["prepare"]
    assert queue.audit_leases == {"r15": [block]}
    assert block.ref_cnt == 1
    assert pool.freed == []

    result = queue.rpc("drop", {"ids": ["r15"]})

    assert result == {"dropped": ["r15"]}
    assert worker_calls == ["prepare", "drop"]
    assert queue.audit_leases == {}
    assert block.ref_cnt == 0
    assert pool.freed == [15]


def test_reset_keeps_audit_pages_for_the_reset_ownership_transaction(monkeypatch):
    pool = FakeBlockPool()
    queue = make_queue(pool)
    block = FakeBlock(16)
    queue.audit_leases = {"r16": [block]}
    queue._audit_cleanup_safe = False
    worker_calls = []

    def base_rpc(self, operation, payload=None):
        worker_calls.append(operation)
        return {"reset": True}

    monkeypatch.setattr(PersistentDraftQueue, "rpc", base_rpc)

    result = queue.rpc("reset")

    assert result == {"reset": True}
    assert worker_calls == ["reset"]
    assert queue._audit_cleanup_safe is True
    # PersistentDraftQueue.reset owns the snapshot and frees these pages
    # after the reset acknowledgement; the RPC wrapper must not free them.
    assert queue.audit_leases == {"r16": [block]}
    assert block.ref_cnt == 1
    assert pool.freed == []


def test_failed_prepare_rejects_partial_drop_before_audit_release(monkeypatch):
    pool = FakeBlockPool()
    queue = make_queue(pool)
    blocks = [FakeBlock(17), FakeBlock(18)]
    queue.audit_leases = {
        "r17": [blocks[0]],
        "r18": [blocks[1]],
    }
    queue._audit_cleanup_safe = False
    worker_calls = []

    def base_rpc(self, operation, payload=None):
        worker_calls.append(operation)
        return {"dropped": payload["ids"]}

    monkeypatch.setattr(PersistentDraftQueue, "rpc", base_rpc)

    with pytest.raises(RuntimeError, match="complete Worker Drop"):
        queue.rpc("drop", {"ids": ["r17"]})

    assert worker_calls == []
    assert queue.audit_leases == {
        "r17": [blocks[0]],
        "r18": [blocks[1]],
    }
    assert [block.ref_cnt for block in blocks] == [1, 1]
    assert pool.freed == []


def test_incomplete_worker_drop_ack_keeps_failed_prepare_audit_pages(monkeypatch):
    pool = FakeBlockPool()
    queue = make_queue(pool)
    blocks = [FakeBlock(30), FakeBlock(31)]
    queue.audit_leases = {
        "r30": [blocks[0]],
        "r31": [blocks[1]],
    }
    queue._audit_cleanup_safe = False
    worker_calls = []

    def base_rpc(self, operation, payload=None):
        worker_calls.append(operation)
        return {"dropped": ["r30"]}

    monkeypatch.setattr(PersistentDraftQueue, "rpc", base_rpc)

    with pytest.raises(RuntimeError, match="Incomplete Worker Drop acknowledgement"):
        queue.rpc("drop", {"ids": ["r30", "r31"]})

    assert worker_calls == ["drop"]
    assert queue._audit_cleanup_safe is False
    assert set(queue.audit_leases) == {"r30", "r31"}
    assert [block.ref_cnt for block in blocks] == [1, 1]
    assert pool.freed == []


def test_incomplete_worker_prepare_ack_keeps_audit_pages(monkeypatch):
    pool = FakeBlockPool()
    queue = make_queue(pool)
    block = FakeBlock(33)
    queue.audit_leases = {"r33": [block]}
    queue._audit_cleanup_safe = False
    worker_calls = []

    def base_rpc(self, operation, payload=None):
        worker_calls.append(operation)
        return {"prepared": []}

    monkeypatch.setattr(PersistentDraftQueue, "rpc", base_rpc)

    with pytest.raises(RuntimeError, match="Incomplete Worker Prepare acknowledgement"):
        queue.rpc(
            "prepare",
            {"plans": [{"req_id": "r33"}]},
        )

    assert worker_calls == ["prepare"]
    assert queue._audit_cleanup_safe is False
    assert queue.audit_leases == {"r33": [block]}
    assert block.ref_cnt == 1
    assert pool.freed == []


def test_unacknowledged_worker_reset_keeps_audit_pages(monkeypatch):
    pool = FakeBlockPool()
    queue = make_queue(pool)
    block = FakeBlock(32)
    queue.audit_leases = {"r32": [block]}
    queue._audit_cleanup_safe = False
    worker_calls = []

    def base_rpc(self, operation, payload=None):
        worker_calls.append(operation)
        return {"reset": False}

    monkeypatch.setattr(PersistentDraftQueue, "rpc", base_rpc)

    with pytest.raises(RuntimeError, match="Worker Reset did not acknowledge"):
        queue.rpc("reset")

    assert worker_calls == ["reset"]
    assert queue._audit_cleanup_safe is False
    assert queue.audit_leases == {"r32": [block]}
    assert block.ref_cnt == 1
    assert pool.freed == []


def test_close_retries_cached_lease_cleanup_without_double_free(monkeypatch):
    pool = FakeBlockPool(fail_on_call=2)
    queue = make_queue(pool)
    blocks = [FakeBlock(20), FakeBlock(21), FakeBlock(22)]
    queue.cached_leases = {
        "r20": [blocks[0]],
        "r21": [blocks[1]],
        "r22": [blocks[2]],
    }

    monkeypatch.setattr(
        PersistentDraftQueue,
        "rpc",
        lambda self, operation, payload=None: {"dropped": payload["ids"]}
        if operation == "drop" else {"ok": True},
    )

    with pytest.raises(RuntimeError, match="simulated free failure"):
        queue.close()

    assert list(queue.cached_leases) == ["r21", "r22"]
    assert [block.ref_cnt for block in blocks] == [0, 1, 1]
    assert pool.freed == [20]

    pool.fail_on_call = None
    queue.close()

    assert queue.cached_leases == {}
    assert [block.ref_cnt for block in blocks] == [0, 0, 0]
    assert pool.freed == [20, 21, 22]
