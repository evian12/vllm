# SPDX-License-Identifier: Apache-2.0
"""Decode-only graph eligibility must never admit ragged Prefill rows."""

import json
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.spec_decode import cdb_av_refill
from vllm.v1.spec_decode.cdb_av_refill import RefillSession


@pytest.mark.parametrize(
    ("rows", "tokens", "query", "max_seq", "width", "expected"),
    [
        (1, 1, 1, 20, 2, (1, 32)),
        (4, 4, 1, 129, 9, (4, 256)),
        (2, 3, 2, 40, 3, None),
        (3, 3, 2, 40, 3, None),
        (5, 5, 1, 40, 3, None),
        (1, 1, 1, 1024, 65, None),
    ],
)
def test_decode_graph_key(rows, tokens, query, max_seq, width, expected):
    session = RefillSession.__new__(RefillSession)
    session.capacity = 4
    session.block_size = 16
    session.p = SimpleNamespace(max_model_len=1024)
    cad = SimpleNamespace(num_reqs=rows, num_actual_tokens=tokens,
                          max_query_len=query, max_seq_len=max_seq,
                          block_table_tensor=SimpleNamespace(shape=(rows, width)))
    assert session._decode_graph_key(cad, tokens) == expected


def test_decode_graph_storage_detects_reallocated_kv():
    session = RefillSession.__new__(RefillSession)
    layer = SimpleNamespace(kv_cache=[torch.zeros(2, 2)])
    positions = torch.zeros(4, dtype=torch.int64)
    session.p = SimpleNamespace(
        input_ids=torch.zeros(4, dtype=torch.int32),
        _slot_mapping_buffer=torch.zeros(4, dtype=torch.int64),
        _get_positions=lambda n: positions[:n],
        attn_layer_names=['layer'],
        vllm_config=SimpleNamespace(compilation_config=SimpleNamespace(
            static_forward_context={'layer': layer})),
    )
    original = session._decode_graph_storage()
    assert original == session._decode_graph_storage()
    layer.kv_cache[0] = torch.ones(3, 3)
    assert original != session._decode_graph_storage()


@pytest.mark.parametrize(
    ("qlens", "max_seq", "expected"),
    [
        ((1, 1, 1, 2), 120, (5, 128, (1, 1, 1, 2))),
        ((1, 1, 2, 2), 130, (6, 256, (1, 1, 2, 2))),
        ((1, 2), 20, (3, 32, (1, 2))),
        ((1, 1, 1, 1), 20, None),
        ((1, 3), 20, None),
        ((2, 2, 2, 2, 1), 20, None),
    ],
)
def test_mixed_graph_key_requires_exact_one_or_two_token_layout(
    qlens, max_seq, expected,
):
    session = RefillSession.__new__(RefillSession)
    session.capacity = 4
    session.block_size = 16
    session.p = SimpleNamespace(max_model_len=1024)
    starts = [0]
    for qlen in qlens:
        starts.append(starts[-1] + qlen)
    n = starts[-1]
    cad = SimpleNamespace(
        num_reqs=len(qlens), num_actual_tokens=n,
        max_query_len=max(qlens), max_seq_len=max_seq,
        query_start_loc_cpu=torch.tensor(starts, dtype=torch.int32),
        block_table_tensor=SimpleNamespace(
            shape=(len(qlens), (max_seq + 15) // 16)),
    )
    assert session._mixed_graph_key(cad, n) == expected


def test_dynamic_mixed_graph_key_shares_permuted_query_layouts():
    session = RefillSession.__new__(RefillSession)
    session.capacity = 4
    session.block_size = 16
    session.p = SimpleNamespace(max_model_len=1024)
    keys = []
    for qlens in ((1, 1, 2), (1, 2, 1), (2, 1, 1)):
        starts = [0]
        for qlen in qlens:
            starts.append(starts[-1] + qlen)
        cad = SimpleNamespace(
            num_reqs=3, num_actual_tokens=4, max_query_len=2,
            max_seq_len=48,
            query_start_loc_cpu=torch.tensor(starts, dtype=torch.int32),
            block_table_tensor=SimpleNamespace(shape=(3, 3)),
        )
        keys.append(session._mixed_graph_key(cad, 4, dynamic=True))
    assert keys == [(4, 64, 3, 'dynamic2')] * 3


@pytest.mark.parametrize(
    ("phase", "has_logits", "expected"),
    [
        ("decode", True, True),
        ("decode", False, False),
        ("mixed_prefill_decode", True, True),
        ("mixed_prefill", True, True),
        ("mixed_prefill", False, False),
        ("reuse_batch_prefill", False, False),
        ("p2_prepare", False, False),
        ("reuse_shape_replay", False, False),
    ],
)
def test_graph_phase_is_limited_to_decode_and_one_token_kv_fill(
    phase, has_logits, expected,
):
    sentinel = object() if has_logits else None
    assert RefillSession._graph_phase_eligible(phase, sentinel) is expected


def q4_session_and_metadata(qlens):
    session = RefillSession.__new__(RefillSession)
    session.capacity = 4
    session.block_size = 16
    session.p = SimpleNamespace(max_model_len=1024)
    starts = [0]
    for length in qlens:
        starts.append(starts[-1] + length)
    cad = SimpleNamespace(num_reqs=len(qlens), num_actual_tokens=starts[-1],
                          max_query_len=max(qlens), max_seq_len=48,
                          query_start_loc_cpu=torch.tensor(starts, dtype=torch.int32),
                          block_table_tensor=SimpleNamespace(shape=(len(qlens), 3)))
    return session, cad, starts[-1]


def test_q4_graph_reuses_key_across_two_three_four_query_lengths():
    keys = []
    for qlens in ((2, 2, 2), (1, 2, 3), (1, 1, 4)):
        session, cad, n = q4_session_and_metadata(qlens)
        key = session._mixed_graph_key(cad, n, dynamic=True, max_query=4)
        keys.append(key)
        # A first capture whose actual max is2 must already reserve bound4.
        assert session._graph_static_query_bound(key, qlens) == 4
    assert keys == [(6, 64, 3, 'dynamic4')] * 3


@pytest.mark.parametrize('qlens,expected', [
    ((4, 4, 4, 4), (16, 64, 4, 'dynamic4')),
    ((1, 1, 5), None), ((1, 0, 4), None),
    ((1, 1, 1, 1, 4), None), ((1, 1, 1, 1), None),
])
def test_q4_graph_exact_token_capacity_and_layout(qlens, expected):
    session, cad, n = q4_session_and_metadata(qlens)
    assert session._mixed_graph_key(cad, n, dynamic=True, max_query=4) == expected


def test_default_and_exact_layout_paths_still_reject_query3():
    session, cad, n = q4_session_and_metadata((1, 1, 3))
    assert session._mixed_graph_key(cad, n, dynamic=True) is None
    assert session._mixed_graph_key(cad, n, dynamic=False, max_query=4) is None
    session, cad, n = q4_session_and_metadata((1, 2))
    assert session._mixed_graph_key(cad, n, dynamic=False, max_query=4) == (3, 64, (1, 2))


def test_q4_graph_rejects_inconsistent_shadow_metadata():
    session, cad, n = q4_session_and_metadata((1, 2, 3))
    cad.max_query_len = 2
    assert session._mixed_graph_key(cad, n, dynamic=True, max_query=4) is None
    cad.max_query_len = 3
    cad.num_actual_tokens = n + 1
    assert session._mixed_graph_key(cad, n, dynamic=True, max_query=4) is None


@pytest.mark.parametrize('key,qlens', [
    ((6, 64, 3, 'dynamic2'), (1, 1, 4)),
    ((6, 64, 4, 'dynamic4'), (1, 1, 4)),
    ((7, 64, 3, 'dynamic4'), (1, 1, 4)),
    ((6, 64, 3, 'dynamic8'), (1, 1, 4)),
])
def test_static_query_bound_rejects_invalid_capture(key, qlens):
    with pytest.raises(ValueError):
        RefillSession._graph_static_query_bound(key, qlens)



def test_draft_graph_runtime_diagnostic_is_bounded_and_reports_runtime_config(
    monkeypatch,
):
    proposer = SimpleNamespace()
    records = []
    monkeypatch.setenv("CDB_AV_DRAFT_DECODE_GRAPH", "0")
    monkeypatch.setenv("CDB_AV_DRAFT_RAGGED_GRAPH", "1")
    monkeypatch.setenv("CDB_AV_DRAFT_RAGGED_DYNAMIC", "1")
    monkeypatch.setenv("CDB_AV_DRAFT_RAGGED_MAX_QUERY", "4")
    monkeypatch.setenv("CDB_AV_RAGGED_GRAPH_MAX_KEYS", "12")
    monkeypatch.setenv("CDB_AV_RAGGED_GRAPH_MIN_HITS", "8")
    monkeypatch.setattr(
        cdb_av_refill.logger,
        "info",
        lambda fmt, raw: records.append((fmt, json.loads(raw))),
    )

    common = dict(phase="decode", request_rows=4, token_rows=4)
    cdb_av_refill._log_draft_graph_runtime_once(
        proposer, "eager", graph_key=(4, 64),
        reason="ragged_warmup", **common)
    cdb_av_refill._log_draft_graph_runtime_once(
        proposer, "eager", graph_key=(4, 64),
        reason="ragged_warmup", **common)
    cdb_av_refill._log_draft_graph_runtime_once(
        proposer, "eager", graph_key=(5, 64, 3, "dynamic4"),
        reason="ragged_warmup", phase="mixed_prefill_decode",
        request_rows=3, token_rows=5)
    cdb_av_refill._log_draft_graph_runtime_once(
        proposer, "eager", reason="draft_graph_disabled", **common)
    cdb_av_refill._log_draft_graph_runtime_once(
        proposer, "capture", graph_key=(4, 64), **common)
    cdb_av_refill._log_draft_graph_runtime_once(
        proposer, "capture", graph_key=(4, 64), **common)
    cdb_av_refill._log_draft_graph_runtime_once(
        proposer, "capture", graph_key=(3, 64),
        phase="decode", request_rows=3, token_rows=3)
    cdb_av_refill._log_draft_graph_runtime_once(
        proposer, "replay", graph_key=(4, 64), **common)

    assert len(records) == 6
    assert all(fmt == "[CDB_DRAFT_GRAPH_RUNTIME] %s"
               for fmt, _ in records)
    payloads = [payload for _, payload in records]
    assert [payload["action"] for payload in payloads] == [
        "eager", "eager", "eager", "capture", "capture", "replay"]
    assert [payload["reason"] for payload in payloads[:3]] == [
        "ragged_warmup", "ragged_warmup", "draft_graph_disabled"]
    assert payloads[0]["graph_key"] == [4, 64]
    assert payloads[1]["graph_key"] == [5, 64, 3, "dynamic4"]
    assert payloads[3]["graph_key"] == [4, 64]
    assert payloads[4]["graph_key"] == [3, 64]
    assert all(payload["pid"] > 0 for payload in payloads)
    config = payloads[0]["config"]
    assert config["CDB_AV_DRAFT_DECODE_GRAPH"] == "0"
    assert config["CDB_AV_DRAFT_RAGGED_GRAPH"] == "1"
    assert config["CDB_AV_DRAFT_RAGGED_DYNAMIC"] == "1"
    assert config["CDB_AV_DRAFT_RAGGED_MAX_QUERY"] == "4"
    assert config["CDB_AV_RAGGED_GRAPH_MAX_KEYS"] == "12"
    assert config["CDB_AV_RAGGED_GRAPH_MIN_HITS"] == "8"


def test_draft_graph_runtime_diagnostic_has_hard_record_cap(monkeypatch):
    proposer = SimpleNamespace()
    records = []
    monkeypatch.setattr(
        cdb_av_refill.logger,
        "info",
        lambda fmt, raw: records.append((fmt, json.loads(raw))),
    )

    for key in range(140):
        cdb_av_refill._log_draft_graph_runtime_once(
            proposer, "replay", phase="decode", request_rows=1,
            token_rows=1, graph_key=(key, 64))

    assert len(records) == 128
    assert len(proposer._cdb_draft_graph_runtime_seen) == 128
