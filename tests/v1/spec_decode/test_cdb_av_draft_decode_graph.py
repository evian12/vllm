# SPDX-License-Identifier: Apache-2.0
"""Decode-only graph eligibility must never admit ragged Prefill rows."""

from types import SimpleNamespace

import pytest
import torch

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
