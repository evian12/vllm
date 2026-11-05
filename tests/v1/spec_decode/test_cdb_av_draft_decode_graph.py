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
