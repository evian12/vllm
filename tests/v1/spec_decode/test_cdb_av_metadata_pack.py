# SPDX-License-Identifier: Apache-2.0
"""Packed Draft attention metadata must match the legacy construction."""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.spec_decode.cdb_av_refill import RefillSession


@pytest.mark.parametrize(
    ("blocks", "seqs", "qlens", "slots"),
    [
        ([[1]], [1], [1], [16]),
        ([[1, 2, 3], [9], [10, 11]], [7, 1, 4], [2, 1, 1], [18, 19, 144, 163]),
        ([[2, 3], [4, 5]], [17, 32], [1, 3], [33, 64, 65, 66]),
    ],
)
def test_packed_metadata_equals_legacy(
    monkeypatch, blocks, seqs, qlens, slots
):
    session = RefillSession.__new__(RefillSession)
    session.p = SimpleNamespace(device=torch.device("cpu"))

    monkeypatch.delenv("CDB_AV_PACK_METADATA", raising=False)
    legacy = session._metadata(blocks, seqs, qlens, slots)
    monkeypatch.setenv("CDB_AV_PACK_METADATA", "1")
    packed = session._metadata(blocks, seqs, qlens, slots)

    for field in (
        "query_start_loc", "query_start_loc_cpu", "seq_lens",
        "block_table_tensor", "slot_mapping",
    ):
        left, right = getattr(legacy, field), getattr(packed, field)
        assert left.dtype == right.dtype
        assert left.shape == right.shape
        assert torch.equal(left, right), field
    for field in (
        "num_reqs", "num_actual_tokens", "max_query_len", "max_seq_len",
        "causal",
    ):
        assert getattr(legacy, field) == getattr(packed, field)
