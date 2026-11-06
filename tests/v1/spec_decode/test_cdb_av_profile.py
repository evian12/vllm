# SPDX-License-Identifier: Apache-2.0
"""Profiling must preserve real KV invalidation and the empty-range no-op."""
from collections import Counter
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.spec_decode.cdb_av_kv_reuse import ReuseDraftSession


@pytest.mark.parametrize("enabled", [False, True])
def test_invalidation_profile_preserves_owned_slot_writes(monkeypatch, enabled):
    monkeypatch.setenv("CDB_AV_PROFILE_DRAFT", "1" if enabled else "0")
    session = ReuseDraftSession.__new__(ReuseDraftSession)
    session.p = SimpleNamespace(device=torch.device("cpu"))
    session.block_size = 2
    session._profile = Counter()
    kv = torch.arange(2 * 8 * 2 * 3, dtype=torch.float32).reshape(2, 8, 2, 1, 3)
    expected = kv.clone()
    # Logical positions 1 and 2 of lease [2, 4] are physical (2,1),(4,0).
    expected[:, 2, 1] = 0
    expected[:, 4, 0] = 0
    session.layers = lambda: [("layer0", kv)]
    session._clear_slots([2, 4], 1, 3)
    assert torch.equal(kv, expected)
    if enabled:
        assert session._profile["invalidate_calls"] == 1
        assert session._profile["invalidate_tokens"] == 2
        assert session._profile["invalidate_host_ns"] > 0
    else:
        assert session._profile == {}


def test_empty_invalidation_does_not_touch_model_or_add_profile_sample(monkeypatch):
    monkeypatch.setenv("CDB_AV_PROFILE_DRAFT", "1")
    session = ReuseDraftSession.__new__(ReuseDraftSession)
    session._profile = Counter()
    # No device or layer access is legal for an empty invalidation interval.
    session._clear_slots([], 0, 0)
    assert session._profile == {}
