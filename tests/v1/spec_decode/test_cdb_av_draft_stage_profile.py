# SPDX-License-Identifier: Apache-2.0
"""CPU-only protocol tests for bounded Draft-stage instrumentation."""
import json
from types import SimpleNamespace

import pytest

from vllm.v1.spec_decode import cdb_av_draft_stage_profile as profile


def test_profile_disabled_by_default(monkeypatch):
    monkeypatch.delenv("CDB_AV_DRAFT_STAGE_PROFILE", raising=False)
    proposer = SimpleNamespace()
    assert profile.get_draft_stage_profiler(proposer) is None
    assert profile.get_draft_stage_profiler(proposer) is None


@pytest.mark.parametrize("enabled,limit", [
    ("invalid", "20"),
    ("1", "abc"),
    ("1", "0"),
    ("1", "101"),
])
def test_profile_rejects_invalid_config(monkeypatch, enabled, limit):
    monkeypatch.setenv("CDB_AV_DRAFT_STAGE_PROFILE", enabled)
    monkeypatch.setenv("CDB_AV_DRAFT_STAGE_PROFILE_MAX_SAMPLES", limit)
    with pytest.raises(ValueError):
        profile.get_draft_stage_profiler(SimpleNamespace())


def test_profile_counter_persists_between_windows(monkeypatch):
    monkeypatch.setenv("CDB_AV_DRAFT_STAGE_PROFILE", "1")
    monkeypatch.setenv("CDB_AV_DRAFT_STAGE_PROFILE_MAX_SAMPLES", "2")
    proposer = SimpleNamespace()
    shared = profile.get_draft_stage_profiler(proposer)
    assert profile.get_draft_stage_profiler(proposer) is shared
    for idx in range(1, 3):
        sample = shared.begin(
            window=idx, step=1, phase="decode", request_rows=2,
            token_rows=2, graph_action="eager", graph_key=None,
            metadata_pack_cpu_ms=0.1,
        )
        assert sample.fields["sample_index"] == idx
    assert not shared.has_capacity
    assert shared.begin(
        window=3, step=1, phase="decode", request_rows=1,
        token_rows=1, graph_action="eager", graph_key=None,
        metadata_pack_cpu_ms=0.0,
    ) is None


def test_sample_emits_cpu_and_gpu_span_without_real_cuda(monkeypatch):
    class FakeEvent:
        def __init__(self, enable_timing):
            assert enable_timing
        def record(self):
            pass
        def synchronize(self):
            pass
        def elapsed_time(self, other):
            assert isinstance(other, FakeEvent)
            return 0.25

    records = []
    monkeypatch.setattr(profile.torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(profile.logger, "info",
                        lambda fmt, raw: records.append((fmt, json.loads(raw))))
    sample = profile.DraftStageProfiler(1).begin(
        window=2, step=3, phase="mixed_prefill_decode",
        request_rows=4, token_rows=7, graph_action="replay",
        graph_key=(7, 64, 4, "dynamic4"),
        metadata_pack_cpu_ms=0.6,
    )
    sample.start("backbone")
    sample.end("backbone")
    sample.cpu_span("pre_d2h_gpu_wait_cpu_ms", 0.0)
    sample.cpu_span("d2h_materialize_cpu_ms", 0.0)
    sample.cpu_span("d2h_wait_cpu_ms", 0.0)
    sample.finish()
    assert len(records) == 1
    fmt, data = records[0]
    assert fmt.startswith("[CDB_DRAFT_STAGE_TIMING]")
    assert data["graph_action"] == "replay"
    assert data["metadata_pack_cpu_ms"] == 0.6
    assert data["measurements"]["backbone"]["stream_span_ms"] == 0.25
    assert data["measurements"]["pre_d2h_gpu_wait_cpu_ms"] >= 0
    assert data["measurements"]["d2h_materialize_cpu_ms"] >= 0
    assert data["measurements"]["d2h_wait_cpu_ms"] >= 0
    with pytest.raises(RuntimeError, match="finished twice"):
        sample.finish()


def test_stage_lifecycle_rejects_incomplete_measurements(monkeypatch):
    class FakeEvent:
        def __init__(self, enable_timing):
            pass
        def record(self):
            pass

    monkeypatch.setattr(profile.torch.cuda, "Event", FakeEvent)
    sample = profile.DraftStageProfiler(1).begin(
        window=1, step=1, phase="decode", request_rows=1,
        token_rows=1, graph_action="eager", graph_key=None,
        metadata_pack_cpu_ms=0.0,
    )
    sample.start("backbone")
    with pytest.raises(RuntimeError, match="left open"):
        sample.finish()


def test_profiler_reserves_replay_samples_after_eager_warmup():
    profiler = profile.DraftStageProfiler(20)
    params = dict(window=1, step=1, phase="decode",
                  request_rows=4, token_rows=4, graph_key=(4, 64),
                  metadata_pack_cpu_ms=0.1)
    for _ in range(10):
        assert profiler.begin(graph_action="eager", **params) is not None
    assert profiler.begin(graph_action="eager", **params) is None
    assert profiler.samples == 10
    for _ in range(10):
        assert profiler.begin(graph_action="replay", **params) is not None
    assert profiler.begin(graph_action="replay", **params) is None
    assert not profiler.has_capacity
    assert profiler.action_samples == {"eager": 10, "replay": 10}
