# SPDX-License-Identifier: Apache-2.0
"""The optional low-log mode preserves cumulative evidence."""

import json

from vllm.v1.spec_decode import cdb_av_refill as refill


def _clear():
    for counter in (
        refill._LOW_LOG_COUNTS,
        refill._LOW_LOG_K,
        refill._LOW_LOG_REASONS,
        refill._LOW_LOG_REUSE,
    ):
        counter.clear()


def test_low_log_aggregates_dynamic_k_and_reuse(monkeypatch):
    _clear()
    monkeypatch.setenv("CDB_AV_LOW_LOG", "1")
    monkeypatch.delenv("CDB_AV_REFILL_TRACE", raising=False)
    lines = []
    monkeypatch.setattr(refill.logger, "info", lambda *args: lines.append(args))

    refill.emit("p2_reuse_plan", reused_tokens=17, prefill_tokens=2)
    refill.emit("p2_verified", accepted=2, drafts=[1, 2, 3])
    for _ in range(1021):
        refill.emit("ready", k_actual=3, reason="max_k", req_id="r")
    assert lines == []
    refill.emit("ready", k_actual=1, reason="low_confidence", req_id="s")

    assert len(lines) == 1
    assert lines[0][0] == "[CDB_AGG] %s"
    payload = json.loads(lines[0][1])
    assert payload["events"]["__total__"] == 1024
    assert payload["k_actual"] == {"3": 1021, "1": 1}
    assert payload["stop_reasons"] == {"max_k": 1021, "low_confidence": 1}
    assert payload["token_counts"] == {
        "reused_tokens": 17,
        "prefill_tokens": 2,
        "accepted_tokens": 2,
        "drafted_tokens": 3,
    }


def test_default_and_trace_modes_keep_detailed_events(monkeypatch):
    _clear()
    lines = []
    monkeypatch.setattr(refill.logger, "info", lambda *args: lines.append(args))
    monkeypatch.delenv("CDB_AV_LOW_LOG", raising=False)
    refill.emit("ready", req_id="r", k_actual=1, reason="max_k")
    assert lines[-1][0] == "[CDB_REFILL] %s"

    monkeypatch.setenv("CDB_AV_LOW_LOG", "1")
    monkeypatch.setenv("CDB_AV_REFILL_TRACE", "1")
    refill.emit("forward", req_ids=["r"])
    assert lines[-1][0] == "[CDB_REFILL] %s"
