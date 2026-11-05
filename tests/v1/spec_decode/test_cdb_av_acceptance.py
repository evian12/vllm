# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for exact-context acceptance replay semantics."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[3] / "vllm/v1/spec_decode/cdb_av_acceptance.py"
spec = importlib.util.spec_from_file_location("cdb_av_acceptance_test", SOURCE)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_conservative_cap_and_censoring():
    assert module.conservative_replay_cap(0, 8, 8) == 1
    assert module.conservative_replay_cap(2, 8, 8) == 3
    assert module.conservative_replay_cap(8, 8, 8) == 8
    with pytest.raises(ValueError):
        module.conservative_replay_cap(9, 8, 8)
    with pytest.raises(ValueError):
        module.conservative_replay_cap(1, 2, 1)


def test_replay_requires_exact_request_and_context(tmp_path):
    p = tmp_path / "reference.jsonl"
    p.write_text(json.dumps(dict(trace_key="stress512:7", fingerprint="a" * 32,
                                 context_tokens=44, observed_k=8, accepted=2)) + "\n")
    replay = module.AcceptanceReplay.from_jsonl(p, 8)
    assert replay.resolve("stress512:7", "a" * 32, 44, 8).cap == 3
    assert replay.resolve("stress512:7", "a" * 32, 44, 8).source == "replay"
    assert replay.resolve("stress512:8", "a" * 32, 44, 8).source == "missing_fallback"
    assert replay.resolve("stress512:7", "b" * 32, 44, 8).source == "missing_fallback"
    assert replay.resolve("stress512:7", "a" * 32, 45, 8).source == "missing_fallback"


def test_full_accept_is_right_censored(tmp_path):
    p = tmp_path / "reference.jsonl"
    p.write_text(json.dumps(dict(trace_key="k", fingerprint="b" * 32,
                                 context_tokens=10, observed_k=4, accepted=4)) + "\n")
    decision = module.AcceptanceReplay.from_jsonl(p, 8).resolve("k", "b" * 32, 10, 8)
    assert (decision.cap, decision.source, decision.censored) == (4, "replay", True)


def test_conflicting_same_context_falls_back(tmp_path):
    p = tmp_path / "reference.jsonl"
    base = dict(trace_key="k", fingerprint="c" * 32, context_tokens=10,
                observed_k=8)
    p.write_text("\n".join(json.dumps(dict(base, accepted=a)) for a in (1, 3)) + "\n")
    replay = module.AcceptanceReplay.from_jsonl(p, 8)
    assert replay.resolve("k", "c" * 32, 10, 8).source == "ambiguous_fallback"


def test_bad_record_and_policy_stop(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"trace_key":"x","accepted":1}\n')
    with pytest.raises(ValueError):
        module.AcceptanceReplay.from_jsonl(p, 8)
    assert not module.should_stop("replay_acceptance", "replay", True)
    assert module.should_stop("replay_acceptance", "hf_fallback", True)
    assert module.should_stop("hf_confidence", "configured", True)


def test_server_log_parser_keeps_only_keyed_verified_rows():
    script = SOURCE.parents[3] / "scripts/cdb_av/build_acceptance_replay.py"
    loader = importlib.util.spec_from_file_location("cdb_accept_parser_test", script)
    parser = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(parser)
    event = dict(event="p2_verified", trace_key="stress512:0",
                 fingerprint="d" * 32, context_tokens=12,
                 drafts=[1, 2, 3], accepted=2)
    lines = ["noise", "[CDB_REFILL] " + json.dumps(event),
             "[CDB_REFILL] " + json.dumps(dict(event, trace_key=None))]
    assert list(parser.extract_records(lines)) == [dict(
        trace_key="stress512:0", fingerprint="d" * 32,
        context_tokens=12, observed_k=3, accepted=2)]
