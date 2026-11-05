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


def test_expand_only_true_accepted_prefixes():
    from vllm.v1.spec_decode.cdb_av_fingerprint import token_fingerprint

    script = SOURCE.parents[3] / "scripts/cdb_av/build_acceptance_replay.py"
    loader = importlib.util.spec_from_file_location("cdb_accept_dense_test", script)
    parser = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(parser)
    prefix = [10, 11, 12]
    output = [21, 22, 99, 30]
    record = dict(trace_key="stress512:0",
                  fingerprint=token_fingerprint(prefix),
                  context_tokens=3, observed_k=4, accepted=2,
                  drafts=[21, 22, 23, 24], generated=[21, 22, 99])
    rows = list(parser.expand_accepted_prefixes(
        [record], {"stress512:0": prefix + output}))
    assert [(x["context_tokens"], x["accepted"], x["observed_k"],
             x["drafts"], x["derived_shift"]) for x in rows] == [
        (3, 2, 4, [21, 22, 23, 24], 0),
        (4, 1, 3, [22, 23, 24], 1),
        (5, 0, 2, [23, 24], 2),
    ]
    assert [x["fingerprint"] for x in rows] == [
        token_fingerprint((prefix + output)[:n]) for n in (3, 4, 5)]
    assert [module.conservative_replay_cap(
        x["accepted"], x["observed_k"], 8) for x in rows] == [3, 2, 1]

    with pytest.raises(ValueError, match="accepted-prefix mismatch"):
        list(parser.expand_accepted_prefixes(
            [record], {"stress512:0": prefix + [21, 50, 99]}))


def test_expand_full_accept_stays_censored():
    from vllm.v1.spec_decode.cdb_av_fingerprint import token_fingerprint

    script = SOURCE.parents[3] / "scripts/cdb_av/build_acceptance_replay.py"
    loader = importlib.util.spec_from_file_location("cdb_accept_censored_test", script)
    parser = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(parser)
    record = dict(trace_key="one", fingerprint=token_fingerprint([5]),
                  context_tokens=1, observed_k=2, accepted=2,
                  drafts=[6, 7], generated=[6, 7, 8])
    rows = list(parser.expand_accepted_prefixes([record], {"one": [5, 6, 7, 8]}))
    assert [(x["accepted"], x["observed_k"]) for x in rows] == [(2, 2), (1, 1)]
    # The API may stop at max_tokens after only one of the accepted tokens.
    limited = list(parser.expand_accepted_prefixes(
        [record], {"one": [5, 6]}))
    assert [x["derived_shift"] for x in limited] == [0, 1]
