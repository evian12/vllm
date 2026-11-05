# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for per-request historical Target acceptance K."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


avg = load("cdb_avg_test", ROOT / "vllm/v1/spec_decode/cdb_av_request_average.py")
accept = load("cdb_accept_avg_test", ROOT / "vllm/v1/spec_decode/cdb_av_acceptance.py")
builder = load("cdb_avg_builder_test", ROOT / "scripts/cdb_av/build_request_average.py")


def test_mean_k_bounds_and_rounding():
    assert [avg.mean_to_k(total, 4, 8) for total in (0, 1, 4, 5, 32)] == [1, 1, 1, 2, 8]
    for args in ((-1, 1, 8), (9, 1, 8), (1, 0, 8), (1, 1, 0), (True, 1, 8)):
        with pytest.raises(ValueError):
            avg.mean_to_k(*args)


def test_loader_match_and_fallback(tmp_path):
    path = tmp_path / "avg.jsonl"
    row = dict(trace_key="stress512:1", prompt_fingerprint="a" * 32,
               accepted_total=5, verified_count=2, chosen_k=3, server_max_k=8)
    path.write_text(json.dumps(row) + "\n")
    source = avg.RequestAverageK.from_jsonl(path, 8)
    assert source.resolve("stress512:1", "a" * 32).cap == 3
    assert source.resolve("stress512:1", "a" * 32).source == "request_average"
    assert source.resolve("stress512:1", "b" * 32).source == "prompt_mismatch_fallback"
    assert source.resolve("stress512:2", "a" * 32).source == "missing_fallback"
    with pytest.raises(ValueError):
        avg.RequestAverageK.from_jsonl(path, 4)


def test_builder_aggregates_only_real_target_verification():
    key = "stress512:1"
    prompt = [1, 2, 3]
    client = [json.dumps(dict(trace_key=key, prompt_token_ids=prompt)),
              json.dumps(dict(trace_key="stress512:2", prompt_token_ids=[4]))]
    server = ["noise", *["[CDB_REFILL] " + json.dumps(dict(
        event="p2_verified", trace_key=key, req_id="physical-1",
        accepted=a, drafts=[1, 2, 3, 4])) for a in (2, 4)]]
    rows, no_verify = builder.build_rows(server, client, 8)
    assert len(rows) == 1 and no_verify == 1
    assert rows[0]["accepted_total"] == 6
    assert rows[0]["verified_count"] == 2
    assert rows[0]["chosen_k"] == 3
    assert rows[0]["prompt_fingerprint"] == builder.token_fingerprint(prompt)
    assert rows[0]["full_accept_count"] == 1


def test_builder_rejects_ambiguous_identity_and_bad_acceptance():
    client = [json.dumps(dict(trace_key="x", prompt_token_ids=[1]))]
    with pytest.raises(ValueError):
        builder.build_rows(["[CDB_REFILL] " + json.dumps(dict(
            event="p2_verified", trace_key="x", req_id="r", accepted=5,
            drafts=[1]))], client, 8)
    event = lambda rid: "[CDB_REFILL] " + json.dumps(dict(
        event="p2_verified", trace_key="x", req_id=rid, accepted=0,
        drafts=[1]))
    with pytest.raises(ValueError):
        builder.build_rows([event("r1"), event("r2")], client, 8)


def test_mixed_batch_confidence_stop_sources():
    assert not accept.should_stop("request_avg_acceptance", "request_average", True)
    assert accept.should_stop("request_avg_acceptance", "hf_fallback", True)
    assert accept.should_stop("hf_confidence", "configured", True)
