# SPDX-License-Identifier: Apache-2.0
import os
from types import SimpleNamespace as NS

import pytest
import torch
import torch.nn.functional as F

from vllm.v1.spec_decode.cdb_av_post_graph import (
    DraftPostGraph, consume_post_packet, get_draft_post_graph, native_hf_packet)


@pytest.mark.parametrize("rows", [1, 2, 3, 4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_native_packet_retains_existing_math(rows, dtype):
    torch.manual_seed(123)
    logits = torch.randn(rows, 151, dtype=dtype)
    fp = logits.float()
    mx, ids = fp.max(-1)
    expected = torch.stack((ids.double(), (mx-torch.logsumexp(fp, -1)).exp().double()), 1)
    assert torch.equal(native_hf_packet(logits).view(torch.uint8),
                       expected.view(torch.uint8))


def test_ties_extremes_and_threshold():
    logits = torch.tensor([[0., 0., 0.], [10000., -10000., 0.]])
    packet = native_hf_packet(logits)
    assert packet[:, 0].tolist() == [0, 0]
    assert (packet[:, 1] < .8).tolist() == [True, False]


def test_pending_packet_identity_not_shape_only():
    x = torch.randn(2, 10)
    packet = native_hf_packet(x)
    assert consume_post_packet((x, packet), x) is packet
    assert consume_post_packet(None, x) is None
    with pytest.raises(RuntimeError, match="identity"):
        consume_post_packet((x, packet), x.clone())
    with pytest.raises(RuntimeError, match="row"):
        consume_post_packet((x, packet[:1]), x)


def test_disabled_factory_no_cuda_or_model_access(monkeypatch):
    monkeypatch.setenv("CDB_AV_DRAFT_POST_GRAPH", "0")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing",
                        lambda: pytest.fail("CUDA touched"))
    assert get_draft_post_graph(object(), "hf_confidence") is None


@pytest.mark.parametrize("value", ["", "2", "true"])
def test_bad_flags_rejected(value, monkeypatch):
    monkeypatch.setenv("CDB_AV_DRAFT_POST_GRAPH", value)
    with pytest.raises(ValueError):
        get_draft_post_graph(object(), "hf_confidence")


@pytest.mark.parametrize("policy", ["manual", "replay_acceptance"])
def test_policy_gate(policy, monkeypatch):
    monkeypatch.setenv("CDB_AV_DRAFT_POST_GRAPH", "1")
    with pytest.raises(ValueError, match="persistent HF"):
        get_draft_post_graph(object(), policy)


def test_refill_helper_consumes_only_actual_graph_result():
    from vllm.v1.spec_decode.cdb_av_refill import RefillSession
    s = object.__new__(RefillSession)
    s.p = NS(model=NS(compute_logits=lambda h: h + 1))
    h = torch.randn(2, 7)
    s._post_graph = None
    assert torch.equal(s._compute_draft_logits(h), h+1)
    out = h+2
    packet = native_hf_packet(out)
    s._post_graph = lambda hidden, **kwargs: (out, packet)
    s._post_graph_context = None
    assert s._compute_draft_logits(h) is out
    assert consume_post_packet(s._post_graph_pending, out) is packet


def model(device):
    weight = torch.randn(31, 8, device=device, dtype=torch.bfloat16)
    m = NS(lm_head=NS(weight=weight, quant_method=object()),
           logits_processor=NS(scale=1., soft_cap=None, org_vocab_size=31))
    m.compute_logits = lambda h: F.linear(h, m.lm_head.weight)
    return m


GPU = pytest.mark.skipif(os.getenv("CDB_POST_GPU_TEST") != "1",
                        reason="explicit isolated CUDA run only")


@GPU
@pytest.mark.parametrize("rows", [1, 2, 3, 4])
def test_cuda_changed_hidden_and_head_storage(rows, monkeypatch):
    monkeypatch.setenv("CDB_AV_DRAFT_POST_VALIDATE", "1")
    monkeypatch.setenv("CDB_AV_DRAFT_POST_VALIDATE_SAMPLES", "2")
    m = model("cuda:0")
    post = DraftPostGraph(m, "cuda:0", hidden_size=8, vocab_size=31)
    with torch.inference_mode():
        for i in range(4):
            h = torch.randn(rows, 8, device="cuda", dtype=torch.bfloat16)
            if i == 2:
                m.lm_head.weight = m.lm_head.weight.clone()
            logits, packet = post(h)
            expected = m.compute_logits(h)
            assert torch.equal(logits.view(torch.uint8), expected.view(torch.uint8))
            assert torch.equal(packet.view(torch.uint8),
                               native_hf_packet(expected).view(torch.uint8))
    assert post.captures == 2 and post.validated == 2
    assert len(post.entries) == 1


@GPU
def test_cuda_row_cache_and_stream_guard(monkeypatch):
    monkeypatch.setenv("CDB_AV_DRAFT_POST_VALIDATE", "0")
    post = DraftPostGraph(model("cuda:0"), "cuda:0", hidden_size=8, vocab_size=31)
    with torch.inference_mode():
        for nr in [4, 1, 2, 3, 1, 4]:
            post(torch.randn(nr, 8, device="cuda", dtype=torch.bfloat16))
        assert len(post.entries) == 4 and post.captures == 4
        other = torch.cuda.Stream()
        with torch.cuda.stream(other):
            with pytest.raises(RuntimeError, match="one serial"):
                post(torch.randn(1, 8, device="cuda", dtype=torch.bfloat16))


@GPU
def test_cuda_failure_poisoned_and_bad_shape(monkeypatch):
    post = DraftPostGraph(model("cuda:0"), "cuda:0", hidden_size=8, vocab_size=31)
    h = torch.randn(1, 8, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        post(h.float())
    post.model.compute_logits = lambda h: (_ for _ in ()).throw(RuntimeError("head failure"))
    with pytest.raises(RuntimeError, match="head failure"):
        post(h)
    with pytest.raises(RuntimeError, match="poisoned"):
        post(h)
