# SPDX-License-Identifier: Apache-2.0
"""Opt-in Draft LM-head/native HF graph; owns no request, KV or scheduler state."""
import json
import os
import threading

import torch

from vllm.logger import init_logger
from vllm.v1.spec_decode.cdb_av_inputs import validate_staging_scope

logger = init_logger(__name__)


def native_hf_packet(logits):
    """Exactly the existing float32 max/logsumexp and float64 transport."""
    fp32 = logits.float()
    maximum, tokens = fp32.max(dim=-1)
    confidence = (maximum - torch.logsumexp(fp32, dim=-1)).exp()
    return torch.stack(
        (tokens.to(torch.float64), confidence.to(torch.float64)), dim=1)


def consume_post_packet(pending, logits):
    if pending is None:
        return None
    source, packet = pending
    if source is not logits:
        raise RuntimeError("Draft post graph/logits identity mismatch")
    if packet.shape != (logits.shape[0], 2):
        raise RuntimeError("Draft post graph/request row mismatch")
    return packet


class DraftPostGraph:
    """Serial Worker-only static hidden buffers, at most four row shapes.

    Outputs are borrowed until the next call. Refill consumes them via its
    existing blocking D2H before any subsequent Draft call. Inputs are copied
    afresh on the same stream; no request IDs or KV addresses are captured.
    """
    def __init__(self, model, device, *, hidden_size=2048, vocab_size=151936):
        self.model = model
        self.device = torch.device(device)
        self.hidden_size, self.vocab_size = hidden_size, vocab_size
        self.owner = threading.get_ident()
        self.stream = None
        self.storage = None
        self.entries = {}
        self.failed = False
        self.calls = self.captures = self.validated = 0
        self.validate = os.getenv("CDB_AV_DRAFT_POST_VALIDATE", "0") == "1"
        self.limit = int(os.getenv("CDB_AV_DRAFT_POST_VALIDATE_SAMPLES", "64"))
        if not 1 <= self.limit <= 256:
            raise ValueError("Invalid Draft post validation limit")

    def _storage(self):
        head = self.model.lm_head
        weight = head.weight
        processor = self.model.logits_processor
        return (id(self.model), id(head), id(head.quant_method),
                weight.data_ptr(), tuple(weight.shape), tuple(weight.stride()),
                weight.dtype, weight.device, id(processor),
                processor.scale, processor.soft_cap, processor.org_vocab_size)

    def __call__(self, hidden, *, context=None):
        if self.failed:
            raise RuntimeError("Draft post graph poisoned by previous failure")
        if threading.get_ident() != self.owner:
            raise RuntimeError("Draft post graph used by another thread")
        if (hidden.ndim != 2 or not 1 <= hidden.shape[0] <= 4
                or hidden.shape[1] != self.hidden_size
                or hidden.dtype != torch.bfloat16
                or hidden.device != self.device or not hidden.is_contiguous()):
            raise ValueError("Draft post graph hidden shape/dtype/device/layout")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Draft post graph called inside another capture")
        stream = torch.cuda.current_stream(self.device).cuda_stream
        if self.stream is not None and self.stream != stream:
            raise RuntimeError("Draft post graph requires one serial CUDA stream")
        self.stream = stream
        storage = self._storage()
        if storage != self.storage:
            self.entries.clear()
            self.storage = storage
        rows = hidden.shape[0]
        action = "replay"
        try:
            entry = self.entries.get(rows)
            if entry is None:
                static = torch.empty_like(hidden)
                static.copy_(hidden)
                # Warm only the stateless head/reductions; never run backbone/KV.
                self.model.compute_logits(static)
                torch.cuda.synchronize(self.device)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    logits = self.model.compute_logits(static)
                    packet = native_hf_packet(logits)
                entry = dict(input=static, graph=graph, logits=logits,
                             packet=packet)
                if logits.shape != (rows, self.vocab_size):
                    raise ValueError("Draft post graph vocabulary shape")
                self.entries[rows] = entry
                self.captures += 1
                action = "capture"
            entry["input"].copy_(hidden)
            entry["graph"].replay()
            logits, packet = entry["logits"], entry["packet"]
            self.calls += 1
            if self.validate and self.validated < self.limit:
                reference = self.model.compute_logits(hidden)
                reference_packet = native_hf_packet(reference)
                logits_exact = torch.equal(logits.contiguous().view(torch.uint8),
                                           reference.contiguous().view(torch.uint8))
                packet_exact = torch.equal(packet.view(torch.uint8),
                                           reference_packet.view(torch.uint8))
                self.validated += 1
                logger.info("[CDB_DRAFT_POST_VALIDATE] %s", json.dumps(dict(
                    source=__file__, sample=self.validated, rows=rows,
                    logits_exact=logits_exact, packet_exact=packet_exact,
                    head_storage_checked=True, context=context)))
                if not (logits_exact and packet_exact):
                    raise RuntimeError("Draft post graph/native byte mismatch")
            if action == "capture" or self.calls % 1024 == 0:
                logger.info("[CDB_DRAFT_POST_GRAPH] %s", json.dumps(dict(
                    source=__file__, action=action, calls=self.calls,
                    rows=rows, captures=self.captures,
                    cached_shapes=len(self.entries), validated=self.validated,
                    stateless_head_and_confidence=True)))
            return logits, packet
        except BaseException:
            self.failed = True
            raise


def get_draft_post_graph(proposer, policy):
    flag = os.getenv("CDB_AV_DRAFT_POST_GRAPH", "0")
    if flag not in ("0", "1"):
        raise ValueError("Invalid CDB_AV_DRAFT_POST_GRAPH")
    if flag == "0":
        return None
    if (policy != "hf_confidence"
            or os.getenv("CDB_AV_PERSISTENT", "0") != "1"
            or os.getenv("CDB_AV_REFILL", "0") != "1"):
        raise ValueError("Draft post graph requires persistent HF refill")
    validate_staging_scope(proposer.vllm_config)
    model = proposer.model
    if (type(model).__name__ != "Qwen3ForCausalLM"
            or type(model.lm_head.quant_method).__name__ != "UnquantizedEmbeddingMethod"
            or model.lm_head.weight.shape != (151936, 2048)
            or model.lm_head.weight.dtype != torch.bfloat16):
        raise ValueError("Draft post graph requires unquantized Qwen3-1.7B BF16")
    result = getattr(proposer, "_cdb_draft_post_graph", None)
    if result is None or result.model is not model:
        result = DraftPostGraph(model, proposer.device)
        proposer._cdb_draft_post_graph = result
    return result
