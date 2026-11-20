# SPDX-License-Identifier: Apache-2.0
"""Experimental full-operation greedy rejection Graph; not wired into runner.

The exact existing sampler, including softmax, runs in capture and eager paths.
GPU gates and end-to-end evidence are required before model integration.
"""
from dataclasses import dataclass, replace
import threading

import torch

from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.logits_processor.builtin import (
    LogitBiasLogitsProcessor, MinTokensLogitsProcessor)


_META_FIELDS = ('draft_token_ids', 'cu_num_draft_tokens',
                'cu_num_sampled_tokens', 'target_logits_indices',
                'bonus_logits_indices', 'logits_indices')


@dataclass(frozen=True)
class VerifyGraphKey:
    device: torch.device
    dtype: torch.dtype
    requests: int
    draft_tokens: int
    max_k: int
    vocab: int
    metadata_dtypes: tuple


def _inactive_processors(processors):
    # These exact built-ins return the input unchanged when their CPU state is
    # empty. Custom classes/subclasses and active states always retain eager.
    for proc in processors:
        if type(proc) is MinTokensLogitsProcessor and not proc.min_toks:
            continue
        if type(proc) is LogitBiasLogitsProcessor and not proc.biases:
            continue
        return False
    return True


def verify_graph_key(metadata, draft_probs, logits, sampling):
    """Only shape/CPU flags: never inspect GPU values or synchronize."""
    if (logits is None or draft_probs is not None or not sampling.all_greedy
            or sampling.all_random or sampling.max_num_logprobs is not None
            or not sampling.no_penalties or sampling.bad_words_token_ids
            or sampling.allowed_token_ids_mask is not None
            or not _inactive_processors(sampling.logitsprocs.non_argmax_invariant)):
        return None
    counts = metadata.num_draft_tokens
    if (not counts or not 1 <= len(counts) <= 4
            or any(type(n) is not int or not 0 <= n <= 8 for n in counts)):
        return None
    nr, nt, mk = len(counts), sum(counts), max(counts)
    if not 1 <= mk <= 8 or metadata.max_spec_len != mk:
        return None
    if (logits.device.type != 'cuda' or logits.dtype not in (torch.float32, torch.bfloat16)
            or not logits.is_contiguous() or logits.ndim != 2
            or logits.shape[0] != nt + nr
            or not 1 <= logits.shape[1] <= 128256):
        return None
    shapes = [(nt,), (nr,), (nr,), (nt,), (nr,), (nt+nr,)]
    dtypes = []
    for name, shape in zip(_META_FIELDS, shapes):
        value = getattr(metadata, name)
        if (value.device != logits.device or tuple(value.shape) != shape
                or value.dtype not in (torch.int32, torch.int64)
                or not value.is_contiguous()):
            return None
        dtypes.append(value.dtype)
    return VerifyGraphKey(logits.device, logits.dtype, nr, nt, mk,
                          logits.shape[1], tuple(dtypes))


class GreedyVerifyGraph:
    """Isolated primitive; caller must reserve the device before GPU use.

    Cache is bounded by key count and conservative estimated bytes. Fresh output
    clones prevent a later replay from overwriting a previously returned result.
    No request or KV ownership state is modified by this class.
    """
    def __init__(self, sampler, *, min_hits=8, max_keys=8, max_bytes=512<<20):
        if min_hits < 1 or not 1 <= max_keys <= 16 or max_bytes < 1:
            raise ValueError('Invalid Verify Graph cache bounds')
        self.sampler = sampler
        self._captured_sampler = sampler
        self._captured_inner_sampler = sampler.sampler
        self.min_hits, self.max_keys, self.max_bytes = min_hits,max_keys,max_bytes
        self.owner = threading.get_ident()
        self.cache = {}
        self.hits = {}
        self.estimated_bytes = 0
        self.stream = None
        self.mode = (sampler.sampler.logprobs_mode,
                     sampler.is_processed_logprobs_mode,
                     sampler.is_logits_logprobs_mode)

    @staticmethod
    def _copy(entry, metadata, logits):
        entry['logits'].copy_(logits)
        for name in _META_FIELDS:
            getattr(entry['metadata'], name).copy_(getattr(metadata, name))

    def _capture(self, key, metadata, logits, sampling):
        tensors = {name: torch.empty_like(getattr(metadata,name))
                   for name in _META_FIELDS}
        static = replace(metadata, **tensors,
                         num_draft_tokens=list(metadata.num_draft_tokens))
        entry = dict(logits=torch.empty_like(logits), metadata=static)
        # Drop per-request CPU lists from captured Python metadata. These fields
        # are not read in the explicitly guarded penalty-free greedy path.
        scoped = replace(sampling, output_token_ids=[[] for _ in range(key.requests)],
                         spec_token_ids=None, bad_words_token_ids={}, generators={})
        self._copy(entry, metadata, logits)
        # Warm the exact kernels before capture; sampling is greedy and stateless.
        self.sampler(static, None, entry['logits'], scoped)
        torch.cuda.synchronize(logits.device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.device(logits.device), torch.cuda.graph(graph):
            output = self.sampler(static, None, entry['logits'], scoped)
        if output.logprobs_tensors is not None:
            raise RuntimeError('Unexpected logprobs in greedy Verify capture')
        entry.update(graph=graph, output=output)
        return entry

    @torch.inference_mode()
    def __call__(self, metadata, draft_probs, logits, sampling):
        key = verify_graph_key(metadata,draft_probs,logits,sampling)
        mode = (self.sampler.sampler.logprobs_mode,
                self.sampler.is_processed_logprobs_mode,
                self.sampler.is_logits_logprobs_mode)
        if (key is None or mode != self.mode
                or self.sampler is not self._captured_sampler
                or self.sampler.sampler is not self._captured_inner_sampler
                or threading.get_ident() != self.owner):
            return self.sampler(metadata,draft_probs,logits,sampling)
        if torch.cuda.is_current_stream_capturing():
            return self.sampler(metadata,draft_probs,logits,sampling)
        stream = torch.cuda.current_stream(logits.device)
        if self.stream is None:
            self.stream = stream
        elif self.stream != stream:
            return self.sampler(metadata,draft_probs,logits,sampling)
        entry = self.cache.get(key)
        if entry is None:
            if key not in self.hits and len(self.hits) >= 256:
                self.hits.pop(next(iter(self.hits)))
            self.hits[key] = self.hits.get(key,0) + 1
            # The sampler promotes BF16 logits internally. Budget against FP32
            # intermediates rather than the smaller raw input element size.
            estimate = logits.numel()*4*8 + 4096
            if (self.hits[key] < self.min_hits or len(self.cache) >= self.max_keys
                    or self.estimated_bytes + estimate > self.max_bytes):
                return self.sampler(metadata,draft_probs,logits,sampling)
            entry = self._capture(key,metadata,logits,sampling)
            self.cache[key] = entry
            self.estimated_bytes += estimate
        else:
            self._copy(entry, metadata, logits)
        entry['graph'].replay()
        return SamplerOutput(sampled_token_ids=entry['output'].sampled_token_ids.clone(),
                             logprobs_tensors=None)
