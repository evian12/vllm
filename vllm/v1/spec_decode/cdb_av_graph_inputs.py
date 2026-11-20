# SPDX-License-Identifier: Apache-2.0
"""Graph-owned integer staging primitive, not enabled in the model hot path.

One instance belongs to one Graph entry. All returned views are persistent and
are overwritten on the next stage call. They must never be used as independent
request-owned metadata. The entry must outlive every capture referencing them.
"""
import threading
import struct

import torch

from vllm.v1.spec_decode.cdb_av_inputs import encode_input_plan, make_input_plan


class GraphInputStage:
    def __init__(self, device, *, requests, tokens, width, max_context=1024):
        if not (1 <= requests <= 4 and requests <= tokens <= 32
                and 1 <= width <= 64 and 1 <= max_context <= 1024):
            raise ValueError('Graph input stage outside bounded shape')
        self.device = torch.device(device)
        if self.device.type != 'cuda':
            raise ValueError('Graph input stage requires CUDA')
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('Stage allocation cannot occur inside capture')
        self.requests, self.tokens, self.width = requests, tokens, width
        self.max_context = max_context
        self.owner = threading.get_ident()
        self.stream = torch.cuda.current_stream(self.device)
        # Only the structural layout of this seed plan is used.
        qs = [1] * (requests - 1) + [tokens - requests + 1]
        starts = [0]
        for q in qs:
            starts.append(starts[-1] + q)
        plan = make_input_plan([0]*tokens, [0]*tokens,
            [[0]*width for _ in qs], qs, qs, [0]*tokens,
            [x-1 for x in starts[1:]], max_context=max_context)
        with torch.inference_mode(False):
            self.host = torch.empty(plan.nbytes, dtype=torch.uint8,
                                    device='cpu', pin_memory=True)
            self.payload = torch.empty(plan.nbytes, dtype=torch.uint8,
                                       device=self.device)
            self.query_cpu = torch.empty(requests+1, dtype=torch.int32,
                                         device='cpu')
        self.host_view = memoryview(self.host.numpy()).cast('B')
        self.query_view = memoryview(self.query_cpu.numpy()).cast('B')
        self.fields = {f.name: self.payload[f.offset:f.offset+f.nbytes]
                       .view(f.dtype).view(f.shape) for f in plan.fields}
        self.fields['query_start_loc_cpu'] = self.query_cpu
        self.event = torch.cuda.Event(enable_timing=False)
        self.pending = self.failed = False
        self.calls = self.source_waits = 0

    def stage(self, ids, positions, blocks, seqs, qlens, slots, last_indices):
        if self.failed:
            raise RuntimeError('Graph input stage poisoned by failed copy')
        if threading.get_ident() != self.owner:
            raise RuntimeError('Graph input stage used by another thread')
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('Host staging cannot run inside capture')
        if torch.cuda.current_stream(self.device) != self.stream:
            raise RuntimeError('Graph input stage requires its owning stream')
        if (len(qlens) != self.requests or sum(qlens) != self.tokens
                or any(not row or len(row) > self.width for row in blocks)):
            raise ValueError('Graph input stage shape changed')
        padded = [list(row) + [0]*(self.width-len(row)) for row in blocks]
        plan = make_input_plan(ids, positions, padded, seqs, qlens, slots,
            last_indices, max_context=self.max_context, max_tokens=32)
        if plan.nbytes != self.payload.numel():
            raise ValueError('Graph input stage layout changed')
        if self.pending and not self.event.query():
            self.source_waits += 1
            self.event.synchronize()
        encode_input_plan(plan, self.host_view)
        # CPU metadata is only consumed by host logic outside capture.
        struct.pack_into(f'={self.requests+1}i', self.query_view, 0, *plan.starts)
        try:
            # Same-stream ordering protects earlier Graph consumers, while the
            # event protects pinned host bytes against an outstanding DMA read.
            self.payload.copy_(self.host, non_blocking=True)
            self.event.record(self.stream)
        except BaseException:
            self.failed = True
            raise
        self.pending = True
        self.calls += 1
        return self.fields
