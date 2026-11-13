# SPDX-License-Identifier: Apache-2.0
"""Opt-in complete Draft-step staging;no KV or scheduling state is owned here."""
import json
import operator
import os
import struct
import threading
from dataclasses import dataclass

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class InputField:
    name: str
    code: str
    dtype: torch.dtype
    shape: tuple
    values: tuple
    offset: int
    nbytes: int


@dataclass(frozen=True)
class InputPlan:
    fields: tuple
    starts: tuple
    requests: int
    tokens: int
    width: int
    nbytes: int


def make_input_plan(ids, positions, blocks, seqs, qlens, slots, last_indices,
                    *, max_requests=4, max_tokens=4096, max_context=1024):
    """CPU-only shape/layout planning;no CUDA or implicit default device."""
    qs = tuple(operator.index(q) for q in qlens)
    nr = len(qs)
    if not 1 <= nr <= max_requests or min(qs) < 1:
        raise ValueError("Invalid staged request/query count")
    n = sum(qs)
    if (n > max_tokens or len(ids) != n or len(positions) != n
            or len(slots) != n or len(blocks) != nr or len(seqs) != nr
            or len(last_indices) != nr or any(not row for row in blocks)):
        raise ValueError("Staged input dimensions differ")
    if any(not 1 <= operator.index(s) <= max_context for s in seqs):
        raise ValueError("Staged sequence length outside context")
    starts = [0]
    for q in qs:
        starts.append(starts[-1] + q)
    if list(last_indices) != [s - 1 for s in starts[1:]]:
        raise ValueError("Staged last-token indices differ from request offsets")
    width = max(map(len, blocks))
    table = tuple(v for row in blocks for v in (*row, *([0] * (width-len(row)))))
    raw = [
        ("ids", "i", torch.int32, (n,), tuple(ids)),
        ("positions", "q", torch.int64, (n,), tuple(positions)),
        ("query_start_loc", "i", torch.int32, (nr+1,), tuple(starts)),
        ("seq_lens", "i", torch.int32, (nr,), tuple(seqs)),
        ("block_table_tensor", "i", torch.int32, (nr,width), table),
        ("slot_mapping", "q", torch.int64, (n,), tuple(slots)),
        ("last_indices", "q", torch.int64, (nr,), tuple(last_indices)),
    ]
    fields = []
    offset = 0
    for name, code, dtype, shape, values in raw:
        offset = (offset + 15) // 16 * 16
        nbytes = len(values) * struct.calcsize("="+code)
        fields.append(InputField(name,code,dtype,shape,values,offset,nbytes))
        offset += nbytes
    return InputPlan(tuple(fields),tuple(starts),nr,n,width,(offset+15)//16*16)


def encode_input_plan(plan, destination, *, clear=True):
    """Write exact int32/int64 bytes with range checking by struct.pack_into."""
    view = memoryview(destination).cast("B")
    if view.readonly or len(view) < plan.nbytes:
        raise ValueError("Staged destination is readonly or too small")
    if clear:
        view[:plan.nbytes] = bytes(plan.nbytes)
    for f in plan.fields:
        # '=' uses native byte order and standard4/8-byte integer widths.
        # It rejects non-integers and overflow instead of silently wrapping.
        struct.pack_into(f"={len(f.values)}{f.code}",view,f.offset,*f.values)


def validate_staging_scope(cfg):
    pc = cfg.parallel_config
    if (not cfg.model_config.enforce_eager
            or cfg.model_config.dtype != torch.bfloat16
            or cfg.scheduler_config.async_scheduling
            or cfg.scheduler_config.max_num_seqs != 4
            or not 1 <= cfg.model_config.max_model_len <= 1024
            or cfg.lora_config is not None
            or getattr(cfg.model_config, "enable_sleep_mode", False)
            or any(type(x) is not int or x != 1 for x in (pc.tensor_parallel_size,
                    pc.pipeline_parallel_size,pc.data_parallel_size,
                    pc.decode_context_parallel_size))):
        raise ValueError("Pinned input experiment requires static eager BF16 capacity4/TP-PP-DP-DCP1,no async/LoRA/sleep")


class PinnedDraftInputStager:
    """Reusable pinned source;fresh GPU payload and CPUquery shadow per call.

    A source-copy event MUST complete before host bytes are changed or grown.
    Fresh device storage avoids changing metadata retained by audit/graph users.
    The class owns no request,token,KV block or lease state.
    """
    def __init__(self, device, *, max_requests=4, max_tokens=4096,
                 max_context=1024, max_bytes=1<<20):
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("Pinned Draft input staging requires CUDA")
        self.max_requests = max_requests
        self.max_tokens = max_tokens
        self.max_context = max_context
        self.max_bytes = max_bytes
        self.owner = threading.get_ident()
        self.host = self.host_view = self.zero_view = None
        self.event = torch.cuda.Event(enable_timing=False)
        self.validation_enabled = os.getenv("CDB_AV_VALIDATE_METADATA","0") == "1"
        self.validation_limit = (int(os.getenv("CDB_AV_INPUT_VALIDATE_SAMPLES","64"))
                                 if self.validation_enabled else 64)
        if not 1 <= self.validation_limit <= 256:
            raise ValueError("Invalid input validation sample limit")
        self.validation_samples = 0
        self.pending = False
        self.failed = False
        self.calls = self.source_waits = self.growths = self.bytes_copied = 0

    def pack(self, ids, positions, blocks, seqs, qlens, slots, last_indices):
        if self.failed:
            raise RuntimeError("Pinned input stager poisoned by failed CUDA copy")
        if threading.get_ident() != self.owner:
            raise RuntimeError("Pinned input stager used by another thread")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Host input staging cannot run inside CUDA capture")
        plan = make_input_plan(ids,positions,blocks,seqs,qlens,slots,last_indices,
            max_requests=self.max_requests,max_tokens=self.max_tokens,
            max_context=self.max_context)
        if plan.nbytes > self.max_bytes:
            raise ValueError("Pinned input payload exceeds bounded allocation")
        if self.pending and not self.event.query():
            self.source_waits += 1
            self.event.synchronize()
        if self.host is None or self.host.numel() < plan.nbytes:
            capacity = max(1024,1 << (plan.nbytes-1).bit_length())
            capacity = min(capacity,self.max_bytes)
            # Normal CPUstorage remains reusable across inference-mode scopes.
            with torch.inference_mode(False):
                self.host = torch.empty(capacity,dtype=torch.uint8,
                                        device="cpu",pin_memory=True)
            self.host_view = memoryview(self.host.numpy()).cast("B")
            self.zero_view = memoryview(bytes(capacity))
            self.growths += 1
        self.host_view[:plan.nbytes] = self.zero_view[:plan.nbytes]
        encode_input_plan(plan,self.host_view,clear=False)
        payload = torch.empty(plan.nbytes,dtype=torch.uint8,device=self.device)
        try:
            payload.copy_(self.host[:plan.nbytes],non_blocking=True)
            self.event.record(torch.cuda.current_stream(self.device))
        except BaseException:
            # No recovery can reuse an unacknowledged DMA source.
            self.failed = True
            raise
        self.pending = True
        fields = {f.name:payload[f.offset:f.offset+f.nbytes].view(f.dtype).view(f.shape)
                  for f in plan.fields}
        fields["query_start_loc_cpu"] = torch.tensor(
            plan.starts,dtype=torch.int32,device="cpu")
        self.calls += 1
        self.bytes_copied += plan.nbytes
        if self.calls == 1 or self.calls % 1024 == 0:
            logger.info("[CDB_INPUT_STAGING] %s",json.dumps(dict(
                source=__file__,calls=self.calls,h2d_enqueues=self.calls,
                bytes_copied=self.bytes_copied,source_waits=self.source_waits,
                capacity=self.host.numel(),growths=self.growths,
                pinned=True,non_blocking=True,fresh_device_payload=True,
                fresh_cpu_query_shadow=True,device=str(self.device))))
        if self.validation_enabled and self.validation_samples < self.validation_limit:
            checks = []
            for f in plan.fields:
                ref = torch.tensor(f.values,dtype=f.dtype,device="cpu").view(f.shape)
                actual = fields[f.name].cpu()
                exact = (actual.dtype == ref.dtype and actual.shape == ref.shape
                         and actual.stride() == ref.stride()
                         and torch.equal(actual.contiguous().view(torch.uint8),
                                         ref.contiguous().view(torch.uint8)))
                checks.append(dict(field=f.name,shape=list(f.shape),
                                   dtype=str(f.dtype),byte_exact=exact))
            shadow = fields["query_start_loc_cpu"]
            shadow_ok = (shadow.device.type == "cpu"
                         and shadow.tolist() == list(plan.starts)
                         and shadow.untyped_storage().data_ptr()
                             != self.host.untyped_storage().data_ptr())
            if not shadow_ok or not all(c["byte_exact"] for c in checks):
                raise RuntimeError("Pinned input/native CPU byte mismatch")
            self.validation_samples += 1
            logger.info("[CDB_INPUT_VALIDATE] %s",json.dumps(dict(
                call=self.calls,sample=self.validation_samples,
                requests=plan.requests,tokens=plan.tokens,
                checks=checks,cpu_shadow_independent=shadow_ok)))
        return fields


def get_input_stager(proposer):
    flag = os.getenv("CDB_AV_PINNED_INPUTS","0")
    if flag not in ("0","1"):
        raise ValueError("CDB_AV_PINNED_INPUTS must be0or1")
    if flag == "0":
        return None
    if any(os.getenv(k,"0") != "1" for k in ("CDB_AV_REFILL","CDB_AV_PERSISTENT")):
        raise ValueError("Pinned input experiment requires persistent refill")
    validate_staging_scope(proposer.vllm_config)
    current = getattr(proposer,"_cdb_input_stager",None)
    if current is None:
        current = PinnedDraftInputStager(proposer.device,
            max_tokens=4*proposer.max_model_len,
            max_context=proposer.max_model_len)
        proposer._cdb_input_stager = current
    return current
