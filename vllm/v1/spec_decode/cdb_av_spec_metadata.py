# SPDX-License-Identifier: Apache-2.0
"""Opt-in packed Target speculative metadata; no scheduling or KV state."""
from dataclasses import dataclass
import json
import os
import threading

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

logger=init_logger(__name__)
NAMES=('cu_num_draft_tokens','cu_num_sampled_tokens','logits_indices',
       'target_logits_indices','bonus_logits_indices','draft_token_indices')


def enabled(name,env=None):
    value=(os.environ if env is None else env).get(name,'0')
    if value not in ('0','1'):raise ValueError(f'{name} must be 0 or 1')
    return value=='1'


@dataclass(frozen=True)
class Field:
    name: str
    array: np.ndarray
    offset: int


@dataclass(frozen=True)
class Packet:
    fields: tuple
    nbytes: int


def packet(values):
    if len(values)!=6:raise ValueError('Expected six speculative metadata arrays')
    fields=[];offset=0
    for name,array in zip(NAMES,values):
        if (not isinstance(array,np.ndarray) or array.ndim!=1
                or not array.flags.c_contiguous or not array.dtype.isnative
                or array.dtype not in (np.dtype('int32'),np.dtype('int64'))
                or array.size>36):
            raise ValueError(f'Unsupported speculative metadata vector: {name}')
        offset=(offset+15)//16*16
        fields.append(Field(name,array,offset));offset+=array.nbytes
    nr=values[0].size
    if (not 1<=nr<=4 or values[1].size!=nr or values[4].size!=nr
            or int(values[0][-1])+nr!=int(values[1][-1])
            or values[2].size!=int(values[1][-1])
            or values[3].size!=int(values[0][-1]) or values[5].size!=values[3].size):
        raise ValueError('Speculative metadata vector dimensions disagree')
    return Packet(tuple(fields),(offset+15)//16*16)


def encode(plan,destination):
    view=memoryview(destination).cast('B')
    if view.readonly or len(view)<plan.nbytes:raise ValueError('Metadata destination too small/readonly')
    view[:plan.nbytes]=bytes(plan.nbytes)
    for f in plan.fields:
        if f.array.nbytes:
            view[f.offset:f.offset+f.array.nbytes]=memoryview(f.array).cast('B')


def direct_indices(logits_indices,target_indices,input_length):
    # Same integer indexing as input_ids[logits][target+1], before any H2D.
    direct=logits_indices[target_indices+1]
    if direct.size and (int(direct.min())<0 or int(direct.max())>=input_length):
        raise ValueError('Draft token index outside actual model input')
    return direct


class Stager:
    """One pinned upload, fresh GPU storage; protect host bytes until copy ends."""
    def __init__(self,device):
        self.device=torch.device(device)
        if self.device.type!='cuda':raise ValueError('Packed metadata requires CUDA')
        self.owner=threading.get_ident()
        self.host=self.host_view=None
        self.event=torch.cuda.Event(enable_timing=False)
        self.pending=self.failed=False
        self.calls=self.source_waits=self.growths=self.audits=0
        self.audit=enabled('CDB_AV_SPEC_METADATA_VALIDATE')

    def copy(self,plan):
        if self.failed:raise RuntimeError('Packed metadata stager poisoned after failed CUDA copy')
        if threading.get_ident()!=self.owner:raise RuntimeError('Packed metadata used by another thread')
        if torch.cuda.is_current_stream_capturing():raise RuntimeError('Metadata staging inside CUDA capture')
        if plan.nbytes>4096:raise ValueError('Packed metadata exceeds allocation bound')
        if self.pending and not self.event.query():
            self.source_waits+=1;self.event.synchronize()
        if self.host is None or self.host.numel()<plan.nbytes:
            size=max(1024,1<<(plan.nbytes-1).bit_length())
            with torch.inference_mode(False):
                self.host=torch.empty(size,dtype=torch.uint8,device='cpu',pin_memory=True)
            self.host_view=self.host.numpy();self.growths+=1
        encode(plan,self.host_view)
        payload=torch.empty(plan.nbytes,dtype=torch.uint8,device=self.device)
        try:
            payload.copy_(self.host[:plan.nbytes],non_blocking=True)
            self.event.record(torch.cuda.current_stream(self.device));self.pending=True
        except BaseException:
            self.failed=True
            raise
        self.calls+=1
        return {f.name:payload.narrow(0,f.offset,f.array.nbytes).view(
            torch.int32 if f.array.dtype.itemsize==4 else torch.int64).reshape(f.array.shape)
            for f in plan.fields}


def build(runner,num_draft_tokens,cu_draft,cu_sampled,logits,target,bonus,input_length):
    if not enabled('CDB_AV_PACKED_SPEC_METADATA'):raise RuntimeError('Packed metadata called while disabled')
    stager=getattr(runner,'_cdb_spec_metadata_stager',None)
    if stager is None:
        from vllm.v1.spec_decode.cdb_av_inputs import validate_staging_scope
        validate_staging_scope(runner.vllm_config)
        cfg=runner.vllm_config.speculative_config
        if cfg is None or not 1<=cfg.num_speculative_tokens<=8:
            raise ValueError('Packed metadata requires speculative cap1..8')
        stager=runner._cdb_spec_metadata_stager=Stager(runner.device)
    if not 0<input_length<=runner.input_ids.gpu.numel():raise ValueError('Invalid actual model input length')
    direct=direct_indices(logits,target,input_length)
    plan=packet((cu_draft,cu_sampled,logits,target,bonus,direct))
    gpu=stager.copy(plan)
    draft_ids=runner.input_ids.gpu[gpu['draft_token_indices']]
    result=SpecDecodeMetadata(draft_token_ids=draft_ids,num_draft_tokens=num_draft_tokens.tolist(),
        cu_num_draft_tokens=gpu['cu_num_draft_tokens'],cu_num_sampled_tokens=gpu['cu_num_sampled_tokens'],
        target_logits_indices=gpu['target_logits_indices'],bonus_logits_indices=gpu['bonus_logits_indices'],
        logits_indices=gpu['logits_indices'])
    if stager.audit and stager.audits<32:
        stager.audits+=1
        # Independent unchanged native transfer/index chain, diagnostic only.
        native=[torch.from_numpy(a).to(runner.device,non_blocking=True)
                for a in (cu_draft,cu_sampled,logits,target,bonus)]
        native_ids=runner.input_ids.gpu[native[2]][native[3]+1]
        checks={name:bool(torch.equal(gpu[name],ref)) and gpu[name].dtype==ref.dtype
                for name,ref in zip(NAMES,native)}
        checks['draft_token_ids']=torch.equal(draft_ids,native_ids) and draft_ids.dtype==native_ids.dtype
        logger.info('[CDB_SPEC_METADATA_VALIDATE] %s',json.dumps(dict(
            sample=stager.audits,req_ids=list(runner.input_batch.req_ids[:len(num_draft_tokens)]),
            num_draft_tokens=num_draft_tokens.tolist(),checks=checks,
            metadata_payload_bytes=plan.nbytes,optimized_payload_copy_calls=1,native_transfer_calls=5,
            source_waits=stager.source_waits,note='Extra native copies/gathers excluded from performance.')))
        if not all(checks.values()):raise RuntimeError('Packed/native speculative metadata mismatch')
    return result
