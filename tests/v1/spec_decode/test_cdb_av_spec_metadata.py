# SPDX-License-Identifier: Apache-2.0
"""Independent native-index reference and serialization tests, no GPU/model."""
import ast
import os
import threading
from pathlib import Path
from types import SimpleNamespace as NS, MethodType

import numpy as np
import pytest
import torch

from vllm.v1.spec_decode import cdb_av_spec_metadata as mod
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata


def native_runner():
    path=Path(__file__).parents[3]/'vllm/v1/worker/gpu_model_runner.py'
    tree=ast.parse(path.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='GPUModelRunner')
    methods={n.name:n for n in cls.body if isinstance(n,ast.FunctionDef)
             and n.name in ('_get_cumsum_and_arange','_calc_spec_decode_metadata')}
    ns=dict(np=np,torch=torch,os=os,SpecDecodeMetadata=SpecDecodeMetadata)
    for name,node in methods.items():
        exec(compile('from __future__ import annotations\n'+ast.unparse(node),str(path),'exec'),ns)
    runner=NS(arange_np=np.arange(4096,dtype=np.int64),device='cpu')
    runner._get_cumsum_and_arange=MethodType(ns['_get_cumsum_and_arange'],runner)
    return runner,ns['_calc_spec_decode_metadata']


def case(drafts,extras,dtype=np.int32):
    runner,calc=native_runner()
    draft=np.array(drafts,dtype=dtype)
    cumulative=np.cumsum(draft+1+np.array(extras,dtype=dtype),dtype=dtype)
    total=int(cumulative[-1]);runner.input_ids=NS(gpu=torch.arange(total,dtype=torch.int32)*7+20)
    meta=calc(runner,draft,cumulative)
    values=[getattr(meta,n).numpy() for n in mod.NAMES[:5]]
    values.append(mod.direct_indices(values[2],values[3],total))
    return runner,meta,tuple(values)


@pytest.mark.parametrize('drafts,extras',[
    ([3,0,2,0],[0,99,0,99]),([0,0,0,0],[20,30,40,50]),
    ([8,8,8,8],[0,0,0,0]),([1],[0]),([0],[99]),([4,1,2],[40,0,20]),
    ([0,8,0,1],[1,40,100,0]),
])
@pytest.mark.parametrize('dtype',[np.int32,np.int64])
def test_packed_fields_and_direct_token_ids_match_original_native_function(monkeypatch,drafts,extras,dtype):
    monkeypatch.setenv('CDB_AV_PACKED_SPEC_METADATA','0')
    runner,reference,values=case(drafts,extras,dtype)
    before=[x.copy() for x in values]
    plan=mod.packet(values);buffer=bytearray(plan.nbytes);mod.encode(plan,buffer)
    decoded={f.name:torch.from_numpy(np.frombuffer(buffer,dtype=f.array.dtype,count=f.array.size,offset=f.offset))
             for f in plan.fields}
    for name in mod.NAMES[:5]:
        assert torch.equal(decoded[name],getattr(reference,name))
        assert decoded[name].dtype==getattr(reference,name).dtype
    assert torch.equal(runner.input_ids.gpu[decoded['draft_token_indices']],reference.draft_token_ids)
    assert all(np.array_equal(a,b) for a,b in zip(values,before))
    regions=sorted((f.offset,f.offset+f.array.nbytes) for f in plan.fields)
    assert all(a[1]<=b[0] for a,b in zip(regions,regions[1:]))
    assert all(f.offset%16==0 for f in plan.fields)


def test_full_signed_integer_bits_are_preserved(monkeypatch):
    monkeypatch.setenv('CDB_AV_PACKED_SPEC_METADATA','0')
    _,_,values=case([0,0],[0,0])
    values=list(values);values[2]=np.array([-(2**31),2**31-1],dtype=np.int32)
    plan=mod.packet(values);buf=bytearray(plan.nbytes);mod.encode(plan,buf)
    field=plan.fields[2]
    assert np.array_equal(np.frombuffer(buf,dtype=np.int32,count=2,offset=field.offset),values[2])


@pytest.mark.parametrize('kind',['float','unsigned','byteswap','noncontiguous','matrix','too_many','dimension'])
def test_invalid_vectors_rejected(monkeypatch,kind):
    monkeypatch.setenv('CDB_AV_PACKED_SPEC_METADATA','0')
    _,_,values=case([2,0],[0,0]);values=list(values)
    if kind=='float':values[2]=values[2].astype(np.float32)
    if kind=='unsigned':values[2]=values[2].astype(np.uint32)
    if kind=='byteswap':values[2]=values[2].astype('>i4')
    if kind=='noncontiguous':values[2]=np.arange(8,dtype=np.int32)[::2]
    if kind=='matrix':values[2]=values[2].reshape(2,2)
    if kind=='too_many':values[2]=np.zeros(37,dtype=np.int32)
    if kind=='dimension':values[4]=np.zeros(3,dtype=np.int32)
    with pytest.raises(ValueError):mod.packet(values)


def test_encode_destination_guard_and_empty_suffix(monkeypatch):
    monkeypatch.setenv('CDB_AV_PACKED_SPEC_METADATA','0')
    _,_,values=case([0],[0]);plan=mod.packet(values)
    with pytest.raises(ValueError):mod.encode(plan,bytes(plan.nbytes))
    with pytest.raises(ValueError):mod.encode(plan,bytearray(plan.nbytes-1))
    mod.encode(plan,bytearray(plan.nbytes))


@pytest.mark.parametrize('indices',[[0,-1],[0,100]])
def test_actual_input_bound_rejected_before_gpu_access(indices):
    with pytest.raises(ValueError):mod.direct_indices(np.array(indices,dtype=np.int32),np.array([0],dtype=np.int32),10)


def test_cpu_device_rejected_without_cuda_initialization():
    with pytest.raises(ValueError,match='requires CUDA'):mod.Stager('cpu')


def test_default_off_and_invalid_switch():
    assert not mod.enabled('FLAG',{}) and mod.enabled('FLAG',{'FLAG':'1'})
    with pytest.raises(ValueError):mod.enabled('FLAG',{'FLAG':'true'})


def test_actual_runner_dispatch_preserves_cpu_inputs(monkeypatch):
    monkeypatch.setenv('CDB_AV_PACKED_SPEC_METADATA','1')
    runner,calc=native_runner();runner.input_ids=NS(gpu=torch.arange(20,dtype=torch.int32))
    sentinel=object();calls=[]
    def build(*args):calls.append(args);return sentinel
    monkeypatch.setattr(mod,'build',build)
    draft=np.array([1,2],dtype=np.int32);scheduled=np.array([5,9],dtype=np.int32)
    assert calc(runner,draft,scheduled) is sentinel
    assert calls[0][0] is runner and calls[0][1] is draft and calls[0][-1]==9
    assert np.array_equal(draft,[1,2]) and np.array_equal(scheduled,[5,9])


def test_poison_thread_capture_and_size_guards_run_before_gpu_access(monkeypatch):
    stage=mod.Stager.__new__(mod.Stager)
    stage.failed=True
    with pytest.raises(RuntimeError,match='poisoned'):stage.copy(None)
    stage.failed=False;stage.owner=threading.get_ident()+1
    with pytest.raises(RuntimeError,match='another thread'):stage.copy(None)
    stage.owner=threading.get_ident()
    monkeypatch.setattr(torch.cuda,'is_current_stream_capturing',lambda:True)
    with pytest.raises(RuntimeError,match='inside CUDA capture'):stage.copy(None)
    monkeypatch.setattr(torch.cuda,'is_current_stream_capturing',lambda:False)
    with pytest.raises(ValueError,match='allocation bound'):stage.copy(mod.Packet((),5000))


def test_failed_copy_propagates_and_prevents_later_host_reuse(monkeypatch):
    monkeypatch.setenv('CDB_AV_PACKED_SPEC_METADATA','0')
    _,_,values=case([0],[0]);plan=mod.packet(values)
    stage=mod.Stager.__new__(mod.Stager)
    stage.failed=stage.pending=False;stage.owner=threading.get_ident()
    stage.host=torch.empty(1024,dtype=torch.uint8);stage.host_view=stage.host.numpy()
    stage.device=torch.device('cuda:0')
    class BrokenPayload:
        def copy_(self,*args,**kwargs):raise ArithmeticError('copy failure')
    monkeypatch.setattr(torch.cuda,'is_current_stream_capturing',lambda:False)
    monkeypatch.setattr(torch,'empty',lambda *args,**kwargs:BrokenPayload())
    with pytest.raises(ArithmeticError,match='copy failure'):stage.copy(plan)
    assert stage.failed
    with pytest.raises(RuntimeError,match='poisoned'):stage.copy(plan)
