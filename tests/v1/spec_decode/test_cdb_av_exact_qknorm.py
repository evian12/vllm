# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
import hashlib
import json
from pathlib import Path
import pytest
import torch
from torch import nn
from vllm.model_executor.layers import cdb_exact_qknorm as mod

def valid_scope():
    return dict(dtype=torch.bfloat16,eager=True,asynchronous=False,prefix_cache=False,
        lora=False,quantized=False,parallel_sizes=(1,1,1,1),capacity=4)

def test_valid_static_scope():
    mod.validate_scope(**valid_scope())

@pytest.mark.parametrize('name,value',[
 ('dtype',torch.float16),('eager',False),('asynchronous',True),
 ('prefix_cache',True),('lora',True),('quantized',True),('capacity',8),
 ('parallel_sizes',(2,1,1,1)),('parallel_sizes',(1,1,1)),
 ('parallel_sizes',(True,1,1,1)),('batch_invariant',True)])
def test_unsupported_scope_rejected(name,value):
    args=valid_scope();args[name]=value
    with pytest.raises(ValueError,match='Exact QK requires'):
        mod.validate_scope(**args)

def test_disabled_factory_never_loads_binary(monkeypatch):
    monkeypatch.setenv('CDB_AV_EXACT_QKNORM','0')
    monkeypatch.setattr(mod,'_load',lambda _:pytest.fail('disabled path loaded extension'))
    assert mod.make_exact_qknorm(None,'',None) is None

def test_bad_switch_rejected(monkeypatch):
    monkeypatch.setenv('CDB_AV_EXACT_QKNORM','yes')
    with pytest.raises(ValueError,match='Invalid exact QK switch'):
        mod.make_exact_qknorm(None,'',None)

def test_non_cdb_use_rejected_before_config(monkeypatch):
    monkeypatch.setenv('CDB_AV_EXACT_QKNORM','1')
    monkeypatch.setenv('CDB_AV_REFILL','0')
    with pytest.raises(ValueError,match='persistent refill'):
        mod.make_exact_qknorm(None,'',None)

def receipt_fixture(tmp_path,monkeypatch):
    required=['experiments/cdb_exact_qknorm/binding.cpp',
       'experiments/cdb_exact_qknorm/kernel.cu','csrc/cub_helpers.h',
       'csrc/quantization/vectorization_utils.cuh','csrc/quantization/vectorization.cuh']
    for name in required:
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(name)
    path=tmp_path/'operator.so';path.write_bytes(b'fixture binary, not loaded')
    receipt=dict(format=1,module=mod.NAME,torch=torch.__version__,
        binary_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        sources={name:hashlib.sha256((tmp_path/name).read_bytes()).hexdigest() for name in required})
    path.with_suffix('.json').write_text(json.dumps(receipt))
    monkeypatch.setattr(mod,'ROOT',tmp_path)
    return path,receipt

def test_receipt_provenance(tmp_path,monkeypatch):
    path,receipt=receipt_fixture(tmp_path,monkeypatch)
    assert mod.validate_receipt(path)==(path,receipt)

@pytest.mark.parametrize('change',['binary','source','incomplete','torch'])
def test_bad_receipt_is_rejected_before_dlopen(tmp_path,monkeypatch,change):
    path,receipt=receipt_fixture(tmp_path,monkeypatch)
    if change=='binary':path.write_bytes(b'changed')
    elif change=='source':(tmp_path/'experiments/cdb_exact_qknorm/kernel.cu').write_text('changed')
    elif change=='incomplete':receipt['sources'].pop('csrc/cub_helpers.h')
    else:receipt['torch']='different'
    path.with_suffix('.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        mod.validate_receipt(path)

def test_row_domain_falls_back_to_complete_native_work(monkeypatch):
    class Rotary(nn.Module):
        def forward(self,pos,q,k):
            return q,k
    att=SimpleNamespace(training=False,q_size=2048,kv_size=1024,num_heads=16,
        num_kv_heads=8,q_norm=nn.Identity(),k_norm=nn.Identity(),rotary_emb=Rotary())
    kernel=SimpleNamespace(forward=lambda *args:pytest.fail('out of domain used kernel'))
    callback=mod.ExactQK(kernel,'draft','fixture')
    data=torch.arange(1025*4096,dtype=torch.float32).reshape(1025,4096)
    q,k=callback(att,torch.zeros(1025,dtype=torch.int64),data)
    assert torch.equal(q,data[:,:2048])
    assert torch.equal(k,data[:,2048:3072])
    assert callback.fallback_seen=={'row_or_layout_domain'}

@pytest.mark.parametrize('hook',[False,True])
def test_qwen_dispatch_preserves_value_and_projection_count(hook):
    from vllm.model_executor.models.qwen3 import Qwen3Attention
    class Project(nn.Module):
        def __init__(self):
            super().__init__();self.calls=0
        def forward(self,x):
            self.calls+=1
            return x,None
    class Attend(nn.Module):
        def forward(self,q,k,v):
            self.inputs=(q,k,v)
            return q
    class Rotate(nn.Module):
        def forward(self,pos,q,k):
            return q,k
    att=Qwen3Attention.__new__(Qwen3Attention);nn.Module.__init__(att)
    att.q_size=2048;att.kv_size=1024;att.head_dim=128
    att.qkv_proj=Project();att.o_proj=Project()
    att.q_norm=nn.Identity();att.k_norm=nn.Identity();att.rotary_emb=Rotate()
    att.attn=Attend()
    def replacement(module,pos,qkv):
        return qkv[:,:2048]+1,qkv[:,2048:3072]+1
    att._cdb_exact_qknorm=replacement if hook else None
    x=torch.randn(3,4096)
    out=att.forward(torch.arange(3),x)
    q,k,v=att.attn.inputs
    assert torch.equal(q,x[:,:2048]+int(hook))
    assert torch.equal(k,x[:,2048:3072]+int(hook))
    assert torch.equal(v,x[:,3072:])
    assert torch.equal(out,q)
    assert att.qkv_proj.calls==att.o_proj.calls==1
