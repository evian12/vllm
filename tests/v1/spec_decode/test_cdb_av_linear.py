# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace as NS
import threading

import pytest
import torch

from vllm.v1.spec_decode import cdb_av_linear as mod


def config():
    return NS(model_config=NS(dtype=torch.bfloat16,quantization=None,enforce_eager=True,
        max_model_len=1024,enable_sleep_mode=False,hf_text_config=NS(hidden_size=2048,
        intermediate_size=8192,num_hidden_layers=16,vocab_size=128256)),
        scheduler_config=NS(async_scheduling=False),lora_config=None,
        parallel_config=NS(tensor_parallel_size=1,pipeline_parallel_size=1,
                           data_parallel_size=1,decode_context_parallel_size=1))


@pytest.mark.parametrize('value,expected',[('0',False),('1',True)])
def test_explicit_switch(value,expected):
    assert mod.enabled({'CDB_AV_DRAFT_DOWN_PAD32':value}) is expected
    assert not mod.enabled({})


@pytest.mark.parametrize('value',['2','true','',' 1'])
def test_bad_switch_fails(value):
    with pytest.raises(ValueError):mod.enabled({'CDB_AV_DRAFT_DOWN_PAD32':value})


@pytest.mark.parametrize('where,key,value',[
    ('model_config','dtype',torch.float16),('model_config','quantization','fp8'),
    ('model_config','enforce_eager',False),('model_config','enable_sleep_mode',True),
    ('model_config','max_model_len',2048),('scheduler_config','async_scheduling',True),
    ('parallel_config','tensor_parallel_size',2),('parallel_config','decode_context_parallel_size',2),
])
def test_unsupported_runtime_rejected(where,key,value):
    cfg=config();setattr(getattr(cfg,where),key,value)
    with pytest.raises(ValueError):mod.validate_scope(cfg,'LlamaForCausalLM')


@pytest.mark.parametrize('key,value',[('hidden_size',4096),('intermediate_size',6144),
                                    ('num_hidden_layers',28),('vocab_size',151936)])
def test_wrong_model_dimensions_rejected(key,value):
    cfg=config();setattr(cfg.model_config.hf_text_config,key,value)
    with pytest.raises(ValueError):mod.validate_scope(cfg,'LlamaForCausalLM')


def test_supported_scope_and_other_model_guard():
    mod.validate_scope(config(),'LlamaForCausalLM')
    with pytest.raises(ValueError):mod.validate_scope(config(),'Qwen3ForCausalLM')
    cfg=config();cfg.lora_config=object()
    with pytest.raises(ValueError):mod.validate_scope(cfg,'LlamaForCausalLM')


@pytest.mark.parametrize('m',[2,3,8,16])
@pytest.mark.parametrize('bias',[False,True])
def test_math_crops_only_extra_projection_rows_and_preserves_inputs(m,bias):
    torch.manual_seed(4)
    x=torch.randn(m,13,dtype=torch.float64);w=torch.randn(7,13,dtype=torch.float64)
    b=torch.randn(7,dtype=torch.float64) if bias else None
    saved=x.clone();weights=w.clone()
    calls=[]
    def original(layer,xx,bb):
        calls.append(tuple(xx.shape))
        assert not torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        return torch.nn.functional.linear(xx,w,bb)
    out=mod.padded_call(original,None,x,b)
    assert calls==[(32,13)] and out.shape==(m,7)
    torch.testing.assert_close(out,torch.nn.functional.linear(x,w,b))
    assert torch.equal(x,saved) and torch.equal(w,weights)


@pytest.mark.parametrize('before',[False,True])
def test_global_accumulation_flag_restored_on_error(before):
    old=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    try:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=before
        def original(*args):
            assert not torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
            raise ArithmeticError('native failure')
        with pytest.raises(ArithmeticError,match='native failure'):
            mod.padded_call(original,None,torch.zeros(3,2),None)
        assert torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction is before
    finally:torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=old


class TensorDescription:
    ndim=2
    is_cuda=True
    dtype=torch.bfloat16
    def __init__(self,m):self.shape=(m,8192)
    def is_contiguous(self):return True


@pytest.mark.parametrize('m',[0,1,2,3,10,11,12,13,14,15,16,17,32])
def test_actual_dispatch_boundary_without_a_gpu(monkeypatch,m):
    called=[]
    x=TensorDescription(m)
    def original(layer,value,bias):called.append('native');return value
    def padded(original,layer,value,bias):called.append('padded');return value
    monkeypatch.setattr(mod,'padded_call',padded)
    fn=mod.wrap_projection(original,threading.get_ident())
    assert fn(None,x) is x
    assert called==(['padded'] if m in (2,3,4,5,6,7,8,9,10,12,16) else ['native'])


def test_thread_guard_and_unsupported_tensor_fallback(monkeypatch):
    fn=mod.wrap_projection(lambda layer,x,bias:x,threading.get_ident())
    x=torch.zeros(3,2)  # CPU and wrong dtype/width: untouched native path.
    assert fn(None,x) is x
    fn=mod.wrap_projection(lambda *args:None,threading.get_ident()+1)
    with pytest.raises(RuntimeError,match='different thread'):fn(None,x)


def make_model(bad_index=None):
    method_class=type('UnquantizedLinearMethod',(),{'apply':lambda self,layer,x,bias=None:x})
    layers=[]
    for i in range(16):
        weight=NS(shape=(2048,8192),is_cuda=i!=bad_index,dtype=torch.bfloat16,is_contiguous=lambda:True)
        layers.append((f'model.layers.{i}.mlp.down_proj',NS(weight=weight,quant_method=method_class())))
    return type('LlamaForCausalLM',(),{'training':False,'named_modules':lambda self:iter(layers)})(),layers


def test_off_does_not_inspect_model(monkeypatch):
    monkeypatch.delenv('CDB_AV_DRAFT_DOWN_PAD32',raising=False)
    assert mod.configure(None,None)==0


def test_failed_full_validation_leaves_all_modules_untouched(monkeypatch):
    monkeypatch.setenv('CDB_AV_DRAFT_DOWN_PAD32','1')
    model,layers=make_model(15)
    methods=[x.quant_method.apply for _,x in layers]
    with pytest.raises(ValueError,match='projection storage'):mod.configure(model,config())
    assert all(layer.quant_method.apply==old for (_,layer),old in zip(layers,methods))
    assert all(not getattr(layer,'_cdb_down_pad32_installed',False) for _,layer in layers)


def test_installation_changes_only_methods_and_rejects_reinstallation(monkeypatch):
    monkeypatch.setenv('CDB_AV_DRAFT_DOWN_PAD32','1')
    model,layers=make_model();weights=[x.weight for _,x in layers]
    assert mod.configure(model,config())==16
    assert all(layer.weight is old for (_,layer),old in zip(layers,weights))
    with pytest.raises(ValueError):mod.configure(model,config())


def test_extra_native_audit_never_runs_inside_graph_capture(monkeypatch):
    audit=[0];x=TensorDescription(4)
    monkeypatch.setattr(mod,'padded_call',lambda original,layer,x,bias:x)
    monkeypatch.setattr(torch.cuda,'is_current_stream_capturing',lambda:True)
    def forbidden(*args):raise AssertionError('extra native forward inside capture')
    fn=mod.wrap_projection(forbidden,threading.get_ident(),audit)
    assert fn(None,x) is x and audit==[0]


def test_training_model_rejected_before_install(monkeypatch):
    monkeypatch.setenv('CDB_AV_DRAFT_DOWN_PAD32','1')
    model,layers=make_model();model.training=True
    with pytest.raises(ValueError,match='inference mode'):mod.configure(model,config())
    assert all(not getattr(layer,'_cdb_down_pad32_installed',False) for _,layer in layers)
