# SPDX-License-Identifier: Apache-2.0
import os
import threading
from types import SimpleNamespace as NS

import pytest
import torch

from vllm.v1.sample.cdb_av_verify_graph import GreedyVerifyGraph, verify_graph_key


def fake_scope(counts=(2,1,3), vocab=128256):
    dev=torch.device('cuda:0')
    def tensor(shape,dtype=torch.int32):
        return NS(device=dev,dtype=dtype,shape=shape,ndim=len(shape),is_contiguous=lambda:True)
    n=sum(counts);r=len(counts)
    m=NS(num_draft_tokens=list(counts),max_spec_len=max(counts),
        draft_token_ids=tensor((n,)),cu_num_draft_tokens=tensor((r,)),
        cu_num_sampled_tokens=tensor((r,)),target_logits_indices=tensor((n,)),
        bonus_logits_indices=tensor((r,)),logits_indices=tensor((n+r,)))
    logits=tensor((n+r,vocab),torch.float32)
    s=NS(all_greedy=True,all_random=False,max_num_logprobs=None,no_penalties=True,
         bad_words_token_ids={},allowed_token_ids_mask=None,
         logitsprocs=NS(non_argmax_invariant=[]))
    return m,logits,s


@pytest.mark.parametrize('dtype,eligible',[(torch.bfloat16,True),(torch.float32,True),
                                          (torch.float16,False),(torch.float64,False)])
def test_raw_lm_head_dtype_scope_and_cache_key(dtype,eligible):
    m,l,s=fake_scope();fp32=verify_graph_key(m,None,l,s);l.dtype=dtype
    key=verify_graph_key(m,None,l,s)
    assert (key is not None)==eligible
    if dtype==torch.bfloat16:
        assert key!=fp32 and key.dtype==torch.bfloat16


def test_key_uses_shapes_not_request_partition_values():
    m,l,s=fake_scope();a=verify_graph_key(m,None,l,s)
    assert a is not None
    m.num_draft_tokens=[3,2,1]
    assert verify_graph_key(m,None,l,s)==a
    m.num_draft_tokens=[0,3,3]
    assert verify_graph_key(m,None,l,s)==a
    m.num_draft_tokens=[2,2,2];m.max_spec_len=2
    assert verify_graph_key(m,None,l,s)!=a


@pytest.mark.parametrize('field,value',[
    ('all_greedy',False),('all_random',True),('max_num_logprobs',0),
    ('no_penalties',False),('bad_words_token_ids',{0:[[2]]}),
    ('allowed_token_ids_mask',object()),
])
def test_unsupported_sampling_flags_fall_back(field,value):
    m,l,s=fake_scope();setattr(s,field,value)
    assert verify_graph_key(m,None,l,s) is None


def test_custom_processor_draft_probs_cpu_and_wrong_shape_fall_back():
    m,l,s=fake_scope();s.logitsprocs.non_argmax_invariant=[object()]
    assert verify_graph_key(m,None,l,s) is None
    s.logitsprocs.non_argmax_invariant=[]
    assert verify_graph_key(m,object(),l,s) is None
    l.device=torch.device('cpu');assert verify_graph_key(m,None,l,s) is None
    l.device=torch.device('cuda:0');m.bonus_logits_indices.shape=(4,)
    assert verify_graph_key(m,None,l,s) is None


@pytest.mark.parametrize('counts',[[],[9],[1,1,1,1,1],[0,0],[1.0],[True],[-1,2]])
def test_unsupported_draft_counts_fall_back(counts):
    m,l,s=fake_scope();m.num_draft_tokens=counts
    assert verify_graph_key(m,None,l,s) is None


def test_fallback_preserves_identity_and_does_not_touch_cuda(monkeypatch):
    m,l,s=fake_scope();s.max_num_logprobs=0
    marker=object();calls=[]
    class Sampler:
        sampler=NS(logprobs_mode='raw_logprobs')
        is_processed_logprobs_mode=False;is_logits_logprobs_mode=False
        def __call__(self,*a):calls.append(a);return marker
    monkeypatch.setattr(torch.cuda,'is_current_stream_capturing',
                        lambda:pytest.fail('fallback touched CUDA'))
    wrapper=GreedyVerifyGraph(Sampler())
    assert wrapper(m,None,l,s) is marker and calls==[(m,None,l,s)]
    assert wrapper(m,None,None,s) is marker
    calls.pop()
    s.max_num_logprobs=None;wrapper.sampler.sampler.logprobs_mode='processed_logits'
    assert wrapper(m,None,l,s) is marker and len(calls)==2


def test_thread_nested_capture_and_other_stream_use_original(monkeypatch):
    m,l,s=fake_scope();marker=object();calls=[]
    class Sampler:
        sampler=NS(logprobs_mode='raw_logprobs')
        is_processed_logprobs_mode=False;is_logits_logprobs_mode=False
        def __call__(self,*a):calls.append(a);return marker
    wrapper=GreedyVerifyGraph(Sampler())
    monkeypatch.setattr(torch.cuda,'is_current_stream_capturing',
                        lambda:pytest.fail('foreign thread touched CUDA'))
    returned=[]
    thread=threading.Thread(target=lambda:returned.append(wrapper(m,None,l,s)))
    thread.start();thread.join()
    assert returned==[marker] and not wrapper.cache
    monkeypatch.setattr(torch.cuda,'is_current_stream_capturing',lambda:True)
    assert wrapper(m,None,l,s) is marker
    monkeypatch.setattr(torch.cuda,'is_current_stream_capturing',lambda:False)
    wrapper.stream=object()
    monkeypatch.setattr(torch.cuda,'current_stream',lambda device:object())
    assert wrapper(m,None,l,s) is marker
    assert len(calls)==3 and not wrapper.cache


def test_warmup_and_memory_and_full_cache_fallbacks(monkeypatch):
    m,l,s=fake_scope();l.numel=lambda:l.shape[0]*l.shape[1]
    marker=object()
    class Sampler:
        sampler=NS(logprobs_mode='raw_logprobs')
        is_processed_logprobs_mode=False;is_logits_logprobs_mode=False
        def __call__(self,*a):return marker
    stream=object()
    monkeypatch.setattr(torch.cuda,'is_current_stream_capturing',lambda:False)
    monkeypatch.setattr(torch.cuda,'current_stream',lambda device:stream)
    warm=GreedyVerifyGraph(Sampler())
    for _ in range(7):assert warm(m,None,l,s) is marker
    assert not warm.cache
    wrapper=GreedyVerifyGraph(Sampler(),max_bytes=1)
    for _ in range(12):assert wrapper(m,None,l,s) is marker
    assert not wrapper.cache and wrapper.estimated_bytes==0
    full=GreedyVerifyGraph(Sampler(),min_hits=1,max_keys=1)
    full.cache[object()]=object()
    assert full(m,None,l,s) is marker and len(full.cache)==1
    bounded=GreedyVerifyGraph(Sampler(),min_hits=1000000)
    for vocab in range(1,261):
        l.shape=(l.shape[0],vocab)
        assert bounded(m,None,l,s) is marker
    assert len(bounded.hits)==256 and not bounded.cache


def test_sampler_replacement_preserves_eager_semantics(monkeypatch):
    m,l,s=fake_scope();first=object();second=object()
    class Sampler:
        is_processed_logprobs_mode=False;is_logits_logprobs_mode=False
        def __init__(self,marker):self.marker=marker;self.sampler=NS(logprobs_mode='raw_logprobs')
        def __call__(self,*a):return self.marker
    original=Sampler(first);wrapper=GreedyVerifyGraph(original)
    monkeypatch.setattr(torch.cuda,'is_current_stream_capturing',
                        lambda:pytest.fail('replacement reached old Graph path'))
    wrapper.sampler=Sampler(second)
    assert wrapper(m,None,l,s) is second
    wrapper.sampler=original;original.sampler=NS(logprobs_mode='raw_logprobs')
    assert wrapper(m,None,l,s) is first


def test_python_forward_hooks_are_preserved_by_eager_fallback(monkeypatch):
    m,l,s=fake_scope();marker=object();seen=[]
    class Sampler(torch.nn.Module):
        is_processed_logprobs_mode=False;is_logits_logprobs_mode=False
        def __init__(self):
            super().__init__();self.sampler=NS(logprobs_mode='raw_logprobs')
        def forward(self,*a):return marker
    sampler=Sampler();wrapper=GreedyVerifyGraph(sampler)
    monkeypatch.setattr(torch.cuda,'is_current_stream_capturing',
                        lambda:pytest.fail('hook reached Graph path'))
    hook=sampler.register_forward_hook(lambda *a:seen.append('local'))
    assert wrapper(m,None,l,s) is marker
    hook.remove()
    hook=torch.nn.modules.module.register_module_forward_hook(lambda *a:seen.append('global'))
    try:assert wrapper(m,None,l,s) is marker
    finally:hook.remove()
    assert seen==['local','global']


def test_real_inactive_builtins_and_custom_subclasses():
    from vllm.v1.sample.logits_processor.builtin import (
        MinTokensLogitsProcessor,LogitBiasLogitsProcessor)
    m,l,s=fake_scope()
    minimum=MinTokensLogitsProcessor.__new__(MinTokensLogitsProcessor)
    minimum.min_toks={}
    bias=LogitBiasLogitsProcessor.__new__(LogitBiasLogitsProcessor)
    bias.biases={}
    s.logitsprocs.non_argmax_invariant=[minimum,bias]
    assert verify_graph_key(m,None,l,s) is not None
    # Validate the actual built-in no-op contract on CPU, without CUDA init.
    x=torch.ones(2,3)
    assert minimum.apply(x) is x and bias.apply(x) is x
    minimum.min_toks[0]=(3,[],{1})
    assert verify_graph_key(m,None,l,s) is None
    minimum.min_toks.clear();bias.biases[0]={1:2.0}
    assert verify_graph_key(m,None,l,s) is None
    class Custom(LogitBiasLogitsProcessor):
        pass
    custom=Custom.__new__(Custom);custom.biases={}
    s.logitsprocs.non_argmax_invariant=[custom]
    assert verify_graph_key(m,None,l,s) is None


GPU=pytest.mark.skipif(os.getenv('CDB_VERIFY_GPU_TEST')!='1',
                       reason='requires explicit isolated GPU3 lock')


def gpu_inputs(counts,vocab=256,dtype=torch.float32):
    from vllm.v1.sample.metadata import SamplingMetadata
    from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
    r=len(counts);n=sum(counts);cd=[];cs=[];targets=[];bonus=[];d=s=0
    for k in counts:
        targets.extend(range(s,s+k));bonus.append(s+k);d+=k;s+=k+1
        cd.append(d);cs.append(s)
    def t(x):return torch.tensor(x,dtype=torch.int32,device='cuda')
    m=SpecDecodeMetadata(draft_token_ids=t([i%vocab for i in range(n)]),
        num_draft_tokens=list(counts),cu_num_draft_tokens=t(cd),
        cu_num_sampled_tokens=t(cs),target_logits_indices=t(targets),
        bonus_logits_indices=t(bonus),logits_indices=t(list(range(n+r))))
    sampling=SamplingMetadata(temperature=None,all_greedy=True,all_random=False,
        top_p=None,top_k=None,generators={},max_num_logprobs=None,no_penalties=True,
        prompt_token_ids=None,frequency_penalties=torch.empty(0),
        presence_penalties=torch.empty(0),repetition_penalties=torch.empty(0),
        output_token_ids=[[] for _ in counts],allowed_token_ids_mask=None,
        bad_words_token_ids={},logitsprocs=NS(non_argmax_invariant=[],argmax_invariant=[]))
    return m,torch.randn(n+r,vocab,device='cuda',dtype=dtype),sampling


@GPU
@pytest.mark.parametrize('dtype',[torch.bfloat16,torch.float32])
def test_dynamic_partition_output_lifetime_and_exact_near_ties(dtype):
    assert os.environ['CUDA_VISIBLE_DEVICES']=='3'
    from vllm.v1.sample.sampler import Sampler
    from vllm.v1.sample.rejection_sampler import RejectionSampler
    sampler=RejectionSampler(Sampler());wrapper=GreedyVerifyGraph(sampler,min_hits=1)
    saved=[]
    for i in range(36):
        counts=[[2,1,3],[3,2,1],[1,3,2],[0,3,3],[3,0,3],[3,3,0]][i%6]
        m,logits,s=gpu_inputs(counts,dtype=dtype)
        if i%4==0:
            # Preserve S01's exact FP32 softmax rounding behavior.
            logits.fill_(-10);logits[:,0]=0;logits[:,1]=1e-8
        elif i%4==1:logits[0].fill_(float('-inf'))
        elif i%4==2:logits[0,2]=float('nan')
        before=logits.clone()
        expected=sampler(m,None,logits,s).sampled_token_ids
        actual=wrapper(m,None,logits,s).sampled_token_ids
        assert torch.equal(expected,actual)
        assert torch.equal(logits.view(torch.uint8),before.view(torch.uint8))
        saved.append((actual,expected.clone()))
    assert len(wrapper.cache)==1
    assert all(torch.equal(a,b) for a,b in saved)
    assert len({a.data_ptr() for a,b in saved})==len(saved)


@GPU
@pytest.mark.parametrize('dtype',[torch.bfloat16,torch.float32])
def test_real_vocab_graph_cache_and_memory_bound(dtype):
    from vllm.v1.sample.sampler import Sampler
    from vllm.v1.sample.rejection_sampler import RejectionSampler
    sampler=RejectionSampler(Sampler());wrapper=GreedyVerifyGraph(sampler,min_hits=1,max_keys=1)
    for counts in [[1,1,1,1],[1,1,1,1],[2,2,2,2],[1,1,1,1]]:
        m,l,s=gpu_inputs(counts,128256,dtype=dtype)
        assert torch.equal(sampler(m,None,l,s).sampled_token_ids,
                           wrapper(m,None,l,s).sampled_token_ids)
    assert len(wrapper.cache)==1 and wrapper.estimated_bytes<=wrapper.max_bytes
    limited=GreedyVerifyGraph(sampler,min_hits=1,max_bytes=1)
    assert torch.equal(limited(m,None,l,s).sampled_token_ids,
                       sampler(m,None,l,s).sampled_token_ids)
    assert not limited.cache
