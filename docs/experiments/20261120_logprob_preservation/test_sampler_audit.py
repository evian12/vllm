from types import SimpleNamespace
import pytest
import torch
import sampler_audit

def patched():
    class Sampler:
        def __init__(self):self.skip_unused_logprobs=True;self.calls=[];self.fail=False
        def forward(self,metadata,draft_probs,logits,params):
            self.calls.append(self.skip_unused_logprobs)
            if self.fail:raise RuntimeError('reference failed')
            return SimpleNamespace(sampled_token_ids=torch.tensor([[1,2]]),logprobs_tensors=None)
    observer=SimpleNamespace(PATCHES={})
    sampler_audit.install(observer)
    observer.PATCHES['vllm.v1.sample.rejection_sampler'](SimpleNamespace(RejectionSampler=Sampler))
    return Sampler()

def test_random_dummy_and_logprob_requests_passthrough():
    s=patched()
    for params in [SimpleNamespace(all_greedy=False,max_num_logprobs=None),SimpleNamespace(all_greedy=True,max_num_logprobs=0)]:s.forward(None,None,None,params)
    assert s.calls==[True,True]

def test_same_input_references_are_bounded():
    s=patched();p=SimpleNamespace(all_greedy=True,max_num_logprobs=None)
    for _ in range(33):s.forward(None,None,None,p)
    assert s.calls==[False,True]*32+[True]
    assert s.skip_unused_logprobs

def test_reference_failure_restores_flag():
    s=patched();s.fail=True
    with pytest.raises(RuntimeError):s.forward(None,None,None,SimpleNamespace(all_greedy=True,max_num_logprobs=None))
    assert s.skip_unused_logprobs
