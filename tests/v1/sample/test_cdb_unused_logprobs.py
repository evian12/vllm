from dataclasses import dataclass
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from vllm.v1.sample import rejection_sampler as m

@dataclass
class Params:
    max_num_logprobs: int | None

class Bonus(nn.Module):
    def __init__(self,mode):
        super().__init__();self.logprobs_mode=mode;self.preserve=[]
    def forward(self,logits,sampling_metadata,**kwargs):
        self.preserve.append(sampling_metadata.max_num_logprobs)
        # Simulate in-place processing and independent raw/processed preservation.
        raw=logits.float().clone();logits=logits.float();logits.add_(torch.arange(logits.shape[-1]))
        kept=logits.clone() if kwargs['logprobs_mode_override']=='processed_logits' else raw
        return SimpleNamespace(sampled_token_ids=logits.argmax(-1).reshape(-1,1),
            logprobs_tensors=None if sampling_metadata.max_num_logprobs is None else SimpleNamespace(logprobs=kept))

@pytest.mark.parametrize('mode',['raw_logits','raw_logprobs','processed_logits','processed_logprobs'])
@pytest.mark.parametrize('requested',[None,-1,0,2])
@pytest.mark.parametrize('dtype',[torch.bfloat16,torch.float32])
def test_preservation_and_processed_probability_equivalence(monkeypatch,mode,requested,dtype):
    inputs=torch.tensor([[1.,2.,3.,4.],[2.,4.,0.,1.],[5.,1.,2.,3.]],dtype=dtype)
    params=Params(requested);meta=SimpleNamespace(max_spec_len=2,bonus_logits_indices=torch.tensor([2]),target_logits_indices=torch.tensor([0,1]),draft_token_ids=torch.tensor([1,2]),num_draft_tokens=[2],cu_num_draft_tokens=torch.tensor([2]))
    seen=[];captured=[]
    def reject(*args):
        seen.append((args[5].clone(),args[6].clone()))
        return torch.tensor([[1,2,3]])
    monkeypatch.setattr(m,'rejection_sample',reject)
    monkeypatch.setattr(m,'apply_sampling_constraints',lambda logits,*args:logits)
    results=[]
    for flag in ['0','1']:
        monkeypatch.setenv('CDB_AV_SKIP_UNUSED_LOGPROBS',flag)
        bonus=Bonus(mode);sampler=m.RejectionSampler(bonus)
        def processors(logits,*args):
            logits.add_(torch.arange(logits.shape[-1]))
            return logits
        sampler.apply_logits_processors=processors
        def logprobs(*args):
            captured.append((args[3].clone(),args[4].clone()))
            return 'requested'
        sampler._get_logprobs_tensors=logprobs
        results.append(sampler(meta,None,inputs.clone(),params))
        assert bonus.preserve==([None] if flag=='1' and requested is None else [-1])
    assert torch.equal(seen[0][0],seen[1][0])
    assert torch.equal(seen[0][1],seen[1][1])
    assert torch.equal(results[0].sampled_token_ids,results[1].sampled_token_ids)
    if requested is None:
        assert not captured and results[1].logprobs_tensors is None
    else:
        assert torch.equal(captured[0][0],captured[1][0])
        assert torch.equal(captured[0][1],captured[1][1])
        expected=inputs[:2].float()
        if mode.startswith('processed'):expected=expected+torch.arange(4)
        assert torch.equal(captured[1][0],expected)

@pytest.mark.parametrize('flag',['','2','true'])
def test_rejects_invalid_flag(monkeypatch,flag):
    monkeypatch.setenv('CDB_AV_SKIP_UNUSED_LOGPROBS',flag)
    with pytest.raises(ValueError):m.RejectionSampler(Bonus('raw_logits'))
