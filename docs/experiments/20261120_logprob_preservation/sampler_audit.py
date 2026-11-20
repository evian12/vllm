"""Diagnostic same-input comparison, greedy-only, no performance use."""
import json,logging
from functools import wraps

def install(observer):
    def patch(module):
        import torch
        old=module.RejectionSampler.forward
        counter=0
        @wraps(old)
        def forward(self,metadata,draft_probs,logits,sampling_metadata):
            nonlocal counter
            if (counter>=32 or not sampling_metadata.all_greedy
                    or sampling_metadata.max_num_logprobs is not None):
                return old(self,metadata,draft_probs,logits,sampling_metadata)
            assert self.skip_unused_logprobs
            # All indexing paths make private copies before in-place processing.
            try:
                self.skip_unused_logprobs=False
                ref=old(self,metadata,draft_probs,logits,sampling_metadata)
            finally:self.skip_unused_logprobs=True
            got=old(self,metadata,draft_probs,logits,sampling_metadata)
            exact=torch.equal(ref.sampled_token_ids,got.sampled_token_ids)
            assert exact and ref.logprobs_tensors is None and got.logprobs_tensors is None
            counter+=1
            logging.getLogger(__name__).info('[CDB_LOGPROB_SAVE_VALIDATE] %s',json.dumps(dict(sample=counter,token_exact=exact,rows=got.sampled_token_ids.shape[0])))
            return got
        module.RejectionSampler.forward=forward
    observer.PATCHES['vllm.v1.sample.rejection_sampler']=patch
