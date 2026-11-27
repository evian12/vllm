"""Exercise production Target Graph shape and replay-metadata checks on CPU."""
from dataclasses import dataclass, replace
from itertools import accumulate
import random
from types import SimpleNamespace as S

import torch
import pytest
from vllm.v1.spec_decode import cdb_av_target_graph as graph


@dataclass
class Metadata:
    num_actual_tokens: int
    max_query_len: int
    max_seq_len: int
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    max_num_splits: int = 0


def make(qlens, seqs, salt=0):
    n=sum(qlens)
    starts=[0,*accumulate(qlens)]
    width=(max(seqs)+15)//16
    rows=len(qlens)
    return Metadata(n,max(qlens),max(seqs),
        torch.tensor(starts,dtype=torch.int32),
        torch.tensor(seqs,dtype=torch.int32),
        torch.arange(salt,salt+rows*width,dtype=torch.int32).reshape(rows,width),
        torch.arange(salt,salt+n,dtype=torch.int64))


def key(meta, cap=4096):
    return graph.graph_key(meta, meta.query_start_loc.tolist(),
        n=meta.num_actual_tokens,capacity=4,max_query=9,max_model_len=cap,
        dynamic=True,seq_floor=1024)


def reject(func):
    try:
        func()
    except (ValueError, RuntimeError):
        return
    raise AssertionError('unsafe graph metadata or runner was accepted')


def check_shape_guards():
    for limit, bucket in ((1024,1024),(1025,2048),(2048,2048),(2049,4096),(4096,4096)):
        regular=make([1],[limit])
        speculative=make([2,9],[limit,limit-16])
        assert key(regular)[2:4]==(1,bucket)
        assert key(speculative)[2:4]==(9,bucket)
        if limit>1024:
            assert key(speculative,1024) is None
    for qlens, seqs in (([10],[1025]),([1],[4097]),([1,1,1,1,1],[2048]*5)):
        assert key(make(qlens,seqs)) is None

    rng=random.Random(42)
    fields=('query_start_loc','seq_lens','block_table','slot_mapping')
    for _ in range(256):
        qlens=[rng.randrange(1,10) for _ in range(rng.randrange(1,5))]
        seqs=[rng.randrange(1025,4097) for _ in qlens]
        original=make(qlens,seqs)
        live=make(qlens,seqs,salt=16)
        shape=key(original)
        static=graph.clone_metadata(original,shape[3],query_bound=shape[2])
        ptrs={field:getattr(static,field).data_ptr() for field in fields}
        graph.update_metadata(static,live,mixed_bound=9)
        for field in fields:
            tgt=getattr(static,field)
            assert tgt.data_ptr()==ptrs[field] and torch.equal(tgt,getattr(live,field))
        reject(lambda: graph.update_metadata(static,replace(live,max_seq_len=shape[3]+1),mixed_bound=9))
        reject(lambda: graph.update_metadata(static,replace(live,max_num_splits=1),mixed_bound=9))
        reject(lambda: graph.update_metadata(static,replace(live,block_table=live.block_table[:,:-1]),mixed_bound=9))


def runner(max_len, fa=2, capacity=4):
    class Qwen3ForCausalLM:
        training=False
        def modules(self):
            return [self,layer]
    layer=S(kv_cache=(),impl=S(vllm_flash_attn_version=fa))
    config=S(parallel_config=S(tensor_parallel_size=1,pipeline_parallel_size=1,
          data_parallel_size=1,decode_context_parallel_size=1),model_config=S(
          enforce_eager=True,dtype=torch.bfloat16,quantization=None,max_model_len=max_len,
          hf_config=S(num_hidden_layers=1)),
          cache_config=S(enable_prefix_caching=False),lora_config=None,
          scheduler_config=S(async_scheduling=False,max_num_seqs=capacity),
          speculative_config=S(num_speculative_tokens=8),
          compilation_config=S(static_forward_context={'layer':layer}))
    return S(cdb_persistent=True,cdb_refill=True,vllm_config=config,model=Qwen3ForCausalLM())


@pytest.mark.parametrize('value,expected', [('0', 1024), ('1', 4096)])
def test_context_flag_is_explicit(value, expected):
    assert graph.target_graph_context_limit({'CDB_AV_TARGET_GRAPH_LONG_CONTEXT': value}) == expected


@pytest.mark.parametrize('value', ['', 'true', '2', 'false'])
def test_context_flag_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        graph.target_graph_context_limit({'CDB_AV_TARGET_GRAPH_LONG_CONTEXT': value})


def test_default_domain_and_opt_in_constructor(monkeypatch):
    monkeypatch.delenv('CDB_AV_TARGET_GRAPH_LONG_CONTEXT', raising=False)
    monkeypatch.setenv('CDB_AV_TARGET_GRAPH_SEQ_FLOOR', '0')
    assert graph.target_graph_context_limit({}) == 1024
    graph.TargetFullGraph(runner(1024))
    with pytest.raises(ValueError):
        graph.TargetFullGraph(runner(4096))
    monkeypatch.setenv('CDB_AV_TARGET_GRAPH_LONG_CONTEXT', '1')
    for limit in (1025, 2048, 4096):
        obj = graph.TargetFullGraph(runner(limit))
        assert obj.max_model_len == limit and obj.context_limit == 4096
    for bad in (runner(0), runner(4097), runner(4096, fa=3), runner(4096, capacity=5)):
        with pytest.raises(ValueError):
            graph.TargetFullGraph(bad)


def test_real_long_metadata_copies_and_reject_boundaries():
    check_shape_guards()
