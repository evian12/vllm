# SPDX-License-Identifier: Apache-2.0
"""CPU-only schema/storage tests; model correctness requires the GPU gate."""
from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.spec_decode.cdb_av_target_graph import (
    validate_scope,
    graph_key,
    clone_metadata,
    update_metadata,
    metadata_has_no_planner,
    query_domain_matches,
    validate_sequence_floor,
    target_graph_timing_config,
)


@dataclass
class Meta:
    num_actual_tokens: int
    max_query_len: int
    max_seq_len: int
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    max_num_splits: int = 0


def metadata(qlens=(1, 2, 3)):
    starts = [0]
    for q in qlens:
        starts.append(starts[-1] + q)
    nr = len(qlens)
    return Meta(starts[-1], max(qlens), 61,
        torch.tensor(starts,dtype=torch.int32),torch.full((nr,),61,dtype=torch.int32),
        torch.ones((nr,4),dtype=torch.int32),torch.arange(starts[-1],dtype=torch.int64))


def key(m):
    return graph_key(m,m.query_start_loc.tolist(),n=m.num_actual_tokens,
        capacity=4,max_query=9,max_model_len=1024)


def test_varlen_same_maximum_layouts_share_storage_domain():
    assert key(metadata((1,2,3))) == key(metadata((2,1,3)))


def test_fa2_packed_gqa_and_other_maxima_are_separate():
    assert key(metadata((1,1,1))) != key(metadata((1,2)))
    assert key(metadata((2,2,2))) != key(metadata((1,1,4)))


@pytest.mark.parametrize('changes', [
    {'num_actual_tokens':7}, {'max_query_len':2}, {'max_seq_len':0},
    {'max_seq_len':1025}, {'block_table':torch.ones((3,0),dtype=torch.int32)},
    {'block_table':torch.ones((2,4),dtype=torch.int32)},
    {'query_start_loc':torch.tensor([0,1,1,6],dtype=torch.int32)},
    {'query_start_loc':torch.tensor([0,1,3,7],dtype=torch.int32)},
    {'query_start_loc':torch.tensor([0,1,3,6],dtype=torch.int64)},
    {'slot_mapping':torch.zeros(5,dtype=torch.int64)},
])
def test_invalid_metadata_falls_back(changes):
    assert key(replace(metadata(),**changes)) is None


def test_capacity_and_long_query_fall_back():
    assert key(metadata((1,1,1,1,1))) is None
    assert key(metadata((10,))) is None


def test_current_slot_and_row_mapping_copied_without_buffer_replacement():
    live = metadata()
    static = clone_metadata(live,64)
    fields = ('query_start_loc','seq_lens','block_table','slot_mapping')
    pointers = {f:getattr(static,f).data_ptr() for f in fields}
    other = metadata((2,1,3))
    other.seq_lens.fill_(63)
    other.block_table.fill_(17)
    other.slot_mapping.add_(100)
    update_metadata(static,other)
    for f in fields:
        assert getattr(static,f).data_ptr() == pointers[f]
        assert torch.equal(getattr(static,f),getattr(other,f))
        assert getattr(static,f).data_ptr() != getattr(other,f).data_ptr()
    assert torch.equal(live.query_start_loc,torch.tensor([0,1,3,6],dtype=torch.int32))
    assert static.max_seq_len == 64


@pytest.mark.parametrize('changes', [
    {'max_query_len':4}, {'max_seq_len':65}, {'num_actual_tokens':7},
    {'slot_mapping':torch.zeros(5,dtype=torch.int64)},
])
def test_replay_rejects_scalar_or_storage_mismatch(changes):
    live = metadata()
    with pytest.raises(RuntimeError,match='Target Graph metadata'):
        update_metadata(clone_metadata(live,64),replace(live,**changes))


@pytest.mark.parametrize('changes', [
    {}, {'persistent':False}, {'refill':False}, {'eager':False},
    {'async_scheduling':True}, {'prefix_caching':True}, {'lora':True},
    {'model':'OtherModel'}, {'dtype':torch.float16}, {'quantization':'fp8'},
    {'parallel_sizes':(2,1,1,1)}, {'parallel_sizes':(1,1,1,2)},
])
def test_supported_scope_is_explicit(changes):
    args=dict(persistent=True,refill=True,eager=True,async_scheduling=False,
        prefix_caching=False,lora=False,model='Qwen3ForCausalLM',dtype=torch.bfloat16,
        quantization=None,parallel_sizes=(1,1,1,1))
    args.update(changes)
    if changes:
        with pytest.raises(ValueError,match='Target full Graph'):
            validate_scope(**args)
    else:
        validate_scope(**args)


@pytest.mark.parametrize('changes,expected', [
    ({}, True), ({'max_dcp_context_kv_len':None}, True),
    ({'max_dcp_context_kv_len':1}, False),
    ({'dcp_context_kv_lens':torch.tensor([1])}, False),
    ({'scheduler_metadata':torch.tensor([0])}, False),
    ({'use_cascade':True}, False), ({'common_prefix_len':1},False),
    ({'causal':False},False),
])
def test_inactive_dcp_zero_is_supported_but_planner_state_is_not(changes, expected):
    fields=dict(max_dcp_context_kv_len=0,dcp_context_kv_lens=None,
        scheduler_metadata=None,prefix_scheduler_metadata=None,
        cu_prefix_query_lens=None,prefix_kv_lens=None,suffix_kv_lens=None,
        use_cascade=False,common_prefix_len=0,causal=True)
    fields.update(changes)
    assert metadata_has_no_planner(SimpleNamespace(**fields)) is expected


def test_model_replacement_recreates_graph_owner(monkeypatch):
    # Exercise the actual runner hook without constructing a model or CUDA state.
    import ast
    import os
    from pathlib import Path
    import vllm.v1.spec_decode.cdb_av_target_graph as module
    path = Path(__file__).parents[3] / 'vllm/v1/worker/gpu_model_runner.py'
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='GPUModelRunner')
    method = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_model_forward')
    text = 'from __future__ import annotations\n' + ast.unparse(method)
    namespace = dict(os=os)
    exec(compile(text,str(path),'exec'),namespace)
    created = []
    class Cache:
        def __init__(self,r):
            import weakref
            self.model_ref=weakref.ref(r.model)
            created.append(r.model)
        def forward(self,*args):return 'graph'
    class Model:
        def __call__(self,**kwargs):return 'eager'
    monkeypatch.setattr(module,'TargetFullGraph',Cache)
    runner=SimpleNamespace(model=Model())
    forward=namespace['_model_forward']
    monkeypatch.setenv('CDB_AV_TARGET_FULL_GRAPH','1')
    assert forward(runner)=='graph'
    assert forward(runner)=='graph'
    assert len(created)==1
    runner.model=Model()
    assert forward(runner)=='graph'
    assert len(created)==2
    monkeypatch.setenv('CDB_AV_TARGET_FULL_GRAPH','0')
    assert forward(runner)=='eager'
    assert len(created)==2


def test_dynamic_mixed_domain_shares_key_across_actual_maxima():
    a,b=metadata((3,3,3,3)),metadata((1,1,1,9))
    def dynamic_key(m):
        return graph_key(m,m.query_start_loc.tolist(),n=m.num_actual_tokens,
            capacity=4,max_query=9,max_model_len=1024,dynamic=True)
    assert key(a)!=key(b)
    assert dynamic_key(a)==dynamic_key(b)
    assert dynamic_key(a)[2]==9
    assert dynamic_key(metadata((1,1,1,1)))[2]==1
    static=clone_metadata(a,64,query_bound=9)
    assert a.max_query_len==3 and static.max_query_len==9
    update_metadata(static,b,mixed_bound=9)
    assert torch.equal(static.query_start_loc,b.query_start_loc)
    assert static.max_query_len==9


@pytest.mark.parametrize('captured,actual,bound,expected', [
    (1,1,9,True),(1,2,9,False),(9,1,9,False),
    (9,2,9,True),(9,9,9,True),(9,10,9,False),
    (9,2,None,False),(2,2,None,True),(3,2,None,False),(8,2,9,False),
])
def test_bounded_query_domain_preserves_packed_gqa_boundary(captured,actual,bound,expected):
    assert query_domain_matches(captured,actual,bound) is expected


@pytest.mark.parametrize('floor', [-1, 3, 255, 2048, True, 256.0])
def test_sequence_floor_rejects_invalid_config(floor):
    with pytest.raises(ValueError, match='Target Graph sequence floor'):
        validate_sequence_floor(floor,1024)


@pytest.mark.parametrize('length,bucket', [(1,256),(61,256),(128,256),
                                        (256,256),(257,512),(1024,1024)])
def test_sequence_floor_is_only_a_launch_bound(length,bucket):
    live=metadata((1,1,1) if length==1 else (1,2,3))
    live.max_seq_len=length
    live.seq_lens.fill_(length)
    before={f:getattr(live,f).clone() for f in
            ('seq_lens','query_start_loc','slot_mapping','block_table')}
    bounded=graph_key(live,live.query_start_loc.tolist(),n=live.num_actual_tokens,capacity=4,
        max_query=9,max_model_len=1024,dynamic=True,seq_floor=256)
    assert bounded[3]==bucket
    for field,value in before.items():
        assert torch.equal(getattr(live,field),value)
    static=clone_metadata(live,bucket,query_bound=bounded[2])
    update_metadata(static,live,mixed_bound=9)
    assert static.max_seq_len==bucket
    assert torch.equal(static.seq_lens,live.seq_lens)


def test_sequence_floor_shares_keys_without_changing_default():
    a,b=metadata(),metadata()
    b.max_seq_len=129
    b.seq_lens.fill_(129)
    def get(m,floor):
        return graph_key(m,m.query_start_loc.tolist(),n=6,capacity=4,
            max_query=9,max_model_len=1024,dynamic=True,seq_floor=floor)
    assert get(a,0)!=get(b,0)
    assert get(a,256)==get(b,256)
    assert get(a,0)==graph_key(a,a.query_start_loc.tolist(),n=6,capacity=4,
        max_query=9,max_model_len=1024,dynamic=True)
    assert graph_key(a,a.query_start_loc.tolist(),n=6,capacity=4,max_query=9,
        max_model_len=1000,seq_floor=256)[3]==256
    too_long=metadata();too_long.max_seq_len=999
    assert graph_key(too_long,too_long.query_start_loc.tolist(),n=6,capacity=4,
        max_query=9,max_model_len=1000,seq_floor=256) is None


def test_target_graph_timing_is_disabled_by_default():
    assert target_graph_timing_config({}) == (False, 5)


def test_target_graph_timing_requires_validation():
    with pytest.raises(ValueError, match='requires CDB_AV_TARGET_GRAPH_VALIDATE=1'):
        target_graph_timing_config({'CDB_AV_TARGET_GRAPH_TIMING': '1'})


@pytest.mark.parametrize('limit', ['1', '5', '32'])
def test_target_graph_timing_accepts_bounded_sample_limit(limit):
    assert target_graph_timing_config({
        'CDB_AV_TARGET_GRAPH_TIMING': '1',
        'CDB_AV_TARGET_GRAPH_VALIDATE': '1',
        'CDB_AV_TARGET_GRAPH_TIMING_MAX_SAMPLES': limit,
    }) == (True, int(limit))


@pytest.mark.parametrize('limit', ['0', '-1', '33', 'many'])
def test_target_graph_timing_rejects_invalid_sample_limit(limit):
    with pytest.raises(ValueError, match='timing sample limit'):
        target_graph_timing_config({
            'CDB_AV_TARGET_GRAPH_TIMING': '1',
            'CDB_AV_TARGET_GRAPH_VALIDATE': '1',
            'CDB_AV_TARGET_GRAPH_TIMING_MAX_SAMPLES': limit,
        })


def test_target_graph_timing_keeps_correctness_validation_unbounded_when_disabled():
    assert target_graph_timing_config({
        'CDB_AV_TARGET_GRAPH_TIMING': '0',
        'CDB_AV_TARGET_GRAPH_VALIDATE': '1',
    }) == (False, 5)


def test_target_graph_timing_ignores_sample_limit_when_disabled():
    assert target_graph_timing_config({
        'CDB_AV_TARGET_GRAPH_TIMING': '0',
        'CDB_AV_TARGET_GRAPH_TIMING_MAX_SAMPLES': 'not-used',
    }) == (False, 5)
