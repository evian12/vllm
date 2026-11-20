# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import torch

from vllm.v1.spec_decode.cdb_av_graph_inputs import GraphInputStage


@pytest.mark.parametrize('shape',[(0,1,1),(5,5,1),(1,33,1),(4,3,1),(1,1,65)])
def test_shape_rejected_before_cuda(shape):
    with pytest.raises(ValueError):
        GraphInputStage('cuda', requests=shape[0], tokens=shape[1], width=shape[2])


def test_cpu_device_rejected():
    with pytest.raises(ValueError):
        GraphInputStage('cpu', requests=1, tokens=1, width=1)


GPU = pytest.mark.skipif(os.getenv('CDB_PINNED_GPU_TEST') != '1',
                        reason='explicit isolated GPU3 reservation required')


@GPU
def test_graph_owned_views_dynamic_layout_and_delayed_dma():
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '3'
    stage = GraphInputStage('cuda:0', requests=2, tokens=4, width=4)
    pointers = {name: t.data_ptr() for name, t in stage.fields.items()}
    args = dict(ids=[10,11,12,13],positions=[0,1,2,3],blocks=[[2],[3,4]],
        seqs=[1,3],qlens=[1,3],slots=[32,48,49,50],last_indices=[0,3])
    fields=stage.stage(**args)
    # Capture a consumer with stable pointers, then verify 100 changing layouts
    # and contents through real replay, including a delayed prior GPU consumer.
    output=torch.empty_like(fields['ids'])
    torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output.copy_(fields['ids'])
    for i in range(100):
        q=[1,3] if i%2 else [2,2]
        data=dict(args,ids=[i,i+1,i+2,i+3],qlens=q,seqs=q,
                  last_indices=[q[0]-1,3])
        if i==10: torch.cuda._sleep(10_000_000)
        current=stage.stage(**data)
        graph.replay()
        assert output.cpu().tolist()==data['ids']
        assert current['query_start_loc'].cpu().tolist()==[0,q[0],4]
        assert current['query_start_loc_cpu'].tolist()==[0,q[0],4]
        assert current['seq_lens'].cpu().tolist()==q
        assert current['positions'].cpu().tolist()==data['positions']
        assert current['slot_mapping'].cpu().tolist()==data['slots']
        assert current['last_indices'].cpu().tolist()==data['last_indices']
        assert current['block_table_tensor'].cpu().tolist()==[[2,0,0,0],[3,4,0,0]]
        assert {name:t.data_ptr() for name,t in current.items()}==pointers
    with torch.cuda.stream(torch.cuda.Stream()):
        with pytest.raises(RuntimeError,match='owning stream'):stage.stage(**args)
    with pytest.raises(ValueError):stage.stage(**dict(args,qlens=[1,2]))


@GPU
def test_same_stream_reuse_preserves_earlier_consumers_without_host_wait():
    stage=GraphInputStage('cuda:0',requests=1,tokens=1,width=1)
    output=[]
    for i in range(50):
        fields=stage.stage([i],[0],[[1]],[1],[1],[16],[0])
        output.append(fields['ids'].clone())
    assert [t.item() for t in output]==list(range(50))


def test_cached_input_stage_uses_only_live_exact_graph_entry():
    from types import SimpleNamespace as NS
    from vllm.v1.spec_decode.cdb_av_refill import RefillSession
    s=RefillSession.__new__(RefillSession)
    s.graph_input_stage=True;s.block_size=16
    storage=(123,456);s._decode_graph_storage=lambda:storage
    calls=[]
    def stage(*args):
        calls.append(args)
        return persistent
    persistent={'ids':object(),'positions':object(),'last_indices':object()}
    stager=NS(stage=stage,calls=2)
    s.p=NS(_cdb_decode_graph_storage=storage,
           _cdb_decode_graphs={(2,32):{'input_stage':stager}})
    args=([9,10],[1,17],[[1],[2,3]],[2,18],[1,1],[17,49],[0,1])
    fields=s._cached_graph_inputs(*args)
    assert calls==[args] and fields==persistent and fields is not persistent
    fields.pop('ids');assert 'ids' in persistent
    s.p._cdb_decode_graph_storage=(999,)
    assert s._cached_graph_inputs(*args) is None
    s.p._cdb_decode_graph_storage=storage
    changed=list(args);changed[4]=[1,2]
    assert s._cached_graph_inputs(*changed) is None
    changed=list(args);changed[3]=[2,35]
    assert s._cached_graph_inputs(*changed) is None
    s.graph_input_stage=False
    assert s._cached_graph_inputs(*args) is None
    assert len(calls)==1


def test_ragged_only_stage_reuses_key_across_query_layouts():
    from types import SimpleNamespace as NS
    from vllm.v1.spec_decode.cdb_av_refill import RefillSession
    s=RefillSession.__new__(RefillSession)
    s.graph_input_stage=False;s.ragged_input_stage=True;s.block_size=16
    s.capacity=4;s.graph_query_bound=8;s.ragged_sequence_floor=1024
    storage=(111,222);s._decode_graph_storage=lambda:storage
    calls=[];persistent={'ids':object()}
    stage=NS(stage=lambda *a:(calls.append(a) or persistent),calls=2)
    s.p=NS(max_model_len=1024,_cdb_decode_graph_storage=storage,
        _cdb_decode_graphs={(4,1024,2,'dynamic8'):{'input_stage':stage}})
    for q in [[1,3],[2,2],[3,1]]:
        result=s._cached_graph_inputs([1,2,3,4],[0,1,2,3],[[1],[2]],
            [3,3],q,[16,32,33,34],[q[0]-1,3])
        assert result==persistent and result is not persistent
    assert len(calls)==3
    # The rejected exact one-query path stays disabled.
    assert s._cached_graph_inputs([1,2],[0,1],[[1],[2]],[1,2],
        [1,1],[16,33],[0,1]) is None
    s.p._cdb_decode_graph_storage=(999,)
    assert s._cached_graph_inputs([1,2,3,4],[0,1,2,3],[[1],[2]],
        [3,3],[1,3],[16,32,33,34],[0,3]) is None
    assert len(calls)==3
    s.p._cdb_decode_graph_storage=storage
    # Unsupported long queries and changed token-row counts must not alias.
    assert s._cached_graph_inputs(list(range(10)),list(range(10)),[[1],[2]],
        [1,9],[1,9],list(range(10)),[0,9]) is None
    assert s._cached_graph_inputs(list(range(5)),list(range(5)),[[1],[2]],
        [2,3],[2,3],list(range(5)),[1,4]) is None
    assert len(calls)==3


@GPU
@pytest.mark.parametrize('layouts',[
    [[1,3],[2,2],[3,1]],
    [[1,2,1,3],[2,1,3,1],[1,1,1,4]],
    [[8,8,7,8],[7,8,8,8],[8,7,8,8]],
])
def test_graph_reads_dynamic_offsets_and_last_indices(layouts):
    nr=len(layouts[0]);n=sum(layouts[0])
    stage=GraphInputStage('cuda:0',requests=nr,tokens=n,width=64)
    def inputs(q,i):
        starts=[0]
        for x in q:starts.append(starts[-1]+x)
        return dict(ids=list(range(i,i+n)),positions=list(range(n)),
            blocks=[[r+1] for r in range(nr)],seqs=q,qlens=q,
            slots=list(range(n)),last_indices=[v-1 for v in starts[1:]])
    f=stage.stage(**inputs(layouts[0],0))
    selected=torch.empty(nr,dtype=torch.int32,device='cuda')
    offsets=torch.empty(nr+1,dtype=torch.int32,device='cuda')
    torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        selected.copy_(f['ids'][f['last_indices']])
        offsets.copy_(f['query_start_loc'])
    pointers={k:v.data_ptr() for k,v in f.items()}
    for i in range(60):
        q=layouts[i%len(layouts)];args=inputs(q,i)
        f=stage.stage(**args);graph.replay()
        assert selected.cpu().tolist()==[args['ids'][j] for j in args['last_indices']]
        assert offsets.cpu().tolist()==f['query_start_loc_cpu'].tolist()
        assert {k:v.data_ptr() for k,v in f.items()}==pointers
