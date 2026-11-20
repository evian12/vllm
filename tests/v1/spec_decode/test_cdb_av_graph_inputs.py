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
