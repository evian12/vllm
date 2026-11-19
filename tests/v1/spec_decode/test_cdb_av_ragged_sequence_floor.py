from types import SimpleNamespace as NS

import pytest
import torch

from vllm.v1.spec_decode.cdb_av_config import get_ragged_sequence_floor
from vllm.v1.spec_decode.cdb_av_refill import RefillSession


@pytest.mark.parametrize('value',['0','16','128','512','1024'])
def test_floor_parsing(value):
    assert get_ragged_sequence_floor(1024,{'CDB_AV_RAGGED_GRAPH_SEQ_FLOOR':value})==int(value)
    assert get_ragged_sequence_floor(1024,{})==0


@pytest.mark.parametrize('value',['-1','3','1000','2048','',None,True,256,'nan'])
def test_invalid_floor_rejected(value):
    with pytest.raises(ValueError):get_ragged_sequence_floor(1024,{'CDB_AV_RAGGED_GRAPH_SEQ_FLOOR':value})


def setup(qlens=(1,1,1,2),seq=65):
    session=RefillSession.__new__(RefillSession)
    session.capacity=4;session.block_size=16;session.p=NS(max_model_len=1024)
    starts=[0]
    for q in qlens:starts.append(starts[-1]+q)
    cad=NS(num_reqs=len(qlens),num_actual_tokens=starts[-1],max_query_len=max(qlens),
           max_seq_len=seq,query_start_loc_cpu=torch.tensor(starts,dtype=torch.int32),
           block_table_tensor=torch.zeros((len(qlens),(seq+15)//16),dtype=torch.int32))
    return session,cad


def test_dynamic_floor_shares_only_sequence_bucket_and_does_not_mutate_live_metadata():
    keys=[]
    for seq in [65,129,300,700]:
        session,cad=setup(seq=seq);session.ragged_sequence_floor=1024
        before=(cad.query_start_loc_cpu.clone(),cad.block_table_tensor.clone(),cad.max_seq_len)
        keys.append(session._mixed_graph_key(cad,5,dynamic=True,max_query=8))
        assert torch.equal(cad.query_start_loc_cpu,before[0])
        assert torch.equal(cad.block_table_tensor,before[1]) and cad.max_seq_len==before[2]
    assert keys==[(5,1024,4,'dynamic8')]*4


def test_off_exact_and_decode_paths_keep_original_keys():
    session,cad=setup();assert session._mixed_graph_key(cad,5,dynamic=True,max_query=8)==(5,128,4,'dynamic8')
    session.ragged_sequence_floor=1024
    assert session._mixed_graph_key(cad,5,dynamic=False)==(5,128,(1,1,1,2))
    session,cad=setup((1,1,1,1));session.ragged_sequence_floor=1024
    assert session._decode_graph_key(cad,4)==(4,128)
    assert session._mixed_graph_key(cad,4,dynamic=True,max_query=8) is None


def test_request_count_token_count_query_bound_domains_stay_separate():
    keys=[]
    for qs in [(1,1,1,2),(1,2,2),(1,1,2,2)]:
        session,cad=setup(qs);session.ragged_sequence_floor=1024
        keys.append(session._mixed_graph_key(cad,sum(qs),dynamic=True,max_query=8))
    assert len(set(keys))==3
    assert RefillSession._graph_static_query_bound(keys[0],(1,1,1,2))==8


def test_block_table_ownership_capacity_still_bounds_the_capture():
    session,cad=setup();session.ragged_sequence_floor=1024
    cad.block_table_tensor=torch.zeros((4,65),dtype=torch.int32)
    assert session._mixed_graph_key(cad,5,dynamic=True,max_query=8) is None
    session,cad=setup();session.ragged_sequence_floor=1024;session.p.max_model_len=512
    assert session._mixed_graph_key(cad,5,dynamic=True,max_query=8) is None
