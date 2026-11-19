# SPDX-License-Identifier: Apache-2.0
"""Fixed Target bound only; no model/GPU or scheduling simulation."""
from dataclasses import dataclass

import pytest
import torch

from vllm.v1.spec_decode.cdb_av_target_graph import graph_key,clone_metadata,update_metadata


@dataclass
class Meta:
    num_actual_tokens: int
    max_query_len: int
    max_seq_len: int
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    max_num_splits: int=0


def meta(qlens=(1,3,1,3),length=128,width=64,splits=0):
    starts=[0]
    for n in qlens:starts.append(starts[-1]+n)
    nr=len(qlens)
    return Meta(num_actual_tokens=sum(qlens),max_query_len=max(qlens),max_seq_len=length,
              query_start_loc=torch.tensor(starts,dtype=torch.int32),
              seq_lens=torch.arange(length-nr+1,length+1,dtype=torch.int32),
              block_table=torch.arange(100,100+nr*width,dtype=torch.int32).view(nr,width),
              slot_mapping=torch.arange(1000,1000+sum(qlens),dtype=torch.int64),max_num_splits=splits)


def key(m,floor=0):
    return graph_key(m,m.query_start_loc.tolist(),n=m.num_actual_tokens,capacity=4,
                     max_query=9,max_model_len=1024,dynamic=True,seq_floor=floor)


def test_merge_only_sequence_bounds_at_real_maximum_context():
    values=[meta(length=n) for n in [61,128,256,512,1024]]
    assert len({key(m) for m in values})==5
    assert len({key(m,1024) for m in values})==1
    for m in values:
        native,bounded=key(m),key(m,1024)
        assert bounded[3]==1024 and native[:3]+native[4:]==bounded[:3]+bounded[4:]


@pytest.mark.parametrize('changes',[dict(width=32),dict(splits=4),dict(qlens=(1,1,1,1))])
def test_other_domains_not_merged(changes):
    assert key(meta(),1024)!=key(meta(**changes),1024)


def test_ar_and_k8_bonus_layouts_copy_exactly_into_same_bounded_capture():
    first=meta((3,3,3,3),length=128)
    second=meta((9,1,1,1),length=1024)
    assert key(first,1024)==key(second,1024)
    fields=('query_start_loc','seq_lens','slot_mapping','block_table')
    before={f:getattr(first,f).clone() for f in fields}
    captured=clone_metadata(first,1024,query_bound=9)
    pointers={f:getattr(captured,f).data_ptr() for f in fields}
    update_metadata(captured,second,mixed_bound=9)
    assert captured.max_seq_len==1024 and captured.max_query_len==9
    for f in fields:
        assert torch.equal(getattr(first,f),before[f])
        assert torch.equal(getattr(captured,f),getattr(second,f))
        assert getattr(captured,f).data_ptr()==pointers[f]
    assert captured.query_start_loc.tolist()==[0,9,10,11,12]


def test_disabled_floor_matches_original_default():
    m=meta(length=512)
    assert key(m)==graph_key(m,m.query_start_loc.tolist(),n=8,capacity=4,max_query=9,
                            max_model_len=1024,dynamic=True)
