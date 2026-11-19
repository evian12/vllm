# SPDX-License-Identifier: Apache-2.0
"""No GPU/model required: copy domain, bounds and default-path regression."""
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from vllm.v1.spec_decode import cdb_av_draft_meta_copy as m


def tensor(shape, dtype, **kw):
    return NS(shape=shape, dtype=dtype, is_cuda=kw.get('is_cuda', True),
              device=kw.get('device', 'cuda:0'),
              is_contiguous=lambda: kw.get('contiguous', True))


def metadata(nr=4, n=12, width=4):
    return NS(num_reqs=nr, num_actual_tokens=n,
              query_start_loc=tensor((nr+1,), torch.int32),
              seq_lens=tensor((nr,), torch.int32),
              block_table_tensor=tensor((nr, width), torch.int32),
              slot_mapping=tensor((n,), torch.int64))


@pytest.mark.parametrize('nr,n,sw,dw', [(1,1,1,1),(4,32,1,64),
                                       (3,7,17,64),(4,4,1024,1024)])
def test_extent_domain(nr, n, sw, dw):
    assert m.layout(metadata(nr,n,sw), metadata(nr,n,dw)) == (nr,n,sw,dw)


@pytest.mark.parametrize('nr,n', [(0,1),(5,5),(3,2),(4,33),(True,1)])
def test_invalid_rows(nr, n):
    with pytest.raises(ValueError):
        m.layout(metadata(nr,n), metadata(nr,n))


@pytest.mark.parametrize('sw,dw', [(0,4),(5,4),(1,1025)])
def test_width_guards(sw, dw):
    with pytest.raises(ValueError):
        m.layout(metadata(width=sw), metadata(width=dw))


@pytest.mark.parametrize('field,change', [
    ('query_start_loc',dict(shape=(4,),dtype=torch.int32)),
    ('seq_lens',dict(shape=(4,),dtype=torch.int64)),
    ('block_table_tensor',dict(shape=(4,4),dtype=torch.int32,is_cuda=False)),
    ('slot_mapping',dict(shape=(12,),dtype=torch.int64,device='cuda:1')),
    ('slot_mapping',dict(shape=(12,),dtype=torch.int64,contiguous=False)),
    ('block_table_tensor',dict(shape=(16,),dtype=torch.int32)),
])
def test_layout_before_write(field, change):
    a,b=metadata(),metadata()
    setattr(a,field,tensor(**change))
    with pytest.raises(ValueError):m.layout(a,b)


def test_destination_is_independently_checked():
    a,b=metadata(),metadata()
    b.slot_mapping=tensor((12,),torch.int32)
    with pytest.raises(ValueError):m.layout(a,b)
    b=metadata(nr=3,n=12)
    with pytest.raises(ValueError):m.layout(a,b)


@pytest.mark.parametrize('value', ['true','2','',None])
def test_strict_flag(value):
    with pytest.raises(ValueError):m.flag('X',{'X':value})


def test_default_does_not_touch_proposer(monkeypatch):
    monkeypatch.delenv('CDB_AV_DRAFT_META_COPY',raising=False)
    assert m.get_metadata_copier(None,None) is None
    assert not m.flag('X',{})
    assert m.flag('X',{'X':'1'})


def test_invalid_layout_never_launches(monkeypatch):
    monkeypatch.delenv('CDB_AV_DRAFT_META_COPY_VALIDATE',raising=False)
    c=m.DraftMetadataCopier()
    monkeypatch.setattr(m,'_copy_metadata_kernel',None)
    with pytest.raises(ValueError):c(metadata(n=33),metadata(n=33))
    assert c.calls==0


def test_original_copy_branch_retained():
    p=Path(m.__file__).with_name('cdb_av_refill.py')
    text=p.read_text()
    assert "getattr(self, '_draft_metadata_copier', None)" in text
    assert 'static_cad.block_table_tensor.zero_()' in text
    assert 'static_cad.slot_mapping.copy_(cad.slot_mapping)' in text
