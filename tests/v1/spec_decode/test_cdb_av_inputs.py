# SPDX-License-Identifier: Apache-2.0
import os
import struct
from types import SimpleNamespace as NS

import pytest
import torch

from vllm.v1.spec_decode.cdb_av_inputs import (
    PinnedDraftInputStager,encode_input_plan,get_input_stager,
    make_input_plan,validate_staging_scope)


def case(qlens):
    starts=[0]
    for q in qlens:starts.append(starts[-1]+q)
    seqs=[max(q,32+16*i) for i,q in enumerate(qlens)]
    blocks=[list(range(100*i+1,100*i+2+(s+15)//16)) for i,s in enumerate(seqs)]
    pos=[p for s,q in zip(seqs,qlens) for p in range(s-q,s)]
    slots=[b[p//16]*16+p%16 for b,s,q in zip(blocks,seqs,qlens) for p in range(s-q,s)]
    n=sum(qlens)
    return dict(ids=list(range(101,101+n)),positions=pos,blocks=blocks,seqs=seqs,
        qlens=list(qlens),slots=slots,last_indices=[s-1 for s in starts[1:]])


def expected(args):
    b=args['blocks'];width=max(map(len,b));starts=[0]
    for q in args['qlens']:starts.append(starts[-1]+q)
    def cpu(x,dtype):return torch.tensor(x,dtype=dtype,device='cpu')
    return dict(ids=cpu(args['ids'],torch.int32),
        positions=cpu(args['positions'],torch.int64),
        query_start_loc=cpu(starts,torch.int32),
        query_start_loc_cpu=cpu(starts,torch.int32),
        seq_lens=cpu(args['seqs'],torch.int32),
        block_table_tensor=cpu([row+[0]*(width-len(row)) for row in b],torch.int32),
        slot_mapping=cpu(args['slots'],torch.int64),
        last_indices=cpu(args['last_indices'],torch.int64))


@pytest.mark.parametrize('qlens',[(1,),(1,1,1,1),(1,2,4),(8,1,2,7),(256,256,256,256)])
def test_cpu_plan_matches_native_typed_values(qlens):
    args=case(qlens);ref=expected(args);plan=make_input_plan(**args)
    dest=bytearray([255]*plan.nbytes);encode_input_plan(plan,dest)
    for f in plan.fields:
        assert f.offset%16==0
        actual=torch.frombuffer(dest,dtype=f.dtype,count=len(f.values),offset=f.offset).view(f.shape)
        assert actual.dtype==ref[f.name].dtype and actual.shape==ref[f.name].shape
        assert torch.equal(actual.view(torch.uint8),ref[f.name].view(torch.uint8))
    assert plan.nbytes%16==0


def test_integer_boundaries_preserve_all_bits():
    args=case([1]);args.update(ids=[2**31-1],positions=[-2**63],
        blocks=[[2**31-1]],slots=[2**63-1])
    plan=make_input_plan(**args);dest=bytearray(plan.nbytes);encode_input_plan(plan,dest)
    values={f.name:struct.unpack_from(f"={len(f.values)}{f.code}",dest,f.offset) for f in plan.fields}
    assert values['ids']==(2**31-1,) and values['positions']==(-2**63,)
    assert values['slot_mapping']==(2**63-1,)


@pytest.mark.parametrize('field,value',[('ids',2**31),('slots',2**63),('ids',1.25)])
def test_encoder_rejects_overflow_or_noninteger(field,value):
    args=case([1]);args[field]=[value];plan=make_input_plan(**args)
    with pytest.raises((struct.error,OverflowError,TypeError)):
        encode_input_plan(plan,bytearray(plan.nbytes))


@pytest.mark.parametrize('change',[
 dict(qlens=[]),dict(ids=[]),dict(positions=[]),dict(slots=[]),
 dict(blocks=[[]]),dict(last_indices=[1]),dict(seqs=[1025]),
 dict(qlens=[0]),dict(qlens=[1,1,1,1,1]),
])
def test_invalid_shapes_rejected(change):
    args=case([1]);args.update(change)
    with pytest.raises((ValueError,TypeError)):make_input_plan(**args)


def test_destination_contract_and_plan_snapshot():
    args=case([1,2]);plan=make_input_plan(**args)
    with pytest.raises(ValueError):encode_input_plan(plan,bytes(plan.nbytes))
    with pytest.raises(ValueError):encode_input_plan(plan,bytearray(plan.nbytes-1))
    old=plan.fields[0].values;args['ids'][0]=999
    assert plan.fields[0].values==old


def config():
    return NS(model_config=NS(enforce_eager=True,dtype=torch.bfloat16,
                  max_model_len=1024,enable_sleep_mode=False),
        scheduler_config=NS(async_scheduling=False,max_num_seqs=4),
        parallel_config=NS(tensor_parallel_size=1,pipeline_parallel_size=1,
            data_parallel_size=1,decode_context_parallel_size=1),lora_config=None)


def test_disabled_factory_never_touches_cuda(monkeypatch):
    monkeypatch.setenv('CDB_AV_PINNED_INPUTS','0')
    monkeypatch.setattr(torch.cuda,'Event',lambda **kw:pytest.fail('CUDA touched'))
    assert get_input_stager(object()) is None


def test_invalid_flag_and_missing_refill(monkeypatch):
    monkeypatch.setenv('CDB_AV_PINNED_INPUTS','x')
    with pytest.raises(ValueError):get_input_stager(object())
    monkeypatch.setenv('CDB_AV_PINNED_INPUTS','1')
    monkeypatch.delenv('CDB_AV_PERSISTENT',raising=False)
    with pytest.raises(ValueError):get_input_stager(object())


@pytest.mark.parametrize('field,value',[
 ('async_scheduling',True),('max_num_seqs',8),('max_model_len',2048),
 ('enable_sleep_mode',True),('dtype',torch.float32),('tensor_parallel_size',2)])
def test_scope_fail_closed(field,value):
    cfg=config()
    for part in (cfg.model_config,cfg.scheduler_config,cfg.parallel_config):
        if hasattr(part,field):setattr(part,field,value)
    with pytest.raises(ValueError):validate_staging_scope(cfg)


def test_cpu_device_stager_rejected_without_cuda():
    with pytest.raises(ValueError):PinnedDraftInputStager('cpu')


GPU = pytest.mark.skipif(os.getenv('CDB_PINNED_GPU_TEST')!='1',
                        reason='explicit isolated CUDA reservation required')


@GPU
@pytest.mark.parametrize('qlens',[(1,),(1,1,1,1),(1,2,4),(8,1,2,7),(256,256,256,256)])
def test_cuda_fields_and_fresh_storage(qlens):
    assert os.environ['CUDA_VISIBLE_DEVICES']=='3'
    args=case(qlens);stager=PinnedDraftInputStager('cuda:0')
    actual=stager.pack(**args);ref=expected(args)
    second=stager.pack(**case([1]))
    for key,value in ref.items():
        a=actual[key].cpu()
        assert a.shape==value.shape and a.dtype==value.dtype
        assert torch.equal(a.contiguous().view(torch.uint8),value.contiguous().view(torch.uint8))
    assert actual['query_start_loc_cpu'].device.type=='cpu'
    assert actual['query_start_loc_cpu'].data_ptr()!=second['query_start_loc_cpu'].data_ptr()
    assert actual['ids'].untyped_storage().data_ptr()!=second['ids'].untyped_storage().data_ptr()
    assert stager.host.is_pinned()


@GPU
def test_pending_source_copy_and_growth_cannot_corrupt_earlier_payload():
    stager=PinnedDraftInputStager('cuda:0')
    warm=stager.pack(**case([1]));warm['ids'].cpu()
    torch.cuda._sleep(50_000_000)
    first=stager.pack(**case([1,2]))
    second=stager.pack(**case([256,256,256,256]))
    assert stager.source_waits>=1 and stager.growths>=2
    for actual,args in [(first,case([1,2])),(second,case([256]*4))]:
        for k,v in expected(args).items():
            a=actual[k].cpu()
            assert torch.equal(a.contiguous().view(torch.uint8),v.contiguous().view(torch.uint8))
    assert first['query_start_loc_cpu'].tolist()==[0,1,3]


@GPU
def test_cpu_shadow_stays_cpu_under_default_cuda_device():
    original=torch.get_default_device()
    try:
        torch.set_default_device('cuda')
        stager=PinnedDraftInputStager('cuda:0')
        actual=stager.pack(**case([1,2]))
        assert actual['query_start_loc_cpu'].device.type=='cpu'
        assert actual['query_start_loc_cpu'].tolist()==[0,1,3]
    finally:
        torch.set_default_device(original)


@pytest.mark.parametrize('staged',[False,True])
def test_real_step_branch_preserves_request_mapping_and_completion(staged,monkeypatch):
    """CPU control regression only;CUDA transport has separate GPU tests."""
    from vllm.v1.spec_decode.cdb_av_refill import RefillSession
    monkeypatch.setenv('CDB_AV_PACKED_METADATA','0')
    s=RefillSession.__new__(RefillSession)
    s.window=1;s.done=False;s.initialized=True;s.steps_completed=0;s.horizon=8
    s.capacity=4;s.block_size=16;s.policy='manual';s.threshold=.8
    s.p=NS(device=torch.device('cpu'));s._draft_stage_profiler=None
    s._draft_stage_pending=None;s._draft_stage_metadata_ms=0
    s.members={
        'A':dict(next_token=9,next_position=10,blocks=[1],ready=False,
                 ready_at=None,cap=1,tokens=[],progress=0,fingerprint='A',source='target_first'),
        'B':dict(next_token=7,next_position=2,blocks=[2],ready=False,
                 ready_at=None,cap=3,tokens=[],progress=0,fingerprint='B',source='target_first')}
    s.active=['A'];s.pending_prefill=[([5,6,7],0,3,[2],'B','reuse')]
    s._input_stager=None
    if staged:
        def pack(ids,positions,blocks,seqs,qlens,slots,last_indices):
            return expected(dict(ids=ids,positions=positions,blocks=blocks,seqs=seqs,
                qlens=qlens,slots=slots,last_indices=last_indices))
        s._input_stager=NS(pack=pack)
    seen=[]
    def forward(ids,pos,cad,indices,rids,phase):
        seen.append(list(rids))
        if len(seen)==1:
            assert ids.tolist()==[9,5,6,7] and pos.tolist()==[10,0,1,2]
            assert indices.tolist()==[0,3]
            assert cad.query_start_loc_cpu.tolist()==[0,1,4]
            assert cad.block_table_tensor.tolist()==[[1],[2]]
            assert cad.slot_mapping.tolist()==[26,32,33,34]
            logits=torch.zeros(2,10);logits[0,3]=1;logits[1,4]=1
        else:
            assert rids==['B'] and indices.tolist()==[0]
            assert pos.tolist()==[len(seen)+1]
            logits=torch.zeros(1,10);logits[0,len(seen)+3]=1
        return logits
    s._forward=forward
    first=s.step();assert first['newly_completed_req_ids']==['A']
    s.step();third=s.step()
    assert third['newly_completed_req_ids']==['B'] and not s.active
    assert seen==[['A','B'],['B'],['B']]
    assert s.members['A']['tokens']==[3] and s.members['B']['tokens']==[4,5,6]

@GPU
def test_bounded_gpu_validation_uses_actual_typed_payload(monkeypatch):
    monkeypatch.setenv("CDB_AV_VALIDATE_METADATA","1")
    monkeypatch.setenv("CDB_AV_INPUT_VALIDATE_SAMPLES","2")
    stager=PinnedDraftInputStager("cuda:0")
    for _ in range(4):stager.pack(**case([1,2,4]))
    assert stager.validation_samples==2
