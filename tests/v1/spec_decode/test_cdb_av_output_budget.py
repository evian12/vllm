# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest

from vllm.v1.request import RequestStatus
from vllm.v1.spec_decode.cdb_av_config import bound_output_budget_cap
from vllm.v1.spec_decode.cdb_av_persistent import PersistentDraftQueue
from vllm.v1.spec_decode.cdb_av_kv_reuse import ReuseDraftQueue


@pytest.mark.parametrize('cap,maximum,output,expected', [
    (8,16,12,3),(8,16,15,0),(1,16,12,1),(8,128,1,8),
    (8,16,16,0),(8,1,0,0),(8,16,0,8),(4,8,6,1),(8,16,14,1),
])
def test_cap_reserves_one_native_target_result(cap,maximum,output,expected):
    assert bound_output_budget_cap(cap,maximum,output)==expected


@pytest.mark.parametrize('args', [
    (0,16,1),(-1,16,1),(True,16,1),(8,0,0),(8,None,0),
    (8,True,0),(8,16,-1),(8,16,17),(8,16,1.0),(8,16,True),
])
def test_invalid_budget_is_rejected(args):
    with pytest.raises(ValueError):bound_output_budget_cap(*args)


def queue(cls):
    q=cls.__new__(cls)
    q.request_average=q.acceptance_replay=None
    q.max_k=8
    return q


def test_default_off_does_not_require_new_request_fields(monkeypatch):
    monkeypatch.delenv('CDB_AV_OUTPUT_BUDGET_CAP',raising=False)
    monkeypatch.setenv('CDB_AV_K_POLICY','hf_confidence')
    req=SimpleNamespace(sampling_params=SimpleNamespace(extra_args=None))
    assert queue(PersistentDraftQueue).resolve_k_cap(req,None)==(8,'configured')


@pytest.mark.parametrize('remaining,expected', [(9,8),(4,3),(2,1),(1,0)])
def test_resolver_preserves_hf_policy_source_and_reports_real_budget(monkeypatch,remaining,expected):
    monkeypatch.setenv('CDB_AV_OUTPUT_BUDGET_CAP','1')
    monkeypatch.setenv('CDB_AV_K_POLICY','hf_confidence')
    req=SimpleNamespace(request_id='real-r',max_tokens=16,num_output_tokens=16-remaining,
        num_tokens=100,sampling_params=SimpleNamespace(extra_args=None))
    assert queue(PersistentDraftQueue).resolve_k_cap(req,None)==(expected,'configured')


@pytest.mark.parametrize('cls',[PersistentDraftQueue,ReuseDraftQueue])
def test_zero_cap_never_allocates_or_reaches_worker_plan(monkeypatch,cls):
    monkeypatch.setenv('CDB_AV_OUTPUT_BUDGET_CAP','1')
    monkeypatch.setenv('CDB_AV_K_POLICY','hf_confidence')
    req=SimpleNamespace(request_id='tail',status=RequestStatus.RUNNING,
        is_finished=lambda:False,num_output_tokens=15,max_tokens=16,spec_token_ids=[],
        num_computed_tokens=34,num_tokens=35,sampling_params=SimpleNamespace(extra_args=None))
    q=queue(cls);q._cdb_admit_limit=None;q.leases={};q.ready={};q.inflight={}
    # No allocator methods exist: any attempted allocation/verification fails.
    q.s=SimpleNamespace(running=[req],kv_cache_manager=SimpleNamespace(block_pool=object()))
    q.unsupported_reason=lambda req:None
    cached={'tail':[object()]};q.cached_leases=cached
    assert q.plans()==[]
    assert q.leases=={} and q.ready=={} and q.inflight=={}
    assert q.cached_leases is cached and len(cached['tail'])==1
    assert q.frozen==set()
