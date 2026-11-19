import pytest
from analyze_actions import summarize


def scope(name,start,dur,cat='user_annotation',**kw):
    return dict(name=name,ph='X',ts=start,dur=dur,cat=cat,pid=1,tid=2,**kw)


def cache():return dict(keys=[[8,4,9,256,64,0,'bounded'],[8,4,9,128,64,0,'bounded']],limit=24,seq_floor=0)


def test_gpu_after_cpu_scope_still_attributed_to_actual_launch():
    events=[scope('CDB::target_forward::t8',10,10),
            scope('CDB::target_dispatch::fallback_budget::[8,4,9,512,64,0,"bounded"]',11,1),
            scope('cudaLaunchKernel',15,1,'cuda_runtime',args={'correlation':7}),
            scope('kernel',80,5,'kernel',args={'correlation':7})]
    x=summarize(events,cache())
    assert x['groups']['target_fallback_budget']['kernel_sum_us']==5
    assert x['cache']['captured']==2 and x['cache']['unique_domains_without_sequence_bound']==1
    assert x['cache']['budget_fallbacks_with_bound_only_match']==1


def test_gpu_mirror_annotation_does_not_count_as_dispatch():
    events=[scope('CDB::target_forward::t8',10,10),scope('CDB::target_dispatch::replay::[8,4,9,256,64,0,"bounded"]',11,1)]
    events.append(scope(events[1]['name'],11,1,'gpu_user_annotation'))
    assert summarize(events,cache())['groups']['target_replay']['calls']==1


def test_rpc_inclusive_is_not_added_as_exclusive_compute():
    events=[scope('CDB::worker_rpc::cdb_refill_rpc::p2_step',0,100),scope('CDB::draft_step_with_sampling',10,80)]
    x=summarize(events,cache())['cpu_scopes']
    assert x['CDB::worker_rpc::cdb_refill_rpc::p2_step']['outside_nested_cdb_ranges_us']==20


def test_missing_marker_is_evidence_gap():
    with pytest.raises(ValueError,match='one actual dispatch'):
        summarize([scope('CDB::target_forward::t8',1,10)],cache())
