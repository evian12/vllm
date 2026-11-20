from audit_verify_graph_trace import simulate


def test_warmup_state_carries_into_measurement_and_cold_capture_threshold():
    key=(4,8,3)
    warm=simulate([('warmup',key)]*8+[('measurement',key)]*3)
    assert warm['counts']['graph_calls']==3 and warm['counts']['captures']==0
    cold=simulate([('measurement',key)]*9)
    assert cold['counts']['graph_calls']==2 and cold['counts']['captures']==1


def test_capacity_and_memory_prevent_capture_and_plain_rows_do_not_fill_cache():
    a=(4,8,3);b=(4,9,3)
    full=simulate([('warmup',a)]*8+[('measurement',b)]*20,max_keys=1)
    assert full['counts']['graph_calls']==0 and len(full['cache_keys'])==1
    limited=simulate([('measurement',a)]*12,max_bytes=1)
    assert not limited['cache_keys'] and limited['counts']['graph_calls']==0
    plain=simulate([('measurement',None)]*3)
    assert plain['counts']['plain_sampler_calls']==3 and not plain['cache_keys']
