from analyze import analyze


def test_budget_tail_and_transition_accounting():
    rows=[]
    def add(t,kind='causal_event',**data):rows.append(dict(audit_ns=t*1000000,kind=kind,**data))
    add(1,event='draft_wait_enqueued',req_id='r')
    add(2,event='draft_wait_admitted',req_id='r')
    add(3,event='ready',req_id='r',fingerprint='f',step=3,reason='low_confidence')
    add(4,event='p2_handoff',req_id='r',fingerprint='f')
    add(5,event='target_ready_select',selected=['r'],ready_queue=1)
    add(6,kind='target_batch',rows={'r':dict(n=4,output_before=8,prompt_tokens=4,max_tokens=10,draft_count=3)},running=1,waiting=0,draft_wait=0,published_ready=1,leases=0,window=1)
    add(7,kind='host_stage',stage='sample',cpu_ns=100,wall_ns=200,cpu_exclusive_ns=50,wall_exclusive_ns=150)
    add(8,event='p2_verified',req_id='r',fingerprint='f',trace_key='measurement:0',drafts_count=3,accepted=3,generated_count=4)
    result=analyze(rows)
    assert result['accounting']['actual_generated_beyond_remaining_budget']==2
    assert result['accounting']['draft_count_mismatches']==0
    assert result['wait_ms']['ready_to_publish_ms']['mean']==1
    assert result['wait_ms']['publish_to_verify_ms']['mean']==4
    assert result['underfull']['underfull_actual_in_tail_running_lt4']==1
