"""Causal-state audit; uses archived observations, never executes models."""
from pathlib import Path
import collections
import json
import math
import sys


def distribution(values):
    values=sorted(values)
    if not values:return dict(n=0)
    return dict(n=len(values),mean=sum(values)/len(values),p50=values[int(.5*(len(values)-1))],p95=values[int(.95*(len(values)-1))],maximum=values[-1])


def analyze(rows):
    rows=sorted(rows,key=lambda x:x['audit_ns'])
    events=[x for x in rows if x.get('kind')=='causal_event']
    verified=[x for x in events if x['event']=='p2_verified' and x.get('trace_key','').startswith('measurement:')]
    ids={x['req_id'] for x in verified}
    assert ids, 'No measured Worker provenance'
    batches=[x for x in rows if x.get('kind')=='target_batch' and set(x['rows'])&ids]
    begin=min(x['audit_ns'] for x in batches)
    end=max(x['audit_ns'] for x in verified)
    core=[x for x in rows if begin<=x['audit_ns']<=end]
    stages=collections.defaultdict(collections.Counter)
    for x in core:
        if x.get('kind')!='host_stage':continue
        d=stages[x['stage']];d['calls']+=1
        for k in ['cpu_ns','wall_ns','cpu_exclusive_ns','wall_exclusive_ns']:d[k]+=x[k]
    stages={stage:{k.replace('_ns','_ms'):(v if k=='calls' else v/1e6) for k,v in d.items()} for stage,d in stages.items()}
    # All request transitions (including warmup) are needed to reconstruct state
    # at the first formal batch. Request identity selects the measured cohort.
    unpublished={};published={};ready={};handoff={};admit={};enqueue={}
    verify_seen=set();wait=collections.defaultdict(list);unders=collections.Counter()
    ready_steps=collections.Counter();reasons=collections.Counter();candidate=collections.defaultdict(collections.Counter)
    budget_excess=accepted=generated=0;prefill=0;verify_rows=0;final_outputs={};batchhist=collections.Counter();rhist=collections.Counter()
    targetcount=0;unknown=[];step=0;horizon=8;window_open=False;coalesce=collections.Counter();remaining_hist=collections.Counter()
    for x in rows:
        kind=x.get('kind');ev=x.get('event');rid=x.get('req_id')
        if kind=='causal_event':
            if ev=='p2_window_begin':step=0;horizon=x['H'];window_open=True
            elif ev=='p2_draft_step':step=x['step']
            elif ev=='p2_boundary':window_open=False
            elif ev=='p2_reset':unpublished.clear();published.clear()
            elif ev=='draft_wait_enqueued':enqueue[rid]=x
            elif ev=='draft_wait_admitted':admit[rid]=x
            elif ev=='ready':
                key=(rid,x['fingerprint']);assert key not in ready
                ready[key]=x;unpublished[rid]=x
                if rid in ids:
                    ready_steps[x['step']]+=1;reasons[x['reason']]+=1
                    if rid in enqueue and rid in admit:
                        wait['enqueue_to_admit_ms'].append((admit[rid]['audit_ns']-enqueue[rid]['audit_ns'])/1e6)
                        wait['admit_to_ready_ms'].append((x['audit_ns']-admit[rid]['audit_ns'])/1e6)
            elif ev=='p2_handoff':
                key=(rid,x['fingerprint']);assert key not in handoff
                handoff[key]=x;unpublished.pop(rid,None);published[rid]=x
                if rid in ids:
                    assert key in ready
                    wait['ready_to_publish_ms'].append((x['audit_ns']-ready[key]['audit_ns'])/1e6)
            elif ev=='p2_verified':
                published.pop(rid,None)
                if rid in ids:
                    key=(rid,x['fingerprint']);assert key in handoff and key not in verify_seen
                    verify_seen.add(key);wait['publish_to_verify_ms'].append((x['audit_ns']-handoff[key]['audit_ns'])/1e6)
                    wait['ready_to_verify_ms'].append((x['audit_ns']-ready[key]['audit_ns'])/1e6)
                    candidate[x['drafts_count']]['verify_rows']+=1
                    candidate[x['drafts_count']]['accepted']+=x['accepted']
                    candidate[x['drafts_count']]['generated']+=x['generated_count']
                    accepted+=x['accepted'];generated+=x['generated_count']
            elif ev=='target_ready_select' and begin<=x['audit_ns']<=end:
                n=len(x['selected']);rhist[n]+=1
                if n<4:
                    unders['all_underfull_selections']+=1
                    unders['missing_rows']+=4-n
                    if x['ready_queue']>=4:unders['published_backlog_but_underfull']+=1
                    if unpublished:unders['underfull_with_unpublished']+=1
                    if len(unpublished)>=4-n:unders['unpublished_sufficient_to_fill']+=1
        elif kind=='target_batch' and set(x['rows'])&ids:
            targetcount+=1;batchhist[len(x['rows'])]+=1
            if len(x['rows'])<4:
                unders['underfull_actual_target_calls']+=1
                remaining_hist[str(horizon-step) if window_open else 'closed']+=1
                for bound in [2,7]:
                    if (window_open and 0<horizon-step<=bound and not x['waiting']
                            and all(b['draft_count']>0 for b in x['rows'].values())
                            and len(x['rows'])+len(unpublished)>=4):
                        coalesce[f'within{bound}steps_calls']+=1
                        coalesce[f'within{bound}steps_rows']+=len(x['rows'])
                        coalesce[f'within{bound}steps_unpublished']+=len(unpublished)
                unders['underfull_actual_with_waiting_draft']+=int(x['draft_wait']>0)
                unders['underfull_actual_with_leases']+=int(x['leases']>0)
                unders['underfull_actual_with_unpublished']+=int(bool(unpublished))
                unders['underfull_actual_running_ge4']+=int(x['running']>=4)
                unders['underfull_actual_in_tail_running_lt4']+=int(x['running']<4 and x['waiting']==0)
            for rid,b in x['rows'].items():
                if rid not in ids:unknown.append(rid);continue
                final_outputs[rid]=b['max_tokens']
                k=b['draft_count']
                if k:
                    verify_rows+=1
                    # Capacity upper bound only; not actual accepted waste.
                    budget_excess+=max(0,k+1-(b['max_tokens']-b['output_before']))
                elif b['output_before']==0:prefill+=1
    assert len(verify_seen)==len(verified)
    # Match each verified event to the preceding scheduled request row by order,
    # and count generated tokens beyond its remaining client output budget.
    pending={};actual_excess=0;mismatches=0;out_after={}
    for x in rows:
        if x.get('kind')=='target_batch':
            for rid,b in x['rows'].items():
                if rid in ids and b['draft_count']:pending[rid]=b
        elif x.get('event')=='p2_verified' and x.get('req_id') in ids:
            rid=x['req_id'];b=pending.pop(rid)
            mismatches+=int(b['draft_count']!=x['drafts_count'])
            actual_excess+=max(0,x['generated_count']-(b['max_tokens']-b['output_before']))
            out_after[rid]=min(b['max_tokens'],b['output_before']+x['generated_count'])
    return dict(scope='instrumented full512, no throughput claim; main-thread CPU only',measured_requests=len(ids),target_calls=targetcount,target_row_hist=dict(batchhist),selection_hist=dict(rhist),underfull=dict(unders),underfull_remaining_steps=dict(remaining_hist),bounded_coalesce_opportunities=dict(coalesce),wait_ms={k:distribution(v) for k,v in wait.items()},ready_step_hist=dict(ready_steps),stop_reasons=dict(reasons),candidate_yield={k:dict(v) for k,v in candidate.items()},accounting=dict(prefill_rows=prefill,verify_rows=verify_rows,accepted=accepted,generated=generated,output_budget_sum=sum(final_outputs.values()),counter_excess=accepted+verify_rows+prefill-sum(final_outputs.values()),actual_generated_beyond_remaining_budget=actual_excess,scheduled_candidate_capacity_beyond_budget=budget_excess,draft_count_mismatches=mismatches,final_reached_budget=sum(out_after.get(r)==b for r,b in final_outputs.items()),unknown_scheduled_rows=len(unknown)),host_stages_ms=stages,interval_span_s=(end-begin)/1e9,records=len(rows))


if __name__=='__main__':
    root=Path(sys.argv[1]);records=[]
    for p in (root/'full512_causal/observer').glob('*_requests.jsonl'):
        for line in p.open():
            x=json.loads(line)
            if 'audit_ns' in x:records.append(x)
    result=analyze(records)
    (root/'CAUSAL_ANALYSIS.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
