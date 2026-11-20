"""Relate real dispatch markers to CPU launch owners, not GPU time containment."""
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import statistics
import sys

from analyze_trace import correlation, union_us


def frozen(x):
    return tuple(frozen(v) for v in x) if isinstance(x,list) else x


def summarize(events, cache):
    complete=[e for e in events if e.get('ph')=='X' and 'dur' in e]
    ranges=[e for e in complete if e.get('cat')=='user_annotation' and e.get('name','').startswith('CDB::')]
    target=[e for e in ranges if e['name'].startswith('CDB::target_forward::')]
    draft=[e for e in ranges if e['name'].startswith('CDB::draft_forward::')]
    def inside(a,b):return a.get('pid')==b.get('pid') and a.get('tid')==b.get('tid') and b['ts']<=a['ts'] and a['ts']+a['dur']<=b['ts']+b['dur']
    groups=defaultdict(lambda:dict(calls=0,cpu_inclusive_us=0.,kernel_count=0,kernel_sum_us=0.,memory_count=0,memory_sum_us=0.,tokens=Counter()))
    assignment={};markers=[]
    for parent in target:
        children=[e for e in ranges if e['name'].startswith('CDB::target_dispatch::') and inside(e,parent)]
        if len(children)!=1:raise ValueError(f'Expected one actual dispatch marker, found {len(children)}')
        parts=children[0]['name'].split('::',3);action=parts[2]
        encoded=parts[3]
        key=json.loads(bytes.fromhex(encoded[4:]).decode() if encoded.startswith('hex_') else encoded)
        n=int(parent['name'].rsplit('::t',1)[1])
        group='target_'+action+('_large_input' if n>36 else '')
        assignment[id(parent)]=group
        groups[group]['calls']+=1;groups[group]['cpu_inclusive_us']+=parent['dur'];groups[group]['tokens'][n]+=1
        markers.append(dict(action=action,key=key,tokens=n))
    for parent in draft:
        children=[e for e in ranges if e['name'].startswith('CDB::draft_graph_forward::') and inside(e,parent)]
        group='draft_graph' if children else 'draft_eager'
        if len(children)>1:raise ValueError('Ambiguous Draft graph parent')
        assignment[id(parent)]=group
        groups[group]['calls']+=1;groups[group]['cpu_inclusive_us']+=parent['dur']
        groups[group]['tokens'][int(parent['name'].rsplit('::t',1)[1])]+=1
    apis=defaultdict(list)
    for e in complete:
        if e.get('cat') in ('cuda_runtime','cuda_driver') and correlation(e) is not None:apis[correlation(e)].append(e)
    unknown=0
    for e in complete:
        if e.get('cat') not in ('kernel','gpu_memcpy','gpu_memset'):continue
        candidates=apis.get(correlation(e),[])
        if len(candidates)!=1:unknown+=1;continue
        api=candidates[0];point=api['ts']+api['dur']/2
        parents=[p for p in target+draft if p.get('pid')==api.get('pid') and p.get('tid')==api.get('tid') and p['ts']<=point<=p['ts']+p['dur']]
        if not parents:continue
        parent=min(parents,key=lambda p:p['dur']);g=groups[assignment[id(parent)]]
        kind='kernel' if e.get('cat')=='kernel' else 'memory'
        g[kind+'_count']+=1;g[kind+'_sum_us']+=e['dur']
    cpu={}
    for prefix in ['worker_rpc','engine_draft_advance','engine_reconcile','draft_step_with_sampling','target_storage_audit','draft_metadata_pack']:
        rs=[e for e in ranges if e['name'].startswith('CDB::'+prefix)]
        by_name=defaultdict(list)
        for e in rs:by_name[e['name']].append(e)
        for name,values in by_name.items():
            exclusive=[]
            for e in values:
                children=[(c['ts'],c['ts']+c['dur']) for c in ranges if c is not e and inside(c,e)]
                exclusive.append(max(0,e['dur']-union_us(children)))
            cpu[name]=dict(calls=len(values),inclusive_sum_us=sum(e['dur'] for e in values),
                           median_inclusive_us=statistics.median(e['dur'] for e in values),
                           outside_nested_cdb_ranges_us=sum(exclusive))
    def no_seq(k):
        if not isinstance(k,(list,tuple)) or len(k)<4:raise ValueError('Unknown Target key schema')
        return frozen(list(k[:3])+list(k[4:]))
    keys=cache['keys'];domains={no_seq(k) for k in keys}
    fallback=[x for x in markers if x['action']=='fallback_budget']
    return dict(groups=dict(groups),cpu_scopes=cpu,dispatches=markers,
                cache=dict(captured=len(keys),unique_domains_without_sequence_bound=len(domains),
                           key_budget=cache['limit'],actual_sequence_floor=cache['seq_floor'],
                           budget_fallbacks=len(fallback),
                           budget_fallbacks_with_bound_only_match=sum(no_seq(x['key']) in domains for x in fallback)),
                ambiguous_gpu_api_correlations=unknown,
                notes=['Diagnostic profiled slice only; no throughput claim.',
                       'Kernel ownership uses unique correlated CPU API inside parent, not GPU start time.',
                       'Outside nested ranges includes framework/profiler/Python work, not proven IPC cost.',
                       'Bound-only match is counterfactual key compatibility, not proof of replay validity or speedup.'])


if __name__=='__main__':
    path=Path(sys.argv[1]);opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'rt') as f:data=json.load(f)
    cache=json.loads(Path(sys.argv[2]).read_text())
    result=summarize(data['traceEvents'],cache)
    Path(sys.argv[3]).write_text(json.dumps(result,indent=2)+'\n')
