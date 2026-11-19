"""Attribute uncovered GPU trace intervals; never call them proven GPU idle."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import statistics


def correlation(e):
    value=e.get('args',{}).get('correlation')
    return str(value) if value is not None else None


def summarize(events):
    complete=[e for e in events if e.get('ph')=='X' and 'dur' in e]
    gpu=sorted((e for e in complete if e.get('cat') in ['kernel','gpu_memcpy','gpu_memset']),key=lambda e:e['ts'])
    apis=defaultdict(list);ranges=defaultdict(list)
    for e in complete:
        if e.get('cat') in ['cuda_runtime','cuda_driver'] and correlation(e) is not None:
            apis[correlation(e)].append(e)
        if e.get('cat')=='user_annotation' and e.get('name','').startswith('CDB::'):
            ranges[(e['pid'],e['tid'])].append(e)
    def owner(e):
        if correlation(e) is None:return None,None,None
        candidates=apis[correlation(e)]
        if len(candidates)!=1:return None,None,None
        api=candidates[0];point=api['ts']+api['dur']/2
        parents=[r for r in ranges[(api['pid'],api['tid'])] if r['ts']<=point<=r['ts']+r['dur']]
        parent=min(parents,key=lambda r:r['dur']) if parents else None
        return api,parent,correlation(e)
    grouped=defaultdict(list);transitions=defaultdict(list);examples=[]
    last=None;end=None
    for e in gpu:
        if last is not None and e['ts']>end:
            duration=e['ts']-end
            a,pa,ca=owner(last);b,pb,cb=owner(e)
            same_range=pa is not None and pa is pb
            if a is not None and b is not None and ca==cb and 'GraphLaunch' in a['name']:
                label='same_graph_launch'
            elif same_range:label='same_cpu_scope_'+pa['name'].split('::')[1]
            else:label='between_scopes_or_unattributed'
            grouped[label].append(duration)
            left=pa['name'].split('::')[1] if pa else 'unknown'
            right=pb['name'].split('::')[1] if pb else 'unknown'
            transitions[left+' -> '+right].append(duration)
            examples.append(dict(us=duration,classification=label,previous=left,next=right,
                                 previous_api=a['name'] if a else None,next_api=b['name'] if b else None))
        if end is None or e['ts']+e['dur']>end:
            end=e['ts']+e['dur'];last=e
    def stats(v):return dict(count=len(v),sum_us=sum(v),median_us=statistics.median(v),max_us=max(v))
    return dict(total_uncovered_us=sum(sum(v) for v in grouped.values()) if gpu else None,
        groups={k:stats(v) for k,v in grouped.items()},
        transitions={k:stats(v) for k,v in sorted(transitions.items(),key=lambda x:-sum(x[1]))},
        largest_intervals=sorted(examples,key=lambda x:-x['us'])[:20],
        note='Observed trace union gaps only. Profiler overhead or incomplete GPU coverage may contribute; these are not proven hardware idle or wholly removable overhead. CPU scopes are launch owners, not causal explanations.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('trace',type=Path);p.add_argument('output',type=Path);args=p.parse_args()
    with gzip.open(args.trace,'rt') as f:data=json.load(f)
    result=summarize(data['traceEvents']);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['largest_intervals','transitions']},indent=2))
