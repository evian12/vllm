"""Read saved Kineto trace, attribute kernels to outer model/shape by launch correlation."""
import gzip,json,collections,statistics,re,hashlib
from pathlib import Path
p=next(Path('/home/yypan/CDB-AV/results/draft_wait_profile_20261120/full512_wait/trace').glob('*.pt.trace.json.gz'))
with gzip.open(p,'rt') as f:events=json.load(f)['traceEvents']
x=[e for e in events if e.get('ph')=='X' and 'dur' in e]
ranges=[e for e in x if e.get('cat')=='user_annotation' and e['name'].startswith('CDB::')]
apis=[e for e in x if e.get('cat') in ('cuda_runtime','cuda_driver')]
by_thread=collections.defaultdict(list);corr=collections.defaultdict(list)
for e in ranges:by_thread[(e.get('pid'),e.get('tid'))].append(e)
for e in apis:
 c=e.get('args',{}).get('correlation')
 if c is not None:corr[str(c)].append(e)
def parents(api):
 point=api['ts']+api['dur']/2
 return [e for e in by_thread[(api.get('pid'),api.get('tid'))] if e['ts']<=point<=e['ts']+e['dur']]
def main_scope(par):
 for prefix in ['CDB::target_logits','CDB::draft_logits','CDB::target_forward::','CDB::draft_forward::']:
  found=[e for e in par if e['name'].startswith(prefix)]
  if found:return min(found,key=lambda e:e['dur'])
 return None
calls=collections.defaultdict(list);kernel_groups=collections.defaultdict(lambda:collections.Counter());api_groups=collections.defaultdict(lambda:dict(count=0,cpu_us=0.0));shapes={}
ambiguous=0
for e in x:
 if e.get('cat')!='kernel':continue
 c=e.get('args',{}).get('correlation');match=corr.get(str(c),[])
 if len(match)!=1:ambiguous+=1;continue
 api=match[0];scope=main_scope(parents(api))
 if scope is None:continue
 key=(scope.get('pid'),scope.get('tid'),scope['ts'],scope['name']);shapes[key]=scope['name'];calls[key].append(e)
 stage=scope['name'].split('::')[1]
 kernel_groups[stage][e['name']]+=e['dur']
for api in apis:
 scope=main_scope(parents(api))
 if scope:
  stage=scope['name'].split('::')[1];g=api_groups[(stage,api['name'])];g['count']+=1;g['cpu_us']+=api['dur']
shape_groups=collections.defaultdict(list)
for key,ks in calls.items():
 shape_groups[shapes[key]].append(dict(kernel_us=sum(e['dur'] for e in ks),gemm_us=sum(e['dur'] for e in ks if 'gemm' in e['name'].lower()),kernels=len(ks)))
result=dict(source=str(p),source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),ambiguous_correlations=ambiguous,
 shape_groups={k:dict(calls=len(v),kernel_mean_us=statistics.mean(e['kernel_us'] for e in v),kernel_sum_us=sum(e['kernel_us'] for e in v),gemm_sum_us=sum(e['gemm_us'] for e in v),kernel_count=sum(e['kernels'] for e in v)) for k,v in shape_groups.items()},
 stage_kernel_totals={s:dict(total_us=sum(v.values()),gemm_us=sum(t for k,t in v.items() if 'gemm' in k.lower()),families=[dict(name=k,us=t) for k,t in v.most_common(12)]) for s,v in kernel_groups.items()},
 api_groups=[dict(stage=k[0],api=k[1],**v) for k,v in api_groups.items()],
 limits=['Existing profiled middle slice, not new performance measurement','Graph kernels linked by unique CPU API correlation; main model labels supply total packed token rows, not per-GEMM matrix M','CPU API times may include GPU waits and cannot be added to kernel times','Kernel sums not hardware DRAM traffic or end-to-end critical path'])
print(json.dumps(result,indent=2))
