"""Exclusive CPU scopes and measured GPU overlap with blocking API spans."""
from pathlib import Path
import gzip,json,collections,hashlib,sys
sys.path.insert(0,'/home/yypan/CDB-AV/results/draft_wait_profile_20261120')
from analyze_trace import union_us,scope_name,correlation
p=next(Path('/home/yypan/CDB-AV/results/draft_wait_profile_20261120/full512_wait/trace').glob('*.pt.trace.json.gz'))
with gzip.open(p,'rt') as f:all_events=json.load(f)['traceEvents']
events=[e for e in all_events if e.get('ph')=='X' and e.get('dur',0)>0]
ranges=[e for e in events if e.get('cat')=='user_annotation' and e['name'].startswith('CDB::')]
apis=[e for e in events if e.get('cat') in ['cuda_runtime','cuda_driver']]
gpu=[e for e in events if e.get('cat') in ['kernel','gpu_memcpy','gpu_memset']]
threads=collections.defaultdict(list)
for e in ranges:threads[(e.get('pid'),e.get('tid'))].append(e)
def inside(e,span):return e['ts']>=span['ts'] and e['ts']+e['dur']<=span['ts']+span['dur']
def parents(e):
 point=e['ts']+e['dur']/2
 return [r for r in threads[(e.get('pid'),e.get('tid'))] if r['ts']<=point<=r['ts']+r['dur']]
def intervals(rows):return [(e['ts'],e['ts']+e['dur']) for e in rows]
def intersect_total(intervals,left,right):return union_us([(max(a,left),min(b,right)) for a,b in intervals if a<right and b>left])
# GPU intervals merged once for inexpensive overlap queries.
merged=[]
for a,b in sorted(intervals(gpu)):
 if merged and a<=merged[-1][1]:merged[-1]=(merged[-1][0],max(b,merged[-1][1]))
 else:merged.append((a,b))
exclusive=collections.defaultdict(lambda:dict(calls=0,inclusive_us=0,exclusive_of_nested_cdb_us=0))
for r in ranges:
 children=[x for x in threads[(r.get('pid'),r.get('tid'))] if x is not r and x['dur']<r['dur'] and inside(x,r)]
 g=exclusive[scope_name(r['name'])];g['calls']+=1;g['inclusive_us']+=r['dur'];g['exclusive_of_nested_cdb_us']+=r['dur']-union_us(intervals(children))
api_map=collections.defaultdict(list)
for a in apis:
 if correlation(a) is not None:api_map[correlation(a)].append(a)
blocking=[]
for e in events:
 if e.get('cat')=='gpu_memcpy' and 'DtoH' in e['name']:
  candidates=api_map.get(correlation(e),[])
  if len(candidates)!=1:continue
  a=candidates[0];par=parents(a);name=scope_name(min(par,key=lambda x:x['dur'])['name']) if par else 'unscoped'
  blocking.append(dict(scope=name,api=a['name'],copy=e['name'],api_cpu_us=a['dur'],gpu_copy_us=e['dur'],gpu_activity_overlap_with_api_us=intersect_total(merged,a['ts'],a['ts']+a['dur']),bytes=e.get('args',{}).get('bytes')))
grouped=collections.defaultdict(lambda:dict(count=0,api_cpu_us=0,gpu_copy_us=0,gpu_activity_overlap_with_api_us=0))
for e in blocking:
 g=grouped[(e['scope'],e['copy'])];g['count']+=1
 for k in ['api_cpu_us','gpu_copy_us','gpu_activity_overlap_with_api_us']:g[k]+=e[k]
# Runtime synchronizations, separately reported; don't sum with DtoH or parent scopes.
sync=collections.defaultdict(lambda:dict(count=0,cpu_us=0,gpu_activity_overlap_us=0))
for a in apis:
 if 'Synchronize' not in a['name']:continue
 par=parents(a);name=scope_name(min(par,key=lambda x:x['dur'])['name']) if par else 'unscoped'
 g=sync[(name,a['name'])];g['count']+=1;g['cpu_us']+=a['dur'];g['gpu_activity_overlap_us']+=intersect_total(merged,a['ts'],a['ts']+a['dur'])
memory_by_scope=collections.defaultdict(lambda:dict(count=0,gpu_us=0,bytes=0))
for e in gpu:
 if e.get('cat') not in ['gpu_memcpy','gpu_memset']:continue
 candidates=api_map.get(correlation(e),[])
 if len(candidates)!=1:continue
 par=parents(candidates[0]);name=scope_name(min(par,key=lambda x:x['dur'])['name']) if par else 'unscoped'
 g=memory_by_scope[(name,e['name'])];g['count']+=1;g['gpu_us']+=e['dur'];g['bytes']+=e.get('args',{}).get('bytes',0)
print(json.dumps(dict(source=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),cpu_scopes=dict(exclusive),memory_by_scope=[dict(scope=k[0],kind=k[1],**v) for k,v in memory_by_scope.items()],d2h=[dict(scope=k[0],copy=k[1],**v) for k,v in grouped.items()],synchronizations=[dict(scope=k[0],api=k[1],**v) for k,v in sync.items()],limits=['Exclusive excludes nested CDB labels only, still includes Torch/framework/profiler and unlabeled waits','GPU overlap reports recorded activities; does not prove physical idle or exact avoidable CPU costs','Do not add API wall time to kernel time or to parent inclusive time']),indent=2))
