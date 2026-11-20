from pathlib import Path
import collections,json,statistics,time,sys
R=Path(sys.argv[1]);W=Path('/home/yypan/CDB-AV/worktrees/cdb-host-critical-audit-20261120')
assert json.loads((R/'STATUS.json').read_text())['state']=='audit_complete'
rows=[]
for p in (R/'audit128/observer').glob('*_requests.jsonl'):
 for line in p.read_text().splitlines():
  x=json.loads(line)
  if x.get('kind')=='host_audit' and x.get('epoch')=='measurement':rows.append(x)
spans=collections.defaultdict(list);samples=[];events=[]
for x in rows:
 if 'wall_ns' in x:spans[x['stage']].append(x)
 elif x['stage']=='cpu_input_sample':samples.append(x['values'])
 elif x['stage']=='algorithm_event':events.append(x)
def desc(vals):
 vals=sorted(vals);n=len(vals)
 return dict(n=n,sum_ms=sum(vals)/1e6,mean_us=statistics.mean(vals)/1e3,p50_us=vals[n//2]/1e3,p95_us=vals[min(n-1,int(n*.95))]/1e3,max_us=max(vals)/1e3)
stats={k:dict(inclusive=desc([x['wall_ns'] for x in a]),unattributed=desc([x['exclusive_wall_ns'] for x in a])) for k,a in spans.items()}
sys.path.insert(0,str(W))
from vllm.v1.spec_decode.cdb_av_inputs import make_input_plan,encode_input_plan
assert samples
plans=[make_input_plan(**a) for a in samples]
buffers=[bytearray(p.nbytes) for p in plans]
def bench(fn):
 out=[]
 for _ in range(5):
  start=time.perf_counter_ns()
  for _ in range(20):fn()
  out.append((time.perf_counter_ns()-start)/(20*len(samples))/1e3)
 return dict(per_call_us=out,median_us=statistics.median(out))
planning=bench(lambda:[make_input_plan(**a) for a in samples])
def encoding():
 for p,b in zip(plans,buffers):encode_input_plan(p,b)
encoding_result=bench(encoding)
result=dict(scope='bounded normal-flags non-Profiler128-request observational audit; host spans are perturbed and cannot be treated as formal throughput',source=json.loads((R/'SOURCE.json').read_text())['commit'],measured_requests=128,stages=stats,cpu_replay=dict(samples=len(samples),plan=planning,encode=encoding_result,shape_counts=dict(collections.Counter(str((tuple(a['qlens']),max(map(len,a['blocks'])))) for a in samples))),caveats=['pack and forward include CUDA APIs/driver waits and cannot be classified as pure Python','unattributed parent includes hook overhead and unobserved/capped children; never sum nested spans','first4096 calls per stage per epoch are sampled; CPU snapshots cover first128 pack calls only','CPU replay of saved values measures host planning and encoding only, no GPU, no TPS claim'])
(R/'HOST_ANALYSIS.json').write_text(json.dumps(result,indent=2)+'\n')
# Independent algorithm evidence from this already-authorized observer, with
# request IDs/window/fingerprint joining and no change to stopping/publication.
ready={};verify={};idle=[];coalesce=[]
for x in events:
 key=(x.get('req_id'),x.get('window'),x.get('fingerprint'))
 if x['event']=='ready':assert key not in ready;ready[key]=x
 if x['event']=='p2_verified':assert key not in verify;verify[key]=x
 if x['event']=='p2_idle_step':idle.append(x)
 if x['event']=='p2_target_coalesce':coalesce.append(x)
matched=ready.keys()&verify.keys()
bins=collections.Counter();accepted=collections.Counter()
for k in matched:
 a,b=ready[k],verify[k]
 assert a['tokens_count']==b['drafts_count']
 if a['k_actual']==1 and a['reason']=='low_confidence':
  bound=next(v for v in [.2,.4,.6,.7,.75,.8,1.01] if a['confidence']<v)
  bins[str(bound)]+=1;accepted[str(bound)]+=b['accepted']
algo=dict(scope='post-coalescing instrumented128-request measurement interval, not throughput or counterfactual simulation',ready_events=len(ready),verified_events=len(verify),matched=len(matched),unmatched_ready=len(ready.keys()-verify.keys()),unmatched_verify=len(verify.keys()-ready.keys()),idle_barrier_events=len(idle),coalesce_events=len(coalesce),k1_low_confidence_counts=dict(bins),k1_low_confidence_accepted=dict(accepted),limits='Ungenerated continuation after a low-confidence stop is unknown. Idle barrier ticks issue no empty GPU forward. Do not infer equivalent GPU savings by deleting H ticks.')
(R/'ALGORITHM_OBSERVATION.json').write_text(json.dumps(algo,indent=2)+'\n')
print(json.dumps(result,indent=2));print(json.dumps(algo,indent=2))
