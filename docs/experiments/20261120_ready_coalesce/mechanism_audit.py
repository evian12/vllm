from pathlib import Path
import collections,json
B=Path('/home/yypan/CDB-AV/results');R=B/'ready_coalesce_20261120'

def counters(path):
 rows=json.loads(path.read_text());c=collections.Counter()
 for row in rows:
  for k,v in row.items():
   if k.startswith('cdb_') and isinstance(v,(int,float)):c[k]+=v
 return c

def extract(path):
 before,after=counters(path/'observer_before.json'),counters(path/'observer_after.json')
 keys=['cdb_target_schedules','cdb_target_verify_rows','cdb_target_prefill_rows','cdb_target_ar_rows','cdb_draft_dispatches','cdb_draft_request_rows','cdb_draft_token_rows','cdb_target_calls_with_verify','cdb_ready_wait_sum_ms','cdb_ready_wait_count','cdb_target_token_rows']
 d={k:after[k]-before[k] for k in keys}
 calls=d['cdb_target_schedules'];v=d['cdb_target_verify_rows']
 def prom(name):
  def get(file):
   return sum(float(line.rsplit(' ',1)[1]) for line in file.read_text().splitlines() if line.startswith('vllm:'+name+'{'))
  return get(path/'metrics_after.prom')-get(path/'metrics_before.prom')
 d['accepted_candidates']=prom('spec_decode_num_accepted_tokens_total')
 d['draft_candidates']=prom('spec_decode_num_draft_tokens_total')
 d['accepted_per_verify']=d['accepted_candidates']/v
 d['mean_verified_K']=d['draft_candidates']/v
 d['target_calls_per_1k_output']=calls/101003*1000
 d['draft_calls_per_1k_output']=d['cdb_draft_dispatches']/101003*1000
 d['verify_rows_per_target']=v/calls
 d['ready_to_target_schedule_ms']=d['cdb_ready_wait_sum_ms']/d['cdb_ready_wait_count']
 return d


baseline=[extract(B/f'draft_wait_20261120_v2/pair_seed{seed}_draft_wait1') for seed in [42,43,44]]
state=json.loads((R/'STATUS.json').read_text())
runs=[extract(Path(x['directory'])) for x in state['results']]
on={k:sum(x[k] for x in runs)/len(runs) for k in runs[0]}
mean={k:sum(x[k] for x in baseline)/3 for k in baseline[0]}
changes={k:100*(on[k]/mean[k]-1) if mean[k] else None for k in on}
a=dict(scope='ON mean vs three historical reference runs; mechanism counts, no paired causal attribution',on_runs=runs,on=on,historical_mean=mean,change_pct=changes)
(R/'MECHANISM_AUDIT.json').write_text(json.dumps(a,indent=2)+'\n');print(json.dumps(a,indent=2))
