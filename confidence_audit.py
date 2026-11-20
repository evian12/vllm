"""Read-only post-coalescing confidence/yield audit; no counterfactual TPS."""
import collections,json,hashlib
from pathlib import Path
P=Path('/home/yypan/CDB-AV/results/ready_coalesce_20261120/gate_coalesce/server.log')
ready={};verified={}
for line in P.read_text().splitlines():
 if '[CDB_REFILL] ' not in line:continue
 x=json.loads(line.split('[CDB_REFILL] ',1)[1])
 key=(x.get('req_id'),x.get('window'),x.get('fingerprint'))
 if x['event']=='ready':
  assert key not in ready;ready[key]=x
 if x['event']=='p2_verified':
  assert key not in verified;verified[key]=x
assert ready.keys()==verified.keys()
bins={};rows=0;traces=set()
for key,v in verified.items():
 if not v['trace_key'].startswith('measurement:'):continue
 x=ready[key];rows+=1;traces.add(v['trace_key'])
 assert x['tokens']==v['drafts'] and len(v['drafts'])==x['k_actual']
 if x['k_actual']!=1 or x['reason']!='low_confidence':continue
 conf=x['confidence'];bound=next(b for b in [.2,.4,.6,.7,.75,.8,1.01] if conf<b)
 b=bins.setdefault(str(bound),dict(rows=0,accepted=0,generated=0))
 b['rows']+=1;b['accepted']+=v['accepted'];b['generated']+=len(v['generated'])
result=dict(scope='saved accepted Ready coalescing functional gate, measured16 short requests only; not formal512 or a throughput trial',source_sha='ccf96aac0064aebebc2077fb47e36929d65aa06a',input=str(P),input_sha256=hashlib.sha256(P.read_bytes()).hexdigest(),matched_all_events=len(verified),measured_traces=len(traces),measured_verification_rows=rows,k1_low_confidence_by_exclusive_upper_bound=bins,counterfactual_limit='The stopped token exists and is verified; tokens after stopping were not generated. No prediction of longer-K acceptance, GPU work or TPS is justified. Threshold must remain0.8 until independently registered algorithm trial.')
print(json.dumps(result,indent=2))
