from pathlib import Path
import json,collections
R=Path('/home/yypan/CDB-AV/results/host_critical_audit_20261120_retry1')
ready={};verify={}
for f in (R/'audit128/observer').glob('*_requests.jsonl'):
 for line in f.read_text().splitlines():
  x=json.loads(line)
  if x.get('kind')!='host_audit' or x.get('epoch')!='measurement' or x.get('stage')!='algorithm_event':continue
  key=(x.get('req_id'),x.get('window'),x.get('fingerprint'))
  if x['event']=='ready':ready[key]=x
  if x['event']=='p2_verified':verify[key]=x
assert ready.keys()==verify.keys()
ks=collections.Counter();allaccepted=0;eligible=[]
for k,a in ready.items():
 if a['reason']=='low_confidence' and .7<=a['confidence']<.8 and a['k_actual']<a['k_cap']:
  b=verify[k];ks[a['k_actual']]+=1
  allaccepted+=b['accepted']==a['k_actual'];eligible.append(a)
x=dict(scope='same instrumented128-request post-coalescing audit; no new GPU experiment',matched_verification_rows=len(verify),hypothetical_HF_0_7_first_changed_stop_rows=len(eligible),fraction_of_verify_rows=len(eligible)/len(verify),by_existing_K=dict(ks),current_candidate_allaccepted_rows=allaccepted,meaning='At these already-observed stopping states, threshold0.7 would permit at least one more Draft step. Existing candidate acceptance is known; additional candidate token, its acceptance, later scheduling and total work are unknown.',decision='A02 threshold0.7 is a possible independent algorithm candidate after historical de-dup; keep H8/cap8/Ready coalescing fixed. Require unchanged guards and actual-model gate, then one ON only. Never treat this observational count as predicted saved Target calls or throughput.',system_line='S03 audit completed independently; next S04 audit must proceed and cannot be replaced by threshold search')
print(json.dumps(x,indent=2));(R/'A02_STOP_AUDIT.json').write_text(json.dumps(x,indent=2)+'\n')
