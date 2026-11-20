from pathlib import Path
import json,collections
R=Path('/home/yypan/CDB-AV/results/ready_coalesce_20261120');text=(R/'gate_coalesce/server.log').read_text()
events=[json.loads(l.split('[CDB_REFILL] ',1)[1]) for l in text.splitlines() if '[CDB_REFILL] ' in l]
deferred={};done=collections.Counter();delays=[];hist=collections.Counter();unsafe=[];open_window=None;step=0
for e in events:
 ev=e['event']
 if ev=='p2_window_begin':open_window=e['window'];step=0
 elif ev=='p2_draft_step':step=e['step']
 elif ev=='p2_target_coalesce':
  assert e['window']==open_window and 0<e['step']<8
  deferred.setdefault(e['window'],set()).update(e['published'])
 elif ev=='p2_boundary':
  if e['window'] in deferred:
   assert deferred[e['window']]<=set(e['ready']), 'Deferred published work lost before boundary'
   done['deferred_windows']+=1;done['deferred_rows_at_boundary']+=len(deferred[e['window']])
  open_window=None
 elif ev=='p2_target_begin' and e['scheduled']:
  hist[len(e['scheduled'])]+=1
  if open_window in deferred and set(e['verify'])&deferred[open_window]:
   # A request arrival, cancellation or changed state may legitimately restore
   # original scheduling before H. Report, never label automatically invalid.
   done['early_fallback_calls']+=1
assert done['deferred_windows']==len(deferred)
a=dict(scope='64warmup + 16 short functional gate, not formal workload or TPS',target_row_hist=dict(hist),**done)
(R/'COALESCE_GATE_STATE_AUDIT.json').write_text(json.dumps(a,indent=2)+'\n');print(json.dumps(a,indent=2))
