from pathlib import Path
import gzip,json,collections,hashlib
p=next(Path('/home/yypan/CDB-AV/results/draft_wait_profile_20261120/full512_wait/trace').glob('*.pt.trace.json.gz'))
with gzip.open(p,'rt') as f:events=json.load(f)['traceEvents']
r=Path('/home/yypan/CDB-AV/results/draft_wait_20261120_v2')
a=json.loads((r/'ANALYSIS.json').read_text());rows=[x for x in a['runs'] if x['draft_wait']==1]
counts=collections.Counter();graphs=collections.Counter()
for row in rows:
 counts.update(row['interior']['token_counts']);graphs.update(row['interior']['target_graph'])
# Record exact source excerpts covering all major stage boundaries.
W=Path('/home/yypan/CDB-AV/worktrees/cdb-draft-wait-20261120');source={}
for name,lo,hi in [('vllm/v1/spec_decode/cdb_av_inputs.py',135,177),('vllm/v1/spec_decode/cdb_av_refill.py',361,394),('vllm/v1/spec_decode/cdb_av_refill.py',420,432),('vllm/v1/spec_decode/cdb_av_refill.py',510,530),('vllm/v1/spec_decode/cdb_av_refill.py',848,894),('vllm/v1/worker/gpu_model_runner.py',3564,3597),('vllm/v1/spec_decode/cdb_av_persistent.py',1190,1213),('vllm/v1/spec_decode/cdb_av_persistent.py',1315,1352),('vllm/v1/engine/core.py',429,448)]:
 lines=(W/name).read_text().splitlines();source[f'{name}:{lo}']=dict(sha256=hashlib.sha256((W/name).read_bytes()).hexdigest(),text='\n'.join(f'{i+1}: {lines[i]}' for i in range(lo-1,hi)))
mem=collections.defaultdict(lambda:dict(count=0,gpu_us=0,bytes=0))
examples=[]
for e in events:
 if e.get('cat') not in ['gpu_memcpy','gpu_memset'] or e.get('ph')!='X':continue
 g=mem[e['name']];g['count']+=1;g['gpu_us']+=e['dur'];g['bytes']+=e.get('args',{}).get('bytes',0)
 if len(examples)<2:examples.append(e.get('args',{}))
print(json.dumps(dict(token_counts_interior=dict(counts),reuse_fraction=counts['reused_tokens']/(counts['reused_tokens']+counts['prefill_tokens']),target_graph_interior=dict(graphs),memory_events=dict(mem),memory_arg_examples=examples,source_excerpts=source,source_commit='73d37c98fb3ad44709a5a4d343c78d6944aa8490',limits=['Formal log interior counters omit tails; not whole-run token conservation','Memory event time GPU-side, API wait must be counted separately','Source excerpts prove paths and guards, not standalone speedup']),indent=2))
