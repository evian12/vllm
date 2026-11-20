from pathlib import Path
import json,hashlib
w=Path('/home/yypan/CDB-AV/worktrees/cdb-draft-wait-20261120');out={}
for n,lo,hi in [('vllm/v1/sample/sampler.py',64,116),('vllm/v1/sample/rejection_sampler.py',92,164),('vllm/v1/sample/rejection_sampler.py',376,402),('vllm/v1/sample/rejection_sampler.py',443,468),('vllm/v1/spec_decode/cdb_av_target_graph.py',171,181),('vllm/v1/spec_decode/cdb_av_persistent.py',1531,1558),('vllm/v1/executor/uniproc_executor.py',55,79)]:
 p=w/n;lines=p.read_text().splitlines();hi=min(hi,len(lines));out[f'{n}:{lo}']=dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),text='\n'.join(f'{i+1}: {lines[i]}' for i in range(lo-1,hi)))
print(json.dumps(out,indent=2))
