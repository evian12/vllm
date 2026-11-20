from pathlib import Path
import json,subprocess,sys,time,hashlib,os,traceback
R=Path('/home/yypan/CDB-AV/results/ready_coalesce_20261120');PY='/home/yypan/CDB-AV/.venv/bin/python'
seed=int(sys.argv[1]);assert seed in [43,44];tag=f'confirm_seed{seed}_on'
state=json.loads((R/'STATUS.json').read_text());assert state['state']=='completed'
old=state.copy();(R/f'STATUS_before_seed{seed}.json').write_text(json.dumps(old,indent=2)+'\n')
assert old['results'][0]['tps']>=312.56473125804325*1.005
state.update(state='confirming',tag=tag,seed=seed,pid=os.getpid(),time=time.time());(R/'STATUS.json').write_text(json.dumps(state,indent=2)+'\n')
cmd=[PY,str(R/'driver.py'),'--tag',tag,'--mode','hf','--k-cap','8','--seed',str(seed),'--ready-coalesce','1']
try:
 for dry in [True,False]:
  with (R/(tag+('_dry_run.log' if dry else '_controller.log'))).open('x') as f:
   subprocess.run(cmd+(['--dry-run'] if dry else []),cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=2700)
 d=R/tag;release=json.loads((d/'released.json').read_text());assert release['port_closed'] and not release['our_gpu_pids']
 text=(d/'server.log').read_text();assert all(x not in text for x in ['Traceback (most recent call last)','CUDA error:','Cross-request draft KV slot collision'])
 summary=json.loads((d/'summary.json').read_text());assert summary['completed']==512 and summary['failed']==0 and summary['output_tokens']==101003
 result=dict(directory=str(d),cap=8,seed=seed,ready_coalesce=1,tps=summary['throughput_tok_s'],e2e_p95=summary['latency_s']['e2e']['p95'],ttft_p95=summary['latency_s']['ttft']['p95'])
 state.update(state='completed',results=old['results']+[result],time=time.time())
 (R/'RESULTS_PARTIAL.json').write_text(json.dumps(state['results'],indent=2)+'\n')
except BaseException:
 state.update(state='failed',error=traceback.format_exc(),time=time.time());raise
finally:
 (R/'STATUS.json').write_text(json.dumps(state,indent=2)+'\n')
