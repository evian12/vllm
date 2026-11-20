"""Profile only after the prescribed unprofiled pairs release their service."""
from pathlib import Path
import json
import os
import subprocess
import time
import traceback

B=Path('/home/yypan/CDB-AV')
R=Path(__file__).resolve().parent
PAIR=B/'results/draft_wait_20261120_v2'
PY=str(B/'.venv/bin/python')


def state(value,**fields):
    (R/'STATUS.json').write_text(json.dumps(dict(state=value,pid=os.getpid(),time=time.time(),**fields),indent=2)+'\n')


try:
    state('waiting_for_unprofiled_pairs')
    deadline=time.monotonic()+7200
    while time.monotonic()<deadline:
        status=json.loads((PAIR/'STATUS.json').read_text())
        if status['state']=='failed':
            raise RuntimeError('Unprofiled pairs failed; no profile service started')
        if status['state']=='completed':
            break
        os.kill(status['pid'],0)
        time.sleep(20)
    else:
        raise TimeoutError('Unprofiled pairs not complete')
    state('full_workload_bounded_trace')
    cmd=[PY,str(R/'profile_driver.py'),'--tag','full512_wait','--mode','hf','--k-cap','8','--requests','512','--seed','42']
    for dry in [True,False]:
        with (R/('dry_run.log' if dry else 'driver.log')).open('w') as log:
            subprocess.run(cmd+(['--dry-run'] if dry else []),cwd=R,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=2400)
    d=R/'full512_wait'
    release=json.loads((d/'released.json').read_text())
    assert release['port_closed'] and not release['our_gpu_pids']
    summary=json.loads((d/'summary.json').read_text())
    assert summary['completed']==512 and summary['failed']==0 and summary['output_tokens']==101003
    traces=list((d/'trace').glob('*.pt.trace.json.gz'))
    assert len(traces)==1
    with (R/'trace_analysis.log').open('w') as log:
        subprocess.run([PY,str(R/'analyze_trace.py'),str(traces[0]),'--output',str(d/'TRACE_SUMMARY.json')],cwd=R,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
    caches=list(d.glob('target_graph_cache_*.json'))
    assert len(caches)==1
    with (R/'action_analysis.log').open('w') as log:
        subprocess.run([PY,str(R/'analyze_actions.py'),str(traces[0]),str(caches[0]),str(d/'ACTION_SUMMARY.json')],cwd=R,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
    state('completed',release=release,trace=str(traces[0]),note='Profiler throughput is diagnostic; use unprofiled pairs for performance claims')
except BaseException:
    state('failed',error=traceback.format_exc())
    raise
