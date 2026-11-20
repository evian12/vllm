"""One GPU input regression plus one bounded profile of the measured winner."""
from pathlib import Path
import fcntl
import json
import os
import signal
import socket
import subprocess
import time
import traceback

R=Path(__file__).resolve().parent
B=Path('/home/yypan/CDB-AV')
W=B/'worktrees/cdb-context-fp-cache-20261120'
TEST_W=B/'worktrees/cdb-draft-meta-copy-20261120'
PY=str(B/'.venv/bin/python')
UUID='GPU-b493bb52-b935-1cf4-47fe-7869125775b9'


def write(name,x):(R/name).write_text(json.dumps(x,indent=2)+'\n')
def state(s,**x):write('STATUS.json',dict(state=s,pid=os.getpid(),time=time.time(),**x))


def idle():
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True)
    memory=int(subprocess.check_output(['nvidia-smi','--id=3','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True))
    with socket.socket() as s:closed=s.connect_ex(('127.0.0.1',18084))!=0
    return UUID not in apps and memory<128 and closed,dict(apps=apps,memory_mib=memory,port_closed=closed)


try:
    assert json.loads((B/'results/context_fp_cache_20261120/STATUS.json').read_text())['state']=='completed'
    state('bounded_profile')
    cmd=[PY,str(R/'profile_driver.py'),'--tag','post_cache_cap8','--mode','hf','--k-cap','8',
         '--diagnostic','--requests','16','--max-tokens','128']
    with (R/'dry_run.log').open('w') as out:subprocess.run(cmd+['--dry-run'],stdout=out,stderr=subprocess.STDOUT,check=True,timeout=30)
    with (R/'driver.log').open('w') as out:subprocess.run(cmd,stdout=out,stderr=subprocess.STDOUT,check=True,timeout=660)
    d=R/'post_cache_cap8';released=json.loads((d/'released.json').read_text())
    assert released['port_closed'] and not released['our_gpu_pids']
    traces=list((d/'trace').glob('*.pt.trace.json.gz'));assert len(traces)==1
    facts=list(d.glob('executor_runtime_*.json'));assert len(facts)==1
    executor=json.loads(facts[0].read_text())
    assert executor['executor']=='vllm.v1.executor.uniproc_executor.UniProcExecutor' and executor['backend']=='uni'
    caches=list(d.glob('target_graph_cache_*.json'));assert len(caches)==1
    subprocess.run([PY,str(R/'analyze_trace.py'),str(traces[0]),'--output',str(d/'TRACE_SUMMARY.json')],stdout=subprocess.DEVNULL,check=True,timeout=120)
    subprocess.run([PY,str(R/'analyze_actions.py'),str(traces[0]),str(caches[0]),str(d/'ACTION_SUMMARY.json')],check=True,timeout=120)
    state('completed',traces=[str(p) for p in traces],release=released,executor=executor)
except BaseException:
    state('failed',error=traceback.format_exc())
    raise
