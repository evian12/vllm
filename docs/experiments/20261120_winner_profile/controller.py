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
W=B/'worktrees/cdb-draft-seq-bound-20261119'
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
    assert (B/'results/draft_meta_copy_20261120/COMPLETE.json').exists()
    assert (W/'vllm/v1/spec_decode/cdb_av_inputs.py').read_bytes()==(TEST_W/'vllm/v1/spec_decode/cdb_av_inputs.py').read_bytes()
    lock=open('/tmp/cdb_gpu3_test.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    ready,info=idle();assert ready,info;write('GPU_UNIT_PREFLIGHT.json',info)
    state('explicit_pinned_input_gpu_tests')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='3',PYTHONPATH=str(TEST_W),CDB_PINNED_GPU_TEST='1')
    with (R/'GPU_INPUT_TESTS.log').open('w') as out:
        p=subprocess.Popen([PY,'-m','pytest','-q','tests/v1/spec_decode/test_cdb_av_inputs.py'],cwd=TEST_W,env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
        try:rc=p.wait(timeout=180)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            raise
    for _ in range(15):
        ready,info=idle()
        if ready:break
        time.sleep(1)
    write('GPU_UNIT_RELEASED.json',dict(released=ready,**info));assert ready,info
    fcntl.flock(lock,fcntl.LOCK_UN)
    assert rc==0,rc
    assert 'skipped' not in (R/'GPU_INPUT_TESTS.log').read_text()
    state('bounded_profile')
    cmd=[PY,str(R/'profile_driver.py'),'--tag','winner_cap8','--mode','hf','--k-cap','8',
         '--diagnostic','--requests','16','--max-tokens','128']
    with (R/'dry_run.log').open('w') as out:subprocess.run(cmd+['--dry-run'],stdout=out,stderr=subprocess.STDOUT,check=True,timeout=30)
    with (R/'driver.log').open('w') as out:subprocess.run(cmd,stdout=out,stderr=subprocess.STDOUT,check=True,timeout=660)
    d=R/'winner_cap8';released=json.loads((d/'released.json').read_text())
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
