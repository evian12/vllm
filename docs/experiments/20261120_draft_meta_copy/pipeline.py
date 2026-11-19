"""Wait for current paired test; validate/checkpoint isolated next candidate.

No signals or changes to the running comparison. GPU work is serialized by the
same shared lock; the new driver cleans only its own service after each test.
"""
from pathlib import Path
import fcntl
import hashlib
import json
import os
import signal
import socket
import subprocess
import time
import traceback

R=Path(__file__).resolve().parent
B=Path('/home/yypan/CDB-AV')
W=B/'worktrees/cdb-draft-meta-copy-20261120'
P=B/'results/spec_metadata_joint_20261120'
PY=str(B/'.venv/bin/python')
env=dict(os.environ,PATH=str(B/'.venv/bin')+':'+os.environ['PATH'],
         PYTHONPATH=str(R)+':'+str(W),CUDA_VISIBLE_DEVICES='3',
         GIT_SSH_COMMAND='ssh -o BatchMode=yes -o ConnectTimeout=15')
UUID='GPU-b493bb52-b935-1cf4-47fe-7869125775b9'


def write(name,obj):
    p=R/name;t=p.with_suffix('.tmp')
    t.write_text(json.dumps(obj,indent=2)+'\n');t.replace(p)


def state(s,**kw):write('STAGE.json',dict(state=s,time=time.time(),pid=os.getpid(),**kw))


def run(cmd,log,timeout=180):
    with (R/log).open('w') as out:
        subprocess.run(cmd,cwd=W,env=env,stdout=out,stderr=subprocess.STDOUT,
                       check=True,timeout=timeout)


def idle():
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True)
    memory=int(subprocess.check_output(['nvidia-smi','--id=3','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True))
    with socket.socket() as s:closed=s.connect_ex(('127.0.0.1',18084))!=0
    return UUID not in apps and memory<128 and closed,dict(memory_mib=memory,apps=apps,port_closed=closed)


lock=(R/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
primitive=None
try:
    state('waiting_for_metadata_comparison')
    deadline=time.monotonic()+7200
    while True:
        p=json.loads((P/'STATUS.json').read_text())
        if p['state']=='failed':raise RuntimeError('Previous paired test failed; do not advance')
        if p['state']=='completed':break
        if time.monotonic()>deadline:raise TimeoutError('Previous test still running; no GPU started')
        time.sleep(15)
    assert (P/'COMPLETE.json').exists()
    # Analyze the earlier comparison before preparing another GPU candidate.
    run([PY,str(P/'analyze.py')],'previous_analysis.log',60)
    state('cpu_validation')
    changed=subprocess.check_output(['git','status','--porcelain'],cwd=W,text=True)
    write('PRE_CPU_STATUS.json',dict(status=changed,head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip()))
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip()=='008c04f89f319b51b26187e81f93406b2322b9c9'
    tests=['tests/v1/spec_decode/test_cdb_av_draft_meta_copy.py',
           'tests/v1/spec_decode/test_cdb_av_ragged_sequence_floor.py',
           'tests/v1/spec_decode/test_cdb_av_draft_decode_graph.py',
           'tests/v1/spec_decode/test_cdb_av_config.py',
           'tests/v1/spec_decode/test_cdb_av_inputs.py',
           'tests/v1/spec_decode/test_cdb_av_kv_reuse.py',
           'tests/v1/spec_decode/test_cdb_av_kv_reuse_ownership.py',
           'tests/v1/spec_decode/test_cdb_av_persistent_queue.py']
    run([PY,'-m','pytest','-q',*tests],'cpu_tests.log',240)
    run(['bash','scripts/validate_changes.sh','--pytest',tests[0]],'validate.log',240)
    paths=['vllm/v1/spec_decode/cdb_av_draft_meta_copy.py',
           'vllm/v1/spec_decode/cdb_av_refill.py',tests[0]]
    run(['git','diff','--',paths[1]],'refill.diff',30)
    for p in [paths[0],paths[2]]:
        # New files are reviewed as exact snapshots, not omitted by git diff.
        write('SNAPSHOT_'+Path(p).name+'.json',dict(path=p,sha256=hashlib.sha256((W/p).read_bytes()).hexdigest(),text=(W/p).read_text()))
    cmd=['bash','scripts/git_checkpoint.sh','--message','perf: gate integer-only fused Draft Graph metadata copy','--pytest',tests[0]]
    for p in paths:cmd+=['--path',p]
    run(cmd,'checkpoint.log',180)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip()
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=W,text=True).strip()
    remote=subprocess.check_output(['git','ls-remote','--heads','myfork','refs/heads/'+branch],cwd=W,env=env,text=True).split()[0]
    assert commit==remote
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=W,text=True)
    old=json.loads((B/'results/draft_seq_bound_20261119/SOURCE.json').read_text())
    tracked=list(old['hashes'])+paths
    hashes={p:hashlib.sha256((W/p).read_bytes()).hexdigest() for p in sorted(set(tracked))}
    source=dict(source=str(W),worktree=str(W),commit=commit,branch=branch,
                remote_head=remote,parent='008c04f89f319b51b26187e81f93406b2322b9c9',hashes=hashes)
    write('SOURCE.json',source)
    plan=dict(source=source,family='llama',GPU=3,port=18084,H=8,k_cap=8,
              HF_threshold=0.8,max_num_seqs=4,max_model_len=1024,
              max_num_batched_tokens=1024,requests=512,client_concurrency=16,
              seeds=[42,43,44],warmup_requests=128,warmup_output_tokens=128,
              output_tokens_per_formal_run=101003,order=[[0,1],[1,0],[0,1]],
              candidate='Fused integer-only copy into existing Draft Graph metadata buffers',
              independent_flag='CDB_AV_DRAFT_META_COPY',fixed_draft_sequence_floor=1024,
              target_spec_metadata_packing=0,request_policy_and_model_math_changes=False,
              client_sha256=hashlib.sha256((R/'benchmark.py').read_bytes()).hexdigest(),
              observer_sha256=hashlib.sha256((R/'manual_observer.py').read_bytes()).hexdigest(),
              data_sha256=hashlib.sha256((R/'sharegpt.jsonl').read_bytes()).hexdigest(),
              target='/disk14T/yypan/models/Llama-3.1-8B-Instruct',
              draft='/disk14T/yypan/CDB-AV-runtime/baseline-comparison-20261113/models/Llama-3.2-1B-Instruct')
    write('PLAN.json',plan)
    write('CPU_ACCEPTED.json',dict(commit=commit,remote_head=remote,tests=tests))
    state('waiting_for_gpu3_primitive')
    deadline=time.monotonic()+7200
    shared=open('/tmp/cdb_gpu3_test.lock','a')
    while True:
        try:
            fcntl.flock(shared,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            if time.monotonic()>deadline:raise TimeoutError('Shared GPU3 test lock unavailable')
            time.sleep(10);continue
        ready,info=idle()
        if ready:break
        fcntl.flock(shared,fcntl.LOCK_UN)
        if time.monotonic()>deadline:raise TimeoutError('GPU3/18084 not free; no foreign job stopped')
        time.sleep(10)
    write('PRIMITIVE_PREFLIGHT.json',info)
    state('gpu_primitive')
    try:
        with (R/'gpu_primitive.log').open('w') as out:
            primitive=subprocess.Popen([PY,str(R/'gpu_primitive.py')],cwd=W,env=env,
                                        stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
            try:rc=primitive.wait(timeout=180)
            except subprocess.TimeoutExpired:
                os.killpg(primitive.pid,signal.SIGTERM)
                try:primitive.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(primitive.pid,signal.SIGKILL);primitive.wait()
                raise
            assert rc==0,rc
    finally:
        for attempt in range(15):
            ready,info=idle()
            if ready:break
            time.sleep(1)
        write('PRIMITIVE_RELEASED.json',dict(released=ready,**info))
        fcntl.flock(shared,fcntl.LOCK_UN)
    assert ready,info
    state('starting_gpu_gate_and_pairs',commit=commit)
    with (R/'controller.log').open('w') as out:
        proc=subprocess.Popen(['python3',str(R/'controller.py')],cwd=R,stdout=out,
                              stderr=subprocess.STDOUT,start_new_session=True)
    (R/'controller.pid').write_text(str(proc.pid)+'\n')
    state('gpu_controller_started',controller_pid=proc.pid,commit=commit)
except BaseException:
    state('failed',error=traceback.format_exc())
    raise
finally:
    fcntl.flock(lock,fcntl.LOCK_UN)
