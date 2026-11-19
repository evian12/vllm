"""Validate existing fixed Target sequence-bound configuration, then GPU gate."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import time
import traceback

B=Path('/home/yypan/CDB-AV')
R=Path(__file__).resolve().parent
W=B/'worktrees/cdb-context-fp-cache-20261120'
PY=str(B/'.venv/bin/python')
env=dict(os.environ,PATH=str(B/'.venv/bin')+':'+os.environ['PATH'],
         PYTHONPATH=str(W),CUDA_VISIBLE_DEVICES='',CDB_AV_FINGERPRINT_BATCH='1',GIT_SSH_COMMAND='ssh -o BatchMode=yes -o ConnectTimeout=15')


def write(name,x):(R/name).write_text(json.dumps(x,indent=2)+'\n')
def state(s,**x):write('STAGE.json',dict(state=s,pid=os.getpid(),time=time.time(),**x))
def run(cmd,log,timeout=240):
    with (R/log).open('w') as out:subprocess.run(cmd,cwd=W,env=env,stdout=out,stderr=subprocess.STDOUT,check=True,timeout=timeout)


try:
    previous=B/'results/target_seq_floor_20261120'
    assert json.loads((previous/'STATUS.json').read_text())['state']=='completed'
    analysis=json.loads((previous/'ANALYSIS.json').read_text())
    assert len(analysis['paired'])==3 and all(x['tps_change_pct']>0 for x in analysis['paired'])
    assert analysis['groups']['1024']['mean_run_e2e_p95']<=analysis['groups']['0']['mean_run_e2e_p95']
    write('RATIONALE.json',dict(previous=str(previous/'ANALYSIS.json'),fixed_target_floor=1024,
          throughput_tradeoff='Prior matched3pairs+1.063%;E2EP95flat,TTFTP95+16.7ms disclosed',
          candidate='Memoized original digest only after COMPLETE typed immutable prefix equality'))
    state('cpu_validation')
    tests=['tests/v1/spec_decode/test_cdb_av_context_cache.py',
           'tests/v1/spec_decode/test_cdb_av_context_wire.py',
           'tests/v1/spec_decode/test_cdb_av_fingerprint_batch.py',
           'tests/v1/spec_decode/test_cdb_av_persistent_queue.py',
           'tests/v1/spec_decode/test_cdb_av_kv_reuse.py',
           'tests/v1/spec_decode/test_cdb_av_kv_reuse_ownership.py',
           'tests/v1/spec_decode/test_cdb_av_target_graph.py',
           'tests/v1/spec_decode/test_cdb_av_target_sequence_floor.py']
    run([PY,'-m','pytest','-q',*tests],'CPU_TESTS.log')
    run(['bash','scripts/validate_changes.sh','--pytest',tests[0],'--pytest',tests[1]],'validate.log')
    paths=['vllm/v1/spec_decode/cdb_av_context_cache.py',
           'vllm/v1/spec_decode/cdb_av_refill_queue.py',tests[0],tests[1]]
    cmd=['bash','scripts/git_checkpoint.sh','--message','perf: gate exact-prefix digest memoization without skipping frozen checks',
         '--pytest',tests[0],'--pytest',tests[1]]
    for p in paths:cmd+=['--path',p]
    run(cmd,'checkpoint.log',180)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip()
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=W,text=True).strip()
    remote=subprocess.check_output(['git','ls-remote','--heads','myfork','refs/heads/'+branch],cwd=W,env=env,text=True).split()[0]
    assert commit==remote and not subprocess.check_output(['git','status','--porcelain'],cwd=W,text=True)
    base=json.loads((B/'results/target_seq_floor_20261120/SOURCE.json').read_text())
    files=list(base['hashes'])+paths+['vllm/v1/spec_decode/cdb_av_fingerprint.py']
    hashes={p:hashlib.sha256((W/p).read_bytes()).hexdigest() for p in sorted(set(files))}
    # Runtime Python is byte-identical to the validated Draft-floor parent.
    for p,expected in base['hashes'].items():assert hashes[p]==expected
    source=dict(source=str(W),worktree=str(W),commit=commit,branch=branch,remote_head=remote,
                parent='23eaa57d50af9c740088d0f375acc59c3cf2474d',hashes=hashes,runtime_code_changed=True)
    write('SOURCE.json',source)
    plan=dict(source=source,H=8,HF_threshold=.8,k_cap=8,requests=512,client_concurrency=16,
              max_num_seqs=4,max_model_len=1024,max_num_batched_tokens=1024,
              target='/disk14T/yypan/models/Llama-3.1-8B-Instruct',
              draft='/disk14T/yypan/CDB-AV-runtime/baseline-comparison-20261113/models/Llama-3.2-1B-Instruct',
              GPU=3,port=18084,dtype='bfloat16',attention_backend='FLASH_ATTN/FA2',
              independent_variable='CDB_AV_CONTEXT_FINGERPRINT_CACHE',values=[0,1],
              fixed_draft_sequence_floor=1024,fixed_target_sequence_floor=1024,other_experimental_optimizations=0,
              repetitions=3,seeds=[42,43,44],order=[[0,1],[1,0],[0,1]],
              warmup_requests=128,warmup_output_tokens=128,measured_output_tokens_per_run=101003,
              client_sha256=hashlib.sha256((R/'benchmark.py').read_bytes()).hexdigest(),
              observer_sha256=hashlib.sha256((R/'manual_observer.py').read_bytes()).hexdigest(),
              data_sha256=hashlib.sha256((R/'sharegpt.jsonl').read_bytes()).hexdigest())
    write('PLAN.json',plan);write('CPU_ACCEPTED.json',dict(source=source,tests=tests))
    with (R/'controller.log').open('w') as out:
        p=subprocess.Popen(['python3',str(R/'controller.py')],cwd=R,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
    (R/'controller.pid').write_text(str(p.pid)+'\n')
    state('gpu_controller_started',controller_pid=p.pid,commit=commit)
except BaseException:
    state('failed',error=traceback.format_exc());raise
