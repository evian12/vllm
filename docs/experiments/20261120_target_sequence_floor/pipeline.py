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
W=B/'worktrees/cdb-target-seq-floor-20261120'
PY=str(B/'.venv/bin/python')
env=dict(os.environ,PATH=str(B/'.venv/bin')+':'+os.environ['PATH'],
         PYTHONPATH=str(W),CUDA_VISIBLE_DEVICES='',GIT_SSH_COMMAND='ssh -o BatchMode=yes -o ConnectTimeout=15')


def write(name,x):(R/name).write_text(json.dumps(x,indent=2)+'\n')
def state(s,**x):write('STAGE.json',dict(state=s,pid=os.getpid(),time=time.time(),**x))
def run(cmd,log,timeout=240):
    with (R/log).open('w') as out:subprocess.run(cmd,cwd=W,env=env,stdout=out,stderr=subprocess.STDOUT,check=True,timeout=timeout)


try:
    profile=B/'results/llama_winner_profile_20261120'
    assert json.loads((profile/'STATUS.json').read_text())['state']=='completed'
    action=profile/'winner_cap8/ACTION_SUMMARY.json'
    evidence=json.loads(action.read_text())
    assert evidence['cache']['captured']==24 and evidence['cache']['unique_domains_without_sequence_bound']==10
    write('RATIONALE.json',dict(evidence=str(action),sha256=hashlib.sha256(action.read_bytes()).hexdigest(),
          observed_cache=evidence['cache'],purpose='Reduce duplicate sequence-bound keys. Existing runtime switch only; no policy/H/K changes.'))
    state('cpu_validation')
    tests=['tests/v1/spec_decode/test_cdb_av_target_sequence_floor.py',
           'tests/v1/spec_decode/test_cdb_av_target_graph.py',
           'tests/v1/spec_decode/test_cdb_av_target_graph_llama.py',
           'tests/v1/spec_decode/test_cdb_av_target_bound_reuse.py',
           'tests/v1/spec_decode/test_cdb_av_ragged_sequence_floor.py',
           'tests/v1/spec_decode/test_cdb_av_kv_reuse.py',
           'tests/v1/spec_decode/test_cdb_av_kv_reuse_ownership.py',
           'tests/v1/spec_decode/test_cdb_av_persistent_queue.py']
    run([PY,'-m','pytest','-q',*tests],'CPU_TESTS.log')
    run(['bash','scripts/validate_changes.sh','--pytest',tests[0]],'validate.log')
    run(['bash','scripts/git_checkpoint.sh','--message','test: cover fixed Target sequence bound metadata invariants',
         '--path',tests[0],'--pytest',tests[0]],'checkpoint.log',180)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip()
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=W,text=True).strip()
    remote=subprocess.check_output(['git','ls-remote','--heads','myfork','refs/heads/'+branch],cwd=W,env=env,text=True).split()[0]
    assert commit==remote and not subprocess.check_output(['git','status','--porcelain'],cwd=W,text=True)
    base=json.loads((B/'results/draft_seq_bound_20261119/SOURCE.json').read_text())
    files=list(base['hashes'])+[tests[0],'vllm/v1/worker/gpu_model_runner.py']
    hashes={p:hashlib.sha256((W/p).read_bytes()).hexdigest() for p in sorted(set(files))}
    # Runtime Python is byte-identical to the validated Draft-floor parent.
    for p,expected in base['hashes'].items():assert hashes[p]==expected
    source=dict(source=str(W),worktree=str(W),commit=commit,branch=branch,remote_head=remote,
                parent='008c04f89f319b51b26187e81f93406b2322b9c9',hashes=hashes,runtime_code_changed=False)
    write('SOURCE.json',source)
    plan=dict(source=source,H=8,HF_threshold=.8,k_cap=8,requests=512,client_concurrency=16,
              max_num_seqs=4,max_model_len=1024,max_num_batched_tokens=1024,
              target='/disk14T/yypan/models/Llama-3.1-8B-Instruct',
              draft='/disk14T/yypan/CDB-AV-runtime/baseline-comparison-20261113/models/Llama-3.2-1B-Instruct',
              GPU=3,port=18084,dtype='bfloat16',attention_backend='FLASH_ATTN/FA2',
              independent_variable='CDB_AV_TARGET_GRAPH_SEQ_FLOOR',values=[0,1024],
              fixed_draft_sequence_floor=1024,other_experimental_optimizations=0,
              repetitions=3,seeds=[42,43,44],order=[[0,1024],[1024,0],[0,1024]],
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
