"""Llama C16 exact formal-prefix digest cache gate and three paired repeats."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

R = Path(__file__).resolve().parent
W = Path('/home/yypan/CDB-AV/worktrees/cdb-draft-chain-20261120')
PY = '/home/yypan/CDB-AV/.venv/bin/python'


def write(name, data):
    path = R / name
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)


def state(value, **data):
    write('STATUS.json', dict(state=value, time=time.time(), pid=os.getpid(), **data))


def wait_gpu(tag):
    deadline = time.monotonic() + 7200
    while True:
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader'], text=True)
        memory = int(subprocess.check_output(['nvidia-smi', '--id=3', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True))
        if 'GPU-b493bb52-b935-1cf4-47fe-7869125775b9' not in apps and memory < 128:
            return
        state('waiting_for_gpu3', next=tag, memory_mib=memory)
        if time.monotonic() >= deadline:
            raise RuntimeError('GPU3 unavailable; no foreign process terminated')
        time.sleep(10)


def execute(tag, cap, seed=42, diagnostic=False, floor=1):
    wait_gpu(tag)
    state('diagnostic' if diagnostic else 'running', tag=tag, cap=cap, seed=seed)
    cmd = [PY, str(R / 'driver.py'), '--tag', tag, '--mode', 'hf', '--k-cap', str(cap), '--seed', str(seed), '--chain-graph', str(floor)]
    if diagnostic:
        cmd += ['--diagnostic', '--requests', '16', '--max-tokens', '64', '--gate-warmup', '64']
    for dry in [True, False]:
        with (R / (tag + ('_dry_run.log' if dry else '_controller.log'))).open('w') as output:
            subprocess.run(cmd + (['--dry-run'] if dry else []), cwd=R, stdout=output, stderr=subprocess.STDOUT, check=True, timeout=2700)
    directory = R / tag
    released = json.loads((directory / 'released.json').read_text())
    assert released['port_closed'] and not released['our_gpu_pids']
    text = (directory / 'server.log').read_text()
    assert all(x not in text for x in ['Traceback (most recent call last)', 'CUDA error:', 'Cross-request draft KV slot collision'])
    summary = json.loads((directory / 'summary.json').read_text())
    assert summary['failed'] == 0
    if diagnostic:
        events = [json.loads(x.split('[CDB_REFILL] ', 1)[1]) for x in text.splitlines() if '[CDB_REFILL] ' in x]
        ready = [x for x in events if x['event'] == 'ready']
        assert ready and all(1 <= x['k_actual'] <= x['k_cap'] == cap for x in ready)
        assert max(x['k_actual'] for x in ready) == cap
        reasons = {x['reason'] for x in ready}
        assert {'low_confidence', 'max_k'} <= reasons
        keys = lambda event: [(x['req_id'], x['fingerprint'], x['window']) for x in events if x['event'] == event]
        handed, verified = keys('p2_handoff'), keys('p2_verified')
        assert len(handed) == len(set(handed)) == len(verified) == len(set(verified))
        assert set(handed) == set(verified)
        reset = [x for x in events if x['event'] == 'p2_reset'][-1]
        assert reset['live_leases'] == reset['ready'] == reset['inflight'] == reset['released_blocks'] == 0
        audits = [x for x in events if x['event'] == 'carry_kv_unchanged']
        scopes = [json.loads(x.split('[CDB_TARGET_GRAPH_SCOPE] ', 1)[1]) for x in text.splitlines() if '[CDB_TARGET_GRAPH_SCOPE] ' in x]
        assert scopes and all(x['model'] == 'LlamaForCausalLM' and x['layers'] == 32 and x['fa_version'] == 2 and x['llama_opt_in'] for x in scopes)
        draft_checks = [json.loads(x.split('[CDB_DRAFT_GRAPH_VALIDATE] ', 1)[1]) for x in text.splitlines() if '[CDB_DRAFT_GRAPH_VALIDATE] ' in x]
        target_checks = [json.loads(x.split('[CDB_TARGET_FULL_GRAPH_VALIDATE] ', 1)[1]) for x in text.splitlines() if '[CDB_TARGET_FULL_GRAPH_VALIDATE] ' in x]
        assert draft_checks and target_checks, 'Llama must execute and validate real Draft and Target Graphs'
        assert all(x['kv_exact'] and x['token_exact'] and x['logits_close'] for x in draft_checks)
        assert all(c['kv_byte_exact'] and c['token_exact'] and c['logits_close'] for x in target_checks for c in x['checks'])
        compared = target_checks[-1]['slot_guard_compared']
        shared = target_checks[-1]['slot_guard_shared_tensor']
        if json.loads((R/'PLAN.json').read_text())['fixed_replay_host']:
            assert compared >= len(target_checks) and shared > 0
        assert audits and all(x['layers'] == 16 for x in audits)
        assert any(x['target_epoch'] > x['previous_target_epoch'] for x in audits)
        audit = dict(model='llama', warmup_requests=64, measured_requests=16, target_scope=scopes, draft_graph_checks=len(draft_checks), target_graph_checks=len(target_checks), cap=cap, completed=summary['completed'], ready=len(ready), verified=len(verified), K_values=sorted({x['k_actual'] for x in ready}), reasons=sorted(reasons), carry_byte_checks=len(audits), cross_target_checks=sum(x['target_epoch'] > x['previous_target_epoch'] for x in audits), cleanup=reset)
        actual=[json.loads(x.split('[CDB_TARGET_FULL_GRAPH] ',1)[1]) for x in text.splitlines() if '[CDB_TARGET_FULL_GRAPH] ' in x]
        captured=[x for x in actual if x['action'] in ('capture','replay')]
        assert captured and all(x['key'][3]==1024 and x['padding']==0 for x in captured)
        audit['target_sequence_floor']=1024
        audit['target_floor_checked_forwards']=len(captured)
        audit['target_floor_cache_keys']=list({json.dumps(x['key']) for x in captured})
        fp_checks=[json.loads(x.split('[CDB_CONTEXT_FP_VALIDATE] ',1)[1]) for x in text.splitlines() if '[CDB_CONTEXT_FP_VALIDATE] ' in x]
        fp_runtime=[json.loads(x.split('[CDB_CONTEXT_FP_CACHE] ',1)[1]) for x in text.splitlines() if '[CDB_CONTEXT_FP_CACHE] ' in x]
        assert len(fp_checks)==32 and all(x['fingerprint_exact'] and x['request_id'] for x in fp_checks)
        assert fp_runtime and any(x['hits']>0 for x in fp_runtime)
        assert all(x['native_exact_type_check'] and x['native_type_calls']>0 and x['native_module'].endswith('/_cdb_av_prefix_types.cpython-311-x86_64-linux-gnu.so') for x in fp_runtime)
        assert all(x['full_typed_prefix_equality'] and x['request_weak_references'] and x['source']==str(W/'vllm/v1/spec_decode/cdb_av_context_cache.py') for x in fp_runtime)
        audit['slot_guard_compared']=compared
        audit['slot_guard_shared_tensor']=shared
        audit['fingerprint_checks']=fp_checks
        audit['fingerprint_runtime']=fp_runtime
        waits=[x for x in events if x['event']=='draft_wait_enqueued']
        admitted=[x for x in events if x['event']=='draft_wait_admitted']
        assert waits and admitted
        snapshots=json.loads((directory/'observer_after.json').read_text())
        unclassified=sum(x.get('cdb_ar_reason_unclassified_rows',0) for x in snapshots)
        assert unclassified==0, 'Every AR row in wait mode needs an explicit fallback'
        audit['chain_graph_enqueues']=len(waits)
        audit['chain_graph_admissions']=len(admitted)
        audit['unclassified_ar_rows']=unclassified
        checks=[json.loads(x.split('[CDB_DRAFT_CHAIN_VALIDATE] ',1)[1]) for x in text.splitlines() if '[CDB_DRAFT_CHAIN_VALIDATE] ' in x]
        assert checks and all(x['logits_byte_exact'] and x['packet_byte_exact'] for x in checks)
        assert any(x['dynamic'] for x in checks), 'Must validate real dynamic row gathers'
        audit['chain_checks']=len(checks)
        audit['chain_dynamic_checks']=sum(x['dynamic'] for x in checks)
        write(tag + '_AUDIT.json', audit)
        return audit
    assert summary['completed'] == 512 and summary['output_tokens'] == 101003
    return dict(directory=str(directory), cap=cap, seed=seed, chain_graph=floor, tps=summary['throughput_tok_s'], e2e_p95=summary['latency_s']['e2e']['p95'], ttft_p95=summary['latency_s']['ttft']['p95'])



results=[]
try:
    write('CRITERIA.json',dict(protocol='one ON screen, user requested no OFF/redundant repetitions',reference_tps=312.56473125804325,reference_sd=.49881798943056715,minimum_change_pct=.5,max_e2e_p95_increase_pct=1,reference_e2e_p95=17.627043815546976,time=time.time()))
    gate=execute('gate_chain',8,diagnostic=True)
    write('FUNCTIONAL_ACCEPTED.json',gate)
    result=execute('screen_seed42_on',8,42,floor=1)
    results.append(result);write('RESULTS_PARTIAL.json',results)
    state('completed',results=results)
except BaseException:
    state('failed',error=traceback.format_exc(),completed=results)
    raise
