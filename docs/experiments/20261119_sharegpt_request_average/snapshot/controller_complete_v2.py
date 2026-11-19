"""Complete real K8 request coverage, then resume paired GPU tests."""
import fcntl
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

R = Path(__file__).resolve().parent
W = Path('/home/yypan/CDB-AV/worktrees/cdb-request-avg-sharegpt-20261119')
H_ROOT = Path('/home/yypan/CDB-AV/results/sharegpt_h_sweep_gpu3_20261119_1016')
PY = '/home/yypan/CDB-AV/.venv/bin/python'
sys.path.insert(0, str(R))


def write(name, data):
    path = R / name
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)


def state(value, **data):
    write('STATUS.json', dict(state=value, time=time.time(), pid=os.getpid(), **data))


def wait_h_sweep():
    deadline = time.monotonic() + 10800
    while True:
        status = json.loads((H_ROOT / 'STATUS.json').read_text())
        if status['state'] == 'failed':
            raise RuntimeError('H sweep failed; new GPU tests are blocked')
        if status['state'] == 'completed':
            assert (H_ROOT / 'COMPLETE.json').is_file()
            return
        state('waiting_for_h_sweep', h_tag=status.get('tag'))
        if time.monotonic() > deadline:
            raise RuntimeError('H sweep did not finish within three hours')
        time.sleep(10)


def prepare_source():
    state('cpu_validation')
    reviewed = json.loads((R / 'REVIEWED_FILES.json').read_text())
    for name, digest in reviewed.items():
        assert hashlib.sha256((W / name).read_bytes()).hexdigest() == digest, name
    status = subprocess.check_output(['git', 'status', '--porcelain'], cwd=W, text=True)
    assert {line[3:] for line in status.splitlines()} == set(reviewed), status
    env = {k: v for k, v in os.environ.items() if not k.startswith('CDB_')}
    env.update(CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(W), CDB_AV_FINGERPRINT_BATCH='1', CDB_AV_KV_CHECK_REUSE='1', CDB_AV_PINNED_INPUTS='0', CDB_AV_EXACT_QKNORM='0')
    env['PATH'] = '/home/yypan/CDB-AV/.venv/bin:' + env['PATH']
    env['GIT_SSH_COMMAND'] = 'ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=10 -o ServerAliveCountMax=2'
    tests = ['tests/v1/spec_decode/' + n for n in ['test_cdb_av_history_k_fallback.py', 'test_cdb_av_config.py', 'test_cdb_av_acceptance.py', 'test_cdb_av_request_average.py', 'test_cdb_av_persistent_queue.py', 'test_cdb_av_kv_reuse.py', 'test_cdb_av_kv_reuse_ownership.py', 'test_cdb_av_ownership_scan.py', 'test_cdb_av_low_log.py']]
    with (R / 'cpu_final.log').open('w') as out:
        subprocess.run([PY, '-m', 'pytest', '-q'] + tests, cwd=W, env=env, stdout=out, stderr=subprocess.STDOUT, check=True, timeout=240)
    local_env = dict(env, PYTHONPATH=str(R) + os.pathsep + str(W), PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    with (R / 'protocol_tests.log').open('w') as out:
        subprocess.run([PY, '-m', 'pytest', '-q', str(R / 'test_protocol.py'), str(R / 'test_build_average.py')], cwd=R, env=local_env, stdout=out, stderr=subprocess.STDOUT, check=True, timeout=60)
    with (R / 'validate.log').open('w') as out:
        subprocess.run(['bash', 'scripts/validate_changes.sh', '--pytest', 'tests/v1/spec_decode/test_cdb_av_history_k_fallback.py'], cwd=W, env=env, stdout=out, stderr=subprocess.STDOUT, check=True, timeout=180)
    cmd = ['bash', 'scripts/git_checkpoint.sh', '--message', 'feat: add explicit fixed fallback for historical K policies', '--pytest', 'tests/v1/spec_decode/test_cdb_av_history_k_fallback.py']
    for name in reviewed:
        cmd += ['--path', name]
    with (R / 'checkpoint.log').open('w') as out:
        subprocess.run(cmd, cwd=W, env=env, stdout=out, stderr=subprocess.STDOUT, check=True, timeout=360)
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=W, text=True)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=W, text=True).strip()
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=W, text=True).strip()
    remote = subprocess.check_output(['git', 'ls-remote', '--heads', 'myfork', branch], cwd=W, env=env, text=True, timeout=45).split()[0]
    assert commit == remote
    source = dict(source=str(W), commit=commit, branch=branch, remote_head=remote, hashes=reviewed)
    write('SOURCE.json', source)
    (R / 'source.patch').write_text(subprocess.check_output(['git', 'show', '--format=fuller', '--stat', '--patch', commit], cwd=W, text=True))


def wait_gpu(tag):
    deadline = time.monotonic() + 7200
    while True:
        processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader'], text=True)
        memory = int(subprocess.check_output(['nvidia-smi', '--id=3', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True))
        if 'GPU-b493bb52-b935-1cf4-47fe-7869125775b9' not in processes and memory < 128:
            return
        state('waiting_for_gpu3', next=tag, memory_mib=memory)
        if time.monotonic() > deadline:
            raise RuntimeError('GPU3 remained occupied; no foreign process stopped')
        time.sleep(10)


def execute(tag, mode, seed=42, *, diagnostic=False, average_file=None, missing=False):
    wait_gpu(tag)
    state('diagnostic' if diagnostic else 'running', tag=tag, mode=mode, seed=seed)
    cmd = [PY, str(R / 'driver.py'), '--tag', tag, '--mode', mode, '--seed', str(seed)]
    if diagnostic:
        cmd += ['--diagnostic', '--requests', '16', '--max-tokens', '64']
    if average_file:
        cmd += ['--average-file', str(average_file)]
    if missing:
        cmd += ['--inject-missing-first']
    dry = list(cmd) + ['--dry-run']
    with (R / (tag + '_dry_run.log')).open('w') as out:
        subprocess.run(dry, cwd=R, stdout=out, stderr=subprocess.STDOUT, check=True, timeout=30)
    with (R / (tag + '_controller.log')).open('w') as out:
        subprocess.run(cmd, cwd=R, stdout=out, stderr=subprocess.STDOUT, check=True, timeout=2700)
    directory = R / tag
    released = json.loads((directory / 'released.json').read_text())
    assert released['port_closed'] and not released['our_gpu_pids']
    log = (directory / 'server.log').read_text(errors='replace')
    for failure in ['Traceback (most recent call last)', 'CUDA error:', 'Cross-request draft KV slot collision']:
        assert failure not in log, failure
    summary = json.loads((directory / 'summary.json').read_text())
    assert summary['failed'] == 0
    if not diagnostic:
        assert summary['completed'] == 512 and summary['output_tokens'] == 101003
    return dict(mode=mode, seed=seed, directory=str(directory), tps=summary['throughput_tok_s'], e2e_p95=summary['latency_s']['e2e']['p95'], ttft_p95=summary['latency_s']['ttft']['p95'])


def audit_gate(tag, mode):
    directory = R / tag
    events = [json.loads(line.split('[CDB_REFILL] ', 1)[1]) for line in (directory / 'server.log').read_text().splitlines() if '[CDB_REFILL] ' in line]
    identities = [json.loads(x) for x in (directory / 'measurement/identities.jsonl').read_text().splitlines()]
    expected = {x['trace_key'] for x in identities}
    verified = [x for x in events if x['event'] == 'p2_verified']
    assert verified and all(x['trace_key'] in expected for x in verified)
    handoff_keys = [(x['req_id'], x['fingerprint'], x['window']) for x in events if x['event'] == 'p2_handoff']
    verified_keys = [(x['req_id'], x['fingerprint'], x['window']) for x in verified]
    assert len(handoff_keys) == len(set(handoff_keys))
    assert len(verified_keys) == len(set(verified_keys))
    assert set(handoff_keys) == set(verified_keys)
    ready = [x for x in events if x['event'] == 'ready']
    assert ready and all(x['reason'] != 'low_confidence' for x in ready)
    if mode == 'reference':
        assert all(x['k_actual'] == 8 and x['k_cap'] == 8 for x in ready)
    else:
        decisions = {x['req_id']: x for x in events if x['event'] == 'k_request_average_decision'}
        for row in ready:
            decision = decisions[row['req_id']]
            assert row['k_cap'] == decision['cap']
            assert row['k_actual'] == row['k_cap']
            assert row['confidence'] is None, 'Matched/fixed historical work must skip confidence calculation'
            expected_source = 'request_average' if decision['source'] == 'request_average' else 'fixed_max_fallback'
            assert row['k_source'] == expected_source
        assert any(x['k_source'] == 'fixed_max_fallback' for x in ready)
        assert any(x['k_source'] == 'request_average' for x in ready)
    reset = [x for x in events if x['event'] == 'p2_reset'][-1]
    assert reset['live_leases'] == reset['ready'] == reset['inflight'] == reset['released_blocks'] == 0
    audits = [x for x in events if x['event'] == 'carry_kv_unchanged']
    result = dict(mode=mode, ready=len(ready), verified=len(verified), K_values=sorted({x['k_actual'] for x in ready}), observed_trace_keys=len({x['trace_key'] for x in verified}), requested_trace_keys=len(expected), carry_byte_checks=len(audits), cross_target_checks=sum(x['target_epoch'] > x['previous_target_epoch'] for x in audits), cleanup=reset)
    write(tag + '_AUDIT.json', result)
    return result


def check_source():
    source = json.loads((R / 'SOURCE.json').read_text())
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=W, text=True).strip() == source['commit']
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=W, text=True)
    for name, digest in source['hashes'].items():
        assert hashlib.sha256((W / name).read_bytes()).hexdigest() == digest


def supplement():
    tag = 'reference_missing8_c1_seed42'
    wait_gpu(tag)
    state('collecting_missing_reference', tag=tag, mode='reference', concurrency=1)
    cmd = [PY, str(R / 'driver.py'), '--tag', tag, '--mode', 'reference', '--reference-missing-summary', str(R / 'REQUEST_AVERAGE.summary.json')]
    if not (R / tag / 'DONE.json').exists():
        for dry in [True, False]:
            with (R / (tag + ('_dry_run.log' if dry else '_controller.log'))).open('w') as out:
                subprocess.run(cmd + (['--dry-run'] if dry else []), cwd=R, stdout=out, stderr=subprocess.STDOUT, check=True, timeout=2700)
    directory = R / tag
    released = json.loads((directory / 'released.json').read_text())
    assert released['port_closed'] and not released['our_gpu_pids']
    selection = json.loads((directory / 'reference_selection.json').read_text())
    summary = json.loads((directory / 'summary.json').read_text())
    assert summary['completed'] == len(selection['trace_keys']) == 75 and summary['failed'] == 0
    text = (directory / 'server.log').read_text()
    assert all(x not in text for x in ['Traceback (most recent call last)', 'CUDA error:', 'Cross-request draft KV slot collision'])
    events = [json.loads(x.split('[CDB_REFILL] ', 1)[1]) for x in text.splitlines() if '[CDB_REFILL] ' in x]
    verified = [x for x in events if x['event'] == 'p2_verified']
    assert {x['trace_key'] for x in verified} == set(selection['trace_keys'])
    ready = [x for x in events if x['event'] == 'ready']
    assert ready and all(x['k_actual'] == x['k_cap'] == 8 and x['reason'] != 'low_confidence' for x in ready)
    handoffs = [(x['req_id'], x['fingerprint'], x['window']) for x in events if x['event'] == 'p2_handoff']
    keys = [(x['req_id'], x['fingerprint'], x['window']) for x in verified]
    assert len(keys) == len(set(keys)) == len(handoffs) == len(set(handoffs)) and set(keys) == set(handoffs)
    from audit_supplement import audit_events
    result = audit_events(events, set(selection['trace_keys']))
    for metric in ['num_requests_running', 'num_requests_waiting', 'kv_cache_usage_perc']:
        values = [float(line.rsplit(' ', 1)[1]) for line in (directory / 'metrics_after.prom').read_text().splitlines() if line.startswith('vllm:' + metric + '{')]
        assert values and all(value == 0 for value in values), (metric, values)
    result.update(output_tokens=summary['output_tokens'], concurrency=1, final_service_metrics_zero=True, service_released=released)
    write('SUPPLEMENT_AUDIT.json', result)
    return directory


lock = (R / 'sequence.lock').open('a')
# Wait for the already running HF arm to finish and its own service to release.
# The old controller's next average dry-run is blocked by the coverage guard.
write('CONTINUATION_STATUS.json', dict(state='waiting_for_original_controller', time=time.time(), pid=os.getpid()))
fcntl.flock(lock, fcntl.LOCK_EX)
results = []
try:
    previous = json.loads((R / 'STATUS.json').read_text())
    assert previous['state'] == 'failed', previous
    assert 'IndexError' in previous['error'] and "p2_reset" in previous['error']
    assert (R / 'reference_missing8_c1_seed42/DONE.json').exists()
    write('STATUS.supplement_audit_failure.json', previous)
    check_source()
    existing = json.loads((R / 'RESULTS_PARTIAL.json').read_text())
    assert len(existing) == 1 and existing[0]['mode'] == 'hf' and existing[0]['seed'] == 42
    hf = R / 'pair0_hf_seed42'
    assert (hf / 'DONE.json').exists() and json.loads((hf / 'released.json').read_text())['port_closed']
    results.extend(existing)
    directory = supplement()
    from build_complete import build_complete
    average_file = R / 'REQUEST_AVERAGE_COMPLETE.jsonl'
    complete = build_complete(R / 'reference_full8_seed42', directory, average_file)
    assert complete['observed_requests'] == 512 and not complete['requests_without_verification']
    write('AVERAGE_COMPLETE_FILE.json', dict(path=str(average_file), sha256=hashlib.sha256(average_file.read_bytes()).hexdigest(), summary=complete))
    for block, seed, mode in [(0, 42, 'average'), (1, 43, 'average'), (1, 43, 'hf'), (2, 44, 'hf'), (2, 44, 'average')]:
        assert hashlib.sha256(average_file.read_bytes()).hexdigest() == complete['sha256']
        result = execute(f'pair{block}_{mode}_seed{seed}', mode, seed, average_file=average_file if mode == 'average' else None)
        results.append(result)
        write('RESULTS_PARTIAL.json', results)
        print('RESULT', json.dumps(result), flush=True)
    write('COMPLETE.json', dict(results=results, time=time.time(), reference_complete=True))
    state('completed', results=results)
except BaseException:
    state('failed', error=traceback.format_exc(), completed=results)
    raise
finally:
    fcntl.flock(lock, fcntl.LOCK_UN)
    lock.close()
