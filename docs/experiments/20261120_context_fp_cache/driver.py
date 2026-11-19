"""Isolated Llama ShareGPT CDB test; matched native dataset/client settings."""
import argparse
import asyncio
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import time
import traceback
import urllib.request

from benchmark import run

R = Path(__file__).resolve().parent
W = Path('/home/yypan/CDB-AV/worktrees/cdb-context-fp-cache-20261120')
PY = '/home/yypan/CDB-AV/.venv/bin/python'
GPU_UUID = 'GPU-b493bb52-b935-1cf4-47fe-7869125775b9'
BASE = 'http://127.0.0.1:18084'
CFG = Path('/home/yypan/CDB-AV/results/cpu_hotpath_20261113/pair0_both/perf_r0_configuration.json')
REF = Path('/home/yypan/CDB-AV/baseline-comparison-20261113/baseline_experiments/results/closed_loop')


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def metrics(path):
    with urllib.request.urlopen(BASE + '/metrics', timeout=10) as res:
        path.write_bytes(res.read())


def observer_snapshot(obs):
    return [json.loads(p.read_text()) for p in obs.glob('*.json')]


def check_output(dest, count, expected_tokens):
    records = [json.loads(x) for x in (dest / 'requests.jsonl').read_text().splitlines()]
    assert len(records) == count and all(x['ok'] for x in records)
    assert sum(x['output_tokens'] for x in records) == expected_tokens
    assert all(sum(e['n'] for e in x['token_events']) == x['output_tokens'] for x in records)
    assert all(all(0 <= t < 128256 for e in x['token_events'] for t in e['tokens']) for x in records)


def preflight(source):
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader'], text=True)
    memory = int(subprocess.check_output(['nvidia-smi', '--id=3', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True))
    assert GPU_UUID not in processes and memory < 128, (memory, processes)
    with socket.socket() as sock:
        assert sock.connect_ex(('127.0.0.1', 18084)) != 0
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=W, text=True).strip() == source['commit']
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=W, text=True)
    for name, expected in source['hashes'].items():
        assert hashlib.sha256((W / name).read_bytes()).hexdigest() == expected
    bindings = json.loads((R / 'RUNTIME_BINDINGS.json').read_text())
    for entry in bindings['bindings']:
        link = W / entry['path']
        target = Path(entry['target'])
        assert link.is_symlink() and link.resolve() == target
        for name, digest in entry['sha256'].items():
            path = target / name if target.is_dir() else target
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    return dict(memory_mib=memory, time=time.time(), gpu_uuid=GPU_UUID)


def healthy(proc):
    for _ in range(120):
        if proc.poll() is not None:
            raise RuntimeError('service exited ' + str(proc.returncode))
        try:
            with urllib.request.urlopen(BASE + '/health', timeout=2) as res:
                if res.status == 200:
                    return
        except Exception:
            pass
        time.sleep(3)
    raise RuntimeError('health timeout')


parser = argparse.ArgumentParser()
parser.add_argument('--tag', required=True)
parser.add_argument('--context-cache', type=int, choices=[0,1], required=True)
parser.add_argument('--k-cap', type=int, choices=[4,8], required=True)
parser.add_argument('--mode', choices=['reference', 'average', 'hf'], required=True)
parser.add_argument('--seed', type=int, choices=[42, 43, 44], default=42)
parser.add_argument('--requests', type=int, default=512)
parser.add_argument('--max-tokens', type=int)
parser.add_argument('--average-file', type=Path)
parser.add_argument('--diagnostic', action='store_true')
parser.add_argument('--gate-warmup', type=int, choices=[0, 64], default=0)
parser.add_argument('--inject-missing-first', action='store_true')
parser.add_argument('--dry-run', action='store_true')
parser.add_argument('--reference-missing-summary', type=Path)
args = parser.parse_args()
assert args.mode == 'hf' and not args.average_file and not args.reference_missing_summary
assert not (R / 'STOP_NEW_GPU_SERVICES.json').exists()
assert args.tag.replace('_', '').replace('-', '').isalnum()
assert 1 <= args.requests <= 512
assert not args.gate_warmup or args.diagnostic
assert not args.inject_missing_first or (args.diagnostic and args.mode == 'average')
source = json.loads((R / 'SOURCE.json').read_text())
plan_config = json.loads((R / 'PLAN.json').read_text())
assert hashlib.sha256((R / 'benchmark.py').read_bytes()).hexdigest() == plan_config['client_sha256']
assert hashlib.sha256((R / 'manual_observer.py').read_bytes()).hexdigest() == plan_config['observer_sha256']
cfg = json.loads(CFG.read_text())
rows = [json.loads(x) for x in (R / 'sharegpt.jsonl').read_text().splitlines()]
assert hashlib.sha256((R / 'sharegpt.jsonl').read_bytes()).hexdigest() == '59e2e785a02b33966d3a5a794896a38be1226420fa75e6280bf576ed96ce6e4d'
reference_selection = None
concurrency = 16
if args.reference_missing_summary:
    assert args.mode == 'reference' and not args.diagnostic and args.max_tokens is None
    missing_summary = json.loads(args.reference_missing_summary.read_text())
    missing = set(missing_summary['requests_without_verification'])
    assert missing and all(x.startswith('measurement:') for x in missing)
    rows = [x for x in rows if 'measurement:' + str(x['id']) in missing]
    assert len(rows) == len(missing)
    args.requests = len(rows)
    concurrency = 1
    reference_selection = dict(summary=str(args.reference_missing_summary.resolve()), sha256=hashlib.sha256(args.reference_missing_summary.read_bytes()).hexdigest(), trace_keys=sorted(missing), concurrency=concurrency, note='Coverage supplement only; not a performance measurement; unchanged prompts and original output budgets.')
if args.max_tokens is not None:
    assert args.diagnostic and 1 <= args.max_tokens <= 128
    rows = [dict(x, max_tokens=min(x['max_tokens'], args.max_tokens)) for x in rows]
if args.mode == 'average':
    assert args.average_file and args.average_file.is_file()
    if args.average_file.name == 'REQUEST_AVERAGE.jsonl':
        expected_average = json.loads((R / 'AVERAGE_FILE.json').read_text())
        assert hashlib.sha256(args.average_file.read_bytes()).hexdigest() == expected_average['sha256']
    if not args.diagnostic:
        coverage = json.loads(args.average_file.with_suffix('.summary.json').read_text())
        assert coverage['measured_requests'] == coverage['observed_requests'] == 512 and not coverage['requests_without_verification'], 'Reference coverage incomplete: do not substitute fallback K for unobserved request means'

env = {k: v for k, v in os.environ.items() if not k.startswith(('CDB_', 'BASELINE_'))}
env.update(cfg['environment'])
env.update(PYTHONPATH=str(R) + os.pathsep + str(W), CUDA_VISIBLE_DEVICES='3', CUDA_DEVICE_ORDER='PCI_BUS_ID',
           CDB_AV_WINDOW_H='8',
           CDB_AV_K_POLICY={'reference': 'manual', 'average': 'request_avg_acceptance', 'hf': 'hf_confidence'}[args.mode])
if args.average_file:
    env['CDB_AV_REQUEST_AVG_FILE'] = str(args.average_file.resolve())
if args.diagnostic:
    env.update(CDB_AV_CARRY_VALIDATE='1', CDB_AV_VALIDATE_METADATA='1', CDB_AV_LOW_LOG='0', CDB_AV_REFILL_TRACE='1')
elif args.mode == 'reference':
    env.update(CDB_AV_LOW_LOG='0', CDB_AV_REFILL_TRACE='0')
cmd = cfg['command'].copy()
cmd[cmd.index('serve') + 1] = '/disk14T/yypan/models/Llama-3.1-8B-Instruct'
cmd[cmd.index('--served-model-name') + 1] = 'target'
cmd[cmd.index('--seed') + 1] = str(args.seed)
spec_index = cmd.index('--speculative-config') + 1
spec = json.loads(cmd[spec_index])
spec['num_speculative_tokens'] = args.k_cap
spec['model'] = '/disk14T/yypan/CDB-AV-runtime/baseline-comparison-20261113/models/Llama-3.2-1B-Instruct'
for name in list(env):
    if name.startswith('CDB_AV_EXACT_QK'):
        env.pop(name)
env.update(CDB_AV_EXACT_QKNORM='0', CDB_AV_TARGET_GRAPH_LLAMA='1', CDB_AV_TARGET_GRAPH_REUSE_BOUNDS='0', CDB_AV_RAGGED_GRAPH_SEQ_FLOOR='1024', CDB_AV_PACKED_SPEC_METADATA='0', CDB_AV_DRAFT_META_COPY='0', CDB_AV_TARGET_GRAPH_SEQ_FLOOR='1024', CDB_AV_CONTEXT_FINGERPRINT_CACHE=str(args.context_cache), CDB_AV_CONTEXT_FINGERPRINT_VALIDATE='1' if args.diagnostic and args.context_cache else '0', CDB_AV_SPEC_METADATA_VALIDATE='0', CDB_AV_DRAFT_META_COPY_VALIDATE='0', CDB_AV_LOWLEVEL_TORCH_PROBE='0')
if args.diagnostic:
    env.update(CDB_AV_DRAFT_GRAPH_VALIDATE='1', CDB_AV_TARGET_GRAPH_VALIDATE='1', CDB_AV_TRACE_TARGET_GRAPH='1')
cmd[spec_index] = json.dumps(spec)
cmd += ['--dtype', 'bfloat16', '--max-num-batched-tokens', '1024']
assert env['CDB_AV_HF_THRESHOLD'] == '0.8'
assert json.loads(cmd[cmd.index('--speculative-config') + 1])['num_speculative_tokens'] == args.k_cap
if args.inject_missing_first:
    os.environ['CDB_TRACE_MISSING_FIRST'] = '1'
if args.dry_run:
    import random
    indices = list(range(len(rows)))
    random.Random(args.seed).shuffle(indices)
    print(json.dumps(dict(mode=args.mode, family='llama', command=cmd, H=8, cap=args.k_cap, requests=args.requests, concurrency=concurrency, gate_warmup=args.gate_warmup, reference_selection=reference_selection, total_output_tokens=sum(rows[i]['max_tokens'] for i in indices[:args.requests]), first_trace_key='measurement:' + str(rows[indices[0]]['id']), environment_delta={k: v for k, v in env.items() if k.startswith('CDB_') and cfg['environment'].get(k) != v})))
    raise SystemExit(0)


def expired(signum, frame):
    raise TimeoutError('Per-service 40 minute deadline')


signal.signal(signal.SIGALRM, expired)
signal.alarm(360 if args.diagnostic else 2400)
lock = open('/tmp/cdb_gpu3_test.lock', 'a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
d = R / args.tag
d.mkdir(exist_ok=False)
proc = sampler = None
try:
    write(d / 'preflight.json', preflight(source))
    obs = d / 'observer'
    obs.mkdir()
    (obs / 'epoch.txt').write_text('warmup')
    env['BASELINE_OBSERVE_DIRECTORY'] = str(obs)
    safe_env = {k: v for k, v in env.items() if k.startswith(('CDB_', 'BASELINE_')) or k in ['PYTHONPATH', 'CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER', 'VLLM_ATTENTION_BACKEND']}
    launch = 'cd ' + shlex.quote(str(W)) + '\n' + shlex.join(['env'] + [k + '=' + v for k, v in safe_env.items()] + cmd) + '\n'
    (d / 'launch.sh').write_text(launch)
    print(launch, flush=True)
    write(d / 'reference_selection.json', reference_selection)
    write(d / 'manifest.json', dict(context_cache=args.context_cache, mode=args.mode, k_cap=args.k_cap, diagnostic=args.diagnostic, seed=args.seed, source=str(W), source_commit=source['commit'], environment=safe_env, command=cmd, data_sha256=hashlib.sha256((R / 'sharegpt.jsonl').read_bytes()).hexdigest(), client_sha256=hashlib.sha256((R / 'benchmark.py').read_bytes()).hexdigest(), observer_sha256=hashlib.sha256((R / 'manual_observer.py').read_bytes()).hexdigest(), driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), bindings_sha256=hashlib.sha256((R / 'RUNTIME_BINDINGS.json').read_bytes()).hexdigest(), average_file_sha256=None if not args.average_file else hashlib.sha256(args.average_file.read_bytes()).hexdigest(), inject_missing_first=args.inject_missing_first))
    track = ['bash', 'scripts/experiment_manifest.sh', '--experiment-name', args.tag, '--vllm-version', '0.15.1', '--target-model', '/disk14T/yypan/models/Llama-3.1-8B-Instruct', '--draft-model', '/disk14T/yypan/CDB-AV-runtime/baseline-comparison-20261113/models/Llama-3.2-1B-Instruct', '--gpu', '3', '--dynamic-k-policy', env['CDB_AV_K_POLICY'], '--dataset', str(R / 'sharegpt.jsonl'), '--command', launch, '--run-dir', str(d / 'tracking'), '--source-import-path', str(W), '--window-h', '8', '--request-count', str(args.requests), '--test-status', 'not_run']
    with (d / 'tracking.log').open('w') as out:
        subprocess.run(track, cwd=W, env=env, stdout=out, stderr=subprocess.STDOUT, check=True, timeout=30)
    probe = "import json,torch,vllm;from vllm.v1.spec_decode import cdb_av_refill as r,cdb_av_persistent as p,cdb_av_context_cache as m;print(json.dumps(dict(torch=torch.__version__,vllm=vllm.__version__,source=vllm.__file__,refill=r.__file__,persistent=p.__file__,context_cache_module=m.__file__)))"
    with (d / 'runtime_imports.json').open('w') as out:
        subprocess.run([PY, '-c', probe], cwd=W, env=env, stdout=out, stderr=subprocess.STDOUT, check=True, timeout=30)
    log = (d / 'server.log').open('w')
    proc = subprocess.Popen(cmd, cwd=W, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write(d / 'pid.json', dict(pid=proc.pid))
    healthy(proc)
    gpu = (d / 'gpu.csv').open('w')
    sampler = subprocess.Popen(['nvidia-smi', '--id=3', '--query-gpu=timestamp,utilization.gpu,memory.used,power.draw,temperature.gpu,clocks.sm', '--format=csv,noheader,nounits', '--loop=1'], stdout=gpu, stderr=subprocess.STDOUT)

    async def execute():
        if args.gate_warmup:
            warm = [dict(x, max_tokens=min(x['max_tokens'], 64)) for x in rows]
            result = await run(BASE, warm, 'llama', 'closed', 16, args.gate_warmup, args.seed, d / 'warmup')
            assert result['failed'] == 0
            warm_plan = json.loads((d / 'warmup/plan.json').read_text())
            check_output(d / 'warmup', args.gate_warmup, sum(warm[x['source']]['max_tokens'] for x in warm_plan))
        if not args.diagnostic and reference_selection is None:
            warm = [dict(x, max_tokens=128) for x in rows]
            result = await run(BASE, warm, 'llama', 'closed', 16, 128, args.seed, d / 'warmup')
            assert result['failed'] == 0
            check_output(d / 'warmup', 128, 16384)
        (obs / 'epoch.txt').write_text('measurement')
        await asyncio.sleep(.3)
        write(d / 'observer_before.json', observer_snapshot(obs))
        metrics(d / 'metrics_before.prom')
        result = await run(BASE, rows, 'llama', 'closed', concurrency, args.requests, args.seed, d / 'measurement')
        assert result['failed'] == 0
        plan = json.loads((d / 'measurement/plan.json').read_text())
        check_output(d / 'measurement', args.requests, sum(rows[x['source']]['max_tokens'] for x in plan))
        if args.requests == 512:
            ref = json.loads((REF / f'llama_sharegpt_closed16_sd1_seed{args.seed}' / 'summary.json').read_text())
            assert result['plan_sha256'] == ref['plan_sha256']
        metrics(d / 'metrics_after.prom')
        await asyncio.sleep(.2)
        write(d / 'observer_after.json', observer_snapshot(obs))
        write(d / 'summary.json', result)
        return result

    result = asyncio.run(execute())
    write(d / 'DONE.json', dict(time=time.time(), requests=args.requests))
    print('RESULT', args.mode, result['throughput_tok_s'], result['latency_s'], flush=True)
except BaseException:
    write(d / 'FAILED.json', dict(error=traceback.format_exc(), time=time.time()))
    raise
finally:
    if sampler is not None:
        sampler.terminate()
        sampler.wait(timeout=10)
    if proc is not None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
    time.sleep(3)
    memory = int(subprocess.check_output(['nvidia-smi', '--id=3', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True))
    with socket.socket() as sock:
        closed = sock.connect_ex(('127.0.0.1', 18084)) != 0
    owned = []
    for line in subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader'], text=True).splitlines():
        if GPU_UUID not in line:
            continue
        pid = int(line.split(',')[0])
        try:
            if proc is not None and os.getpgid(pid) == proc.pid:
                owned.append(pid)
        except ProcessLookupError:
            pass
    write(d / 'released.json', dict(port_closed=closed, memory_mib=memory, our_gpu_pids=owned, time=time.time()))
    fcntl.flock(lock, fcntl.LOCK_UN)
    lock.close()
    assert closed and not owned
