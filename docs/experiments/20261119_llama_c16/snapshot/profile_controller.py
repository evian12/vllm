"""Wait for Llama formal tests; collect one bounded Llama trace on GPU3."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

R = Path(__file__).resolve().parent
CAP = Path('/home/yypan/CDB-AV/results/sharegpt_llama_cdb_gpu3_20261119')
PY = '/home/yypan/CDB-AV/.venv/bin/python'


def state(value, **data):
    (R / 'PROFILE_STATUS.json').write_text(json.dumps(dict(state=value, time=time.time(), pid=os.getpid(), **data), indent=2) + '\n')


lock = (R / 'profile_controller.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
try:
    state('waiting_for_llama_performance')
    deadline = time.monotonic() + 14400
    while True:
        status = json.loads((CAP / 'STATUS.json').read_text())
        if status['state'] == 'failed':
            raise RuntimeError('Llama experiment failed; profiling does not bypass its audit gate')
        if status['state'] == 'completed':
            assert (CAP / 'COMPLETE.json').is_file()
            break
        if time.monotonic() > deadline:
            raise RuntimeError('Llama experiment not complete after four hours')
        time.sleep(10)
    for cap in [8]:
        deadline = time.monotonic() + 7200
        while True:
            apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader'], text=True)
            memory = int(subprocess.check_output(['nvidia-smi', '--id=3', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True))
            if 'GPU-b493bb52-b935-1cf4-47fe-7869125775b9' not in apps and memory < 128:
                break
            if time.monotonic() > deadline:
                raise RuntimeError('GPU3 occupied; foreign experiment is preserved')
            time.sleep(10)
        tag = 'torch_profile_cap' + str(cap)
        state('profiling', cap=cap, tag=tag)
        cmd = [PY, str(R / 'profile_driver.py'), '--tag', tag, '--mode', 'hf', '--k-cap', str(cap), '--diagnostic', '--requests', '16', '--max-tokens', '64']
        with (R / (tag + '_driver.log')).open('w') as log:
            subprocess.run(cmd, cwd=R, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1500)
        directory = R / tag
        released = json.loads((directory / 'released.json').read_text())
        assert released['port_closed'] and not released['our_gpu_pids']
        text = (directory / 'server.log').read_text()
        assert all(x not in text for x in ['Failed to start profiler', 'Failed to stop profiler', 'CUDA error:', 'Cross-request draft KV slot collision', 'Traceback (most recent call last)'])
        traces = list((directory / 'trace').glob('*.pt.trace.json*'))
        assert traces
        for index, trace in enumerate(traces):
            output = directory / ('TRACE_SUMMARY_' + str(index) + '.json')
            with (directory / ('TRACE_ANALYSIS_' + str(index) + '.log')).open('w') as log:
                subprocess.run([PY, str(R / 'analyze_trace.py'), str(trace), '--output', str(output)], cwd=R, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
            summary = json.loads(output.read_text())
            assert summary['gpu_kernel_count'] > 0 and summary['cdb_ranges_count'] > 0, 'CPU-only or unlabeled trace is insufficient'
    state('traces_collected', profiles=[8], note='Diagnostic timings only; no kernel optimization claimed. Llama traces collected after unprofiled C16 repeats.')
except BaseException:
    state('failed', error=traceback.format_exc())
    raise
finally:
    fcntl.flock(lock, fcntl.LOCK_UN)
    lock.close()
