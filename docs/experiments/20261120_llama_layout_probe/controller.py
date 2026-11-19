"""Wait for the active C16 experiment; reserve/release only our GPU3 probe."""
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import traceback

R=Path(__file__).resolve().parent
PREVIOUS=R.parent/'context_fp_cache_20261120'
PY='/home/yypan/CDB-AV/.venv/bin/python'
UUID='GPU-b493bb52-b935-1cf4-47fe-7869125775b9'


def write(name,x):(R/name).write_text(json.dumps(x,indent=2)+'\n')
def status(value,**x):write('STATUS.json',dict(state=value,pid=os.getpid(),time=time.time(),**x))
def gpu():
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True)
    memory=int(subprocess.check_output(['nvidia-smi','--id=3','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True))
    return apps,memory


lock=(R/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
try:
    assert not (R/'RESULTS.json').exists()
    limit=time.monotonic()+7200
    status('waiting_for_existing_paired_experiment')
    while True:
        previous=json.loads((PREVIOUS/'STATUS.json').read_text())
        if previous['state']=='completed':break
        if previous['state']=='failed':raise RuntimeError('Prior paired experiment failed; no GPU probe launched')
        if time.monotonic()>limit:raise TimeoutError('Prior controller still running; no GPU probe launched')
        time.sleep(20)
    assert (PREVIOUS/'FUNCTIONAL_ACCEPTED.json').exists()
    with (R/'CPU_TESTS.log').open('w') as out:
        subprocess.run([PY,'-m','pytest','-q','test_plan.py'],cwd=R,stdout=out,stderr=subprocess.STDOUT,check=True,timeout=60)
    with open('/tmp/cdb_gpu3_test.lock','a') as reservation:
        fcntl.flock(reservation,fcntl.LOCK_EX|fcntl.LOCK_NB)
        apps,memory=gpu()
        assert UUID not in apps and memory<128, (memory,apps)
        with socket.socket() as s:assert s.connect_ex(('127.0.0.1',18084))!=0
        write('PREFLIGHT.json',dict(memory_mib=memory,apps=apps,previous_completed=previous,port_closed=True))
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='3',CUDA_DEVICE_ORDER='PCI_BUS_ID')
        status('bounded_gpu_component_probe')
        with (R/'probe.log').open('w') as out:
            process=subprocess.Popen([PY,str(R/'probe.py')],cwd=R,env=env,stdout=out,stderr=subprocess.STDOUT)
            write('PROCESS.json',dict(pid=process.pid))
            try:code=process.wait(timeout=300)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:process.kill();process.wait(timeout=10)
                raise
        time.sleep(2)
        apps,memory=gpu()
        assert str(process.pid)+',' not in apps
        write('RELEASED.json',dict(own_pid=process.pid,own_process_exited=True,memory_mib=memory,apps=apps,returncode=code))
        assert code==0, 'See retained probe.log; no runtime optimization installed'
        result=json.loads((R/'RESULTS.json').read_text())
        assert len(result['records'])==20
        assert all(x['logical_weight_bytes_exact'] and x['finite'] for x in result['records'])
        status('completed',comparisons=len(result['records']),runtime_modified=False)
except BaseException:
    status('failed',error=traceback.format_exc());raise
finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
