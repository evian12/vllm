"""Real CUDA copy/index/lifetime test; does not claim model inference acceptance."""
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

R=Path(__file__).resolve().parent
W=Path('/home/yypan/CDB-AV/worktrees/cdb-spec-metadata-20261119')


def main():
    import numpy as np
    import torch
    from vllm.v1.spec_decode import cdb_av_spec_metadata as mod
    spec=importlib.util.spec_from_file_location('native_spec_reference',W/'tests/v1/spec_decode/test_cdb_av_spec_metadata.py')
    reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
    assert os.environ['CUDA_VISIBLE_DEVICES']=='3'
    assert Path(mod.__file__).resolve()==(W/'vllm/v1/spec_decode/cdb_av_spec_metadata.py').resolve()
    source={n:hashlib.sha256((W/n).read_bytes()).hexdigest() for n in
            ['vllm/v1/spec_decode/cdb_av_spec_metadata.py','vllm/v1/worker/gpu_model_runner.py']}
    os.environ['CDB_AV_PACKED_SPEC_METADATA']='0'
    os.environ['CDB_AV_SPEC_METADATA_VALIDATE']='0'
    rng=np.random.default_rng(42)
    runner,calc=reference.native_runner()
    cpu_ids=torch.arange(1024,dtype=torch.int32)*17+3
    runner.input_ids=reference.NS(gpu=cpu_ids)
    gpu_ids=cpu_ids.cuda()
    stager=mod.Stager('cuda:0')
    cases=[];kept=[]
    for iteration in range(128):
        nr=int(rng.integers(1,5));dtype=np.int32 if iteration%2==0 else np.int64
        drafts=rng.integers(0,9,size=nr).astype(dtype)
        extras=rng.integers(0,50,size=nr).astype(dtype)
        if iteration==0:drafts=np.zeros(4,dtype=dtype);extras=np.zeros(4,dtype=dtype)
        if iteration==1:drafts=np.full(4,8,dtype=dtype);extras=np.zeros(4,dtype=dtype)
        cumulative=np.cumsum(drafts+1+extras,dtype=dtype)
        native=calc(runner,drafts,cumulative)
        arrays=[getattr(native,n).numpy() for n in mod.NAMES[:5]]
        arrays.append(mod.direct_indices(arrays[2],arrays[3],int(cumulative[-1])))
        plan=mod.packet(arrays)
        if iteration==0:
            # A deliberately small primitive-test source exercises safe growth.
            stager.host=torch.empty(plan.nbytes,dtype=torch.uint8,pin_memory=True)
            stager.host_view=stager.host.numpy()
            assert hasattr(torch.cuda,'_sleep'),'Cannot exercise delayed source ownership'
            torch.cuda._sleep(5_000_000)
        gpu=stager.copy(plan)
        ids=gpu_ids[gpu['draft_token_indices']]
        kept.append((gpu,ids,[a.copy() for a in arrays],native.draft_token_ids.clone()))
        cases.append(dict(requests=len(drafts),drafts=drafts.tolist(),scheduled=cumulative.tolist(),dtype=str(dtype),bytes=plan.nbytes))
    # No readback occurs during the burst; all previous GPU payloads must live.
    for gpu,ids,arrays,expected_ids in kept:
        for name,expected in zip(mod.NAMES,arrays):
            actual=gpu[name].cpu()
            assert np.array_equal(actual.numpy(),expected) and actual.numpy().dtype==expected.dtype,name
        assert torch.equal(ids.cpu(),expected_ids)
    assert stager.source_waits>0 and stager.growths>0
    assert len({v[0]['cu_num_draft_tokens'].data_ptr() for v in kept})==128
    torch.cuda.synchronize()
    (R/'GPU_PRIMITIVE.json').write_text(json.dumps(dict(cases=cases,passed=128,
        source_waits=stager.source_waits,growths=stager.growths,source=source,
        torch=torch.__version__,cuda=torch.version.cuda,actual_module=mod.__file__,
        held_payloads=128,all_metadata_and_draft_ids_exact=True,
        note='CUDA primitive correctness/lifetime only, no model or performance claim.'),indent=2)+'\n')
    print('128 CUDA metadata/index/lifetime cases passed',flush=True)


if __name__=='__main__':
    def expire(*args):raise TimeoutError('Three-minute primitive deadline')
    signal.signal(signal.SIGALRM,expire);signal.alarm(180)
    with open('/tmp/cdb_gpu3_test.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True)
        memory=int(subprocess.check_output(['nvidia-smi','--id=3','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True))
        assert 'GPU-b493bb52-b935-1cf4-47fe-7869125775b9' not in apps and memory<128
        with socket.socket() as sock:assert sock.connect_ex(('127.0.0.1',18084))!=0
        main()
