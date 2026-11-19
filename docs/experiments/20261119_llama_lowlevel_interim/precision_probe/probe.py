"""Bounded synthetic linear-row padding probe, not an inference benchmark.

Model and runtime files are never modified. Own process holds the shared GPU
lock; its allocations disappear on exit. Graph timings include padding work.
"""
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import time


R = Path(__file__).resolve().parent


def cases(configs):
    cfg = configs['draft']
    return [('draft','down',m,cfg['hidden_size'],cfg['intermediate_size'])
            for m in [1,2,3,4,5,6,7,8,9,10,12,16,24]]


def run():
    import torch
    import torch.nn.functional as F
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '3'
    assert not (R/'RESULTS.json').exists()
    configs = {}
    provenance = {}
    for label, directory in [('target','/disk14T/yypan/models/Llama-3.1-8B-Instruct'),
                             ('draft','/disk14T/yypan/CDB-AV-runtime/baseline-comparison-20261113/models/Llama-3.2-1B-Instruct')]:
        p = Path(directory)/'config.json'
        configs[label] = json.loads(p.read_text())
        assert configs[label]['model_type'] == 'llama'
        provenance[label] = dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                                 config=configs[label])
    (R/'MODEL_CONFIGS.json').write_text(json.dumps(provenance,indent=2)+'\n')
    (R/'RUNTIME.json').write_text(json.dumps(dict(torch=torch.__version__,cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(0),dtype='torch.bfloat16',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        preferred_blas=str(torch.backends.cuda.preferred_blas_library())),indent=2)+'\n')
    torch.manual_seed(42)
    records = []
    with torch.inference_mode():
        for label, projection, m, n, k in cases(configs):
            # Stream at least256MiB of weights, larger than 4090 L2.
            bank = max(1, math.ceil(256*1024**2/(n*k*2)))
            weights = [torch.randn(n,k,device='cuda',dtype=torch.bfloat16) / math.sqrt(k)
                       for _ in range(bank)]
            column = [w.t().contiguous().t() for w in weights]
            assert all(torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)) for a,b in zip(weights,column))
            x = torch.randn(m,k,device='cuda',dtype=torch.bfloat16)
            default_reduce = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
            variants = dict(native=(m,False,default_reduce),pad32=(32,False,default_reduce),
                            native_fp32=(m,False,False),pad32_fp32=(32,False,False),
                            column=(m,True,default_reduce),column_fp32=(m,True,False))
            graphs,outputs = {},{}
            def forward(bound,use_column):
                padded = F.pad(x,(0,0,0,bound-m)) if bound > m else x
                return [F.linear(padded,w)[:m] for w in (column if use_column else weights)]
            try:
                for name,(bound,use_column,reduced) in variants.items():
                    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = reduced
                    for _ in range(3): out=forward(bound,use_column)
                    torch.cuda.synchronize()
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):out=forward(bound,use_column)
                    graphs[name],outputs[name]=graph,out
                    graph.replay()
            finally:
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=default_reduce
            samples = {name:[] for name in variants}
            for trial in range(8):
                names = list(variants)
                if trial % 2: names.reverse()
                for name in names:
                    for _ in range(4): graphs[name].replay()
                    begin,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    begin.record()
                    for _ in range(16): graphs[name].replay()
                    end.record();end.synchronize()
                    samples[name].append(begin.elapsed_time(end)*1000/(16*bank))
            ref = torch.cat(outputs['native']).contiguous()
            comparisons = {}
            for name in variants:
                candidate = torch.cat(outputs[name]).contiguous()
                delta = candidate.double()-ref.double()
                norm = ref.double().norm().item()
                comparisons[name] = dict(byte_exact=torch.equal(ref.view(torch.uint8),candidate.view(torch.uint8)),
                    different_bytes=int((ref.view(torch.uint8)!=candidate.view(torch.uint8)).sum()),
                    finite=bool(torch.isfinite(candidate).all()),max_abs=delta.abs().max().item(),
                    rms=delta.square().mean().sqrt().item(),rel_l2=delta.norm().item()/norm if norm else None,
                    argmax_equal_rows=int((ref.argmax(-1)==candidate.argmax(-1)).sum()),compared_rows=ref.shape[0],
                    median_us=statistics.median(samples[name]),samples_us=samples[name],
                    speedup=statistics.median(samples['native'])/statistics.median(samples[name]))
            records.append(dict(model=label,projection=projection,M=m,N=n,K=k,bank=bank,
                                streamed_weight_bytes=bank*n*k*2,comparisons=comparisons))
            (R/'RESULTS_PARTIAL.json').write_text(json.dumps(records,indent=2)+'\n')
            print(label,projection,m,{a:round(v['speedup'],3) for a,v in comparisons.items()},flush=True)
            del graphs,outputs,weights,column,out,graph,x,ref,candidate,delta
            gc.collect();torch.cuda.empty_cache()
    (R/'RESULTS.json').write_text(json.dumps(dict(records=records,torch=torch.__version__,cuda=torch.version.cuda,
        scope='Synthetic Draft down projection. Padding included, exact logical BF16 weights. Native FP32-reduction and physical-column variants compare against original native default. Graphs retain kernels selected during capture; process global reduction flag restored immediately after capture. Not model/TPS evidence.'),indent=2)+'\n')


if __name__ == '__main__':
    def expire(*args): raise TimeoutError('15 minute bounded component probe')
    signal.signal(signal.SIGALRM,expire);signal.alarm(900)
    with open('/tmp/cdb_gpu3_test.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True)
        memory=int(subprocess.check_output(['nvidia-smi','--id=3','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True))
        assert 'GPU-b493bb52-b935-1cf4-47fe-7869125775b9' not in apps and memory <128
        with socket.socket() as s: assert s.connect_ex(('127.0.0.1',18084)) !=0
        (R/'PREFLIGHT.json').write_text(json.dumps(dict(memory_mib=memory,apps=apps,time=time.time()),indent=2))
        run()
