"""Read actual BF16 Target weights; compare native/column physical strides.

Standalone component experiment, never installs into vLLM. The controller must
reserve GPU3 and wait for the existing paired experiment to finish first.
"""
import gc
import hashlib
import json
import math
from pathlib import Path
import signal
import statistics

from plan import cases

R = Path(__file__).resolve().parent
MODEL = Path('/disk14T/yypan/models/Llama-3.1-8B-Instruct')


def run():
    import os
    assert os.environ['CUDA_VISIBLE_DEVICES']=='3'
    import torch
    from safetensors import safe_open
    import torch.nn.functional as F
    signal.signal(signal.SIGALRM, lambda *args: (_ for _ in ()).throw(TimeoutError('Bounded layout probe')))
    signal.alarm(240)
    assert not (R/'RESULTS.json').exists()
    raw=(MODEL/'config.json').read_bytes(); config=json.loads(raw)
    index=json.loads((MODEL/'model.safetensors.index.json').read_bytes())['weight_map']
    assert (config['hidden_size'], config['intermediate_size'], config['num_hidden_layers'])==(4096,14336,32)
    records=[]; provenance={}

    def load(name):
        path=MODEL/index[name]
        with safe_open(path,framework='pt',device='cpu') as f:
            tensor=f.get_tensor(name)
        assert tensor.dtype==torch.bfloat16 and tensor.ndim==2
        provenance[name]=dict(file=str(path),shape=list(tensor.shape),dtype=str(tensor.dtype))
        return tensor

    def weight(projection, layer):
        prefix=f'model.layers.{layer}.'
        if projection=='qkv':
            return torch.cat([load(prefix+'self_attn.'+n+'_proj.weight') for n in ['q','k','v']])
        if projection=='gate_up':
            return torch.cat([load(prefix+'mlp.'+n+'_proj.weight') for n in ['gate','up']])
        if projection=='down':return load(prefix+'mlp.down_proj.weight')
        if projection=='o':return load(prefix+'self_attn.o_proj.weight')
        return load('lm_head.weight')

    torch.manual_seed(42)
    with torch.inference_mode():
        rows=cases(config)
        for projection in dict.fromkeys(x['projection'] for x in rows):
            selected=[x for x in rows if x['projection']==projection]
            bank=selected[0]['bank']
            # LM head is already >256MiB; every other bank uses distinct layers.
            native=[weight(projection,j).to('cuda') for j in range(bank)]
            n,k=native[0].shape
            assert all(w.shape==(n,k) and w.is_contiguous() for w in native)
            column=[w.t().contiguous().t() for w in native]
            assert all(torch.equal(a.view(torch.uint8),b.contiguous().view(torch.uint8))
                       for a,b in zip(native,column))
            for case in selected:
                assert (case['n'],case['k'])==(n,k)
                x=torch.randn(case['m'],k,device='cuda',dtype=torch.bfloat16)/math.sqrt(k)
                graphs={}; outputs={}
                for label,weights in [('native',native),('column',column)]:
                    for _ in range(3): out=[F.linear(x,w) for w in weights]
                    torch.cuda.synchronize()
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):out=[F.linear(x,w) for w in weights]
                    graphs[label]=graph;outputs[label]=out;graph.replay()
                samples={'native':[],'column':[]}
                for trial in range(8):
                    for label in (['native','column'] if trial%2==0 else ['column','native']):
                        for _ in range(3):graphs[label].replay()
                        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        start.record()
                        for _ in range(12):graphs[label].replay()
                        end.record();end.synchronize()
                        samples[label].append(start.elapsed_time(end)*1000/(12*bank))
                a=torch.cat(outputs['native']).contiguous()
                b=torch.cat(outputs['column']).contiguous()
                delta=b.double()-a.double();norm=a.double().norm().item()
                record=dict(case,logical_weight_bytes_exact=True,
                    native_stride=list(native[0].stride()),column_stride=list(column[0].stride()),
                    native_us=statistics.median(samples['native']),column_us=statistics.median(samples['column']),
                    samples_us=samples,output_byte_exact=torch.equal(a.view(torch.uint8),b.view(torch.uint8)),
                    different_bytes=int((a.view(torch.uint8)!=b.view(torch.uint8)).sum()),
                    finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
                    max_abs=delta.abs().max().item(),rms=delta.square().mean().sqrt().item(),
                    rel_l2=delta.norm().item()/norm if norm else None,
                    argmax_equal_rows=int((a.argmax(-1)==b.argmax(-1)).sum()),compared_rows=len(a))
                record['gain_pct']=100*(record['native_us']/record['column_us']-1)
                records.append(record)
                (R/'RESULTS_PARTIAL.json').write_text(json.dumps(records,indent=2)+'\n')
                print(json.dumps({k:v for k,v in record.items() if k!='samples_us'}),flush=True)
                del graphs,outputs,out,graph,a,b,delta,x
            del native,column
            gc.collect();torch.cuda.empty_cache()
    signal.alarm(0)
    result=dict(records=records,config_sha256=hashlib.sha256(raw).hexdigest(),weights=provenance,
        torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Actual BF16 logical weights; random synthetic hidden states; component Graph timings only, not inference TPS or token correctness; no vLLM/model files modified')
    (R/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':run()
