# SPDX-License-Identifier: Apache-2.0
import argparse,hashlib,json,math,os,statistics,subprocess,socket,time
from pathlib import Path
import torch
from safetensors import safe_open
from vllm import _custom_ops as ops
from build import load_built,module_path

def same(a,b):
    return torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))

def stats(a,b):
    if not a.numel():
        return dict(applicable=False,byte_exact=None,max_abs=None,rms=None,rel_l2=None)
    aa=a.float();bb=b.float();delta=aa-bb;ref=bb.norm().item()
    finite=bool(torch.isfinite(aa).all() and torch.isfinite(bb).all())
    return dict(applicable=True,byte_exact=same(a,b),finite=finite,
        max_abs=delta.abs().max().item() if finite else None,
        rms=delta.square().mean().sqrt().item() if finite else None,
        rel_l2=delta.norm().item()/ref if finite and ref else None)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args()
    out=Path(args.out);out.mkdir()
    used=int(subprocess.check_output(['nvidia-smi','--id=3','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True))
    uuid=subprocess.check_output(['nvidia-smi','--id=3','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    assert used<1000
    assert uuid not in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).splitlines()
    with socket.socket() as sock:assert sock.connect_ex(('127.0.0.1',18084))!=0
    assert os.environ['CUDA_VISIBLE_DEVICES']=='3' and torch.cuda.device_count()==1
    ext=load_built();torch.manual_seed(42);records=[];bench=[]
    configs={}
    for name in ('draft','target'):
        path=Path('/disk14T/yypan/models')/('Qwen3-1.7B' if name=='draft' else 'Qwen3-8B')
        c=json.loads((path/'config.json').read_text());idx=json.loads((path/'model.safetensors.index.json').read_text())['weight_map']
        configs[name]=dict(path=str(path),config=c,config_sha256=hashlib.sha256((path/'config.json').read_bytes()).hexdigest())
        nq=c['num_attention_heads'];nk=c['num_key_value_heads'];assert c['head_dim']==128
        weights=[]
        for layer in range(c['num_hidden_layers']):
            pair=[]
            for kind in ('q','k'):
                key=f'model.layers.{layer}.self_attn.{kind}_norm.weight'
                with safe_open(path/idx[key],framework='pt',device='cpu') as f:
                    pair.append(f.get_tensor(key).to(device='cuda',dtype=torch.bfloat16))
            weights.append(pair)
        theta=c.get('rope_theta',1e6)
        inv=1/(theta**(torch.arange(0,128,2,device='cuda',dtype=torch.float32)/128))
        freq=torch.arange(1024,device='cuda',dtype=torch.float32)[:,None]*inv[None,:]
        cache=torch.cat([freq.cos(),freq.sin()],-1).to(torch.bfloat16).contiguous()
        width=(nq+2*nk)*128

        def case(rows,weight_pair,label,pattern=None):
            x=torch.randn(rows,width,device='cuda',dtype=torch.bfloat16)
            if pattern=='zero':x.zero_()
            elif pattern=='tiny':x.mul_(1e-20)
            elif pattern=='subnormal':x.fill_(1e-40)
            elif pattern=='large':x.mul_(1e20)
            elif pattern=='max':x.fill_(torch.finfo(torch.bfloat16).max)
            elif pattern=='constant':x.fill_(1)
            elif pattern=='alternating':x[:,::2]=1;x[:,1::2]=-1
            pos=torch.randint(0,1024,(rows,),device='cuda',dtype=torch.int64)
            if rows==4:pos.copy_(torch.tensor([0,1,1022,1023],device='cuda'))
            qw,kw=weight_pair
            qo=torch.empty(rows,nq*128,device='cuda',dtype=torch.bfloat16)
            ko=torch.empty(rows,nk*128,device='cuda',dtype=torch.bfloat16)
            rq=torch.empty_like(qo);rk=torch.empty_like(ko)
            before=x.clone();wq=qw.clone();wk=kw.clone();cache_before=cache.clone();pos_before=pos.clone()
            q,k,v=x.split([nq*128,nk*128,nk*128],-1)
            def reference():
                if rows:
                    ops.rms_norm(rq.view(rows,nq,128),q.view(rows,nq,128),qw,1e-6)
                    ops.rms_norm(rk.view(rows,nk,128),k.view(rows,nk,128),kw,1e-6)
                    ops.rotary_embedding(pos,rq,rk,128,cache,True)
            def fused():
                ext.forward(x,qw,kw,cache,pos,qo,ko,1e-6,nq,nk)
            reference();fused();torch.cuda.synchronize()
            record=dict(model=name,label=label,rows=rows,Q=stats(qo,rq),K=stats(ko,rk),
                input_byte_preserved=same(x,before),weights_preserved=same(qw,wq) and same(kw,wk),
                positions_preserved=same(pos,pos_before),cache_preserved=same(cache,cache_before))
            records.append(record)
            return reference,fused,record

        ones=(torch.ones(128,device='cuda',dtype=torch.bfloat16),torch.ones(128,device='cuda',dtype=torch.bfloat16))
        for rows in (1,4,8,16,64,256):
            for label,pair in [('ones',ones),('layer0',weights[0])]:
                case(rows,pair,label)
        for i,pair in enumerate(weights):
            case(4,pair,f'learned_layer{i}')
        for pattern in ('zero','tiny','subnormal','large','max','constant','alternating'):
            case(4,weights[0],pattern,pattern)
        zeros=(torch.zeros(128,device='cuda',dtype=torch.bfloat16),torch.zeros(128,device='cuda',dtype=torch.bfloat16))
        case(4,zeros,'zero_weights')
        case(0,ones,'empty')
        (out/'accuracy.json').write_text(json.dumps(dict(records=records,configs=configs),indent=2))
        bad=[z for z in records if any(z[k]['applicable'] and not z[k]['byte_exact'] for k in ('Q','K')) or not all(z[k] for k in ('input_byte_preserved','weights_preserved','positions_preserved','cache_preserved'))]
        print('ACCURACY',name,'cases',len(records),'mismatches',len(bad),flush=True)

    bad=[z for z in records if any(z[k]['applicable'] and not z[k]['byte_exact'] for k in ('Q','K')) or not all(z[k] for k in ('input_byte_preserved','weights_preserved','positions_preserved','cache_preserved'))]
    assert not bad,'Exact-fusion comparison failed; do not benchmark/promote'
    print('ALL EXACT',len(records),flush=True)
    (out/'ACCEPTED.json').write_text(json.dumps(dict(cases=len(records),all_byte_exact=True,
        binary=str(module_path()),binary_sha256=hashlib.sha256(module_path().read_bytes()).hexdigest(),
        torch=torch.__version__,scope='Independent operators only; no model or TPS validation.'),indent=2))

if __name__=='__main__':
    with torch.inference_mode():main()
