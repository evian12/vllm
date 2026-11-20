"""Bounded algorithm audit: actual BF16 weights, synthetic hidden inputs; not inference."""
from pathlib import Path
import ctypes as C,json,subprocess,fcntl,time,traceback,statistics,hashlib,math,gc,os
r=Path(__file__).resolve().parent
lock=open('/tmp/cdb_gpu3_test.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def save(name,data):(r/name).write_text(json.dumps(data,indent=2)+'\n')
uuid=subprocess.check_output(['nvidia-smi','--id=3','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
assert uuid not in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
assert int(subprocess.check_output(['nvidia-smi','--id=3','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True))<128
import torch
from safetensors import safe_open
lib=C.CDLL(str(r/'lt_probe.so'));lib.configure.argtypes=[C.c_int,C.c_int,C.c_int,C.c_size_t];lib.run.argtypes=[C.c_int]+[C.c_void_p]*4+[C.c_size_t,C.c_void_p];lib.algorithm_id.argtypes=[C.c_int]
torch.manual_seed(42);records=[];scratch=torch.empty(32*1024**2,dtype=torch.uint8,device='cuda');stream=torch.cuda.current_stream().cuda_stream

def call(i,x,w,y):
 code=lib.run(i,x.data_ptr(),w.data_ptr(),y.data_ptr(),scratch.data_ptr(),scratch.numel(),torch.cuda.current_stream().cuda_stream)
 if code:raise RuntimeError(('cublasLt status',code))
def timed(action,count):
 graph=torch.cuda.CUDAGraph()
 with torch.cuda.graph(graph):action()
 for _ in range(5):graph.replay()
 torch.cuda.synchronize();samples=[]
 for _ in range(5):
  a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record()
  for __ in range(10):graph.replay()
  b.record();b.synchronize();samples.append(a.elapsed_time(b)*1000/(10*count))
 del graph
 return samples

def weights(base,kind):
 files=list(base.glob('*.safetensors'));mapping={}
 for p in files:
  with safe_open(p,framework='pt',device='cpu') as f:
   for k in f.keys():mapping[k]=p
 def get(key):
  with safe_open(mapping[key],framework='pt',device='cpu') as f:return f.get_tensor(key)
 keys=['gate_proj','up_proj'] if kind=='gate_up' else ['down_proj']
 first=torch.cat([get('model.layers.0.mlp.'+k+'.weight') for k in keys],dim=0);n,k=first.shape;bank=max(2,math.ceil(256*1024**2/(n*k*2)));result=[first.cuda()]
 for layer in range(1,bank):result.append(torch.cat([get(f'model.layers.{layer}.mlp.'+key+'.weight') for key in keys],dim=0).cuda())
 return result,n,k
try:
 save('STATUS.json',dict(state='running',pid=os.getpid(),scope='actual weights with synthetic hidden rows; algorithm audit only'))
 # Known layout contract check before large real weights.
 x=torch.randn(2,8,device='cuda',dtype=torch.bfloat16);w=torch.randn(6,8,device='cuda',dtype=torch.bfloat16);y=torch.empty(2,6,device='cuda',dtype=torch.bfloat16)
 assert lib.configure(2,6,8,scratch.numel())>0;call(0,x,w,y);torch.cuda.synchronize();assert torch.allclose(y.float(),torch.nn.functional.linear(x,w).float(),atol=.0625,rtol=.01)
 del x,w,y
 for model,base,kind,ms in [('target',Path('/disk14T/yypan/models/Llama-3.1-8B-Instruct'),'gate_up',[12,20]),('target',Path('/disk14T/yypan/models/Llama-3.1-8B-Instruct'),'down',[12]),('draft',Path('/disk14T/yypan/CDB-AV-runtime/baseline-comparison-20261113/models/Llama-3.2-1B-Instruct'),'gate_up',[4,5])]:
  ws,n,k=weights(base,kind)
  for m in ms:
   x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16);refs=[torch.nn.functional.linear(x,w) for w in ws];ys=[torch.empty_like(y) for y in refs]
   def reference():
    for index,w in enumerate(ws):refs[index]=torch.nn.functional.linear(x,w)
   baseline=timed(reference,len(ws));count=lib.configure(m,n,k,scratch.numel());row=dict(model=model,kind=kind,M=m,N=n,K=k,bank=len(ws),bank_bytes=sum(w.numel()*w.element_size() for w in ws),baseline_us=baseline,algorithms=[])
   for i in range(count):
    try:
     def action():
      for w,y in zip(ws,ys):call(i,x,w,y)
     action();torch.cuda.synchronize();exact=all(torch.equal(a,b) for a,b in zip(refs,ys));max_abs=max((a.float()-b.float()).abs().max().item() for a,b in zip(refs,ys))
     samples=timed(action,len(ws));gain=100*(statistics.median(baseline)/statistics.median(samples)-1)
     row['algorithms'].append(dict(index=i,id=lib.algorithm_id(i),exact=exact,max_abs=max_abs,samples_us=samples,gain_pct=gain))
    except RuntimeError as e:row['algorithms'].append(dict(index=i,error=str(e)))
   records.append(row);save('PARTIAL.json',records);print(model,kind,m,'bestexact',max((a.get('gain_pct',-999) for a in row['algorithms'] if a.get('exact')),default=None),flush=True)
   del x,refs,ys;gc.collect();torch.cuda.empty_cache()
  del ws;gc.collect();torch.cuda.empty_cache()
 save('RESULT.json',dict(records=records,limits='Real weights, random hidden inputs; exact on these samples is not model correctness. Micro gains are not system gains. No runtime integration. Existing default backend switch experiments were not repeated; enumerated specific heuristics.',so_sha256=hashlib.sha256((r/'lt_probe.so').read_bytes()).hexdigest()))
 save('STATUS.json',dict(state='completed',pid=os.getpid()))
except BaseException:
 save('STATUS.json',dict(state='failed',error=traceback.format_exc()));raise
finally:
 torch.cuda.synchronize()
 fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
