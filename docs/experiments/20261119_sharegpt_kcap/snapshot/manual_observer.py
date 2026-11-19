"""Read-only runtime observations at native model-call and scheduler boundaries.
No GPU tensor reads, synchronization, per-step I/O, or decoding changes.
"""
import os,pathlib,json,time,threading,functools,sys,importlib.abc,importlib.machinery
import math
ROOT=pathlib.Path(os.environ['BASELINE_OBSERVE_DIRECTORY'])
LOCK=threading.Lock();STARTED=False;EPOCH='warmup';ROWS=[]
STATS=dict(target_forward_calls=0,draft_forward_calls=0,max_waiting_requests=0,max_running_requests=0)
STATS.update(cdb_draft_dispatches=0,cdb_draft_request_rows=0,cdb_draft_token_rows=0)

def observe_target_schedule(scheduler,output):
 rows=output.num_scheduled_tokens
 if not rows:return
 verify=set(output.scheduled_spec_decode_tokens)
 counts=dict(cdb_target_schedules=1,cdb_target_request_rows=len(rows),cdb_target_token_rows=sum(rows.values()),cdb_target_verify_rows=len(verify),cdb_target_prefill_rows=0,cdb_target_ar_rows=0,cdb_target_other_rows=0)
 waits=[];now=time.monotonic();p2=getattr(scheduler,'_cdb_p2',None)
 for rid,n in rows.items():
  req=scheduler.requests.get(rid)
  if rid in verify:
   ready=getattr(p2,'ready',{}).get(rid,{})
   wait=ready.get('ready_at')
   if wait is not None:
    wait=(now-wait)*1000
    if math.isfinite(wait) and wait>=0:waits.append(wait)
    else:counts['cdb_ready_wait_invalid']=counts.get('cdb_ready_wait_invalid',0)+1
  elif req is not None and req.num_output_tokens==0:counts['cdb_target_prefill_rows']+=1
  elif req is not None and n==1:counts['cdb_target_ar_rows']+=1
  else:counts['cdb_target_other_rows']+=1
 if counts['cdb_target_ar_rows']==len(rows):
  counts['cdb_target_pure_ar_calls']=1
  if scheduler.waiting:counts['cdb_target_pure_ar_with_waiting']=1
 start()
 with LOCK:
  for key,value in counts.items():STATS[key]=STATS.get(key,0)+value
  for wait in waits:
   STATS['cdb_ready_wait_count']=STATS.get('cdb_ready_wait_count',0)+1
   STATS['cdb_ready_wait_sum_ms']=STATS.get('cdb_ready_wait_sum_ms',0)+wait
   STATS['cdb_ready_wait_max_ms']=max(STATS.get('cdb_ready_wait_max_ms',0),wait)
   bound=next((b for b in [1,2,5,10,20,50,100,250,500,1000] if wait<=b),'inf')
   key='cdb_ready_wait_bucket_'+str(bound)+'_ms';STATS[key]=STATS.get(key,0)+1
def start():
 global STARTED
 if not STARTED:
  STARTED=True;threading.Thread(target=writer,daemon=True).start()
def record(key,value=1,peak=False):
 start()
 with LOCK:STATS[key]=max(STATS[key],value) if peak else STATS[key]+value

def writer():
 global EPOCH
 ROOT.mkdir(parents=True,exist_ok=True);p=ROOT/f'{os.getpid()}.json';tmp=p.with_suffix('.tmp')
 while True:
  control=ROOT/'epoch.txt'
  e=control.read_text().strip() if control.exists() else 'warmup'
  with LOCK:
   if e!=EPOCH:EPOCH=e;STATS['max_waiting_requests']=0;STATS['max_running_requests']=0
   snap=dict(pid=os.getpid(),epoch=EPOCH,**STATS);rows=ROWS[:];ROWS.clear()
  tmp.write_text(json.dumps(snap));tmp.replace(p)
  if rows:
   with (ROOT/f'{os.getpid()}_requests.jsonl').open('a') as f:
    for row in rows:f.write(json.dumps(row)+'\n')
  time.sleep(.1)

def patch_runner(module):
 cls=module.GPUModelRunner;old=cls._model_forward
 @functools.wraps(old)
 def forward(self,*args,**kwargs):
  record('target_forward_calls');return old(self,*args,**kwargs)
 cls._model_forward=forward

def patch_draft(module):
 cls=module.DraftModelProposer;old=cls.load_model
 @functools.wraps(old)
 def load(self,*args,**kwargs):
  result=old(self,*args,**kwargs);model=self.model;model._baseline_observed_draft=True;kind=type(model)
  if not getattr(kind,'_baseline_call_wrapped',False):
   original=kind.__call__
   @functools.wraps(original)
   def call(instance,*a,**kw):
    if getattr(instance,'_baseline_observed_draft',False):record('draft_forward_calls')
    return original(instance,*a,**kw)
   kind.__call__=call;kind._baseline_call_wrapped=True
  return result
 cls.load_model=load

def patch_refill(module):
 cls=module.RefillSession;old=cls._forward
 @functools.wraps(old)
 def forward(self,ids,positions,cad,last_indices,req_ids,phase):
  result=old(self,ids,positions,cad,last_indices,req_ids,phase)
  # Tensor shape metadata only: no GPU values, copies, or synchronization.
  # One dispatch may also pay Graph capture work internally; these counters
  # describe logical forwards and real input rows, not capture kernel counts.
  start()
  with LOCK:
   STATS['cdb_draft_dispatches']+=1
   STATS['cdb_draft_request_rows']+=len(req_ids)
   STATS['cdb_draft_token_rows']+=ids.numel()
   for suffix,value in [('calls',1),('request_rows',len(req_ids)),('token_rows',ids.numel())]:
    key='cdb_phase_'+phase+'_'+suffix
    STATS[key]=STATS.get(key,0)+value
  return result
 cls._forward=forward

def patch_scheduler(module):
 cls=module.Scheduler
 def observe(self):
  record('max_waiting_requests',len(self.waiting),True);record('max_running_requests',len(self.running),True)
 for name in ['add_request','schedule']:
  old=getattr(cls,name)
  def build(original,is_schedule):
   @functools.wraps(original)
   def wrapped(self,*args,**kwargs):
    observe(self);result=original(self,*args,**kwargs);observe(self)
    if is_schedule:observe_target_schedule(self,result)
    return result
   return wrapped
  setattr(cls,name,build(old,name=='schedule'))

def patch_output(module):
 cls=module.RequestState;old=cls.make_request_output
 @functools.wraps(old)
 def make(self,*args,**kwargs):
  result=old(self,*args,**kwargs)
  if result is not None and result.finished and result.metrics is not None:
   s=result.metrics
   row=dict(response_id=result.request_id,epoch=EPOCH,arrival_time=s.arrival_time,queued_ts=s.queued_ts,scheduled_ts=s.scheduled_ts,first_token_ts=s.first_token_ts,last_token_ts=s.last_token_ts,queue_time_s=s.scheduled_ts-s.queued_ts,num_generation_tokens=s.num_generation_tokens)
   start()
   with LOCK:ROWS.append(row)
  return result
 cls.make_request_output=make

PATCHES={'vllm.v1.worker.gpu_model_runner':patch_runner,'vllm.v1.spec_decode.draft_model':patch_draft,'vllm.v1.spec_decode.cdb_av_refill':patch_refill,'vllm.v1.core.sched.scheduler':patch_scheduler,'vllm.v1.engine.output_processor':patch_output}
class Loader(importlib.abc.Loader):
 def __init__(self,inner):self.inner=inner
 def create_module(self,spec):return self.inner.create_module(spec)
 def exec_module(self,module):self.inner.exec_module(module);PATCHES[module.__name__](module)
class Finder(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname in PATCHES:
   spec=importlib.machinery.PathFinder.find_spec(fullname,path)
   if spec:spec.loader=Loader(spec.loader)
   return spec
sys.meta_path.insert(0,Finder())
