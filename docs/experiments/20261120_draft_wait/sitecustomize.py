"""Observe eager model forward invocations without changing decoding or tensors.
Enabled only in the private server subprocess. No per-forward I/O or GPU sync.
"""
import os
if os.environ.get('BASELINE_FORWARD_COUNTS'):
 import sys,importlib.abc,importlib.machinery,threading,time,json,pathlib,functools
 _counts={};_started=False
 def _writer():
  root=pathlib.Path(os.environ['BASELINE_FORWARD_COUNTS']);root.mkdir(parents=True,exist_ok=True)
  p=root/f'{os.getpid()}.json';tmp=p.with_suffix('.tmp')
  while True:
   tmp.write_text(json.dumps(dict(pid=os.getpid(),models=list(_counts.values()))));tmp.replace(p);time.sleep(.1)
 def _wrap(cls):
  original=cls.forward
  @functools.wraps(original)
  def wrapped(self,*args,**kwargs):
   global _started
   key=id(self)
   if key not in _counts:
    _counts[key]=dict(model=getattr(self.config,'_name_or_path','unknown'),model_class=type(self).__name__,calls=0,input_tokens=0)
   row=_counts[key];row['calls']+=1
   ids=kwargs.get('input_ids',args[0] if args else None)
   if ids is not None:row['input_tokens']+=ids.numel()
   if not _started:
    _started=True;threading.Thread(target=_writer,daemon=True).start()
   return original(self,*args,**kwargs)
  cls.forward=wrapped
 class _Loader(importlib.abc.Loader):
  def __init__(self,inner):self.inner=inner
  def create_module(self,spec):return self.inner.create_module(spec)
  def exec_module(self,module):
   self.inner.exec_module(module)
   name={'vllm.model_executor.models.qwen3':'Qwen3ForCausalLM','vllm.model_executor.models.llama':'LlamaForCausalLM'}[module.__name__]
   _wrap(getattr(module,name))
 class _Finder(importlib.abc.MetaPathFinder):
  def find_spec(self,fullname,path=None,target=None):
   if fullname in ('vllm.model_executor.models.qwen3','vllm.model_executor.models.llama'):
    spec=importlib.machinery.PathFinder.find_spec(fullname,path)
    if spec:spec.loader=_Loader(spec.loader)
    return spec
 sys.meta_path.insert(0,_Finder())

if os.environ.get('BASELINE_OBSERVE_DIRECTORY'):
 import manual_observer

if (os.environ.get('BASELINE_OBSERVE_DIRECTORY')
        and os.environ.get('CDB_AV_COMPILE_EAGER_AUDIT') == '1'):
 import compile_audit
 compile_audit.install(manual_observer)
