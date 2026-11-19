"""CPU range labels for a bounded private Torch trace; no tensor mutations."""
import functools
import json
import os
from pathlib import Path

import manual_observer

def key_label(key):
    return 'hex_'+json.dumps(key,separators=(',',':')).encode().hex()


TRACE_ACTIVE=False
SNAPSHOTS=set()


def wrap(cls, name, label):
    import torch
    original = getattr(cls, name)

    @functools.wraps(original)
    def observed(self, *args, **kwargs):
        text = label(self, args, kwargs) if callable(label) else label
        with torch.profiler.record_function(text):
            return original(self, *args, **kwargs)

    setattr(cls, name, observed)


def augment(module_name, operation):
    old = manual_observer.PATCHES.get(module_name)

    def patch(module):
        if old is not None:
            old(module)
        operation(module)

    manual_observer.PATCHES[module_name] = patch


def refill(module):
    wrap(module.RefillSession, '_forward',
         lambda self, args, kwargs: 'CDB::draft_forward::' + str(args[5])
         + '::r' + str(len(args[4])) + '::t' + str(args[0].numel()))
    wrap(module.RefillSession, '_decode_graph_forward',
         lambda self,args,kwargs: 'CDB::draft_graph_forward::'+key_label(args[2]))


def reuse(module):
    wrap(module.ReuseDraftSession, 'step', 'CDB::draft_step_with_sampling')
    wrap(module.ReuseDraftSession, 'prepare', 'CDB::draft_prepare')
    wrap(module.ReuseDraftSession, '_clear_slots',
         lambda self, args, kwargs: 'CDB::clear_slots::' + str(args[2] - args[1]))
    wrap(module.ReuseDraftQueue, 'before_target', 'CDB::kv_ownership_before_target')
    wrap(module.ReuseDraftQueue, 'plans', 'CDB::draft_plans')
    wrap(module.ReuseDraftQueue, 'advance', 'CDB::engine_draft_advance')
    wrap(module.ReuseDraftQueue, 'reconcile', 'CDB::engine_reconcile')


def runner(module):
    def target_label(self, args, kwargs):
        ids = kwargs.get('input_ids', args[0] if args else None)
        return 'CDB::target_forward::t' + str(ids.numel() if ids is not None else 'none')

    wrap(module.GPUModelRunner, '_model_forward', target_label)
    wrap(module.GPUModelRunner, 'sample_tokens', 'CDB::target_sampling')
    for method in ['_update_states', '_prepare_input_ids', '_prepare_inputs', '_build_attention_metadata', '_calc_spec_decode_metadata', '_preprocess']:
        wrap(module.GPUModelRunner, method, 'CDB::target_input' + method)



def draft(module):
    original = module.DraftModelProposer.load_model

    @functools.wraps(original)
    def loaded(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        kind = type(self.model)
        if not getattr(kind, '_cdb_profile_logits_wrapped', False):
            wrap(kind, 'compute_logits', lambda instance, args, kwargs:
                 'CDB::draft_logits' if getattr(instance, '_baseline_observed_draft', False)
                 else 'CDB::target_logits')
            kind._cdb_profile_logits_wrapped = True
        return result

    module.DraftModelProposer.load_model = loaded


augment('vllm.v1.worker.gpu_model_runner', runner)
augment('vllm.v1.spec_decode.draft_model', draft)
augment('vllm.v1.spec_decode.cdb_av_refill', refill)
augment('vllm.v1.spec_decode.cdb_av_kv_reuse', reuse)
def target_graph(module):
    wrap(module.TargetFullGraph, '_storage', 'CDB::target_storage_audit')
    original=module.TargetFullGraph._record
    @functools.wraps(original)
    def recorded(self,action,key=None,**kwargs):
        import torch
        with torch.profiler.record_function('CDB::target_dispatch::'+action+'::'+key_label(key)):
            result=original(self,action,key,**kwargs)
        if TRACE_ACTIVE and id(self) not in SNAPSHOTS:
            directory=os.environ['BASELINE_OBSERVE_DIRECTORY']
            path=Path(directory).parent/('target_graph_cache_'+str(os.getpid())+'.json')
            assert not path.exists()
            path.write_text(json.dumps(dict(keys=list(self.cache),limit=self.limit,
                seq_floor=self.seq_floor,dynamic=self.dynamic,first_profile_action=action,
                first_profile_key=key),indent=2)+'\n')
            SNAPSHOTS.add(id(self))
        return result
    module.TargetFullGraph._record=recorded


def executor(module):
    original=module.UniProcExecutor.collective_rpc
    @functools.wraps(original)
    def rpc(self,method,*args,**kwargs):
        import torch
        global TRACE_ACTIVE
        if method=='profile':
            result=original(self,method,*args,**kwargs)
            TRACE_ACTIVE=True
            return result
        operation=kwargs.get('args',())
        suffix=('::'+operation[0] if method=='cdb_refill_rpc'
                and operation and isinstance(operation[0],str) else '')
        with torch.profiler.record_function('CDB::worker_rpc::'+str(method)+suffix):
            return original(self,method,*args,**kwargs)
    module.UniProcExecutor.collective_rpc=rpc


def core(module):
    original=module.EngineCore.__init__
    @functools.wraps(original)
    def initialized(self,*args,**kwargs):
        result=original(self,*args,**kwargs)
        directory=os.getenv('BASELINE_OBSERVE_DIRECTORY')
        if directory:
            obj=self.model_executor
            facts=dict(pid=os.getpid(),executor=type(obj).__module__+'.'+type(obj).__name__,
                       backend=str(obj.parallel_config.distributed_executor_backend),
                       engine_source=module.__file__,
                       note='Actual instance, not inferred from CLI or logical RPC labels.')
            p=Path(directory).parent/('executor_runtime_'+str(os.getpid())+'.json')
            assert not p.exists()
            p.write_text(json.dumps(facts,indent=2)+'\n')
        return result
    module.EngineCore.__init__=initialized


augment('vllm.v1.spec_decode.cdb_av_target_graph',target_graph)
augment('vllm.v1.spec_decode.cdb_av_inputs',
        lambda m: wrap(m.PinnedDraftInputStager, 'pack', 'CDB::draft_metadata_pack'))
augment('vllm.v1.executor.uniproc_executor',executor)
augment('vllm.v1.engine.core',core)
