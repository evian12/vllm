"""Bounded read-only host attribution; no GPU tensor reads or synchronization.

Driver/API/allocator time in pack/forward is NOT pure Python time. Only CPU
planning and encoding are leaf CPU spans. Hook and writer overhead perturb
execution; this run is not an end-to-end throughput measurement.
"""
import functools
import threading
import time


def install(observer, limit=4096):
    tls=threading.local()
    counts={}

    def append(row):
        observer.start()
        with observer.LOCK:
            observer.ROWS.append(dict(kind='host_audit',epoch=observer.EPOCH,
                                      audit_ns=time.monotonic_ns(),**row))

    def wrap(owner, method, label, sample_inputs=False):
        original=getattr(owner,method)
        @functools.wraps(original)
        def observed(*args,**kwargs):
            epoch=observer.EPOCH
            key=(epoch,label)
            count=counts.get(key,0)+1;counts[key]=count
            if count>limit:return original(*args,**kwargs)
            stack=getattr(tls,'stack',None)
            if stack is None:stack=tls.stack=[]
            frame=[time.monotonic_ns(),0];stack.append(frame)
            try:
                result=original(*args,**kwargs)
                return result
            finally:
                elapsed=time.monotonic_ns()-frame[0]
                assert stack.pop() is frame
                if stack:stack[-1][1]+=elapsed
                append(dict(stage=label,ordinal=count,wall_ns=elapsed,
                            exclusive_wall_ns=elapsed-frame[1]))
                if sample_inputs and count<=128:
                    # Lists of Python integers only. Copy after timing so
                    # captured payload cost is not attributed to the method.
                    names=['ids','positions','blocks','seqs','qlens','slots','last_indices']
                    values=args[1:]
                    snapshot={k:v for k,v in zip(names,values)}
                    snapshot.update({k:v for k,v in kwargs.items() if k in names})
                    import copy
                    append(dict(stage='cpu_input_sample',values=copy.deepcopy(snapshot)))
        setattr(owner,method,observed)

    def augment(name,patch):
        old=observer.PATCHES.get(name)
        def patched(module):
            if old is not None:old(module)
            patch(module)
        observer.PATCHES[name]=patched

    def inputs(module):
        wrap(module,'make_input_plan','cpu_plan')
        wrap(module,'encode_input_plan','cpu_encode')
        wrap(module.PinnedDraftInputStager,'pack','draft_input_pack',True)

    def refill(module):
        wrap(module.RefillSession,'_metadata','draft_legacy_metadata')
        wrap(module.RefillSession,'_forward','draft_forward')
        wrap(module.RefillSession,'_decode_graph_forward','draft_graph_forward')
        old=module.emit
        @functools.wraps(old)
        def emitted(event,**data):
            if event in {'ready','p2_handoff','p2_boundary','p2_verified',
                         'p2_target_coalesce','p2_window_begin','p2_idle_step'}:
                x=dict(data)
                for name in ('tokens','drafts','generated'):
                    if name in x:x[name+'_count']=len(x.pop(name))
                append(dict(stage='algorithm_event',event=event,**x))
            return old(event,**data)
        module.emit=emitted

    def persistent(module):
        for name in ['plans','reconcile','refresh_draft_wait','draft_wait_reason']:
            wrap(module.PersistentDraftQueue,name,'queue_'+name)

    def runner(module):
        for name in ['_prepare_inputs','_update_states','_build_attention_metadata',
                     '_calc_spec_decode_metadata']:
            wrap(module.GPUModelRunner,name,'target_'+name)

    augment('vllm.v1.spec_decode.cdb_av_inputs',inputs)
    augment('vllm.v1.spec_decode.cdb_av_refill',refill)
    augment('vllm.v1.spec_decode.cdb_av_persistent',persistent)
    augment('vllm.v1.worker.gpu_model_runner',runner)
