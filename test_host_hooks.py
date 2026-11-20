import threading
from types import SimpleNamespace as NS
from host_hooks import install


def test_hooks_preserve_values_aliases_exception_and_bound():
    observer=NS(LOCK=threading.Lock(),ROWS=[],EPOCH='measurement',PATCHES={},start=lambda:None)
    install(observer,limit=2)
    marker=object()
    def plan(*args):return marker
    def encode(*args):raise ValueError('kept')
    class Stager:
        def pack(self,*args):return module.make_input_plan(*args)
    module=NS(make_input_plan=plan,encode_input_plan=encode,PinnedDraftInputStager=Stager)
    observer.PATCHES['vllm.v1.spec_decode.cdb_av_inputs'](module)
    args=([1],[0],[[2]],[1],[1],[32],[0])
    stage=Stager()
    for _ in range(3):assert stage.pack(*args) is marker
    plans=[x for x in observer.ROWS if x.get('stage')=='cpu_plan']
    packs=[x for x in observer.ROWS if x.get('stage')=='draft_input_pack']
    assert len(plans)==len(packs)==2
    assert all(0<=x['exclusive_wall_ns']<=x['wall_ns'] for x in packs)
    snapshots=[x for x in observer.ROWS if x.get('stage')=='cpu_input_sample']
    args[0][0]=999
    assert snapshots[0]['values']['ids']==[1]
    try:module.encode_input_plan(None)
    except ValueError as e:assert str(e)=='kept'
    else:assert False
    assert any(x.get('stage')=='cpu_encode' for x in observer.ROWS)


def test_real_repository_hook_targets_exist():
    from vllm.v1.spec_decode import cdb_av_persistent as p,cdb_av_refill as r,cdb_av_inputs as i
    from vllm.v1.worker import gpu_model_runner as g
    observer=NS(LOCK=threading.Lock(),ROWS=[],EPOCH='measurement',PATCHES={},start=lambda:None)
    install(observer)
    # Match the real import order; installing must not touch CUDA or run models.
    for module in [i,r,p,g]:observer.PATCHES[module.__name__](module)
