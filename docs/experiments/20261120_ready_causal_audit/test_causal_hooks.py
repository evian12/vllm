from types import SimpleNamespace
from threading import Lock
import pytest
import causal_hooks


def test_normalize_preserves_input_and_counts():
    data=dict(tokens=[1,2], drafts=[1], generated=[1,2], accepted=1)
    result=causal_hooks.normalize('ready',data)
    assert result['tokens_count']==2 and result['generated_count']==2
    assert data['tokens']==[1,2] and data['drafts']==[1]


def test_nested_timer_preserves_result_errors_and_event():
    class Session:
        def _decode_graph_forward(self, fail=False):
            if fail:raise ValueError('sentinel')
            return marker
        def _forward(self, fail=False):
            return self._decode_graph_forward(fail)
    marker=object();seen=[]
    module=SimpleNamespace(RefillSession=Session,emit=lambda event,**data:seen.append((event,data)))
    observer=SimpleNamespace(PATCHES={},start=lambda:None,LOCK=Lock(),ROWS=[],EPOCH='test')
    causal_hooks.install(observer);observer.PATCHES['vllm.v1.spec_decode.cdb_av_refill'](module)
    assert Session()._forward() is marker
    with pytest.raises(ValueError,match='sentinel'):Session()._forward(True)
    assert causal_hooks.TLS.stack==[]
    module.emit('ready',tokens=[2],req_id='r')
    assert len(seen)==1
    timings=[x for x in observer.ROWS if x['kind']=='host_stage']
    assert len(timings)==4
    assert all(x['cpu_ns']>=x['cpu_exclusive_ns']>=0 for x in timings)
    assert all(x['wall_ns']>=x['wall_exclusive_ns']>=0 for x in timings)
    assert sum(x['kind']=='causal_event' for x in observer.ROWS)==1
