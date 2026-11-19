# SPDX-License-Identifier: Apache-2.0
"""Exercise actual caller/frozen invariant bodies without a model or GPU."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from vllm.v1.spec_decode import cdb_av_context_cache as cache_module
from vllm.v1.spec_decode.cdb_av_fingerprint import token_fingerprint

SOURCE=Path(__file__).parents[3]/'vllm/v1/spec_decode'


def method(path,cls,name,namespace):
    tree=ast.parse(path.read_text())
    c=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
    fn=next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name==name)
    fn.decorator_list=[]
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),namespace)
    return namespace[name]


class Request:
    request_id='actual-caller-body'
    status='running'
    num_computed_tokens=2
    def __init__(self):self.prompt_token_ids=[1,2];self.output_token_ids=[3];self.spec_token_ids=[4]
    def is_finished(self):return False


def context(enabled,monkeypatch):
    monkeypatch.setattr(cache_module,'ENABLED',True)
    monkeypatch.setattr(cache_module,'_cache',cache_module.ContextFingerprintCache(validate=True))
    return method(SOURCE/'cdb_av_refill_queue.py','RefillQueue','context',dict(
        token_fingerprint=token_fingerprint,CONTEXT_FP_CACHE=enabled,
        cached_context=cache_module.cached_context))


def frozen(fn,req):
    q=NS(contexts={'A':dict(context=fn(req),computed=2)},s=NS(requests={'A':req}),
         inflight={},ready={},window=1)
    reconcile=method(SOURCE/'cdb_av_persistent.py','PersistentDraftQueue','reconcile',dict(
        RefillQueue=NS(context=fn),RequestStatus=NS(RUNNING='running'),emit=lambda *args,**kw:None))
    return q,reconcile


def test_off_path_is_original_context(monkeypatch):
    fn=context(False,monkeypatch);r=Request()
    assert fn(r)==dict(context_tokens=3,fingerprint=token_fingerprint([1,2,3]))
    assert cache_module._cache.calls==0


def test_actual_caller_on_uses_complete_prefix_and_fresh_dict(monkeypatch):
    fn=context(True,monkeypatch);r=Request()
    a=fn(r);b=fn(r);assert a==b and a is not b
    assert cache_module._cache.hits==1
    r.output_token_ids[0]=7
    assert fn(r)['fingerprint']==token_fingerprint([1,2,7])
    assert cache_module._cache.misses==2


def test_same_length_target_advance_still_raises_frozen_assert(monkeypatch):
    fn=context(True,monkeypatch);r=Request();q,reconcile=frozen(fn,r)
    reconcile(q);assert cache_module._cache.hits>0
    r.output_token_ids[0]=8
    with pytest.raises(RuntimeError,match='advanced a frozen Carry'):reconcile(q)


def test_computed_length_assert_still_runs_on_cache_hit(monkeypatch):
    fn=context(True,monkeypatch);r=Request();q,reconcile=frozen(fn,r)
    r.num_computed_tokens=3
    with pytest.raises(RuntimeError,match='changed Carry computed length'):reconcile(q)
    assert cache_module._cache.hits==1


def test_invalidation_still_drops_work_and_clears_spec(monkeypatch):
    fn=context(True,monkeypatch);r=Request();q,reconcile=frozen(fn,r);events=[]
    q.rpc=lambda op,payload:events.append((op,payload)) or dict(dropped=['A'])
    q.release=lambda rid,reason:events.append((rid,reason))
    r.status='preempted';reconcile(q)
    assert events==[('drop',{'ids':['A']}),('A','invalid')]
    assert r.spec_token_ids==[]
