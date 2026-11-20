import importlib.util
import json
import os
from pathlib import Path
import sys
import types

import pytest


@pytest.fixture
def hooks(monkeypatch):
    names=[]
    class Scope:
        def __init__(self,name):names.append(name)
        def __enter__(self):return self
        def __exit__(self,*args):pass
    monkeypatch.setitem(sys.modules,'manual_observer',types.SimpleNamespace(PATCHES={}))
    monkeypatch.setitem(sys.modules,'torch',types.SimpleNamespace(profiler=types.SimpleNamespace(record_function=Scope)))
    spec=importlib.util.spec_from_file_location('private_hooks',Path(__file__).with_name('profile_hooks.py'))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m,names


def test_executor_calls_original_in_process_and_keeps_exceptions(hooks):
    m,names=hooks;value=object()
    class Executor:
        def collective_rpc(self,method,*args,**kwargs):
            if method=='bad':raise RuntimeError('original')
            assert args==() and kwargs=={'args':('p2_step',value)}
            return value
    m.executor(types.SimpleNamespace(UniProcExecutor=Executor))
    assert Executor().collective_rpc('cdb_refill_rpc',args=('p2_step',value)) is value
    assert names==['CDB::worker_rpc::cdb_refill_rpc::p2_step']
    with pytest.raises(RuntimeError,match='original'):Executor().collective_rpc('bad')


def test_profile_trigger_only_after_successful_native_start(hooks):
    m,names=hooks;value=object()
    class Executor:
        def collective_rpc(self,method,*args,**kwargs):return value
    m.executor(types.SimpleNamespace(UniProcExecutor=Executor))
    assert Executor().collective_rpc('profile',args=(True,)) is value
    assert m.TRACE_ACTIVE and names==[]


def test_real_instance_probe_outside_counter_directory(hooks,monkeypatch,tmp_path):
    m,_=hooks
    directory=tmp_path/'observer';directory.mkdir()
    monkeypatch.setenv('BASELINE_OBSERVE_DIRECTORY',str(directory))
    class Executor:
        parallel_config=types.SimpleNamespace(distributed_executor_backend='uni')
    class Core:
        def __init__(self,value,*,flag):
            assert flag==2;self.model_executor=value
    m.core(types.SimpleNamespace(EngineCore=Core,__file__='actual_core.py'))
    obj=Executor();instance=Core(obj,flag=2)
    assert instance.model_executor is obj
    facts=json.loads((tmp_path/f'executor_runtime_{os.getpid()}.json').read_text())
    assert facts['executor'].endswith('.Executor') and facts['backend']=='uni'
    assert not list(directory.iterdir())


def test_cache_snapshot_does_not_mutate_native_cache(hooks,monkeypatch,tmp_path):
    m,names=hooks;m.TRACE_ACTIVE=True
    monkeypatch.setenv('BASELINE_OBSERVE_DIRECTORY',str(tmp_path/'observer'))
    value=object()
    class Graph:
        cache={(8,4,9,256):value};limit=24;seq_floor=0;dynamic=True
        def _storage(self):return value
        def _record(self,action,key=None,**kwargs):
            assert action=='fallback_budget' and key==(8,4,9,512) and kwargs=={'flag':2}
            return value
    m.target_graph(types.SimpleNamespace(TargetFullGraph=Graph))
    g=Graph();before=dict(g.cache)
    for _ in range(2):assert g._record('fallback_budget',(8,4,9,512),flag=2) is value
    assert g.cache==before
    payload=json.loads((tmp_path/f'target_graph_cache_{os.getpid()}.json').read_text())
    assert payload['keys']==[[8,4,9,256]]
    assert len(m.SNAPSHOTS)==1
