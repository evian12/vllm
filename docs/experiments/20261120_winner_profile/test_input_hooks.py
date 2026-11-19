import importlib.util
from pathlib import Path
import sys
import types


def test_target_preparation_wrappers_preserve_original_calls(monkeypatch):
    names=[]
    class Scope:
        def __init__(self,name):self.name=name
        def __enter__(self):names.append(self.name)
        def __exit__(self,*args):pass
    monkeypatch.setitem(sys.modules,'manual_observer',types.SimpleNamespace(PATCHES={}))
    monkeypatch.setitem(sys.modules,'torch',types.SimpleNamespace(profiler=types.SimpleNamespace(record_function=Scope)))
    spec=importlib.util.spec_from_file_location('test_input_scope',Path(__file__).with_name('profile_hooks.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    value=object()
    def operation(self,*args,**kwargs):
        assert args==(value,) and kwargs=={'flag':2}
        return value
    methods=['_update_states','_prepare_input_ids','_prepare_inputs','_build_attention_metadata',
             '_calc_spec_decode_metadata','_preprocess']
    cls=type('Runner',(),{n:operation for n in methods+['_model_forward','sample_tokens']})
    module.runner(types.SimpleNamespace(GPUModelRunner=cls))
    for name in methods:assert getattr(cls(),name)(value,flag=2) is value
    assert names==['CDB::target_input'+n for n in methods]
