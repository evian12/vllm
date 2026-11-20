"""Real CPU Kineto export; CUDA is explicitly unavailable for this test."""
import importlib.util
import json
from pathlib import Path
import sys
import types


def test_actual_cpu_trace_export_roundtrips_safe_key(monkeypatch,tmp_path):
    import torch
    monkeypatch.setitem(sys.modules,'manual_observer',types.SimpleNamespace(PATCHES={}))
    p=Path(__file__).with_name('profile_hooks_safe_labels.py')
    spec=importlib.util.spec_from_file_location('safe_export_hooks',p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    key=[8,4,9,512,64,0,'bounded']
    name='CDB::target_dispatch::replay::'+m.key_label(key)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        with torch.profiler.record_function(name):torch.ones(4).add_(1)
    dest=tmp_path/'cpu_trace.json';profile.export_chrome_trace(str(dest))
    data=json.loads(dest.read_text())
    events=[e for e in data['traceEvents'] if e.get('name')==name]
    assert len(events)==1
    assert json.loads(bytes.fromhex(events[0]['name'].split('::',3)[3][4:]).decode())==key
