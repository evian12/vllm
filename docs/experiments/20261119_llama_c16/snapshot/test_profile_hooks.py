import importlib.util
from pathlib import Path
import sys
import types

import pytest


@pytest.fixture
def hooks(monkeypatch):
    seen = []

    class Scope:
        def __init__(self, name):
            self.name = name

        def __enter__(self):
            seen.append(('enter', self.name))

        def __exit__(self, kind, error, traceback):
            seen.append(('exit', self.name, kind))

    monkeypatch.setitem(sys.modules, 'manual_observer', types.SimpleNamespace(PATCHES={}))
    monkeypatch.setitem(sys.modules, 'torch', types.SimpleNamespace(profiler=types.SimpleNamespace(record_function=Scope)))
    spec = importlib.util.spec_from_file_location('isolated_profile_hooks', Path(__file__).with_name('profile_hooks.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, seen


def test_wrapper_preserves_return_identity_and_arguments(hooks):
    module, seen = hooks
    result = object()

    class Operator:
        def compute(self, value, *, flag):
            assert value is result and flag == 3
            return value

    module.wrap(Operator, 'compute', 'test')
    assert Operator().compute(result, flag=3) is result
    assert seen == [('enter', 'test'), ('exit', 'test', None)]


def test_original_exception_propagates(hooks):
    module, seen = hooks

    class Operator:
        def compute(self):
            raise ArithmeticError('original failure')

    module.wrap(Operator, 'compute', 'test')
    with pytest.raises(ArithmeticError, match='original failure'):
        Operator().compute()
    assert seen[-1] == ('exit', 'test', ArithmeticError)


def test_logits_labels_distinguish_shared_target_draft_class(hooks):
    module, seen = hooks

    class Model:
        def compute_logits(self, hidden):
            return hidden

    class Proposer:
        def load_model(self):
            self.model = Model()
            self.model._baseline_observed_draft = True

    module.draft(types.SimpleNamespace(DraftModelProposer=Proposer))
    proposer = Proposer()
    proposer.load_model()
    value = object()
    assert proposer.model.compute_logits(value) is value
    assert Model().compute_logits(value) is value
    assert [x[1] for x in seen if x[0] == 'enter'] == ['CDB::draft_logits', 'CDB::target_logits']
