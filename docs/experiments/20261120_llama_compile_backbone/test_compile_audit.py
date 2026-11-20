from contextlib import contextmanager
from types import SimpleNamespace

import pytest

import compile_audit


def test_target_only_validation_model_uses_skip_compiled():
    skipped = []
    logs = []

    @contextmanager
    def native_context(*, skip_compiled=False):
        skipped.append(skip_compiled)
        yield

    class Graph:
        def _validate(self, entry, live, caches, model):
            return model()

    module = SimpleNamespace(TargetFullGraph=Graph, set_forward_context=native_context,
                             logger=SimpleNamespace(info=lambda *args: logs.append(args)))
    compile_audit.target(module)

    def model():
        with module.set_forward_context():
            return 'completion'

    assert model() == 'completion'
    assert Graph()._validate(None, None, None, model) == 'completion'
    assert model() == 'completion'
    assert skipped == [False, True, False]
    assert len(logs) == 1

    def failure():
        with module.set_forward_context():
            raise RuntimeError('original failure')

    with pytest.raises(RuntimeError, match='original failure'):
        Graph()._validate(None, None, None, failure)
    assert model() == 'completion'
    assert skipped[-2:] == [True, False]


def test_draft_capture_stays_compiled_and_two_references_bypass(monkeypatch):
    import torch

    captured = False
    skipped = []

    @contextmanager
    def native_context(*, skip_compiled=False):
        skipped.append(skip_compiled)
        yield

    class Draft:
        def _decode_graph_forward(self):
            nonlocal captured
            captured = True
            with module.set_forward_context():
                pass
            captured = False
            for _ in range(2):
                with module.set_forward_context():
                    pass

    monkeypatch.setattr(torch.cuda, 'is_current_stream_capturing', lambda: captured)
    module = SimpleNamespace(RefillSession=Draft, set_forward_context=native_context,
                             logger=SimpleNamespace(info=lambda *args: None))
    compile_audit.draft(module)
    Draft()._decode_graph_forward()
    with module.set_forward_context():
        pass
    assert skipped == [False, True, True, False]
