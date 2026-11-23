from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from vllm.v1.engine import core


@pytest.mark.parametrize(
    "enabled,bypassed,progressed,connector,model_executed,unfinished,expected",
    [
        (True, True, True, None, False, True, [0]),
        (False, True, True, None, False, True, [0.001]),
        (True, False, True, None, False, True, [0.001]),
        (True, True, False, None, False, True, [0.001]),
        (True, True, True, object(), False, True, [0.001]),
        (True, True, True, None, True, True, []),
        (True, True, True, None, False, False, []),
    ],
)
def test_draft_progress_yield(
    monkeypatch, enabled, bypassed, progressed, connector,
    model_executed, unfinished, expected,
):
    engine = object.__new__(core.EngineCoreProc)
    engine.async_scheduling = False
    engine.scheduler = SimpleNamespace(
        connector=connector, ec_connector=None,
        has_unfinished_requests=lambda: unfinished,
    )
    engine._cdb_persistent = SimpleNamespace(window=1, _p3_steps=7)
    engine._cdb_target_bypassed = True  # Stale marker from the preceding iteration.
    engine.output_queue = Mock()
    calls = []
    monkeypatch.setattr(core, "time", SimpleNamespace(sleep=calls.append))
    monkeypatch.setenv("CDB_AV_DRAFT_PROGRESS_YIELD", str(int(enabled)))

    def step():
        assert not engine._cdb_target_bypassed
        engine._cdb_target_bypassed = bypassed
        return {}, model_executed

    def post_step(executed):
        assert executed is model_executed
        if progressed:
            engine._cdb_persistent._p3_steps = 0  # H boundary closes the window.

    engine.step_fn = step
    engine.post_step = post_step
    assert engine._process_engine_step() is model_executed
    assert calls == expected
    engine.output_queue.put_nowait.assert_not_called()
