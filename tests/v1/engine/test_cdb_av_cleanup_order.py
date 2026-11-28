# SPDX-License-Identifier: Apache-2.0
"""All independent engine resources must be offered shutdown after a failure."""
from unittest.mock import Mock
import pytest
from vllm.v1.engine.core import EngineCore, DPEngineCoreProc


@pytest.mark.parametrize(
    "failure", [None, "draft", "structured", "executor", "scheduler"]
)
def test_engine_shutdown_attempts_every_resource(failure):
    engine = EngineCore.__new__(EngineCore)
    calls = []

    def operation(name):
        def close():
            calls.append(name)
            if failure == name:
                raise RuntimeError(name + " cleanup failed")

        return close

    engine._cdb_persistent = Mock(close=operation("draft"))
    engine.structured_output_manager = Mock(clear_backend=operation("structured"))
    engine.model_executor = Mock(shutdown=operation("executor"))
    engine.scheduler = Mock(shutdown=operation("scheduler"))
    if failure:
        with pytest.raises(RuntimeError, match=failure):
            engine.shutdown()
    else:
        engine.shutdown()
    assert calls == ["draft", "structured", "executor", "scheduler"]


def test_DP_group_cleanup_survives_parent_shutdown_failure(monkeypatch):
    engine = DPEngineCoreProc.__new__(DPEngineCoreProc)
    engine.dp_group = object()

    def fail(self):
        raise RuntimeError("parent cleanup failed")

    monkeypatch.setattr(EngineCore, "shutdown", fail)
    destroy = Mock()
    monkeypatch.setattr(
        "vllm.v1.engine.core.stateless_destroy_torch_distributed_process_group", destroy
    )
    with pytest.raises(RuntimeError, match="parent cleanup failed"):
        engine.shutdown()
    destroy.assert_called_once_with(engine.dp_group)


@pytest.mark.parametrize(
    "missing", ["structured_output_manager", "model_executor", "scheduler"]
)
def test_partial_initialization_still_closes_remaining_resources(missing):
    engine = EngineCore.__new__(EngineCore)
    draft = engine._cdb_persistent = Mock()
    structured = engine.structured_output_manager = Mock()
    executor = engine.model_executor = Mock()
    scheduler = engine.scheduler = Mock()
    delattr(engine, missing)
    engine.shutdown()
    draft.close.assert_called_once()
    for name, resource, operation in [
        ("structured_output_manager", structured, "clear_backend"),
        ("model_executor", executor, "shutdown"),
        ("scheduler", scheduler, "shutdown"),
    ]:
        if name != missing:
            getattr(resource, operation).assert_called_once()


def test_all_cleanup_failures_keep_the_exception_context_chain():
    engine = EngineCore.__new__(EngineCore)
    calls = []

    def operation(name):
        def close():
            calls.append(name)
            raise RuntimeError(name + " failure")

        return close

    engine._cdb_persistent = Mock(close=operation("draft"))
    engine.structured_output_manager = Mock(clear_backend=operation("structured"))
    engine.model_executor = Mock(shutdown=operation("executor"))
    engine.scheduler = Mock(shutdown=operation("scheduler"))
    with pytest.raises(RuntimeError) as info:
        engine.shutdown()
    assert calls == ["draft", "structured", "executor", "scheduler"]
    errors = []
    error = info.value
    while error is not None:
        errors.append(str(error))
        error = error.__context__
    assert errors == [
        "scheduler failure",
        "executor failure",
        "structured failure",
        "draft failure",
    ]
