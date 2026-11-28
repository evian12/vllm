# SPDX-License-Identifier: Apache-2.0
from contextlib import nullcontext
from types import SimpleNamespace as NS
from unittest.mock import Mock
import threading

import pytest
import torch

from vllm.v1.spec_decode import cdb_av_d2h_wait as wait_module


@pytest.fixture
def cuda_stub(monkeypatch):
    monkeypatch.setenv("CDB_AV_BLOCKING_D2H_WAIT", "1")
    monkeypatch.setattr(wait_module, "_LOCAL", NS())
    event = Mock()
    factory = Mock(return_value=event)
    stream = object()
    monkeypatch.setattr(torch.cuda, "Event", factory)
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _: stream)
    return NS(
        tensor=NS(is_cuda=True, device=NS(index=0)),
        event=event,
        factory=factory,
        stream=stream,
    )


def test_disabled_path_does_not_touch_CUDA(cuda_stub, monkeypatch):
    monkeypatch.setenv("CDB_AV_BLOCKING_D2H_WAIT", "0")
    assert wait_module.wait_for_d2h(cuda_stub.tensor, "draft") is False
    cuda_stub.factory.assert_not_called()


def test_cpu_tensor_does_not_touch_CUDA(cuda_stub):
    assert wait_module.wait_for_d2h(torch.tensor([1]), "verify") is False
    cuda_stub.factory.assert_not_called()


def test_event_blocks_current_stream_and_is_reused(cuda_stub):
    for scope in ("draft", "verify", "draft"):
        assert wait_module.wait_for_d2h(cuda_stub.tensor, scope)
    cuda_stub.factory.assert_called_once_with(blocking=True)
    assert cuda_stub.event.record.call_count == 3
    cuda_stub.event.record.assert_called_with(cuda_stub.stream)
    assert cuda_stub.event.synchronize.call_count == 3


@pytest.mark.parametrize("stage", ["record", "synchronize"])
def test_wait_failure_propagates_and_discards_event(cuda_stub, stage):
    getattr(cuda_stub.event, stage).side_effect = RuntimeError("injected event failure")
    with pytest.raises(RuntimeError, match="injected event failure"):
        wait_module.wait_for_d2h(cuda_stub.tensor, "draft")
    assert not wait_module._LOCAL.events
    getattr(cuda_stub.event, stage).side_effect = None
    assert wait_module.wait_for_d2h(cuda_stub.tensor, "draft")
    assert cuda_stub.factory.call_count == 2


def test_capture_guard_precedes_event_record(cuda_stub, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="capture"):
        wait_module.wait_for_d2h(cuda_stub.tensor, "draft")
    cuda_stub.factory.assert_not_called()


def test_invalid_mode_is_rejected(cuda_stub, monkeypatch):
    monkeypatch.setenv("CDB_AV_BLOCKING_D2H_WAIT", "yes")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        wait_module.wait_for_d2h(cuda_stub.tensor, "draft")


def test_wait_events_belong_to_the_calling_thread(cuda_stub, monkeypatch):
    monkeypatch.setattr(wait_module, "_LOCAL", threading.local())
    cuda_stub.factory.side_effect = [Mock(), Mock()]
    owned = []

    def call():
        wait_module.wait_for_d2h(cuda_stub.tensor, "draft")
        owned.append(wait_module._LOCAL.events[0])

    threads = [threading.Thread(target=call) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(owned) == 2 and owned[0] is not owned[1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    "dtype", [torch.int32, torch.int64, torch.float32, torch.float64]
)
def test_GPU_materialization_matches_synchronous_copy(monkeypatch, dtype):
    monkeypatch.setenv("CDB_AV_BLOCKING_D2H_WAIT", "1")
    for i in range(8):
        tensor = torch.arange(36, dtype=dtype, device="cuda").reshape(4, 9) + i
        expected = tensor.cpu()
        tensor = tensor.clone()  # Queue a fresh producer before recording.
        assert wait_module.wait_for_d2h(tensor, "test")
        assert torch.equal(tensor.cpu(), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("discard", [(), (1,)])
def test_GPU_rejection_result_matches_disabled_wait(monkeypatch, discard):
    from vllm.v1.sample.rejection_sampler import RejectionSampler

    output = torch.tensor(
        [[1, 2, -1], [3, -1, -1], [4, 5, 6]], dtype=torch.int32, device="cuda"
    )
    monkeypatch.setenv("CDB_AV_BLOCKING_D2H_WAIT", "0")
    expected = RejectionSampler.parse_output(output, 16, discard)
    monkeypatch.setenv("CDB_AV_BLOCKING_D2H_WAIT", "1")
    assert RejectionSampler.parse_output(output.clone(), 16, discard) == expected
