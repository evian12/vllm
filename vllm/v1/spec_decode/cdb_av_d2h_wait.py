# SPDX-License-Identifier: Apache-2.0
"""Opt-in blocking CPU waits before small synchronous device-to-host results."""
import logging
import os
import threading

import torch

_LOCAL = threading.local()
_LOGGER = logging.getLogger(__name__)


def wait_for_d2h(tensor: torch.Tensor, scope: str) -> bool:
    """Wait for prior current-stream work without changing tensors or streams.

    The following ordinary synchronous CPU copy still owns materialization.
    This event only lets the host sleep while prior GPU work completes.
    """
    mode = os.getenv("CDB_AV_BLOCKING_D2H_WAIT", "0")
    if mode not in ("0", "1"):
        raise ValueError("CDB_AV_BLOCKING_D2H_WAIT must be 0 or 1")
    if mode == "0" or not tensor.is_cuda:
        return False
    with torch.cuda.device(tensor.device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Blocking D2H wait cannot run during CUDA capture")
        events = getattr(_LOCAL, "events", None)
        if events is None:
            events = _LOCAL.events = {}
        key = tensor.device.index
        event = events.get(key)
        if event is None:
            event = events[key] = torch.cuda.Event(blocking=True)
        try:
            event.record(torch.cuda.current_stream(tensor.device))
            event.synchronize()
        except BaseException:
            # A failed record/wait must never reuse the uncertain event.
            events.pop(key, None)
            raise
        logged = getattr(_LOCAL, "logged", None)
        if logged is None:
            logged = _LOCAL.logged = set()
        if (key, scope) not in logged:
            logged.add((key, scope))
            _LOGGER.info(
                "[CDB_BLOCKING_D2H_WAIT] scope=%s device=%s blocking=True",
                scope,
                tensor.device,
            )
    return True
