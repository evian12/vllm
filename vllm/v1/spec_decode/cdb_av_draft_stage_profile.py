# SPDX-License-Identifier: Apache-2.0
"""Opt-in, bounded Draft-stage timing for CDB-AV.

Diagnostic only. CUDA events measure current-stream spans, NOT active kernel time.
The event synchronizations in finish() perturb throughput; never use this mode for
end-to-end performance comparisons. No CUDA events are created when disabled.
"""
import json
import os
import time

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_UNSET = object()


def get_draft_stage_profiler(proposer):
    """Reuse one bounded counter across all RefillSession windows."""
    current = getattr(proposer, "_cdb_draft_stage_profiler", _UNSET)
    if current is not _UNSET:
        return current
    enabled = os.getenv("CDB_AV_DRAFT_STAGE_PROFILE", "0")
    if enabled not in ("0", "1"):
        raise ValueError("CDB_AV_DRAFT_STAGE_PROFILE must be 0 or 1")
    if enabled == "0":
        profiler = None
    else:
        raw = os.getenv("CDB_AV_DRAFT_STAGE_PROFILE_MAX_SAMPLES", "20")
        try:
            limit = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid Draft stage profile sample limit") from exc
        if not 1 <= limit <= 100:
            raise ValueError("Draft stage profile sample limit must be in [1, 100]")
        profiler = DraftStageProfiler(limit)
    proposer._cdb_draft_stage_profiler = profiler
    return profiler


class DraftStageProfiler:
    def __init__(self, limit):
        self.limit = limit
        self.samples = 0

    @property
    def has_capacity(self):
        return self.samples < self.limit

    def begin(self, *, window, step, phase, request_rows, token_rows,
              graph_action, graph_key, metadata_pack_cpu_ms):
        if not self.has_capacity:
            return None
        self.samples += 1
        return DraftStageSample(
            window=window, step=step, phase=phase,
            request_rows=request_rows, token_rows=token_rows,
            graph_action=graph_action, graph_key=graph_key,
            metadata_pack_cpu_ms=metadata_pack_cpu_ms,
            sample_index=self.samples, sample_limit=self.limit,
        )


class DraftStageSample:
    def __init__(self, **fields):
        self.fields = fields
        self.measurements = {}
        self._begun = {}
        self._event_pairs = {}
        self._finished = False

    def start(self, name):
        if name in self._begun or name in self._event_pairs:
            raise RuntimeError("Draft profile stage started twice: " + name)
        before = torch.cuda.Event(enable_timing=True)
        before.record()
        self._begun[name] = (time.perf_counter(), before)

    def end(self, name):
        cpu_start, before = self._begun.pop(name)
        after = torch.cuda.Event(enable_timing=True)
        after.record()
        elapsed_cpu_ms = (time.perf_counter() - cpu_start) * 1000
        self.measurements[name] = {"cpu_enqueue_ms": elapsed_cpu_ms}
        self._event_pairs[name] = (before, after)

    def cpu_span(self, name, start):
        self.measurements[name] = (time.perf_counter() - start) * 1000

    def finish(self):
        if self._finished:
            raise RuntimeError("Draft profile sample finished twice")
        if self._begun:
            raise RuntimeError("Draft profile stages left open: " +
                               ", ".join(sorted(self._begun)))
        self._finished = True
        for name, (before, after) in self._event_pairs.items():
            after.synchronize()
            self.measurements[name]["stream_span_ms"] = before.elapsed_time(after)
        logger.info("[CDB_DRAFT_STAGE_TIMING] %s", json.dumps({
            **self.fields,
            "measurements": self.measurements,
            "interpretation": (
                "CUDA current-stream spans, not kernel active time; "
                "D2H wait includes outstanding upstream GPU work; "
                "bounded diagnostic synchronizations perturb throughput"
            ),
        }))
