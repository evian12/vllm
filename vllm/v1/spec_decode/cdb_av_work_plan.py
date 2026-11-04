# SPDX-License-Identifier: Apache-2.0
"""CDB-AV draft work planning.

This module describes draft work. It does not grant GPU/KV resources.
Admission must be performed by the vLLM Scheduler.
"""

from dataclasses import dataclass
from enum import Enum


class DraftOp(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"


@dataclass(frozen=True)
class DraftWork:
    req_id: str
    epoch: int
    op: DraftOp

    # Total known context that must be processed before a new
    # speculative token can be proposed.
    context_end: int

    # Number of known context tokens already processed by draft.
    draft_computed: int

    # Number of speculative tokens already proposed this epoch.
    spec_progress: int

    k_cap: int

    # Number of input tokens authorized for this forward.
    num_input_tokens: int

    def __post_init__(self):
        if not self.req_id:
            raise ValueError("Empty request ID")

        if self.epoch < 0:
            raise ValueError("Negative epoch")

        if not 0 <= self.draft_computed <= self.context_end:
            raise ValueError("Invalid draft context progress")

        if not 0 <= self.spec_progress < self.k_cap:
            raise ValueError("Invalid speculative progress")

        if self.num_input_tokens < 1:
            raise ValueError("Empty forward is not allowed")

        if self.op == DraftOp.PREFILL:
            remaining = self.context_end - self.draft_computed
            if remaining <= 0:
                raise ValueError("No context remains for prefill")
            if self.num_input_tokens > remaining:
                raise ValueError("Prefill exceeds known context")

        if self.op == DraftOp.DECODE:
            if self.draft_computed != self.context_end:
                raise ValueError("Decode requires prepared context")
            if self.num_input_tokens != 1:
                raise ValueError("Decode must process one token")


@dataclass(frozen=True)
class DraftWorkPlan:
    window_id: int
    works: tuple[DraftWork, ...]

    def __post_init__(self):
        ids = [w.req_id for w in self.works]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate requests in Draft Work Plan")

    @property
    def total_input_tokens(self):
        return sum(w.num_input_tokens for w in self.works)


def make_draft_work(
    *,
    req_id: str,
    epoch: int,
    context_end: int,
    draft_computed: int,
    spec_progress: int,
    k_cap: int,
    prefill_budget: int,
) -> DraftWork:
    """Classify an already admitted request's next draft operation.

    The caller must validate Scheduler authorization and KV resources.
    """
    remaining = context_end - draft_computed

    if remaining > 0:
        if prefill_budget < 1:
            raise ValueError("Prefill budget exhausted")
        op = DraftOp.PREFILL
        n = min(remaining, prefill_budget)
    else:
        op = DraftOp.DECODE
        n = 1

    return DraftWork(
        req_id=req_id,
        epoch=epoch,
        op=op,
        context_end=context_end,
        draft_computed=draft_computed,
        spec_progress=spec_progress,
        k_cap=k_cap,
        num_input_tokens=n,
    )
