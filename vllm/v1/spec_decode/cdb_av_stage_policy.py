# SPDX-License-Identifier: Apache-2.0
"""Experimental AV policy. Pure decision logic; no KV or admission mutation.

Times are seconds on one monotonic clock. Deadlines refer to output events,
not average time per token. A Target action must be executable by the runtime.
The caller owns FIFO batch construction and all feasibility checks.
"""
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class StageState:
    now: float
    target_deadlines: tuple[float, ...]
    target_count: int
    target_capacity: int
    draft_can_progress: bool
    draft_step_time: float
    target_time: float
    # True only if another draft step can contribute to a later target batch.
    draft_can_add_ready: bool


def choose_stage(s: StageState) -> str:
    """Choose one executable stage; do not wait for hypothetical arrivals."""
    if not 0 <= s.target_count <= s.target_capacity or s.target_capacity < 1:
        raise ValueError("invalid target capacity")
    if len(s.target_deadlines) != s.target_count:
        raise ValueError("one deadline per eligible target request required")
    if not all(isfinite(x) for x in (s.now, s.draft_step_time, s.target_time,
                                     *s.target_deadlines)):
        raise ValueError("invalid time")
    if min(s.draft_step_time, s.target_time) < 0:
        raise ValueError("negative duration")
    if not s.target_count:
        return "draft" if s.draft_can_progress else "idle"
    if not s.draft_can_progress:
        return "target"
    if s.target_count == s.target_capacity:
        return "target"
    if not s.draft_can_add_ready:
        return "target"
    if min(s.target_deadlines) <= s.now + s.draft_step_time + s.target_time:
        return "target"
    return "draft"
