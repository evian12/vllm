# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CDB-AV per-request draft state.

Tracks draft execution progress independently of the generator.
Does not own or allocate GPU KV blocks.
"""

from dataclasses import dataclass, field
from enum import Enum


class DraftPhase(str, Enum):
    WAITING = "waiting"
    ACTIVE = "active"
    CARRY = "carry"
    READY = "ready"
    VERIFYING = "verifying"


@dataclass
class DraftState:
    req_id: str
    epoch: int
    k_cap: int
    context_version: int
    status: DraftPhase = DraftPhase.WAITING
    progress: int = 0
    draft_tokens: list[int] = field(default_factory=list)
    kv_valid_upto: int | None = None

    def __post_init__(self):
        if not self.req_id or self.epoch < 0 or self.k_cap < 1:
            raise ValueError("Invalid draft state identity or K")

    def append_token(self, token_id: int, completed: bool):
        if self.status != DraftPhase.ACTIVE:
            raise RuntimeError(
                f"{self.req_id}: cannot append in {self.status}"
            )

        if self.progress >= self.k_cap:
            raise RuntimeError("Draft exceeds K cap")

        self.draft_tokens.append(int(token_id))
        self.progress += 1

        if completed or self.progress == self.k_cap:
            self.status = DraftPhase.READY

    def pause(self):
        if self.status != DraftPhase.ACTIVE:
            raise RuntimeError("Only ACTIVE can become CARRY")
        if not 0 < self.progress < self.k_cap:
            raise RuntimeError("Invalid Carry progress")
        self.status = DraftPhase.CARRY

    def resume(self, context_version: int):
        if self.status != DraftPhase.CARRY:
            raise RuntimeError("Only CARRY can resume")
        if context_version != self.context_version:
            raise RuntimeError("Stale draft context")
        self.status = DraftPhase.ACTIVE


class DraftStateStore:
    def __init__(self):
        self.states: dict[tuple[str, int], DraftState] = {}

    def create(
        self,
        req_id: str,
        epoch: int,
        k_cap: int,
        context_version: int,
    ) -> DraftState:
        key = (req_id, epoch)

        if key in self.states:
            raise RuntimeError(f"Duplicate draft epoch: {key}")

        state = DraftState(
            req_id=req_id,
            epoch=epoch,
            k_cap=k_cap,
            context_version=context_version,
        )
        self.states[key] = state
        return state

    def get(self, req_id: str, epoch: int) -> DraftState:
        return self.states[(req_id, epoch)]

    def discard(self, req_id: str, epoch: int):
        self.states.pop((req_id, epoch), None)
