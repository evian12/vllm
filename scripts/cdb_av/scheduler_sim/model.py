"""State and request models for the CDB-AV scheduler simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Action(str, Enum):
    DRAFT = "draft"
    VERIFY = "verify"


class RequestStatus(str, Enum):
    WAITING = "waiting"
    ELIGIBLE = "eligible"
    ACTIVE_DRAFT = "active_draft"
    READY = "ready"
    FINISHED = "finished"


_DEFAULT_ACCEPTANCE = (0.72, 0.52, 0.36, 0.25, 0.17, 0.11, 0.07, 0.04)


@dataclass(slots=True)
class Request:
    """One request in the simplified Draft/Verify lifecycle.

    acceptance_probs contains conditional acceptance probabilities:
    position j is tested only if all earlier candidates were accepted.
    confidence_by_position is a synthetic confidence trace used by the
    simulator's explicit stop-after-generated-candidate abstraction.
    """

    request_id: str
    arrival_time_ms: float
    prompt_tokens: int = 512
    max_output_tokens: int = 64
    draft_cap: int = 8
    acceptance_probs: tuple[float, ...] = _DEFAULT_ACCEPTANCE
    confidence_by_position: tuple[float, ...] = ()

    status: RequestStatus = RequestStatus.WAITING
    draft_depth: int = 0
    total_draft_tokens: int = 0
    committed_tokens: int = 0
    accepted_candidate_tokens: int = 0
    verify_rounds: int = 0
    ready_since_ms: float | None = None
    first_commit_time_ms: float | None = None
    last_commit_time_ms: float | None = None
    completed_time_ms: float | None = None
    commit_gap_samples_ms: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must be non-empty")
        if self.arrival_time_ms < 0:
            raise ValueError("arrival_time_ms must be non-negative")
        if self.prompt_tokens < 0:
            raise ValueError("prompt_tokens must be non-negative")
        if self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")
        if self.draft_cap < 1:
            raise ValueError("draft_cap must be positive")
        self.acceptance_probs = tuple(float(p) for p in self.acceptance_probs)
        self.confidence_by_position = tuple(
            float(p) for p in self.confidence_by_position
        )
        if not self.acceptance_probs:
            raise ValueError("acceptance_probs must contain at least one value")
        if any(not 0.0 <= p <= 1.0 for p in self.acceptance_probs):
            raise ValueError("acceptance probabilities must be in [0, 1]")
        if any(not 0.0 <= p <= 1.0 for p in self.confidence_by_position):
            raise ValueError("confidence values must be in [0, 1]")

    @property
    def remaining_output_tokens(self) -> int:
        return max(0, self.max_output_tokens - self.committed_tokens)

    def conditional_acceptance_probability(self, position: int) -> float:
        if position < 0:
            raise ValueError("position must be non-negative")
        index = min(position, len(self.acceptance_probs) - 1)
        return self.acceptance_probs[index]

    def confidence_at(self, position: int) -> float:
        if position < 0:
            raise ValueError("position must be non-negative")
        if position >= len(self.confidence_by_position):
            return 1.0
        return self.confidence_by_position[position]

    def expected_accepted_length(self, depth: int | None = None) -> float:
        """Expected accepted prefix length under conditional probabilities."""
        candidate_count = self.draft_depth if depth is None else depth
        candidate_count = min(
            max(0, candidate_count),
            max(0, self.remaining_output_tokens - 1),
        )
        survival_probability = 1.0
        expected_length = 0.0
        for position in range(candidate_count):
            survival_probability *= self.conditional_acceptance_probability(
                position
            )
            expected_length += survival_probability
        return expected_length

    def expected_verify_tokens(self, depth: int | None = None) -> float:
        """Expected committed tokens from one verification, including one
        correction token or bonus token, capped by remaining output length.
        """
        if self.remaining_output_tokens <= 0:
            return 0.0
        return min(
            float(self.remaining_output_tokens),
            1.0 + self.expected_accepted_length(depth),
        )

    def expected_next_candidate_acceptance(self) -> float:
        if self.draft_depth >= self.draft_cap:
            return 0.0
        if self.remaining_output_tokens <= self.draft_depth + 1:
            return 0.0
        return self.conditional_acceptance_probability(self.draft_depth)


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    max_draft_batch_size: int = 4
    max_verify_batch_size: int = 4
    fixed_h: int = 8
    queue_threshold: int = 4
    ready_wait_budget_ms: float = 8.0
    confidence_threshold: float = 0.8
    max_consecutive_verify_forwards: int = 4
    max_steps: int = 200_000
    seed: int = 42

    def __post_init__(self) -> None:
        if self.max_draft_batch_size < 1:
            raise ValueError("max_draft_batch_size must be positive")
        if self.max_verify_batch_size < 1:
            raise ValueError("max_verify_batch_size must be positive")
        if self.fixed_h < 1:
            raise ValueError("fixed_h must be positive")
        if self.queue_threshold < 1:
            raise ValueError("queue_threshold must be positive")
        if self.ready_wait_budget_ms <= 0:
            raise ValueError("ready_wait_budget_ms must be positive")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")
        if self.max_consecutive_verify_forwards < 1:
            raise ValueError("max_consecutive_verify_forwards must be positive")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
