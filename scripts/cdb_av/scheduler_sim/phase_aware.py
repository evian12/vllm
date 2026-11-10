"""Phase-aware CDB-AV scheduling simulator.

The simulator models one serial GPU timeline. Draft is one action; each Target
forward may contain an atomic Verify item, one-token AR items, and chunked
Prefill items in the same batch. Latency coefficients are synthetic unless a
measured profile is supplied.
"""

from __future__ import annotations

import copy
import json
import math
import random
from collections import deque
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from statistics import mean, median
from typing import Protocol, Sequence


class PhaseAction(str, Enum):
    DRAFT = "draft"
    TARGET = "target"


class RequestStatus(str, Enum):
    WAITING = "waiting"
    PREFILL = "prefill"
    DRAFT_ELIGIBLE = "draft_eligible"
    ACTIVE_DRAFT = "active_draft"
    READY = "ready"
    ACTIVE_AR = "active_ar"
    FINISHED = "finished"


class WorkKind(str, Enum):
    PREFILL = "prefill"
    VERIFY = "verify"
    AR = "ar"


_DEFAULT_ACCEPTANCE = (0.72, 0.52, 0.36, 0.25, 0.17, 0.11, 0.07, 0.04)


@dataclass(slots=True)
class PhaseRequest:
    request_id: str
    arrival_time_ms: float
    prompt_tokens: int = 512
    max_output_tokens: int = 64
    draft_cap: int = 8
    acceptance_probs: tuple[float, ...] = _DEFAULT_ACCEPTANCE
    confidence_by_position: tuple[float, ...] = ()
    speculative: bool = True

    status: RequestStatus = RequestStatus.WAITING
    prefill_progress: int = 0
    committed_tokens: int = 0
    draft_depth: int = 0
    total_draft_tokens: int = 0
    accepted_candidate_tokens: int = 0
    ready_since_ms: float | None = None
    prefill_first_start_ms: float | None = None
    prefill_last_scheduled_ms: float | None = None
    prefill_complete_time_ms: float | None = None
    first_token_time_ms: float | None = None
    last_token_time_ms: float | None = None
    completed_time_ms: float | None = None
    commit_gap_samples_ms: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must be non-empty")
        if self.arrival_time_ms < 0:
            raise ValueError("arrival_time_ms must be non-negative")
        if self.prompt_tokens < 1:
            raise ValueError("prompt_tokens must be positive")
        if self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")
        if self.draft_cap < 1:
            raise ValueError("draft_cap must be positive")
        self.acceptance_probs = tuple(float(p) for p in self.acceptance_probs)
        self.confidence_by_position = tuple(
            float(p) for p in self.confidence_by_position
        )
        if not self.acceptance_probs:
            raise ValueError("acceptance_probs must not be empty")
        if any(not 0.0 <= p <= 1.0 for p in self.acceptance_probs):
            raise ValueError("acceptance probabilities must be in [0, 1]")
        if any(not 0.0 <= p <= 1.0 for p in self.confidence_by_position):
            raise ValueError("confidence values must be in [0, 1]")

    @property
    def remaining_prompt_tokens(self) -> int:
        return max(0, self.prompt_tokens - self.prefill_progress)

    @property
    def remaining_output_tokens(self) -> int:
        return max(0, self.max_output_tokens - self.committed_tokens)

    def conditional_acceptance_probability(self, position: int) -> float:
        if position < 0:
            raise ValueError("position must be non-negative")
        return self.acceptance_probs[min(position, len(self.acceptance_probs) - 1)]

    def confidence_at(self, position: int) -> float:
        if position < 0:
            raise ValueError("position must be non-negative")
        if position >= len(self.confidence_by_position):
            return 1.0
        return self.confidence_by_position[position]

    def expected_next_candidate_acceptance(self) -> float:
        if self.draft_depth >= self.draft_cap:
            return 0.0
        if self.remaining_output_tokens <= self.draft_depth + 1:
            return 0.0
        survival = 1.0
        for position in range(self.draft_depth):
            survival *= self.conditional_acceptance_probability(position)
        return survival * self.conditional_acceptance_probability(self.draft_depth)

    def expected_verify_tokens(self) -> float:
        if self.remaining_output_tokens <= 0:
            return 0.0
        survival = 1.0
        expected_accepted = 0.0
        for position in range(min(self.draft_depth, self.remaining_output_tokens - 1)):
            survival *= self.conditional_acceptance_probability(position)
            expected_accepted += survival
        return min(float(self.remaining_output_tokens), 1.0 + expected_accepted)


@dataclass(frozen=True, slots=True)
class PhaseConfig:
    max_num_seqs: int = 4
    max_num_batched_tokens: int = 2048
    max_draft_batch_size: int = 4
    fixed_h: int = 8
    confidence_threshold: float = 0.8
    ready_wait_budget_ms: float = 8.0
    ar_itl_budget_ms: float = 40.0
    prefill_wait_budget_ms: float = 12.0
    max_consecutive_target_forwards: int = 4
    max_steps: int = 200_000
    seed: int = 42
    short_prefill_threshold: int = 512

    def __post_init__(self) -> None:
        for name in ("max_num_seqs", "max_num_batched_tokens", "max_draft_batch_size",
                     "fixed_h", "max_consecutive_target_forwards", "max_steps",
                     "short_prefill_threshold"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        for name in ("ready_wait_budget_ms", "ar_itl_budget_ms", "prefill_wait_budget_ms"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class PhaseExecutionProfile:
    """Shape-aware synthetic service-time model; not a hardware measurement."""

    draft_base_ms: float = 0.60
    draft_per_request_ms: float = 0.22
    draft_per_context_token_ms: float = 0.000015
    target_base_ms: float = 0.90
    target_per_request_ms: float = 0.20
    prefill_per_token_ms: float = 0.0008
    prefill_context_per_token_ms: float = 0.0000005
    verify_per_candidate_ms: float = 0.055
    verify_context_per_token_ms: float = 0.000012
    ar_per_request_ms: float = 0.20
    ar_context_per_token_ms: float = 0.000012

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.draft_base_ms == 0 and self.draft_per_request_ms == 0:
            raise ValueError("Draft profile must have positive execution cost")
        if self.target_base_ms == 0 and self.target_per_request_ms == 0:
            raise ValueError("Target profile must have positive execution cost")

    @staticmethod
    def _context(request: PhaseRequest) -> int:
        return request.prompt_tokens + request.committed_tokens

    def draft_duration_ms(self, requests: Sequence[PhaseRequest]) -> float:
        if not requests:
            raise ValueError("Draft duration requires a non-empty batch")
        context = sum(self._context(req) + req.draft_depth for req in requests)
        duration = (
            self.draft_base_ms
            + self.draft_per_request_ms * len(requests)
            + self.draft_per_context_token_ms * context
        )
        return max(duration, 1e-6)

    def target_duration_ms(self, items: Sequence[TargetWorkItem]) -> float:
        if not items:
            raise ValueError("Target duration requires a non-empty batch")
        duration = self.target_base_ms + self.target_per_request_ms * len(items)
        for item in items:
            req = item.request
            context = self._context(req)
            if item.kind is WorkKind.PREFILL:
                prior = req.prefill_progress
                tokens = item.token_count
                duration += self.prefill_per_token_ms * tokens
                duration += self.prefill_context_per_token_ms * (
                    prior * tokens + 0.5 * tokens * tokens
                )
            elif item.kind is WorkKind.VERIFY:
                candidates = max(0, item.token_count - 1)
                duration += self.verify_per_candidate_ms * candidates
                duration += self.verify_context_per_token_ms * context * item.token_count
            else:
                duration += self.ar_per_request_ms
                duration += self.ar_context_per_token_ms * context
        return max(duration, 1e-6)

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_json(cls, path: str | Path) -> PhaseExecutionProfile:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("profile JSON must contain an object")
        unknown = set(payload) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown profile fields: {sorted(unknown)}")
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class TargetWorkItem:
    request: PhaseRequest
    kind: WorkKind
    token_count: int


@dataclass(frozen=True, slots=True)
class PhaseStepRecord:
    step: int
    action: str
    start_ms: float
    end_ms: float
    duration_ms: float
    request_count: int
    target_tokens: int
    prefill_requests: int
    verify_requests: int
    ar_requests: int
    ready_before: int
    ready_after: int
    active_draft_after: int
    committed_tokens: int


class TargetBatcher(Protocol):
    name: str

    def plan(self, simulator: PhaseAwareSimulator) -> list[TargetWorkItem]:
        ...


class _BaseTargetBatcher:
    name = "vllm_chunked"

    def prefill_candidates(self, simulator: PhaseAwareSimulator) -> list[PhaseRequest]:
        return list(simulator.prefill_queue)

    def prefill_budget_tokens(
        self, simulator: PhaseAwareSimulator, remaining_tokens: int
    ) -> int:
        return remaining_tokens

    def prefill_chunk_limit(
        self, simulator: PhaseAwareSimulator, request: PhaseRequest
    ) -> int:
        return simulator.config.max_num_batched_tokens

    def plan(self, simulator: PhaseAwareSimulator) -> list[TargetWorkItem]:
        cfg = simulator.config
        items: list[TargetWorkItem] = []
        selected: set[str] = set()
        token_budget = cfg.max_num_batched_tokens

        def add(request: PhaseRequest, kind: WorkKind, token_count: int) -> bool:
            nonlocal token_budget
            if request.request_id in selected or len(items) >= cfg.max_num_seqs:
                return False
            if token_count < 1 or token_count > token_budget:
                return False
            items.append(TargetWorkItem(request, kind, token_count))
            selected.add(request.request_id)
            token_budget -= token_count
            return True

        # Running Decode-like work is selected before Waiting Prefill work.
        for request in simulator.ready_queue:
            add(request, WorkKind.VERIFY, request.draft_depth + 1)
        for request in simulator.ar_queue:
            add(request, WorkKind.AR, 1)

        prefill_budget = min(
            token_budget,
            self.prefill_budget_tokens(simulator, token_budget),
        )
        for request in self.prefill_candidates(simulator):
            if len(items) >= cfg.max_num_seqs or prefill_budget <= 0:
                break
            if request.request_id in selected:
                continue
            chunk = min(
                request.remaining_prompt_tokens,
                self.prefill_chunk_limit(simulator, request),
                prefill_budget,
            )
            if add(request, WorkKind.PREFILL, chunk):
                prefill_budget -= chunk

        return items


class VllmChunkedTargetBatcher(_BaseTargetBatcher):
    """FCFS Prefill after Ready/AR work, chunked by remaining token budget."""

    name = "vllm_chunked"


class DeadlineAwareTargetBatcher(_BaseTargetBatcher):
    """Earliest-deadline-first selection across Verify, AR, and chunked Prefill.

    Deadlines are request-local service deadlines derived from the corresponding
    queue's wait budget. Prefill chunks are bounded so one long prompt cannot
    consume the entire Target token budget in a single forward.
    """

    name = "deadline_aware"

    def __init__(self, prefill_chunk_tokens: int = 512) -> None:
        if prefill_chunk_tokens < 1:
            raise ValueError("prefill_chunk_tokens must be positive")
        self.prefill_chunk_tokens = prefill_chunk_tokens

    def plan(self, simulator: PhaseAwareSimulator) -> list[TargetWorkItem]:
        cfg = simulator.config
        candidates: list[
            tuple[float, int, float, str, WorkKind, PhaseRequest]
        ] = []

        for req in simulator.ready_queue:
            base = (
                req.ready_since_ms
                if req.ready_since_ms is not None
                else simulator.now_ms
            )
            candidates.append((
                base + cfg.ready_wait_budget_ms,
                0,
                req.arrival_time_ms,
                req.request_id,
                WorkKind.VERIFY,
                req,
            ))

        for req in simulator.ar_queue:
            base = (
                req.last_token_time_ms
                if req.last_token_time_ms is not None
                else req.arrival_time_ms
            )
            candidates.append((
                base + cfg.ar_itl_budget_ms,
                1,
                req.arrival_time_ms,
                req.request_id,
                WorkKind.AR,
                req,
            ))

        for req in simulator.prefill_queue:
            # TTFT is not emitted until the final Prefill chunk completes, so
            # partial service must not reset the request's absolute TTFT deadline.
            candidates.append((
                req.arrival_time_ms + cfg.prefill_wait_budget_ms,
                2,
                req.arrival_time_ms,
                req.request_id,
                WorkKind.PREFILL,
                req,
            ))

        candidates.sort(key=lambda item: item[:4])
        items: list[TargetWorkItem] = []
        token_budget = cfg.max_num_batched_tokens
        selected: set[str] = set()

        for _, _, _, _, kind, req in candidates:
            if len(items) >= cfg.max_num_seqs or token_budget <= 0:
                break
            if req.request_id in selected:
                continue

            if kind is WorkKind.VERIFY:
                token_count = req.draft_depth + 1
            elif kind is WorkKind.AR:
                token_count = 1
            else:
                token_count = min(
                    req.remaining_prompt_tokens,
                    self.prefill_chunk_tokens,
                    token_budget,
                )

            if token_count < 1 or token_count > token_budget:
                continue
            items.append(TargetWorkItem(req, kind, token_count))
            selected.add(req.request_id)
            token_budget -= token_count

        return items


class SarathiTargetBatcher(_BaseTargetBatcher):
    """Sarathi-inspired stall-free batches with a bounded Prefill chunk."""

    name = "sarathi"

    def __init__(self, chunk_tokens: int = 512) -> None:
        if chunk_tokens < 1:
            raise ValueError("chunk_tokens must be positive")
        self.chunk_tokens = chunk_tokens

    def prefill_chunk_limit(
        self, simulator: PhaseAwareSimulator, request: PhaseRequest
    ) -> int:
        return self.chunk_tokens


class LengthAwareTargetBatcher(_BaseTargetBatcher):
    """LAPS-inspired length buckets, bounded waiting, and adaptive Prefill budget.

    This is an algorithmic analogue for comparison, not a reproduction of LAPS.
    """

    name = "length_aware"

    def __init__(
        self,
        short_threshold: int = 512,
        bucket_width: int = 256,
        base_wait_ms: float = 2.0,
        min_wait_ms: float = 0.5,
        max_wait_ms: float = 5.0,
        min_prefill_budget: int = 256,
        max_prefill_budget: int = 1024,
        long_chunk_tokens: int = 256,
    ) -> None:
        if min(short_threshold, bucket_width, min_prefill_budget,
               max_prefill_budget, long_chunk_tokens) < 1:
            raise ValueError("length-aware limits must be positive")
        if min_wait_ms <= 0 or base_wait_ms <= 0 or max_wait_ms < min_wait_ms:
            raise ValueError("invalid length-aware wait-window bounds")
        if min_prefill_budget > max_prefill_budget:
            raise ValueError("minimum Prefill budget exceeds maximum")
        self.short_threshold = short_threshold
        self.bucket_width = bucket_width
        self.base_wait_ms = base_wait_ms
        self.min_wait_ms = min_wait_ms
        self.max_wait_ms = max_wait_ms
        self.min_prefill_budget = min_prefill_budget
        self.max_prefill_budget = max_prefill_budget
        self.long_chunk_tokens = long_chunk_tokens

    def adaptive_wait_window_ms(self, simulator: PhaseAwareSimulator) -> float:
        intervals = simulator.short_prefill_intervals_ms[-8:]
        if intervals:
            estimate = median(intervals)
            return min(
                self.max_wait_ms,
                max(self.min_wait_ms, 0.25 * estimate),
            )
        return min(self.max_wait_ms, max(self.min_wait_ms, self.base_wait_ms))

    def should_wait_for_short_batch(self, simulator: PhaseAwareSimulator) -> bool:
        short = [
            req for req in simulator.prefill_queue
            if req.prompt_tokens <= self.short_threshold
        ]
        if not short or not simulator.active_draft:
            return False
        if simulator.ready_queue or simulator.ar_queue:
            return False

        def recency(req: PhaseRequest) -> tuple[float, float, str]:
            last = req.prefill_last_scheduled_ms
            return (
                last if last is not None else req.arrival_time_ms,
                req.arrival_time_ms,
                req.request_id,
            )

        oldest = min(short, key=recency)
        bucket = (oldest.prompt_tokens - 1) // self.bucket_width
        same_bucket = [
            req for req in short
            if (req.prompt_tokens - 1) // self.bucket_width == bucket
        ]
        if len(same_bucket) >= 2:
            return False
        last_service = oldest.prefill_last_scheduled_ms
        wait_origin = (
            last_service if last_service is not None else oldest.arrival_time_ms
        )
        wait_ms = simulator.now_ms - wait_origin
        return wait_ms < self.adaptive_wait_window_ms(simulator)

    def prefill_candidates(self, simulator: PhaseAwareSimulator) -> list[PhaseRequest]:
        short = [
            req for req in simulator.prefill_queue
            if req.prompt_tokens <= self.short_threshold
        ]
        long = [
            req for req in simulator.prefill_queue
            if req.prompt_tokens > self.short_threshold
        ]
        if not short and not long:
            return []

        def recency(req: PhaseRequest) -> tuple[float, float, str]:
            last = req.prefill_last_scheduled_ms
            return (
                last if last is not None else req.arrival_time_ms,
                req.arrival_time_ms,
                req.request_id,
            )

        oldest_short = min(short, key=recency) if short else None
        oldest_long = min(long, key=recency) if long else None
        if oldest_short is not None and oldest_long is not None:
            long_service_age = simulator.now_ms - recency(oldest_long)[0]
            short_service_age = simulator.now_ms - recency(oldest_short)[0]
            choose_long = (
                long_service_age >= self.max_wait_ms
                or recency(oldest_long) < recency(oldest_short)
            )
        else:
            choose_long = oldest_long is not None

        pool = long if choose_long else short
        if not pool:
            pool = short or long
        oldest = min(pool, key=recency)
        bucket = (oldest.prompt_tokens - 1) // self.bucket_width
        return sorted(
            [
                req for req in pool
                if (req.prompt_tokens - 1) // self.bucket_width == bucket
            ],
            key=recency,
        )

    def prefill_budget_tokens(
        self, simulator: PhaseAwareSimulator, remaining_tokens: int
    ) -> int:
        backlog = sum(req.remaining_prompt_tokens for req in simulator.prefill_queue)
        if backlog <= 0:
            return 0
        pressure = min(1.0, backlog / max(1, self.max_prefill_budget))
        ready_urgency = min(
            1.0,
            simulator.oldest_ready_wait_ms() / simulator.config.ready_wait_budget_ms,
        )
        ar_urgency = min(
            1.0,
            simulator.oldest_ar_wait_ms() / simulator.config.ar_itl_budget_ms,
        )
        urgency = max(ready_urgency, ar_urgency)
        desired = self.min_prefill_budget + (
            self.max_prefill_budget - self.min_prefill_budget
        ) * pressure * (1.0 - 0.6 * urgency)
        return min(remaining_tokens, max(1, int(math.ceil(desired))))

    def prefill_chunk_limit(
        self, simulator: PhaseAwareSimulator, request: PhaseRequest
    ) -> int:
        if request.prompt_tokens > self.short_threshold:
            return self.long_chunk_tokens
        return simulator.config.max_num_batched_tokens


class PhaseSchedulingPolicy(Protocol):
    name: str
    batcher: TargetBatcher

    def choose_action(self, simulator: PhaseAwareSimulator) -> PhaseAction:
        ...


class FixedHPhasePolicy:
    name = "fixed_h"

    def __init__(self, h: int = 8, batcher: TargetBatcher | None = None) -> None:
        if h < 1:
            raise ValueError("h must be positive")
        self.h = h
        self.batcher = batcher or VllmChunkedTargetBatcher()

    def choose_action(self, simulator: PhaseAwareSimulator) -> PhaseAction:
        actions = simulator.available_actions()
        if actions == [PhaseAction.TARGET]:
            return PhaseAction.TARGET
        if actions == [PhaseAction.DRAFT]:
            return PhaseAction.DRAFT
        if simulator.ready_queue and (
            simulator.drafts_since_verify >= self.h
            or simulator.ready_backlog_after_verify
        ):
            return PhaseAction.TARGET
        if simulator.oldest_ready_wait_ms() >= simulator.config.ready_wait_budget_ms:
            return PhaseAction.TARGET
        if simulator.oldest_ar_wait_ms() >= simulator.config.ar_itl_budget_ms:
            return PhaseAction.TARGET
        if simulator.oldest_prefill_wait_ms() >= simulator.config.prefill_wait_budget_ms:
            return PhaseAction.TARGET
        return PhaseAction.DRAFT


class QueuePressurePhasePolicy:
    name = "queue_pressure"

    def __init__(
        self,
        batcher: TargetBatcher | None = None,
        ready_wait_budget_ms: float | None = None,
        ar_itl_budget_ms: float | None = None,
        prefill_wait_budget_ms: float | None = None,
        max_consecutive_target_forwards: int | None = None,
    ) -> None:
        self.batcher = batcher or VllmChunkedTargetBatcher()
        self.ready_wait_budget_ms = ready_wait_budget_ms
        self.ar_itl_budget_ms = ar_itl_budget_ms
        self.prefill_wait_budget_ms = prefill_wait_budget_ms
        self.max_consecutive_target_forwards = max_consecutive_target_forwards

    def choose_action(self, simulator: PhaseAwareSimulator) -> PhaseAction:
        actions = simulator.available_actions()
        if actions == [PhaseAction.TARGET]:
            return PhaseAction.TARGET
        if actions == [PhaseAction.DRAFT]:
            return PhaseAction.DRAFT

        cfg = simulator.config
        ready_budget = self.ready_wait_budget_ms or cfg.ready_wait_budget_ms
        ar_budget = self.ar_itl_budget_ms or cfg.ar_itl_budget_ms
        prefill_budget = self.prefill_wait_budget_ms or cfg.prefill_wait_budget_ms
        max_target_burst = (
            self.max_consecutive_target_forwards
            or cfg.max_consecutive_target_forwards
        )

        if simulator.oldest_ready_wait_ms() >= ready_budget:
            return PhaseAction.TARGET
        if simulator.oldest_ar_wait_ms() >= ar_budget:
            return PhaseAction.TARGET
        if simulator.oldest_prefill_wait_ms() >= prefill_budget:
            return PhaseAction.TARGET
        if simulator.consecutive_target_forwards >= max_target_burst:
            return PhaseAction.DRAFT

        plan = self.batcher.plan(simulator)
        if not plan:
            return PhaseAction.DRAFT

        target_yield = 0.0
        for item in plan:
            req = item.request
            if item.kind is WorkKind.VERIFY:
                target_yield += req.expected_verify_tokens()
            elif item.kind is WorkKind.AR:
                target_yield += 1.0
            else:
                fraction = item.token_count / max(1, req.remaining_prompt_tokens)
                target_yield += 0.15 + (0.85 if item.token_count >= req.remaining_prompt_tokens else 0.0)
                target_yield += 0.10 * fraction

        target_time = simulator.profile.target_duration_ms(plan)
        target_score = target_yield / target_time
        target_score += 0.10 * min(
            simulator.oldest_ready_wait_ms() / ready_budget, 2.0
        )
        target_score += 0.08 * min(
            simulator.oldest_ar_wait_ms() / ar_budget, 2.0
        )
        target_score += 0.06 * min(
            simulator.oldest_prefill_wait_ms() / prefill_budget, 2.0
        )

        draft_yield = sum(
            req.expected_next_candidate_acceptance()
            for req in simulator.active_draft
        )
        draft_time = simulator.profile.draft_duration_ms(simulator.active_draft)
        eligible_supply = min(
            1.0,
            len(simulator.draft_eligible_queue) / cfg.max_draft_batch_size,
        )
        draft_score = (draft_yield / draft_time) * (1.0 + 0.15 * eligible_supply)

        return (
            PhaseAction.TARGET
            if target_score >= draft_score
            else PhaseAction.DRAFT
        )


class DynamicPhasePolicy:
    """Deadline-aware stage selector with an emergent Draft window.

    It never counts toward a configured H. It compares useful Target work with
    the marginal value of another Draft forward, while checking whether that
    Draft forward would push any queued request beyond its wait budget.
    """

    name = "dynamic"

    def __init__(
        self,
        batcher: TargetBatcher | None = None,
        fill_gain_weight: float = 0.25,
        urgency_weight: float = 0.20,
    ) -> None:
        if fill_gain_weight < 0 or urgency_weight < 0:
            raise ValueError("dynamic scheduling weights must be non-negative")
        self.batcher = batcher or VllmChunkedTargetBatcher()
        self.fill_gain_weight = fill_gain_weight
        self.urgency_weight = urgency_weight

    @staticmethod
    def _target_yield(
        simulator: PhaseAwareSimulator,
        plan: Sequence[TargetWorkItem],
    ) -> float:
        value = 0.0
        for item in plan:
            req = item.request
            if item.kind is WorkKind.VERIFY:
                value += req.expected_verify_tokens()
            elif item.kind is WorkKind.AR:
                value += 1.0
            else:
                fraction = item.token_count / max(1, req.remaining_prompt_tokens)
                completes = item.token_count >= req.remaining_prompt_tokens
                value += 0.15 + (0.85 if completes else 0.0) + 0.10 * fraction
        return value

    @staticmethod
    def _wait_pressure(simulator: PhaseAwareSimulator) -> tuple[float, bool]:
        cfg = simulator.config
        oldest_prefill_ttft_age = max(
            (
                simulator.now_ms - req.arrival_time_ms
                for req in simulator.prefill_queue
            ),
            default=0.0,
        )
        ages_and_budgets = [
            (simulator.oldest_ready_wait_ms(), cfg.ready_wait_budget_ms),
            (simulator.oldest_ar_wait_ms(), cfg.ar_itl_budget_ms),
            (oldest_prefill_ttft_age, cfg.prefill_wait_budget_ms),
        ]
        pressure = max(
            (age / budget for age, budget in ages_and_budgets),
            default=0.0,
        )
        return pressure, any(age >= budget for age, budget in ages_and_budgets)

    def choose_action(self, simulator: PhaseAwareSimulator) -> PhaseAction:
        actions = simulator.available_actions()
        if len(actions) == 1:
            return actions[0]

        plan = self.batcher.plan(simulator)
        if not plan:
            return PhaseAction.DRAFT

        draft_batch = simulator.active_draft
        draft_duration = simulator.profile.draft_duration_ms(draft_batch)
        target_duration = simulator.profile.target_duration_ms(plan)

        # Do not cross a request's configured wait budget just to improve packing.
        cfg = simulator.config
        prefill_ttft_age = max(
            (
                simulator.now_ms - req.arrival_time_ms
                for req in simulator.prefill_queue
            ),
            default=0.0,
        )
        prefill_service = (
            target_duration
            if any(item.kind is WorkKind.PREFILL for item in plan)
            else 0.0
        )
        if (
            simulator.oldest_ready_wait_ms() + draft_duration
            >= cfg.ready_wait_budget_ms
            or simulator.oldest_ar_wait_ms() + draft_duration
            >= cfg.ar_itl_budget_ms
            or prefill_ttft_age + draft_duration + prefill_service
            >= cfg.prefill_wait_budget_ms
        ):
            return PhaseAction.TARGET
        target_fill = len(plan) / cfg.max_num_seqs
        target_score = (
            self._target_yield(simulator, plan)
            / max(target_duration, 1e-6)
        )
        target_score *= 0.55 + 0.45 * target_fill

        wait_pressure, _ = self._wait_pressure(simulator)
        target_score *= 1.0 + self.urgency_weight * min(wait_pressure, 1.0)

        draft_yield = sum(
            req.expected_next_candidate_acceptance()
            for req in draft_batch
        )
        eligible_supply = min(
            1.0,
            len(simulator.draft_eligible_queue) / cfg.max_draft_batch_size,
        )
        draft_score = (
            draft_yield / max(draft_duration, 1e-6)
        ) * (1.0 + 0.15 * eligible_supply)

        # Estimate whether one more Draft forward will materially improve the
        # next Target batch. The clone keeps this decision side-effect free.
        projected = simulator.clone()
        projected.execute_action(PhaseAction.DRAFT)
        projected_plan = self.batcher.plan(projected)
        projected_fill = len(projected_plan) / cfg.max_num_seqs
        fill_gain = max(0.0, projected_fill - target_fill)
        draft_score *= 1.0 + self.fill_gain_weight * fill_gain

        return (
            PhaseAction.DRAFT
            if draft_score > target_score
            else PhaseAction.TARGET
        )


class LoadAdaptivePhasePolicy(DynamicPhasePolicy):
    """Delay Verify only when one more Draft is predicted to improve packing.

    Under sparse load, another Draft does not increase the number of requests
    in the next Target batch, so the policy verifies immediately. Under busy
    load, it may continue Draft when the projected batch gains request slots.
    All queue wait-budget guards remain hard constraints.
    """

    name = "load_adaptive"

    def __init__(
        self,
        batcher: TargetBatcher | None = None,
        min_batch_gain: int = 1,
    ) -> None:
        super().__init__(batcher=batcher, fill_gain_weight=0.0, urgency_weight=0.20)
        if min_batch_gain < 1:
            raise ValueError("min_batch_gain must be positive")
        self.min_batch_gain = min_batch_gain

    def choose_action(self, simulator: PhaseAwareSimulator) -> PhaseAction:
        actions = simulator.available_actions()
        if len(actions) == 1:
            return actions[0]

        current_plan = self.batcher.plan(simulator)
        if not current_plan:
            return PhaseAction.DRAFT

        cfg = simulator.config
        draft_duration = simulator.profile.draft_duration_ms(simulator.active_draft)
        target_duration = simulator.profile.target_duration_ms(current_plan)
        prefill_ttft_age = max(
            (
                simulator.now_ms - req.arrival_time_ms
                for req in simulator.prefill_queue
            ),
            default=0.0,
        )
        prefill_service = (
            target_duration
            if any(item.kind is WorkKind.PREFILL for item in current_plan)
            else 0.0
        )

        # Hard latency guard: never delay a queued class beyond its budget.
        if (
            simulator.oldest_ready_wait_ms() + draft_duration
            >= cfg.ready_wait_budget_ms
            or simulator.oldest_ar_wait_ms() + draft_duration
            >= cfg.ar_itl_budget_ms
            or prefill_ttft_age + draft_duration + prefill_service
            >= cfg.prefill_wait_budget_ms
        ):
            return PhaseAction.TARGET

        if len(current_plan) >= cfg.max_num_seqs:
            return PhaseAction.TARGET

        projected = simulator.clone()
        projected.execute_action(PhaseAction.DRAFT)
        projected_plan = self.batcher.plan(projected)
        marginal_gain = max(0, len(projected_plan) - len(current_plan))

        # Sparse-load rule: if another Draft cannot add a Target request,
        # waiting only adds latency and cannot reduce Target forward count.
        if marginal_gain < self.min_batch_gain:
            return PhaseAction.TARGET

        # If the extra Draft creates useful parallel work, permit one more step.
        return PhaseAction.DRAFT


class JointPhasePolicy(QueuePressurePhasePolicy):
    """Joint stage selection plus LAPS-inspired Prefill batching window."""

    name = "joint_phase_length_aware"

    def __init__(self, batcher: LengthAwareTargetBatcher | None = None) -> None:
        super().__init__(batcher or LengthAwareTargetBatcher())

    def choose_action(self, simulator: PhaseAwareSimulator) -> PhaseAction:
        if (
            isinstance(self.batcher, LengthAwareTargetBatcher)
            and self.batcher.should_wait_for_short_batch(simulator)
        ):
            return PhaseAction.DRAFT
        return super().choose_action(simulator)


class PhaseLookaheadPolicy:
    """Short-horizon model-predictive stage selector for the phase-aware model."""

    name = "lookahead"

    def __init__(
        self,
        horizon: int = 4,
        rollouts: int = 2,
        terminal_discount: float = 0.35,
        rollout_policy: QueuePressurePhasePolicy | None = None,
    ) -> None:
        if horizon < 1 or rollouts < 1:
            raise ValueError("lookahead horizon and rollouts must be positive")
        if not 0.0 <= terminal_discount <= 1.0:
            raise ValueError("terminal_discount must be in [0, 1]")
        self.horizon = horizon
        self.rollouts = rollouts
        self.terminal_discount = terminal_discount
        self.rollout_policy = rollout_policy or QueuePressurePhasePolicy()
        self.batcher = self.rollout_policy.batcher

    @staticmethod
    def _terminal_potential(simulator: PhaseAwareSimulator) -> float:
        value = sum(req.expected_verify_tokens() for req in simulator.ready_queue)
        value += float(len(simulator.ar_queue))
        value += sum(
            0.15 + 0.85 * req.prefill_progress / max(1, req.prompt_tokens)
            for req in simulator.prefill_queue
        )
        value += sum(
            0.20 + req.expected_next_candidate_acceptance()
            for req in simulator.active_draft
        )
        return value

    def choose_action(self, simulator: PhaseAwareSimulator) -> PhaseAction:
        actions = simulator.available_actions()
        if len(actions) == 1:
            return actions[0]

        scores: dict[PhaseAction, float] = {}
        for action in actions:
            rollout_scores: list[float] = []
            for rollout_index in range(self.rollouts):
                clone = simulator.clone()
                clone.policy = self.rollout_policy
                clone.rng.seed(
                    simulator.config.seed
                    + simulator.decision_count * 1009
                    + rollout_index
                )
                start_ms = clone.now_ms
                start_output = clone.total_output_tokens
                clone.execute_action(action)
                forwards = 1

                while forwards < self.horizon and not clone.is_finished:
                    clone.refresh()
                    next_actions = clone.available_actions()
                    if not next_actions:
                        if not clone.advance_to_next_arrival():
                            break
                        continue
                    next_action = self.rollout_policy.choose_action(clone)
                    clone.execute_action(next_action)
                    forwards += 1

                elapsed = max(clone.now_ms - start_ms, 1e-6)
                committed = clone.total_output_tokens - start_output
                terminal = self._terminal_potential(clone)
                rollout_scores.append(
                    (committed + self.terminal_discount * terminal) / elapsed
                )
            scores[action] = mean(rollout_scores)
        return max(scores, key=scores.get)


@dataclass(frozen=True, slots=True)
class _StepMetrics:
    prefill_wait_ms: float
    ready_wait_ms: float


class PhaseAwareSimulator:
    """One-GPU discrete-event model with mixed Target batches."""

    def __init__(
        self,
        requests: Sequence[PhaseRequest],
        config: PhaseConfig,
        profile: PhaseExecutionProfile,
        policy: PhaseSchedulingPolicy,
    ) -> None:
        self.requests = sorted(
            copy.deepcopy(list(requests)),
            key=lambda req: (req.arrival_time_ms, req.request_id),
        )
        self.config = config
        self.profile = profile
        self.policy = policy
        self.rng = random.Random(config.seed)

        for req in self.requests:
            if req.draft_cap + 1 > config.max_num_batched_tokens:
                raise ValueError(
                    f"{req.request_id}: draft_cap + 1 exceeds Target token budget"
                )

        self.now_ms = 0.0
        self.gpu_busy_ms = 0.0
        self.idle_ms = 0.0
        self.pending_arrival_index = 0

        self.prefill_queue: deque[PhaseRequest] = deque()
        self.draft_eligible_queue: deque[PhaseRequest] = deque()
        self.active_draft: list[PhaseRequest] = []
        self.ready_queue: deque[PhaseRequest] = deque()
        self.ar_queue: deque[PhaseRequest] = deque()
        self.short_prefill_arrival_times_ms: list[float] = []
        self.short_prefill_intervals_ms: list[float] = []

        self.draft_forward_count = 0
        self.target_forward_count = 0
        self.drafts_since_verify = 0
        self.draft_window_samples: list[int] = []
        self.ready_backlog_after_verify = False
        self.consecutive_target_forwards = 0
        self.consecutive_draft_forwards = 0
        self.decision_count = 0

        self.total_draft_tokens = 0
        self.accepted_candidate_tokens = 0
        self.total_prefill_tokens = 0
        self.prefill_chunk_count = 0
        self.prefill_request_count = 0
        self.verify_request_count = 0
        self.ar_token_count = 0
        self.mixed_target_forward_count = 0
        self.target_token_counts: list[int] = []
        self.target_batch_sizes: list[int] = []
        self.target_compositions: list[dict[str, int]] = []
        self.prefill_chunk_sizes: list[int] = []
        self.adaptive_prefill_wait_samples_ms: list[float] = []
        self.adaptive_prefill_budget_samples: list[int] = []
        self.prefill_wait_samples_ms: list[float] = []
        self.prefill_latency_samples_ms: list[float] = []
        self.ready_wait_samples_ms: list[float] = []
        self.ar_wait_samples_ms: list[float] = []
        self.ttft_samples_ms: list[float] = []
        self.commit_gap_samples_ms: list[float] = []
        self.step_records: list[PhaseStepRecord] = []

        self.refresh()

    def clone(self) -> PhaseAwareSimulator:
        return copy.deepcopy(self)

    @property
    def total_output_tokens(self) -> int:
        return sum(req.committed_tokens for req in self.requests)

    @property
    def completed_requests(self) -> int:
        return sum(req.status is RequestStatus.FINISHED for req in self.requests)

    @property
    def is_finished(self) -> bool:
        return (
            self.pending_arrival_index >= len(self.requests)
            and all(req.status is RequestStatus.FINISHED for req in self.requests)
        )

    def refresh(self) -> None:
        self._process_arrivals()
        self._refill_active_draft()

    def _process_arrivals(self) -> None:
        while self.pending_arrival_index < len(self.requests):
            req = self.requests[self.pending_arrival_index]
            if req.arrival_time_ms > self.now_ms:
                break
            self.pending_arrival_index += 1
            if req.status is not RequestStatus.WAITING:
                raise RuntimeError(f"request {req.request_id} arrived more than once")
            req.status = RequestStatus.PREFILL
            self.prefill_queue.append(req)
            if req.prompt_tokens <= self.config.short_prefill_threshold:
                times = self.short_prefill_arrival_times_ms
                if times:
                    self.short_prefill_intervals_ms.append(
                        req.arrival_time_ms - times[-1]
                    )
                times.append(req.arrival_time_ms)

    def _refill_active_draft(self) -> None:
        while (
            len(self.active_draft) < self.config.max_draft_batch_size
            and self.draft_eligible_queue
        ):
            req = self.draft_eligible_queue.popleft()
            if req.remaining_output_tokens <= 1:
                req.status = RequestStatus.ACTIVE_AR
                self.ar_queue.append(req)
                continue
            req.status = RequestStatus.ACTIVE_DRAFT
            self.active_draft.append(req)

    def available_actions(self) -> list[PhaseAction]:
        actions: list[PhaseAction] = []
        if self.active_draft:
            actions.append(PhaseAction.DRAFT)
        if self.prefill_queue or self.ready_queue or self.ar_queue:
            actions.append(PhaseAction.TARGET)
        return actions

    def oldest_ready_wait_ms(self) -> float:
        return max(
            (
                self.now_ms - req.ready_since_ms
                for req in self.ready_queue
                if req.ready_since_ms is not None
            ),
            default=0.0,
        )

    def oldest_ar_wait_ms(self) -> float:
        return max(
            (
                self.now_ms - (
                    req.last_token_time_ms
                    if req.last_token_time_ms is not None
                    else req.arrival_time_ms
                )
                for req in self.ar_queue
            ),
            default=0.0,
        )

    def oldest_prefill_wait_ms(self) -> float:
        return max(
            (
                self.now_ms - (
                    req.prefill_last_scheduled_ms
                    if req.prefill_last_scheduled_ms is not None
                    else req.arrival_time_ms
                )
                for req in self.prefill_queue
            ),
            default=0.0,
        )

    def advance_to_next_arrival(self) -> bool:
        if self.pending_arrival_index >= len(self.requests):
            return False
        next_time = self.requests[self.pending_arrival_index].arrival_time_ms
        if next_time > self.now_ms:
            self.idle_ms += next_time - self.now_ms
            self.now_ms = next_time
        self.refresh()
        return True

    def _remove_selected(self, queue: deque[PhaseRequest], req: PhaseRequest) -> None:
        try:
            queue.remove(req)
        except ValueError as exc:
            raise RuntimeError(
                f"selected request {req.request_id} is not in its expected queue"
            ) from exc

    def _finish(self, req: PhaseRequest) -> None:
        req.status = RequestStatus.FINISHED
        req.completed_time_ms = self.now_ms

    def _record_output_commit(self, req: PhaseRequest, count: int) -> None:
        if count <= 0:
            return
        if req.first_token_time_ms is None:
            req.first_token_time_ms = self.now_ms
            self.ttft_samples_ms.append(self.now_ms - req.arrival_time_ms)
        elif req.last_token_time_ms is not None:
            gap = self.now_ms - req.last_token_time_ms
            req.commit_gap_samples_ms.append(gap)
            self.commit_gap_samples_ms.append(gap)
        req.last_token_time_ms = self.now_ms
        req.committed_tokens += count

    def _sample_accepted_prefix(self, req: PhaseRequest) -> int:
        accepted = 0
        for position in range(req.draft_depth):
            if self.rng.random() < req.conditional_acceptance_probability(position):
                accepted += 1
            else:
                break
        return accepted

    def execute_action(self, action: PhaseAction) -> PhaseStepRecord:
        action = PhaseAction(action)
        if action not in self.available_actions():
            raise ValueError(f"action {action.value} is not currently legal")

        start_ms = self.now_ms
        ready_before = len(self.ready_queue)
        output_before = self.total_output_tokens
        prefill_count = verify_count = ar_count = 0
        target_tokens = 0

        if action is PhaseAction.DRAFT:
            batch = list(self.active_draft)
            duration = self.profile.draft_duration_ms(batch)
            self.now_ms += duration
            self.gpu_busy_ms += duration
            self.draft_forward_count += 1
            self.total_draft_tokens += len(batch)
            self.drafts_since_verify += 1
            self.consecutive_draft_forwards += 1
            self.consecutive_target_forwards = 0

            still_active: list[PhaseRequest] = []
            for req in batch:
                req.draft_depth += 1
                req.total_draft_tokens += 1
                confidence = req.confidence_at(req.draft_depth - 1)
                stop = (
                    req.draft_depth >= req.draft_cap
                    or req.remaining_output_tokens <= req.draft_depth + 1
                    or confidence < self.config.confidence_threshold
                )
                if stop:
                    req.status = RequestStatus.READY
                    req.ready_since_ms = self.now_ms
                    self.ready_queue.append(req)
                else:
                    still_active.append(req)
            self.active_draft = still_active
            self.refresh()
            request_count = len(batch)

        else:
            items = self.policy.batcher.plan(self)
            if not items:
                raise RuntimeError("Target action selected but no legal Target work fits")
            request_ids = [item.request.request_id for item in items]
            if len(request_ids) != len(set(request_ids)):
                raise RuntimeError("a request appears more than once in one Target batch")
            if len(items) > self.config.max_num_seqs:
                raise RuntimeError("Target batch exceeds max_num_seqs")
            target_tokens = sum(item.token_count for item in items)
            if target_tokens > self.config.max_num_batched_tokens:
                raise RuntimeError("Target batch exceeds max_num_batched_tokens")

            if (
                isinstance(self.policy.batcher, LengthAwareTargetBatcher)
                and any(item.kind is WorkKind.PREFILL for item in items)
            ):
                batcher = self.policy.batcher
                self.adaptive_prefill_wait_samples_ms.append(
                    batcher.adaptive_wait_window_ms(self)
                )
                non_prefill_tokens = sum(
                    item.token_count for item in items
                    if item.kind is not WorkKind.PREFILL
                )
                available_for_prefill = max(
                    0, self.config.max_num_batched_tokens - non_prefill_tokens
                )
                self.adaptive_prefill_budget_samples.append(
                    min(
                        available_for_prefill,
                        batcher.prefill_budget_tokens(self, available_for_prefill),
                    )
                )

            ready_waits: dict[str, float] = {}
            for item in items:
                req = item.request
                if item.kind is WorkKind.PREFILL:
                    self._remove_selected(self.prefill_queue, req)
                    prefill_count += 1
                elif item.kind is WorkKind.VERIFY:
                    self._remove_selected(self.ready_queue, req)
                    if req.ready_since_ms is None:
                        raise RuntimeError(f"Ready request {req.request_id} has no timestamp")
                    ready_waits[req.request_id] = start_ms - req.ready_since_ms
                    verify_count += 1
                else:
                    self._remove_selected(self.ar_queue, req)
                    ar_count += 1

            duration = self.profile.target_duration_ms(items)
            self.now_ms += duration
            self.gpu_busy_ms += duration
            self.target_forward_count += 1
            self.target_token_counts.append(target_tokens)
            self.target_batch_sizes.append(len(items))
            self.target_compositions.append({
                "prefill": prefill_count,
                "verify": verify_count,
                "ar": ar_count,
            })
            if sum(value > 0 for value in (prefill_count, verify_count, ar_count)) > 1:
                self.mixed_target_forward_count += 1

            for item in items:
                req = item.request
                if item.kind is WorkKind.PREFILL:
                    if req.prefill_first_start_ms is None:
                        req.prefill_first_start_ms = start_ms
                        self.prefill_wait_samples_ms.append(
                            start_ms - req.arrival_time_ms
                        )
                    req.prefill_progress += item.token_count
                    req.prefill_last_scheduled_ms = self.now_ms
                    self.total_prefill_tokens += item.token_count
                    self.prefill_chunk_count += 1
                    self.prefill_chunk_sizes.append(item.token_count)
                    if req.remaining_prompt_tokens > 0:
                        req.status = RequestStatus.PREFILL
                        self.prefill_queue.append(req)
                    else:
                        req.prefill_complete_time_ms = self.now_ms
                        self.prefill_request_count += 1
                        self.prefill_latency_samples_ms.append(
                            self.now_ms - req.arrival_time_ms
                        )
                        self._record_output_commit(req, 1)
                        if req.remaining_output_tokens <= 0:
                            self._finish(req)
                        elif req.speculative and req.remaining_output_tokens > 1:
                            req.status = RequestStatus.DRAFT_ELIGIBLE
                            self.draft_eligible_queue.append(req)
                        else:
                            req.status = RequestStatus.ACTIVE_AR
                            self.ar_queue.append(req)

                elif item.kind is WorkKind.VERIFY:
                    self.ready_wait_samples_ms.append(ready_waits[req.request_id])
                    accepted = self._sample_accepted_prefix(req)
                    req.accepted_candidate_tokens += accepted
                    self.accepted_candidate_tokens += accepted
                    committed = min(req.remaining_output_tokens, accepted + 1)
                    self._record_output_commit(req, committed)
                    req.draft_depth = 0
                    req.ready_since_ms = None
                    self.verify_request_count += 1
                    if req.remaining_output_tokens <= 0:
                        self._finish(req)
                    elif req.speculative and req.remaining_output_tokens > 1:
                        req.status = RequestStatus.DRAFT_ELIGIBLE
                        self.draft_eligible_queue.append(req)
                    else:
                        req.status = RequestStatus.ACTIVE_AR
                        self.ar_queue.append(req)

                else:
                    previous_token_time = (
                        req.last_token_time_ms
                        if req.last_token_time_ms is not None
                        else req.arrival_time_ms
                    )
                    self.ar_wait_samples_ms.append(start_ms - previous_token_time)
                    self._record_output_commit(req, 1)
                    self.ar_token_count += 1
                    if req.remaining_output_tokens <= 0:
                        self._finish(req)
                    else:
                        req.status = RequestStatus.ACTIVE_AR
                        self.ar_queue.append(req)

            if verify_count > 0:
                self.draft_window_samples.append(self.drafts_since_verify)
                self.drafts_since_verify = 0
                self.ready_backlog_after_verify = bool(self.ready_queue)
            self.consecutive_target_forwards += 1
            self.consecutive_draft_forwards = 0
            self.refresh()
            request_count = len(items)

        self.decision_count += 1
        record = PhaseStepRecord(
            step=self.decision_count,
            action=action.value,
            start_ms=start_ms,
            end_ms=self.now_ms,
            duration_ms=duration,
            request_count=request_count,
            target_tokens=target_tokens,
            prefill_requests=prefill_count,
            verify_requests=verify_count,
            ar_requests=ar_count,
            ready_before=ready_before,
            ready_after=len(self.ready_queue),
            active_draft_after=len(self.active_draft),
            committed_tokens=self.total_output_tokens - output_before,
        )
        self.step_records.append(record)
        return record

    def run(self) -> dict[str, object]:
        while not self.is_finished:
            if self.decision_count >= self.config.max_steps:
                raise RuntimeError(f"simulation exceeded max_steps={self.config.max_steps}")
            self.refresh()
            actions = self.available_actions()
            if not actions:
                if self.advance_to_next_arrival():
                    continue
                unfinished = [
                    req.request_id for req in self.requests
                    if req.status is not RequestStatus.FINISHED
                ]
                raise RuntimeError(f"deadlock with unfinished requests: {unfinished[:8]}")
            action = self.policy.choose_action(self)
            self.execute_action(action)
        return self.summary()

    @staticmethod
    def _percentile(values: Sequence[float], q: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = max(0, min(len(ordered) - 1, int(round(q * (len(ordered) - 1)))))
        return float(ordered[index])

    def summary(self) -> dict[str, object]:
        elapsed = max(self.now_ms, 0.0)
        output = self.total_output_tokens
        busy_ratio = self.gpu_busy_ms / elapsed if elapsed > 0 else 0.0
        return {
            "policy": self.policy.name,
            "requests": len(self.requests),
            "completed_requests": self.completed_requests,
            "output_tokens": output,
            "output_tokens_per_s": output / (elapsed / 1000.0) if elapsed > 0 else 0.0,
            "elapsed_ms": elapsed,
            "gpu_busy_ms": self.gpu_busy_ms,
            "gpu_busy_ratio": busy_ratio,
            "idle_ms": self.idle_ms,
            "draft_forwards": self.draft_forward_count,
            "target_forwards": self.target_forward_count,
            "draft_window_count": len(self.draft_window_samples),
            "draft_window_mean": mean(self.draft_window_samples) if self.draft_window_samples else 0.0,
            "draft_window_p50": self._percentile(self.draft_window_samples, 0.50),
            "draft_window_p95": self._percentile(self.draft_window_samples, 0.95),
            "draft_window_max": max(self.draft_window_samples, default=0),
            "mixed_target_forwards": self.mixed_target_forward_count,
            "mixed_target_ratio": (
                self.mixed_target_forward_count / self.target_forward_count
                if self.target_forward_count else 0.0
            ),
            "total_draft_tokens": self.total_draft_tokens,
            "accepted_candidate_tokens": self.accepted_candidate_tokens,
            "candidate_acceptance_rate": (
                self.accepted_candidate_tokens / self.total_draft_tokens
                if self.total_draft_tokens else 0.0
            ),
            "total_prefill_tokens": self.total_prefill_tokens,
            "prefill_chunks": self.prefill_chunk_count,
            "prefill_completed_requests": self.prefill_request_count,
            "ar_tokens": self.ar_token_count,
            "verify_requests": self.verify_request_count,
            "mean_target_batch_size": mean(self.target_batch_sizes) if self.target_batch_sizes else 0.0,
            "mean_target_tokens": mean(self.target_token_counts) if self.target_token_counts else 0.0,
            "target_token_budget_utilization": (
                sum(self.target_token_counts)
                / (self.target_forward_count * self.config.max_num_batched_tokens)
                if self.target_forward_count else 0.0
            ),
            "target_slot_utilization": (
                sum(self.target_batch_sizes)
                / (self.target_forward_count * self.config.max_num_seqs)
                if self.target_forward_count else 0.0
            ),
            "target_compositions": list(self.target_compositions),
            "prefill_chunk_mean_tokens": mean(self.prefill_chunk_sizes) if self.prefill_chunk_sizes else 0.0,
            "adaptive_prefill_wait_window_mean_ms": (
                mean(self.adaptive_prefill_wait_samples_ms)
                if self.adaptive_prefill_wait_samples_ms else 0.0
            ),
            "adaptive_prefill_budget_mean_tokens": (
                mean(self.adaptive_prefill_budget_samples)
                if self.adaptive_prefill_budget_samples else 0.0
            ),
            "prefill_wait_p50_ms": self._percentile(self.prefill_wait_samples_ms, 0.50),
            "prefill_wait_p95_ms": self._percentile(self.prefill_wait_samples_ms, 0.95),
            "prefill_latency_p50_ms": self._percentile(self.prefill_latency_samples_ms, 0.50),
            "prefill_latency_p95_ms": self._percentile(self.prefill_latency_samples_ms, 0.95),
            "ttft_p50_ms": self._percentile(self.ttft_samples_ms, 0.50),
            "ttft_p95_ms": self._percentile(self.ttft_samples_ms, 0.95),
            "ready_wait_p50_ms": self._percentile(self.ready_wait_samples_ms, 0.50),
            "ready_wait_p95_ms": self._percentile(self.ready_wait_samples_ms, 0.95),
            "ar_wait_p50_ms": self._percentile(self.ar_wait_samples_ms, 0.50),
            "ar_wait_p95_ms": self._percentile(self.ar_wait_samples_ms, 0.95),
            "commit_gap_p50_ms": self._percentile(self.commit_gap_samples_ms, 0.50),
            "commit_gap_p95_ms": self._percentile(self.commit_gap_samples_ms, 0.95),
            "step_count": self.decision_count,
        }


def generate_phase_workload(
    request_count: int,
    arrival_rate_rps: float,
    arrival_mode: str = "poisson",
    seed: int = 42,
    max_output_tokens: int = 64,
    draft_caps: tuple[int, ...] = (2, 4, 6, 8),
    prompt_lengths: tuple[int, ...] = (128, 256, 384, 512, 768, 1024, 2048, 4096),
    ar_fraction: float = 0.25,
    acceptance_base: float = 0.72,
    acceptance_decay: float = 0.72,
) -> list[PhaseRequest]:
    if request_count < 0:
        raise ValueError("request_count must be non-negative")
    if arrival_rate_rps <= 0:
        raise ValueError("arrival_rate_rps must be positive")
    if arrival_mode not in {"poisson", "uniform", "bursty"}:
        raise ValueError("arrival_mode must be poisson, uniform, or bursty")
    if not draft_caps or any(cap < 1 for cap in draft_caps):
        raise ValueError("draft_caps must contain positive values")
    if not prompt_lengths or any(length < 1 for length in prompt_lengths):
        raise ValueError("prompt_lengths must contain positive values")
    if not 0.0 <= ar_fraction <= 1.0:
        raise ValueError("ar_fraction must be in [0, 1]")
    if not 0.0 <= acceptance_base <= 1.0 or not 0.0 <= acceptance_decay <= 1.0:
        raise ValueError("acceptance parameters must be in [0, 1]")

    rng = random.Random(seed)
    mean_interarrival = 1000.0 / arrival_rate_rps
    arrival_time = 0.0
    max_cap = max(draft_caps)
    requests: list[PhaseRequest] = []
    for index in range(request_count):
        if index:
            if arrival_mode == "poisson":
                arrival_time += rng.expovariate(1.0 / mean_interarrival)
            elif arrival_mode == "uniform":
                arrival_time += mean_interarrival
            elif index % 8 == 0:
                arrival_time += 7.65 * mean_interarrival
            else:
                arrival_time += 0.05 * mean_interarrival

        cap = rng.choice(draft_caps)
        base = min(1.0, max(0.0, acceptance_base * rng.uniform(0.85, 1.15)))
        acceptance = tuple(
            min(1.0, max(0.0, base * (acceptance_decay ** pos) * rng.uniform(0.97, 1.03)))
            for pos in range(max_cap)
        )
        confidence_base = rng.uniform(0.76, 0.98)
        confidence = tuple(
            min(1.0, max(0.0, confidence_base - 0.006 * pos + rng.uniform(-0.015, 0.015)))
            for pos in range(max_cap)
        )
        requests.append(
            PhaseRequest(
                request_id=f"phase-{index:05d}",
                arrival_time_ms=arrival_time,
                prompt_tokens=rng.choice(prompt_lengths),
                max_output_tokens=max_output_tokens,
                draft_cap=cap,
                acceptance_probs=acceptance,
                confidence_by_position=confidence,
                speculative=(rng.random() >= ar_fraction),
            )
        )
    return requests
