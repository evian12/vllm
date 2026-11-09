"""Single-GPU, serial-forward discrete-event simulation engine."""

from __future__ import annotations

import copy
import random
from collections import deque
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Sequence

from .model import Action, Request, RequestStatus, SimulationConfig
from .profiles import ExecutionProfile
from .policies import SchedulingPolicy


@dataclass(frozen=True, slots=True)
class StepRecord:
    step: int
    action: str
    start_ms: float
    end_ms: float
    duration_ms: float
    batch_size: int
    ready_before: int
    ready_after: int
    active_draft_after: int
    committed_tokens: int


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(values)
    index = max(
        0,
        min(len(ordered) - 1, int(round(quantile * (len(ordered) - 1)))),
    )
    return float(ordered[index])


class Simulator:
    """Simulate serial Draft and Target Verify forwards on one GPU.

    Phase-1 scope: requests are assumed to have completed Target Prefill before
    becoming Draft Eligible. Prefill and AR are intentionally excluded so the
    outer Draft/Verify stage-selection policies can be evaluated in isolation.
    Arrivals during a Forward are admitted at its completion time, but retain
    their original arrival timestamp for waiting-time accounting.
    """

    def __init__(
        self,
        requests: Sequence[Request],
        config: SimulationConfig,
        profile: ExecutionProfile,
        policy: SchedulingPolicy,
    ) -> None:
        self.requests = sorted(
            copy.deepcopy(list(requests)),
            key=lambda request: (request.arrival_time_ms, request.request_id),
        )
        self.config = config
        self.profile = profile
        self.policy = policy
        self.rng = random.Random(config.seed)

        self.now_ms = 0.0
        self.idle_ms = 0.0
        self.gpu_busy_ms = 0.0
        self.pending_arrival_index = 0

        self.draft_eligible_queue: deque[Request] = deque()
        self.active_draft: list[Request] = []
        self.ready_queue: deque[Request] = deque()

        self.draft_forward_count = 0
        self.verify_forward_count = 0
        self.total_draft_tokens = 0
        self.accepted_candidate_tokens = 0
        self.verified_request_count = 0
        self.drafts_since_verify = 0
        self.consecutive_draft_forwards = 0
        self.consecutive_verify_forwards = 0
        self.decision_count = 0

        self.draft_batch_sizes: list[int] = []
        self.verify_batch_sizes: list[int] = []
        self.ready_wait_samples_ms: list[float] = []
        self.first_commit_latencies_ms: list[float] = []
        self.commit_gap_samples_ms: list[float] = []
        self.realized_h_samples: list[int] = []
        self.step_records: list[StepRecord] = []

        self.refresh()

    def clone(self) -> Simulator:
        return copy.deepcopy(self)

    @property
    def total_committed_tokens(self) -> int:
        return sum(request.committed_tokens for request in self.requests)

    @property
    def completed_requests(self) -> int:
        return sum(
            request.status is RequestStatus.FINISHED
            for request in self.requests
        )

    @property
    def is_finished(self) -> bool:
        return (
            self.pending_arrival_index >= len(self.requests)
            and all(
                request.status is RequestStatus.FINISHED
                for request in self.requests
            )
        )

    def refresh(self) -> None:
        self._process_arrivals()
        self._refill_active_draft()

    def _process_arrivals(self) -> None:
        while self.pending_arrival_index < len(self.requests):
            request = self.requests[self.pending_arrival_index]
            if request.arrival_time_ms > self.now_ms:
                break
            self.pending_arrival_index += 1
            if request.status is not RequestStatus.WAITING:
                raise RuntimeError(
                    f"request {request.request_id} arrived more than once"
                )
            request.status = RequestStatus.ELIGIBLE
            self.draft_eligible_queue.append(request)

    def _enqueue_ready(self, request: Request) -> None:
        request.status = RequestStatus.READY
        request.ready_since_ms = self.now_ms
        self.ready_queue.append(request)

    def _refill_active_draft(self) -> None:
        while (
            len(self.active_draft) < self.config.max_draft_batch_size
            and self.draft_eligible_queue
        ):
            request = self.draft_eligible_queue.popleft()
            if request.remaining_output_tokens <= 1:
                self._enqueue_ready(request)
                continue
            if request.draft_depth >= request.draft_cap:
                self._enqueue_ready(request)
                continue
            request.status = RequestStatus.ACTIVE_DRAFT
            self.active_draft.append(request)

    def available_actions(self) -> list[Action]:
        actions: list[Action] = []
        if self.active_draft:
            actions.append(Action.DRAFT)
        if self.ready_queue:
            actions.append(Action.VERIFY)
        return actions

    def advance_to_next_arrival(self) -> bool:
        if self.pending_arrival_index >= len(self.requests):
            return False
        next_arrival = self.requests[
            self.pending_arrival_index
        ].arrival_time_ms
        if next_arrival > self.now_ms:
            self.idle_ms += next_arrival - self.now_ms
            self.now_ms = next_arrival
        self.refresh()
        return True

    def execute_action(self, action: Action) -> StepRecord:
        action = Action(action)
        if action not in self.available_actions():
            raise ValueError(f"action {action.value} is not currently legal")

        start_ms = self.now_ms
        ready_before = len(self.ready_queue)
        committed_before = self.total_committed_tokens

        if action is Action.DRAFT:
            batch = list(self.active_draft)
            duration_ms = self.profile.draft_duration_ms(batch)
            self.now_ms += duration_ms
            self.gpu_busy_ms += duration_ms
            self.draft_forward_count += 1
            self.draft_batch_sizes.append(len(batch))
            self.total_draft_tokens += len(batch)
            self.drafts_since_verify += 1
            self.consecutive_draft_forwards += 1
            self.consecutive_verify_forwards = 0

            still_active: list[Request] = []
            for request in batch:
                request.draft_depth += 1
                request.total_draft_tokens += 1
                confidence = request.confidence_at(request.draft_depth - 1)
                stop_for_cap = request.draft_depth >= request.draft_cap
                stop_for_output_limit = (
                    request.remaining_output_tokens
                    <= request.draft_depth + 1
                )
                stop_for_confidence = (
                    confidence < self.config.confidence_threshold
                )
                if (
                    stop_for_cap
                    or stop_for_output_limit
                    or stop_for_confidence
                ):
                    self._enqueue_ready(request)
                else:
                    still_active.append(request)
            self.active_draft = still_active
            self.refresh()

        else:
            batch = [
                self.ready_queue.popleft()
                for _ in range(
                    min(
                        len(self.ready_queue),
                        self.config.max_verify_batch_size,
                    )
                )
            ]
            duration_ms = self.profile.verify_duration_ms(batch)
            self.now_ms += duration_ms
            self.gpu_busy_ms += duration_ms
            self.verify_forward_count += 1
            self.verify_batch_sizes.append(len(batch))
            self.verified_request_count += len(batch)

            for request in batch:
                if request.ready_since_ms is None:
                    raise RuntimeError(
                        f"Ready request {request.request_id} has no ready timestamp"
                    )
                self.ready_wait_samples_ms.append(
                    self.now_ms - request.ready_since_ms
                )
                accepted = self._sample_accepted_prefix(request)
                request.accepted_candidate_tokens += accepted
                self.accepted_candidate_tokens += accepted

                output_tokens = min(
                    request.remaining_output_tokens,
                    accepted + 1,
                )
                request.committed_tokens += output_tokens
                request.verify_rounds += 1

                if request.first_commit_time_ms is None:
                    request.first_commit_time_ms = self.now_ms
                    self.first_commit_latencies_ms.append(
                        self.now_ms - request.arrival_time_ms
                    )
                elif request.last_commit_time_ms is not None:
                    gap = self.now_ms - request.last_commit_time_ms
                    request.commit_gap_samples_ms.append(gap)
                    self.commit_gap_samples_ms.append(gap)

                request.last_commit_time_ms = self.now_ms
                request.ready_since_ms = None
                request.draft_depth = 0

                if request.remaining_output_tokens <= 0:
                    request.status = RequestStatus.FINISHED
                    request.completed_time_ms = self.now_ms
                else:
                    request.status = RequestStatus.ELIGIBLE
                    self.draft_eligible_queue.append(request)

            self.realized_h_samples.append(self.drafts_since_verify)
            self.drafts_since_verify = 0
            self.consecutive_verify_forwards += 1
            self.consecutive_draft_forwards = 0
            self.refresh()

        self.decision_count += 1
        record = StepRecord(
            step=self.decision_count,
            action=action.value,
            start_ms=start_ms,
            end_ms=self.now_ms,
            duration_ms=duration_ms,
            batch_size=len(batch),
            ready_before=ready_before,
            ready_after=len(self.ready_queue),
            active_draft_after=len(self.active_draft),
            committed_tokens=self.total_committed_tokens - committed_before,
        )
        self.step_records.append(record)
        return record

    def _sample_accepted_prefix(self, request: Request) -> int:
        accepted = 0
        for position in range(request.draft_depth):
            probability = request.conditional_acceptance_probability(position)
            if self.rng.random() <= probability:
                accepted += 1
            else:
                break
        return accepted

    def terminal_potential(self) -> float:
        """Expected committed-token potential in currently held candidates."""
        potential = sum(
            request.expected_verify_tokens()
            for request in self.ready_queue
        )
        potential += sum(
            request.expected_verify_tokens()
            for request in self.active_draft
            if request.draft_depth > 0
        )
        return potential

    def run(self) -> dict[str, object]:
        while not self.is_finished:
            if self.decision_count >= self.config.max_steps:
                raise RuntimeError(
                    f"simulation exceeded max_steps={self.config.max_steps}"
                )
            self.refresh()
            if not self.available_actions():
                if self.advance_to_next_arrival():
                    continue
                unfinished = [
                    request.request_id
                    for request in self.requests
                    if request.status is not RequestStatus.FINISHED
                ]
                raise RuntimeError(
                    f"deadlock with unfinished requests: {unfinished[:8]}"
                )
            action = self.policy.choose_action(self)
            self.execute_action(action)
        return self.summary()

    def summary(self) -> dict[str, object]:
        elapsed_ms = max(self.now_ms, 0.0)
        output_tokens = self.total_committed_tokens
        throughput = (
            output_tokens / (elapsed_ms / 1000.0)
            if elapsed_ms > 0
            else 0.0
        )
        draft_slots = (
            self.draft_forward_count * self.config.max_draft_batch_size
        )
        verify_slots = (
            self.verify_forward_count * self.config.max_verify_batch_size
        )
        all_ready_waits = self.ready_wait_samples_ms
        all_ttft = self.first_commit_latencies_ms
        all_commit_gaps = self.commit_gap_samples_ms

        return {
            "policy": getattr(self.policy, "name", type(self.policy).__name__),
            "requests": len(self.requests),
            "completed_requests": self.completed_requests,
            "output_tokens": output_tokens,
            "accepted_candidate_tokens": self.accepted_candidate_tokens,
            "total_draft_tokens": self.total_draft_tokens,
            "candidate_acceptance_rate": (
                self.accepted_candidate_tokens / self.total_draft_tokens
                if self.total_draft_tokens
                else 0.0
            ),
            "output_tokens_per_s": throughput,
            "elapsed_ms": elapsed_ms,
            "gpu_busy_ms": self.gpu_busy_ms,
            "gpu_busy_ratio": (
                self.gpu_busy_ms / elapsed_ms if elapsed_ms > 0 else 0.0
            ),
            "idle_ms": self.idle_ms,
            "draft_forwards": self.draft_forward_count,
            "verify_forwards": self.verify_forward_count,
            "mean_draft_batch_size": (
                mean(self.draft_batch_sizes) if self.draft_batch_sizes else 0.0
            ),
            "mean_verify_batch_size": (
                mean(self.verify_batch_sizes) if self.verify_batch_sizes else 0.0
            ),
            "draft_slot_utilization": (
                sum(self.draft_batch_sizes) / draft_slots
                if draft_slots
                else 0.0
            ),
            "verify_slot_utilization": (
                sum(self.verify_batch_sizes) / verify_slots
                if verify_slots
                else 0.0
            ),
            "realized_h_samples": list(self.realized_h_samples),
            "mean_realized_h": (
                mean(self.realized_h_samples)
                if self.realized_h_samples
                else 0.0
            ),
            "ready_wait_p50_ms": percentile(all_ready_waits, 0.50),
            "ready_wait_p95_ms": percentile(all_ready_waits, 0.95),
            "ready_wait_max_ms": max(all_ready_waits, default=0.0),
            "first_commit_latency_p50_ms": percentile(all_ttft, 0.50),
            "first_commit_latency_p95_ms": percentile(all_ttft, 0.95),
            "inter_verify_commit_gap_p50_ms": percentile(
                all_commit_gaps, 0.50
            ),
            "inter_verify_commit_gap_p95_ms": percentile(
                all_commit_gaps, 0.95
            ),
            "step_count": self.decision_count,
        }

    def trace(self) -> list[dict[str, object]]:
        return [asdict(record) for record in self.step_records]
