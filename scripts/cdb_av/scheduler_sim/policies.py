"""Scheduling policies for event-driven Draft/Verify decisions."""

from __future__ import annotations

from typing import Protocol, TYPE_CHECKING

from .model import Action

if TYPE_CHECKING:
    from .engine import Simulator


class SchedulingPolicy(Protocol):
    name: str

    def choose_action(self, simulator: Simulator) -> Action:
        ...


class FixedHPolicy:
    """Reference policy: run H Draft Forwards, then issue a Verify batch."""

    name = "fixed_h"

    def __init__(self, h: int = 8) -> None:
        if h < 1:
            raise ValueError("h must be positive")
        self.h = h

    def choose_action(self, simulator: Simulator) -> Action:
        actions = simulator.available_actions()
        if len(actions) == 1:
            return actions[0]
        if simulator.drafts_since_verify >= self.h:
            return Action.VERIFY
        return Action.DRAFT


class QueueThresholdPolicy:
    """Simple event-driven baseline using Ready Queue cardinality."""

    name = "queue_threshold"

    def __init__(self, threshold: int = 4) -> None:
        if threshold < 1:
            raise ValueError("threshold must be positive")
        self.threshold = threshold

    def choose_action(self, simulator: Simulator) -> Action:
        actions = simulator.available_actions()
        if len(actions) == 1:
            return actions[0]
        if len(simulator.ready_queue) >= self.threshold:
            return Action.VERIFY
        return Action.DRAFT


class QueuePressurePolicy:
    """Interpretable queue-pressure heuristic.

    The policy compares expected useful work per estimated millisecond, then
    adjusts Verify value for Ready backlog and age. A Verify-age guard and a
    consecutive-Verify guard prevent unbounded starvation in either stage.
    """

    name = "queue_pressure"

    def __init__(
        self,
        ready_wait_budget_ms: float = 8.0,
        max_consecutive_verify_forwards: int = 4,
        backlog_weight: float = 0.35,
        eligible_supply_weight: float = 0.20,
        age_weight: float = 0.25,
    ) -> None:
        if ready_wait_budget_ms <= 0:
            raise ValueError("ready_wait_budget_ms must be positive")
        if max_consecutive_verify_forwards < 1:
            raise ValueError("max_consecutive_verify_forwards must be positive")
        self.ready_wait_budget_ms = ready_wait_budget_ms
        self.max_consecutive_verify_forwards = max_consecutive_verify_forwards
        self.backlog_weight = backlog_weight
        self.eligible_supply_weight = eligible_supply_weight
        self.age_weight = age_weight

    @staticmethod
    def _oldest_ready_wait_ms(simulator: Simulator) -> float:
        waits = [
            simulator.now_ms - request.ready_since_ms
            for request in simulator.ready_queue
            if request.ready_since_ms is not None
        ]
        return max(waits, default=0.0)

    def choose_action(self, simulator: Simulator) -> Action:
        actions = simulator.available_actions()
        if len(actions) == 1:
            return actions[0]

        ready_count = len(simulator.ready_queue)
        draft_batch = list(simulator.active_draft)
        verify_batch = list(simulator.ready_queue)[
            : simulator.config.max_verify_batch_size
        ]
        oldest_wait = self._oldest_ready_wait_ms(simulator)
        age_ratio = oldest_wait / self.ready_wait_budget_ms

        if age_ratio >= 1.0:
            return Action.VERIFY

        if (
            simulator.consecutive_verify_forwards
            >= self.max_consecutive_verify_forwards
            and age_ratio < 2.0
        ):
            return Action.DRAFT

        draft_yield = sum(
            request.expected_next_candidate_acceptance()
            for request in draft_batch
        )
        verify_yield = sum(
            request.expected_verify_tokens() for request in verify_batch
        )
        draft_time = simulator.profile.draft_duration_ms(draft_batch)
        verify_time = simulator.profile.verify_duration_ms(verify_batch)

        draft_rate = draft_yield / draft_time
        verify_rate = verify_yield / verify_time

        verify_capacity = simulator.config.max_verify_batch_size
        draft_capacity = simulator.config.max_draft_batch_size
        backlog_ratio = min(ready_count / verify_capacity, 2.0)
        eligible_ratio = min(
            len(simulator.draft_eligible_queue) / draft_capacity, 1.0
        )

        verify_score = (
            verify_rate * (1.0 + self.backlog_weight * backlog_ratio)
            + self.age_weight * min(age_ratio, 2.0)
        )
        draft_score = draft_rate * (
            1.0 + self.eligible_supply_weight * eligible_ratio
        )

        return (
            Action.VERIFY
            if verify_score >= draft_score
            else Action.DRAFT
        )


class LookaheadPolicy:
    """Short-horizon rollout policy using QueuePressurePolicy for rollouts."""

    name = "lookahead"

    def __init__(
        self,
        horizon: int = 5,
        rollouts: int = 3,
        terminal_discount: float = 0.35,
        rollout_policy: SchedulingPolicy | None = None,
    ) -> None:
        if horizon < 1:
            raise ValueError("horizon must be positive")
        if rollouts < 1:
            raise ValueError("rollouts must be positive")
        if not 0.0 <= terminal_discount <= 1.0:
            raise ValueError("terminal_discount must be in [0, 1]")
        self.horizon = horizon
        self.rollouts = rollouts
        self.terminal_discount = terminal_discount
        self.rollout_policy = rollout_policy or QueuePressurePolicy()

    def choose_action(self, simulator: Simulator) -> Action:
        actions = simulator.available_actions()
        if len(actions) == 1:
            return actions[0]

        action_scores: dict[Action, float] = {}
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
                start_time = clone.now_ms
                start_committed = clone.total_committed_tokens
                clone.execute_action(action)
                forwards = 1

                while forwards < self.horizon:
                    clone.refresh()
                    if not clone.available_actions():
                        if not clone.advance_to_next_arrival():
                            break
                        continue
                    next_action = self.rollout_policy.choose_action(clone)
                    clone.execute_action(next_action)
                    forwards += 1

                elapsed_ms = max(clone.now_ms - start_time, 1e-6)
                committed = clone.total_committed_tokens - start_committed
                terminal_value = clone.terminal_potential()
                rollout_scores.append(
                    (
                        committed
                        + self.terminal_discount * terminal_value
                    )
                    / elapsed_ms
                )
            action_scores[action] = sum(rollout_scores) / len(rollout_scores)

        return max(action_scores, key=action_scores.get)
