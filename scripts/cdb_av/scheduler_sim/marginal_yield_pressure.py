"""E7.1-A: queue-pressure-corrected marginal useful-output scheduling."""

from __future__ import annotations

from .phase_aware import (
    DeadlineAwareTargetBatcher,
    MarginalYieldPhasePolicy,
    PhaseAction,
    PhaseAwareSimulator,
    TargetBatcher,
    WorkKind,
    _visible_state_clone,
)


class PressureCorrectedMarginalYieldPhasePolicy(MarginalYieldPhasePolicy):
    """E7.1-A: penalize another Draft when it increases Ready backlog/age.

    The original E7 useful-output-rate comparison is retained. The projected
    Draft action's rate is discounted by normalized Ready queue cardinality
    and oldest-request age. Setting both weights to zero exactly recovers E7.
    Existing hard Ready/AR/Prefill guards remain authoritative.
    """

    name = "marginal_yield_pressure"

    def __init__(
        self,
        batcher: TargetBatcher | None = None,
        min_relative_gain: float = 0.01,
        ready_backlog_weight: float = 0.50,
        ready_age_weight: float = 0.50,
    ) -> None:
        super().__init__(batcher=batcher, min_relative_gain=min_relative_gain)
        if ready_backlog_weight < 0 or ready_age_weight < 0:
            raise ValueError("pressure weights must be non-negative")
        self.ready_backlog_weight = ready_backlog_weight
        self.ready_age_weight = ready_age_weight

    def choose_action(self, simulator: PhaseAwareSimulator) -> PhaseAction:
        actions = simulator.available_actions()
        if len(actions) == 1:
            return actions[0]

        current_plan = self.batcher.plan(simulator)
        if not current_plan:
            return PhaseAction.DRAFT

        cfg = simulator.config
        draft_duration = simulator.profile.draft_duration_ms(simulator.active_draft)
        current_target_duration = simulator.profile.target_duration_ms(current_plan)
        prefill_age = self._prefill_age(simulator)
        current_has_prefill = any(
            item.kind is WorkKind.PREFILL for item in current_plan
        )

        if (
            simulator.oldest_ready_wait_ms() + draft_duration
            >= cfg.ready_wait_budget_ms
            or simulator.oldest_ar_wait_ms() + draft_duration
            >= cfg.ar_itl_budget_ms
            or prefill_age + draft_duration
            + (current_target_duration if current_has_prefill else 0.0)
            >= cfg.prefill_wait_budget_ms
        ):
            return PhaseAction.TARGET

        if len(current_plan) >= cfg.max_num_seqs:
            return PhaseAction.TARGET

        current_yield = self._expected_output_yield(current_plan)
        current_rate = current_yield / max(current_target_duration, 1e-6)

        projected = _visible_state_clone(simulator)
        projected.execute_action(PhaseAction.DRAFT)
        projected_plan = self.batcher.plan(projected)
        if not projected_plan:
            return PhaseAction.TARGET

        projected_target_duration = projected.profile.target_duration_ms(projected_plan)
        projected_prefill_age = self._prefill_age(projected)
        projected_has_prefill = any(
            item.kind is WorkKind.PREFILL for item in projected_plan
        )
        projected_prefill_service = (
            projected_target_duration if projected_has_prefill else 0.0
        )
        if (
            projected.oldest_ready_wait_ms() >= cfg.ready_wait_budget_ms
            or projected.oldest_ar_wait_ms() + projected_target_duration
            >= cfg.ar_itl_budget_ms
            or projected_prefill_age + projected_prefill_service
            >= cfg.prefill_wait_budget_ms
        ):
            return PhaseAction.TARGET

        projected_yield = self._expected_output_yield(projected_plan)
        projected_rate = projected_yield / max(
            draft_duration + projected_target_duration, 1e-6
        )

        ready_backlog_ratio = min(
            len(projected.ready_queue) / max(cfg.max_num_seqs, 1), 2.0
        )
        ready_age_ratio = min(
            projected.oldest_ready_wait_ms() / max(cfg.ready_wait_budget_ms, 1e-6),
            2.0,
        )
        pressure_denominator = (
            1.0
            + self.ready_backlog_weight * ready_backlog_ratio
            + self.ready_age_weight * ready_age_ratio
        )
        pressure_adjusted_rate = projected_rate / pressure_denominator

        required_rate = current_rate * (1.0 + self.min_relative_gain)
        return (
            PhaseAction.DRAFT
            if pressure_adjusted_rate > required_rate + 1e-12
            else PhaseAction.TARGET
        )
