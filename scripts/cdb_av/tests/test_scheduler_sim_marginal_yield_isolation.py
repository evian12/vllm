from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cdb_av.scheduler_sim.phase_aware import (
    DeadlineAwareTargetBatcher,
    MarginalYieldPhasePolicy,
    PhaseAction,
    PhaseAwareSimulator,
    PhaseConfig,
    PhaseExecutionProfile,
    PhaseRequest,
    RequestStatus,
)


class MarginalYieldIsolationTests(unittest.TestCase):
    def _simulator(self) -> PhaseAwareSimulator:
        requests = [
            PhaseRequest(
                request_id=f"r-{index}",
                arrival_time_ms=0.0,
                prompt_tokens=128,
                max_output_tokens=12,
                draft_cap=4,
                acceptance_probs=(0.5,) * 8,
            )
            for index in range(4)
        ]
        profile = PhaseExecutionProfile(
            draft_base_ms=1.0,
            draft_per_request_ms=0.1,
            draft_per_context_token_ms=0.0,
            target_base_ms=1.0,
            target_per_request_ms=0.1,
            prefill_per_token_ms=0.001,
            prefill_context_per_token_ms=0.0,
            verify_per_candidate_ms=0.05,
            verify_context_per_token_ms=0.0,
            ar_per_request_ms=0.1,
            ar_context_per_token_ms=0.0,
        )
        policy = MarginalYieldPhasePolicy(
            DeadlineAwareTargetBatcher(prefill_chunk_tokens=512),
            min_relative_gain=0.01,
        )
        sim = PhaseAwareSimulator(
            requests,
            PhaseConfig(
                max_num_seqs=4,
                max_num_batched_tokens=2048,
                max_draft_batch_size=4,
                confidence_threshold=0.0,
                seed=123,
            ),
            profile,
            policy,
        )
        sim.execute_action(PhaseAction.TARGET)
        sim.execute_action(PhaseAction.DRAFT)
        return sim

    @staticmethod
    def _live_state(sim: PhaseAwareSimulator) -> tuple[object, ...]:
        request_state = tuple(
            (
                req.request_id,
                req.status,
                req.prefill_progress,
                req.committed_tokens,
                req.draft_depth,
                req.total_draft_tokens,
                req.ready_since_ms,
            )
            for req in sim.requests
        )
        queue_state = tuple(
            tuple(req.request_id for req in queue)
            for queue in (
                sim.prefill_queue,
                sim.draft_eligible_queue,
                sim.ready_queue,
                sim.ar_queue,
            )
        )
        active_state = tuple(req.request_id for req in sim.active_draft)
        counters = (
            sim.now_ms,
            sim.gpu_busy_ms,
            sim.idle_ms,
            sim.pending_arrival_index,
            sim.draft_forward_count,
            sim.target_forward_count,
            sim.drafts_since_verify,
            sim.decision_count,
            sim.total_draft_tokens,
            sim.accepted_candidate_tokens,
            sim.total_prefill_tokens,
            sim.verify_request_count,
            sim.ar_token_count,
            len(sim.step_records),
        )
        return request_state, queue_state, active_state, counters

    def test_choose_action_does_not_mutate_live_state_or_rng(self) -> None:
        sim = self._simulator()
        before_state = self._live_state(sim)
        before_rng = sim.rng.getstate()

        action = sim.policy.choose_action(sim)

        self.assertIn(action, sim.available_actions())
        self.assertEqual(self._live_state(sim), before_state)
        self.assertEqual(sim.rng.getstate(), before_rng)

    def test_repeated_decisions_are_deterministic_and_rng_isolated(self) -> None:
        sim = self._simulator()
        before_rng = sim.rng.getstate()

        first = sim.policy.choose_action(sim)
        second = sim.policy.choose_action(sim)

        self.assertEqual(first, second)
        self.assertEqual(sim.rng.getstate(), before_rng)

    def test_clone_has_independent_rng_and_request_objects(self) -> None:
        sim = self._simulator()
        clone = sim.clone()

        self.assertIsNot(clone.rng, sim.rng)
        self.assertIsNot(clone.requests[0], sim.requests[0])
        self.assertEqual(clone.rng.getstate(), sim.rng.getstate())

        clone.rng.random()
        self.assertEqual(clone.rng.getstate() != sim.rng.getstate(), True)
        self.assertEqual(self._live_state(sim), self._live_state(sim))


if __name__ == "__main__":
    unittest.main()
