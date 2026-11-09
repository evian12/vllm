from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cdb_av.scheduler_sim.engine import Simulator
from scripts.cdb_av.scheduler_sim.model import (
    Action,
    Request,
    RequestStatus,
    SimulationConfig,
)
from scripts.cdb_av.scheduler_sim.policies import (
    FixedHPolicy,
    LookaheadPolicy,
    QueuePressurePolicy,
    QueueThresholdPolicy,
)
from scripts.cdb_av.scheduler_sim.profiles import ExecutionProfile
from scripts.cdb_av.scheduler_sim.workload import generate_workload


def make_requests(
    count: int,
    *,
    draft_cap: int = 2,
    max_output_tokens: int = 9,
    arrival_gap_ms: float = 0.0,
    acceptance: tuple[float, ...] = (1.0,) * 8,
    confidence: tuple[float, ...] = (1.0,) * 8,
) -> list[Request]:
    return [
        Request(
            request_id=f"r-{index}",
            arrival_time_ms=index * arrival_gap_ms,
            prompt_tokens=128,
            max_output_tokens=max_output_tokens,
            draft_cap=draft_cap,
            acceptance_probs=acceptance,
            confidence_by_position=confidence,
        )
        for index in range(count)
    ]


class SchedulerSimulatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = ExecutionProfile(
            draft_base_ms=1.0,
            draft_per_request_ms=0.1,
            draft_per_context_token_ms=0.0,
            verify_base_ms=1.0,
            verify_per_request_ms=0.1,
            verify_per_candidate_ms=0.05,
            verify_per_context_token_ms=0.0,
        )

    def test_expected_accepted_length_uses_conditional_probabilities(self):
        request = Request(
            request_id="prob",
            arrival_time_ms=0.0,
            max_output_tokens=8,
            draft_cap=3,
            acceptance_probs=(0.5, 0.5, 0.5),
        )
        self.assertAlmostEqual(
            request.expected_accepted_length(depth=3),
            0.875,
        )
        self.assertAlmostEqual(request.expected_verify_tokens(depth=3), 1.875)

    def test_fixed_h_baseline_records_realized_h(self):
        requests = make_requests(
            4,
            draft_cap=2,
            max_output_tokens=3,
            acceptance=(1.0,) * 8,
        )
        simulator = Simulator(
            requests,
            SimulationConfig(
                max_draft_batch_size=4,
                max_verify_batch_size=4,
                fixed_h=2,
                confidence_threshold=0.0,
            ),
            self.profile,
            FixedHPolicy(2),
        )
        result = simulator.run()
        self.assertEqual(result["completed_requests"], 4)
        self.assertEqual(result["realized_h_samples"], [2])
        self.assertEqual(result["output_tokens"], 12)

    def test_verify_batch_never_exceeds_target_capacity(self):
        requests = make_requests(
            8,
            draft_cap=1,
            max_output_tokens=2,
            acceptance=(1.0,) * 8,
        )
        simulator = Simulator(
            requests,
            SimulationConfig(
                max_draft_batch_size=4,
                max_verify_batch_size=4,
                fixed_h=1,
                confidence_threshold=0.0,
            ),
            self.profile,
            FixedHPolicy(1),
        )
        result = simulator.run()
        self.assertEqual(result["completed_requests"], 8)
        self.assertTrue(
            all(size <= 4 for size in simulator.verify_batch_sizes)
        )
        self.assertEqual(simulator.verify_forward_count, 2)

    def test_confidence_stop_moves_request_to_ready_after_candidate(self):
        request = make_requests(
            1,
            draft_cap=8,
            max_output_tokens=8,
            confidence=(0.2,) + (0.99,) * 7,
        )[0]
        simulator = Simulator(
            [request],
            SimulationConfig(
                max_draft_batch_size=1,
                max_verify_batch_size=1,
                confidence_threshold=0.8,
            ),
            self.profile,
            FixedHPolicy(8),
        )
        simulator.execute_action(Action.DRAFT)
        self.assertEqual(len(simulator.ready_queue), 1)
        self.assertEqual(simulator.ready_queue[0].draft_depth, 1)

    def test_output_limit_one_goes_directly_to_verify(self):
        requests = make_requests(
            2,
            draft_cap=8,
            max_output_tokens=1,
        )
        simulator = Simulator(
            requests,
            SimulationConfig(
                max_draft_batch_size=2,
                max_verify_batch_size=2,
            ),
            self.profile,
            QueueThresholdPolicy(1),
        )
        result = simulator.run()
        self.assertEqual(result["output_tokens"], 2)
        self.assertEqual(result["draft_forwards"], 0)

    def test_threshold_policy_switches_at_ready_threshold(self):
        requests = make_requests(
            4,
            draft_cap=1,
            max_output_tokens=2,
        )
        simulator = Simulator(
            requests,
            SimulationConfig(
                max_draft_batch_size=4,
                max_verify_batch_size=4,
                queue_threshold=4,
                confidence_threshold=0.0,
            ),
            self.profile,
            QueueThresholdPolicy(4),
        )
        simulator.execute_action(Action.DRAFT)
        self.assertEqual(len(simulator.ready_queue), 4)
        self.assertEqual(
            simulator.policy.choose_action(simulator),
            Action.VERIFY,
        )

    def test_pressure_policy_obeys_ready_wait_budget(self):
        requests = make_requests(
            2,
            draft_cap=8,
            max_output_tokens=10,
        )
        simulator = Simulator(
            requests,
            SimulationConfig(
                max_draft_batch_size=2,
                max_verify_batch_size=2,
                ready_wait_budget_ms=5.0,
                confidence_threshold=0.0,
            ),
            self.profile,
            QueuePressurePolicy(ready_wait_budget_ms=5.0),
        )
        ready = simulator.active_draft.pop()
        ready.status = RequestStatus.READY
        ready.draft_depth = 1
        ready.ready_since_ms = 0.0
        simulator.ready_queue.append(ready)
        simulator.now_ms = 10.0
        self.assertEqual(
            simulator.policy.choose_action(simulator),
            Action.VERIFY,
        )

    def test_lookahead_policy_does_not_mutate_live_state(self):
        requests = make_requests(
            8,
            draft_cap=2,
            max_output_tokens=5,
        )
        policy = LookaheadPolicy(
            horizon=3,
            rollouts=2,
            rollout_policy=QueuePressurePolicy(),
        )
        simulator = Simulator(
            requests,
            SimulationConfig(
                max_draft_batch_size=4,
                max_verify_batch_size=4,
                confidence_threshold=0.0,
            ),
            self.profile,
            policy,
        )
        before = (
            simulator.now_ms,
            len(simulator.active_draft),
            len(simulator.ready_queue),
            simulator.total_committed_tokens,
            simulator.decision_count,
        )
        action = policy.choose_action(simulator)
        after = (
            simulator.now_ms,
            len(simulator.active_draft),
            len(simulator.ready_queue),
            simulator.total_committed_tokens,
            simulator.decision_count,
        )
        self.assertIn(action, simulator.available_actions())
        self.assertEqual(before, after)

    def test_lookahead_run_completes_all_requests(self):
        simulator = Simulator(
            make_requests(12, draft_cap=2, max_output_tokens=7),
            SimulationConfig(
                max_draft_batch_size=4,
                max_verify_batch_size=4,
                confidence_threshold=0.0,
                max_steps=10_000,
            ),
            self.profile,
            LookaheadPolicy(horizon=3, rollouts=2),
        )
        result = simulator.run()
        self.assertEqual(result["completed_requests"], 12)
        self.assertGreater(result["output_tokens_per_s"], 0.0)

    def test_single_gpu_forwards_are_serial(self):
        simulator = Simulator(
            make_requests(8, draft_cap=2, max_output_tokens=5),
            SimulationConfig(
                max_draft_batch_size=4,
                max_verify_batch_size=4,
                confidence_threshold=0.0,
            ),
            self.profile,
            QueuePressurePolicy(),
        )
        simulator.run()
        previous_end = 0.0
        for record in simulator.step_records:
            self.assertGreaterEqual(record.start_ms, previous_end)
            self.assertGreater(record.end_ms, record.start_ms)
            previous_end = record.end_ms

    def test_deterministic_workload_generation(self):
        first = generate_workload(
            16, 20.0, arrival_mode="bursty", seed=17
        )
        second = generate_workload(
            16, 20.0, arrival_mode="bursty", seed=17
        )
        self.assertEqual(
            [
                (r.request_id, r.arrival_time_ms, r.draft_cap, r.acceptance_probs)
                for r in first
            ],
            [
                (r.request_id, r.arrival_time_ms, r.draft_cap, r.acceptance_probs)
                for r in second
            ],
        )
        self.assertTrue(
            all(
                first[index].arrival_time_ms
                <= first[index + 1].arrival_time_ms
                for index in range(len(first) - 1)
            )
        )

    def test_all_policies_complete_same_workload(self):
        requests = generate_workload(
            24, 30.0, arrival_mode="poisson", seed=23
        )
        policies = (
            FixedHPolicy(4),
            QueueThresholdPolicy(4),
            QueuePressurePolicy(),
            LookaheadPolicy(horizon=3, rollouts=2),
        )
        for policy in policies:
            with self.subTest(policy=policy.name):
                simulator = Simulator(
                    requests,
                    SimulationConfig(
                        max_draft_batch_size=4,
                        max_verify_batch_size=4,
                        fixed_h=4,
                        confidence_threshold=0.8,
                        max_steps=20_000,
                        seed=23,
                    ),
                    self.profile,
                    policy,
                )
                result = simulator.run()
                self.assertEqual(result["completed_requests"], 24)
                self.assertEqual(result["output_tokens"], 24 * 64)
                self.assertGreaterEqual(
                    result["verify_slot_utilization"], 0.0
                )


    def test_ready_wait_excludes_verify_service_time(self):
        simulator = Simulator(
            make_requests(
                1,
                draft_cap=1,
                max_output_tokens=2,
            ),
            SimulationConfig(
                max_draft_batch_size=1,
                max_verify_batch_size=1,
                confidence_threshold=0.0,
            ),
            self.profile,
            FixedHPolicy(1),
        )
        simulator.execute_action(Action.DRAFT)
        simulator.execute_action(Action.VERIFY)
        self.assertEqual(simulator.ready_wait_samples_ms, [0.0])

    def test_arrivals_during_forward_enter_eligible_at_forward_completion(self):
        requests = make_requests(
            2,
            draft_cap=1,
            max_output_tokens=2,
            arrival_gap_ms=0.5,
        )
        simulator = Simulator(
            requests,
            SimulationConfig(
                max_draft_batch_size=1,
                max_verify_batch_size=1,
                confidence_threshold=0.0,
            ),
            self.profile,
            FixedHPolicy(1),
        )
        simulator.execute_action(Action.DRAFT)
        self.assertGreaterEqual(simulator.now_ms, 0.5)
        self.assertEqual(len(simulator.draft_eligible_queue), 0)
        self.assertEqual(len(simulator.active_draft), 1)
        self.assertEqual(
            simulator.active_draft[0].status,
            RequestStatus.ACTIVE_DRAFT,
        )
        self.assertEqual(
            simulator.active_draft[0].arrival_time_ms,
            0.5,
        )


if __name__ == "__main__":
    unittest.main()
