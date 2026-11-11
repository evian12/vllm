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
    PhaseAwareSimulator,
    PhaseConfig,
    PhaseExecutionProfile,
    generate_phase_workload,
)


class MarginalYieldPolicyTests(unittest.TestCase):
    def test_policy_runs_to_completion_and_emits_core_metrics(self):
        requests = generate_phase_workload(
            request_count=12,
            arrival_rate_rps=30,
            arrival_mode="poisson",
            seed=42,
            max_output_tokens=24,
        )
        policy = MarginalYieldPhasePolicy(
            DeadlineAwareTargetBatcher(prefill_chunk_tokens=1024),
            min_relative_gain=0.01,
        )
        simulator = PhaseAwareSimulator(
            requests, PhaseConfig(seed=42), PhaseExecutionProfile(), policy
        )
        result = simulator.run()
        self.assertEqual(result["completed_requests"], 12)
        self.assertGreater(result["output_tokens_per_s"], 0.0)
        self.assertGreater(result["target_forwards"], 0)
        self.assertGreater(result["draft_forwards"], 0)
        self.assertGreaterEqual(result["draft_window_mean"], 0.0)

    def test_arrival_oracle_ablation_completes_same_workload(self):
        requests = generate_phase_workload(
            request_count=24,
            arrival_rate_rps=60,
            arrival_mode="poisson",
            seed=19,
            max_output_tokens=24,
        )
        policy = MarginalYieldPhasePolicy(
            DeadlineAwareTargetBatcher(prefill_chunk_tokens=1024),
            min_relative_gain=0.01,
            allow_future_arrivals=True,
        )
        simulator = PhaseAwareSimulator(
            requests, PhaseConfig(seed=19), PhaseExecutionProfile(), policy
        )
        result = simulator.run()
        self.assertEqual(result["completed_requests"], 24)
        self.assertGreater(result["output_tokens_per_s"], 0.0)

    def test_request_prefill_trace_reconciles_ttft_and_chunks(self):
        requests = generate_phase_workload(
            request_count=16,
            arrival_rate_rps=30,
            arrival_mode="bursty",
            seed=42,
            max_output_tokens=24,
        )
        policy = MarginalYieldPhasePolicy(
            DeadlineAwareTargetBatcher(prefill_chunk_tokens=512),
            min_relative_gain=0.01,
        )
        simulator = PhaseAwareSimulator(
            requests,
            PhaseConfig(seed=42, include_target_forward_trace=True),
            PhaseExecutionProfile(),
            policy,
        )
        result = simulator.run()
        self.assertEqual(result["completed_requests"], 16)

        request_trace = result["prefill_request_trace"]
        self.assertEqual(len(request_trace), 16)
        self.assertEqual(
            len({item["request_id"] for item in request_trace}), 16
        )
        for item in request_trace:
            self.assertGreaterEqual(item["chunk_count"], 1)
            self.assertEqual(item["chunk_count"], len(item["chunks"]))
            self.assertAlmostEqual(
                item["ttft_ms"],
                item["first_chunk_wait_ms"]
                + item["inter_chunk_wait_ms"]
                + item["target_forward_exposure_ms"],
                places=7,
            )
            self.assertLessEqual(item["decomposition_error_ms"], 1e-7)
            self.assertTrue(
                all(chunk["duration_ms"] > 0 for chunk in item["chunks"])
            )
            for deferred in item["deferred_target_forwards"]:
                self.assertIn(
                    deferred["reason"],
                    {
                        "sequence_slots_exhausted",
                        "sequence_and_token_budget_exhausted",
                        "token_budget_exhausted",
                        "not_selected_by_batcher",
                        "policy_deferred",
                        "prefill_policy_budget_exhausted",
                    },
                )
                self.assertGreaterEqual(
                    deferred["end_ms"], deferred["start_ms"]
                )
                self.assertGreaterEqual(deferred["wait_age_ms_at_start"], 0.0)

    def test_minimum_relative_gain_must_be_non_negative(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            MarginalYieldPhasePolicy(min_relative_gain=-0.01)


if __name__ == "__main__":
    unittest.main()
