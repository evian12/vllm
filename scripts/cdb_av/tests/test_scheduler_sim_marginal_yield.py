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

    def test_minimum_relative_gain_must_be_non_negative(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            MarginalYieldPhasePolicy(min_relative_gain=-0.01)


if __name__ == "__main__":
    unittest.main()
