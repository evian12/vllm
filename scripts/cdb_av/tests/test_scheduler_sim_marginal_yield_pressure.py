from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cdb_av.scheduler_sim.marginal_yield_pressure import (
    PressureCorrectedMarginalYieldPhasePolicy,
)
from scripts.cdb_av.scheduler_sim.phase_aware import (
    DeadlineAwareTargetBatcher,
    MarginalYieldPhasePolicy,
    PhaseAwareSimulator,
    PhaseConfig,
    PhaseExecutionProfile,
    generate_phase_workload,
)


class MarginalYieldPressurePolicyTests(unittest.TestCase):
    def _run(self, policy):
        requests = generate_phase_workload(
            request_count=24,
            arrival_rate_rps=30,
            arrival_mode="poisson",
            seed=42,
            max_output_tokens=24,
        )
        simulator = PhaseAwareSimulator(
            requests,
            PhaseConfig(seed=42),
            PhaseExecutionProfile(),
            policy,
        )
        return simulator.run()

    def test_zero_pressure_weights_recover_e7_exactly(self):
        batcher = DeadlineAwareTargetBatcher(prefill_chunk_tokens=1024)
        baseline = self._run(MarginalYieldPhasePolicy(batcher, min_relative_gain=0.01))
        candidate = self._run(
            PressureCorrectedMarginalYieldPhasePolicy(
                batcher,
                min_relative_gain=0.01,
                ready_backlog_weight=0.0,
                ready_age_weight=0.0,
            )
        )
        self.assertEqual(candidate, baseline)

    def test_pressure_weights_must_be_non_negative(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            PressureCorrectedMarginalYieldPhasePolicy(ready_backlog_weight=-0.1)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            PressureCorrectedMarginalYieldPhasePolicy(ready_age_weight=-0.1)

    def test_policy_completes_workload_and_emits_core_metrics(self):
        result = self._run(
            PressureCorrectedMarginalYieldPhasePolicy(
                DeadlineAwareTargetBatcher(prefill_chunk_tokens=1024),
                min_relative_gain=0.01,
                ready_backlog_weight=0.5,
                ready_age_weight=0.5,
            )
        )
        self.assertEqual(result["completed_requests"], 24)
        self.assertGreater(result["output_tokens_per_s"], 0.0)
        self.assertGreater(result["target_forwards"], 0)
        self.assertGreater(result["draft_forwards"], 0)


if __name__ == "__main__":
    unittest.main()
