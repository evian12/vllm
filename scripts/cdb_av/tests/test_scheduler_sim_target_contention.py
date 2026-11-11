from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cdb_av.scheduler_sim.analyze_target_contention import (
    aggregate,
    aggregate_compositions,
    collect_records,
    paired_vs_baseline,
)


def payload(
    chunk: int, rate: int, seed: int, tps: float, prefill_allocated: int = 256
) -> dict:
    return {
        "scope": {
            "scheduling_policy_parameters": {
                "deadline_prefill_chunk_tokens": chunk,
            }
        },
        "workload": {
            "arrival_rate_rps": rate,
            "seed": seed,
        },
        "results": {
            "load_adaptive_deadline": {
                "output_tokens_per_s": tps,
                "ttft_p95_ms": 20.0,
                "ar_wait_p95_ms": 3.0,
                "ready_wait_p95_ms": 2.0,
                "commit_gap_p95_ms": 30.0,
                "target_forwards": 10,
                "prefill_containing_target_forwards": 1,
                "prefill_containing_slot_utilization_mean": 0.5,
                "prefill_containing_slot_utilization_p95": 0.5,
                "prefill_containing_token_budget_utilization_mean": 0.25,
                "prefill_containing_token_budget_utilization_p95": 0.25,
                "prefill_containing_duration_mean_ms": 2.0,
                "prefill_containing_duration_p95_ms": 2.0,
                "ar_excluded_by_sequence_slots_events": 2,
                "ar_excluded_by_token_budget_events": 1,
                "target_exclusion_events": {
                    "ar": {
                        "sequence_slots_exhausted": 2,
                        "token_budget_exhausted": 1,
                    },
                    "prefill": {"policy_deferred": 3},
                },
                "target_compositions": [
                    {"prefill": 1, "verify": 0, "ar": 1},
                    {"prefill": 0, "verify": 0, "ar": 1},
                ],
                "target_forward_trace_enabled": True,
                "target_forward_trace": [
                    {
                        "composition": {
                            "prefill": {"requests": 1, "tokens": prefill_allocated},
                            "verify": {"requests": 0, "tokens": 0},
                            "ar": {"requests": 1, "tokens": 1},
                        },
                        "items": [
                            {
                                "request_id": "prefill",
                                "kind": "prefill",
                                "remaining_prompt_tokens": 1024,
                                "token_count": prefill_allocated,
                                # Deliberately emulate the old, incorrect trace field.
                                "requested_token_count": 2048,
                            },
                            {
                                "request_id": "ar",
                                "kind": "ar",
                                "remaining_prompt_tokens": 0,
                                "token_count": 1,
                            },
                        ],
                    }
                ],
            }
        },
    }


class TargetContentionAnalysisTests(unittest.TestCase):
    def test_collect_aggregate_and_paired_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "chunk256.json").write_text(
                json.dumps(payload(256, 30, 42, 110.0)), encoding="utf-8"
            )
            (root / "chunk1024.json").write_text(
                json.dumps(payload(1024, 30, 42, 100.0, prefill_allocated=512)), encoding="utf-8"
            )

            records = collect_records(root)
            self.assertEqual(len(records), 2)

            summary = aggregate(records)
            self.assertEqual(len(summary), 2)
            candidate = next(row for row in summary if row["chunk_size"] == 256)
            self.assertEqual(candidate["n"], 1)
            self.assertAlmostEqual(candidate["output_tokens_per_s_mean"], 110.0)
            # The stale requested_token_count is 2048 in both fixtures.
            # Reconstruct against the configured chunk cap and actual allocation.
            self.assertEqual(candidate["prefill_chunk_budget_truncations_mean"], 0.0)
            self.assertEqual(candidate["prefill_chunk_budget_shortfall_tokens_mean"], 0.0)
            baseline = next(row for row in summary if row["chunk_size"] == 1024)
            self.assertEqual(baseline["prefill_chunk_budget_truncations_mean"], 1.0)
            self.assertEqual(baseline["prefill_chunk_budget_shortfall_tokens_mean"], 512.0)
            self.assertAlmostEqual(
                candidate["ar_slot_exclusions_per_100_target_forwards_mean"],
                20.0,
            )
            self.assertAlmostEqual(
                candidate["ar_token_exclusions_per_100_target_forwards_mean"],
                10.0,
            )

            compositions = aggregate_compositions(records)
            candidate_compositions = {
                row["composition"]: row["mean_forwards_per_run"]
                for row in compositions
                if row["chunk_size"] == 256
            }
            self.assertEqual(candidate_compositions["prefill_ar"], 1.0)
            self.assertEqual(candidate_compositions["ar_only"], 1.0)

            paired = paired_vs_baseline(records, baseline_chunk=1024)
            self.assertEqual(len(paired), 1)
            self.assertEqual(paired[0]["paired_n"], 1)
            self.assertAlmostEqual(
                paired[0]["output_tokens_per_s_delta_mean"], 10.0
            )

    def test_collect_rejects_runs_without_forward_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            item = payload(256, 30, 42, 100.0)
            result = item["results"]["load_adaptive_deadline"]
            result["target_forward_trace_enabled"] = False
            result.pop("target_forward_trace")
            (root / "missing_trace.json").write_text(
                json.dumps(item), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "include-target-forward-trace"):
                collect_records(root)

    def test_collect_rejects_duplicate_chunk_rate_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            item = json.dumps(payload(256, 30, 42, 100.0))
            (root / "first.json").write_text(item, encoding="utf-8")
            (root / "second.json").write_text(item, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                collect_records(root)


if __name__ == "__main__":
    unittest.main()
