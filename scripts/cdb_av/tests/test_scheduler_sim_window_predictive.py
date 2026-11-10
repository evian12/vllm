from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cdb_av.scheduler_sim.analyze_window_predictive import (
    aggregate,
    collect_records,
    paired_comparison,
)


def payload(rate: int, seed: int, e5_tps: float, e6_tps: float) -> dict:
    def result(policy: str, tps: float, ttft: float) -> dict:
        return {
            "policy": policy,
            "output_tokens_per_s": tps,
            "ttft_p50_ms": ttft / 2,
            "ttft_p95_ms": ttft,
            "ar_wait_p95_ms": 3.0,
            "ready_wait_p95_ms": 2.0,
            "commit_gap_p95_ms": 30.0,
            "target_forwards": 100,
            "draft_forwards": 200,
            "draft_window_mean": 2.0,
            "draft_window_p95": 4.0,
            "mean_target_batch_size": 2.5,
            "candidate_acceptance_rate": 0.4,
            "mixed_target_ratio": 0.2,
        }

    return {
        "workload": {"arrival_rate_rps": rate, "seed": seed},
        "results": {
            "load_adaptive_deadline": result("load_adaptive+deadline_batcher", e5_tps, 20.0),
            "window_predictive": result("window_predictive+deadline_batcher", e6_tps, 18.0),
        },
    }


class WindowPredictiveAnalysisTests(unittest.TestCase):
    def test_collect_aggregate_and_paired_comparison_use_matching_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "r30_s42.json").write_text(
                json.dumps(payload(30, 42, 100.0, 105.0)), encoding="utf-8"
            )
            (root / "r30_s43.json").write_text(
                json.dumps(payload(30, 43, 110.0, 100.0)), encoding="utf-8"
            )

            records = collect_records(root)
            self.assertEqual(len(records), 4)
            summary = aggregate(records)
            self.assertEqual(len(summary), 2)
            e6 = next(row for row in summary if row["policy_key"] == "window_predictive")
            self.assertEqual(e6["n"], 2)
            self.assertAlmostEqual(e6["output_tokens_per_s_mean"], 102.5)

            paired = paired_comparison(records)
            tps = next(
                row for row in paired
                if row["arrival_rate_rps"] == 30
                and row["metric"] == "output_tokens_per_s"
            )
            self.assertEqual(tps["paired_n"], 2)
            self.assertAlmostEqual(tps["e6_minus_e5_mean"], -2.5)
            self.assertEqual(tps["e6_better_seeds"], 1)

            ttft = next(
                row for row in paired
                if row["arrival_rate_rps"] == 30
                and row["metric"] == "ttft_p95_ms"
            )
            self.assertAlmostEqual(ttft["e6_minus_e5_mean"], -2.0)
            self.assertEqual(ttft["e6_better_seeds"], 2)

    def test_collect_rejects_duplicate_rate_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            item = json.dumps(payload(30, 42, 100.0, 105.0))
            (root / "first.json").write_text(item, encoding="utf-8")
            (root / "second.json").write_text(item, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                collect_records(root)

    def test_collect_requires_both_e5_and_e6_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            item = payload(30, 42, 100.0, 105.0)
            item["results"].pop("window_predictive")
            (root / "missing_e6.json").write_text(
                json.dumps(item), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "missing policy results"):
                collect_records(root)


if __name__ == "__main__":
    unittest.main()
