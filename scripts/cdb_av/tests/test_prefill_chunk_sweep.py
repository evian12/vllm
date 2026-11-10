from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.cdb_av.scheduler_sim.analyze_prefill_chunk_sweep import (
    aggregate,
    collect_records,
    paired_vs_baseline,
)


def payload(chunk: int, rate: int, seed: int, tps: float, ttft: float) -> dict:
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
                "ttft_p50_ms": ttft / 2,
                "ttft_p95_ms": ttft,
                "prefill_wait_p95_ms": 1.0,
                "prefill_inter_chunk_wait_p95_ms": 2.0,
                "prefill_target_forward_exposure_p95_ms": 3.0,
                "prefill_chunk_count_mean": 4.0,
                "prefill_decomposition_error_max_ms": 0.0,
                "ar_wait_p95_ms": 1.0,
                "commit_gap_p95_ms": 5.0,
                "target_forwards": 100,
                "mean_target_batch_size": 2.0,
            }
        },
    }


class PrefillChunkSweepAnalysisTest(unittest.TestCase):
    def test_collect_and_aggregate_preserves_chunk_rate_and_seed_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for chunk, tps in ((256, 100.0), (256, 110.0), (512, 90.0), (512, 100.0)):
                seed = 42 if tps in (100.0, 90.0) else 43
                result = payload(chunk, 30, seed, tps, 20.0 + seed - 42)
                folder = root / f"chunk_{chunk}"
                folder.mkdir(exist_ok=True)
                (folder / f"rate_30_seed_{seed}.json").write_text(
                    json.dumps(result), encoding="utf-8"
                )

            records = collect_records(root)
            rows = aggregate(records)
            self.assertEqual(len(records), 4)
            self.assertEqual(len(rows), 2)
            by_chunk = {row["chunk_size"]: row for row in rows}
            self.assertAlmostEqual(by_chunk[256]["output_tokens_per_s_mean"], 105.0)
            self.assertAlmostEqual(by_chunk[512]["output_tokens_per_s_mean"], 95.0)
            self.assertEqual(by_chunk[256]["n"], 2)

    def test_paired_comparison_uses_matching_seed(self):
        records = []
        for seed, base_tps, candidate_tps in (
            (42, 90.0, 100.0),
            (43, 110.0, 105.0),
        ):
            for chunk, tps in ((512, base_tps), (256, candidate_tps)):
                record = {
                    "chunk_size": chunk,
                    "rate": 30,
                    "seed": seed,
                    "output_tokens_per_s": tps,
                    "ttft_p95_ms": 20.0 if chunk == 512 else 18.0,
                    "prefill_wait_p95_ms": 3.0,
                    "prefill_inter_chunk_wait_p95_ms": 2.0,
                    "prefill_target_forward_exposure_p95_ms": 10.0,
                }
                records.append(record)

        rows = paired_vs_baseline(records, baseline_chunk=512)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["paired_n"], 2)
        self.assertAlmostEqual(rows[0]["output_tokens_per_s_delta_mean"], 2.5)
        self.assertAlmostEqual(rows[0]["ttft_p95_ms_delta_mean"], -2.0)


if __name__ == "__main__":
    unittest.main()
