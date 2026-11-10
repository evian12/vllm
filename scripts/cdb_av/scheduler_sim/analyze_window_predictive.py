"""Paired analysis of online E5, clairvoyant E5 oracle, and E6 scheduling."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

POLICIES = (
    "load_adaptive_deadline",
    "load_adaptive_deadline_oracle",
    "window_predictive",
)
COMPARISONS = (
    ("window_predictive_vs_online_e5", "window_predictive", "load_adaptive_deadline"),
    ("oracle_e5_vs_online_e5", "load_adaptive_deadline_oracle", "load_adaptive_deadline"),
)
METRICS = (
    "output_tokens_per_s",
    "ttft_p50_ms",
    "ttft_p95_ms",
    "ar_wait_p95_ms",
    "ready_wait_p95_ms",
    "commit_gap_p95_ms",
    "target_forwards",
    "draft_forwards",
    "draft_window_mean",
    "draft_window_p95",
    "mean_target_batch_size",
    "candidate_acceptance_rate",
    "mixed_target_ratio",
)
# Directional metrics only. Draft-window size and mixed-target ratio are
# diagnostics rather than universally monotonic objectives, so their
# e6_better_seeds values intentionally remain zero.
HIGHER_IS_BETTER = {
    "output_tokens_per_s",
    "mean_target_batch_size",
    "candidate_acceptance_rate",
}
LOWER_IS_BETTER = {
    "ttft_p50_ms",
    "ttft_p95_ms",
    "ar_wait_p95_ms",
    "ready_wait_p95_ms",
    "commit_gap_p95_ms",
    "target_forwards",
    "draft_forwards",
}


def collect_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[tuple[float, int]] = set()

    for path in sorted(root.rglob("*.json")):
        if path.name == "manifest.json":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        scope_params = (
            payload.get("scope", {})
            .get("scheduling_policy_parameters", {})
        )
        projection_mode = scope_params.get("load_adaptive_default_projection_mode")
        if projection_mode != "visible_only":
            raise ValueError(
                f"{path}: E5 projection is not marked visible_only; "
                "rerun the experiment with the non-clairvoyant online policy"
            )

        workload = payload.get("workload", {})
        if "arrival_rate_rps" not in workload or "seed" not in workload:
            raise ValueError(f"{path}: missing arrival_rate_rps or seed")
        rate = float(workload["arrival_rate_rps"])
        seed = int(workload["seed"])
        key = (rate, seed)
        if key in seen:
            raise ValueError(f"Duplicate arrival-rate/seed run: {key}")
        seen.add(key)

        results = payload.get("results", {})
        missing = [policy for policy in POLICIES if policy not in results]
        if missing:
            raise ValueError(
                f"{path}: missing policy results {missing}; run with --policy e5_e6"
            )

        for policy in POLICIES:
            result = results[policy]
            row: dict[str, Any] = {
                "arrival_rate_rps": rate,
                "seed": seed,
                "policy_key": policy,
                "policy": result.get("policy", policy),
            }
            for metric in METRICS:
                if metric not in result:
                    raise ValueError(f"{path}: {policy} result missing metric {metric}")
                row[metric] = float(result[metric])
            records.append(row)

    if not records:
        raise ValueError(f"No online-E5/oracle-E5/E6 result JSON files found in {root}")
    return records


def aggregate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[(float(row["arrival_rate_rps"]), str(row["policy_key"]))].append(row)

    output: list[dict[str, Any]] = []
    for (rate, policy), rows in sorted(groups.items()):
        result: dict[str, Any] = {
            "arrival_rate_rps": rate,
            "policy_key": policy,
            "policy": rows[0]["policy"],
            "n": len(rows),
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in rows]
            result[f"{metric}_mean"] = statistics.mean(values)
            result[f"{metric}_stdev"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
        output.append(result)
    return output


def paired_comparison(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_run: dict[tuple[float, int, str], dict[str, Any]] = {}
    for row in records:
        key = (
            float(row["arrival_rate_rps"]),
            int(row["seed"]),
            str(row["policy_key"]),
        )
        by_run[key] = row

    rates = sorted({float(row["arrival_rate_rps"]) for row in records})
    output: list[dict[str, Any]] = []
    for rate in rates:
        for comparison, left_policy, right_policy in COMPARISONS:
            seeds = sorted(
                seed
                for candidate_rate, seed, policy in by_run
                if candidate_rate == rate
                and policy == left_policy
                and (rate, seed, right_policy) in by_run
            )
            for metric in METRICS:
                deltas = [
                    float(by_run[(rate, seed, left_policy)][metric])
                    - float(by_run[(rate, seed, right_policy)][metric])
                    for seed in seeds
                ]
                better = 0
                if metric in HIGHER_IS_BETTER:
                    better = sum(delta > 0 for delta in deltas)
                elif metric in LOWER_IS_BETTER:
                    better = sum(delta < 0 for delta in deltas)
                output.append({
                    "arrival_rate_rps": rate,
                    "comparison": comparison,
                    "metric": metric,
                    "paired_n": len(deltas),
                    "delta_mean": statistics.mean(deltas) if deltas else 0.0,
                    "delta_stdev": (
                        statistics.stdev(deltas) if len(deltas) > 1 else 0.0
                    ),
                    "better_seeds": better,
                })
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_report(
    summary: list[dict[str, Any]],
    paired: list[dict[str, Any]],
) -> str:
    by_group = {
        (float(row["arrival_rate_rps"]), str(row["policy_key"])): row
        for row in summary
    }
    lines = [
        "# CDB-AV Online E5 vs Oracle E5 vs E6 Scheduling",
        "",
        "CPU-only discrete-event simulation. Coefficients are synthetic unless a measured profile is supplied.",
        "",
        "## Mean ± sample standard deviation",
        "",
        "| Rate | Policy | N | TPS | TTFT P50 | TTFT P95 | AR Wait P95 | Ready Wait P95 | Target Forwards | Draft Window Mean |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    rates = sorted({float(row["arrival_rate_rps"]) for row in summary})
    for rate in rates:
        for policy in POLICIES:
            row = by_group.get((rate, policy))
            if row is None:
                continue
            def fmt(metric: str) -> str:
                return (
                    f"{row[f'{metric}_mean']:.2f} ± "
                    f"{row[f'{metric}_stdev']:.2f}"
                )
            lines.append(
                f"| {rate:g} | {policy} | {row['n']} | {fmt('output_tokens_per_s')} "
                f"| {fmt('ttft_p50_ms')} | {fmt('ttft_p95_ms')} "
                f"| {fmt('ar_wait_p95_ms')} | {fmt('ready_wait_p95_ms')} "
                f"| {fmt('target_forwards')} | {fmt('draft_window_mean')} |"
            )

    lines.extend([
        "",
        "## Paired comparisons",
        "",
        "Positive TPS deltas are better; negative latency deltas are better. Each comparison uses matching arrival rate and seed.",
        "",
        "| Rate | Comparison | Metric | Paired N | Delta Mean | Better Seeds |",
        "|---:|---|---|---:|---:|---:|",
    ])
    selected = {
        "output_tokens_per_s",
        "ttft_p95_ms",
        "ar_wait_p95_ms",
        "ready_wait_p95_ms",
        "commit_gap_p95_ms",
        "target_forwards",
        "draft_window_mean",
    }
    for row in paired:
        if row["metric"] not in selected:
            continue
        lines.append(
            f"| {row['arrival_rate_rps']:g} | {row['comparison']} | {row['metric']} "
            f"| {row['paired_n']} | {row['delta_mean']:+.2f} "
            f"| {row['better_seeds']}/{row['paired_n']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate paired CDB-AV E5/E6 scheduler simulations."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    records = collect_records(args.input_dir)
    summary = aggregate(records)
    paired = paired_comparison(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "e5_e6_summary.csv", summary)
    _write_csv(args.output_dir / "e5_e6_paired.csv", paired)
    report = render_report(summary, paired)
    (args.output_dir / "e5_e6_report.md").write_text(report, encoding="utf-8")
    print(f"runs: {len(records) // len(POLICIES)}")
    print(f"saved: {args.output_dir / 'e5_e6_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
