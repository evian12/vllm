#!/usr/bin/env python3
"""Aggregate phase-aware Target Forward contention traces across CPU runs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

METRICS = (
    "output_tokens_per_s",
    "ttft_p95_ms",
    "ar_wait_p95_ms",
    "ready_wait_p95_ms",
    "commit_gap_p95_ms",
    "target_forwards",
    "prefill_containing_target_forwards",
    "prefill_containing_slot_utilization_mean",
    "prefill_containing_slot_utilization_p95",
    "prefill_containing_token_budget_utilization_mean",
    "prefill_containing_token_budget_utilization_p95",
    "prefill_containing_duration_mean_ms",
    "prefill_containing_duration_p95_ms",
    "prefill_chunk_budget_truncations",
    "prefill_chunk_budget_shortfall_tokens",
    "ar_excluded_by_sequence_slots_events",
    "ar_excluded_by_token_budget_events",
    "ar_slot_exclusions_per_100_target_forwards",
    "ar_token_exclusions_per_100_target_forwards",
    "verify_slot_exclusions_per_100_target_forwards",
    "verify_token_exclusions_per_100_target_forwards",
    "prefill_slot_exclusions_per_100_target_forwards",
    "prefill_token_exclusions_per_100_target_forwards",
    "prefill_policy_deferred_per_100_target_forwards",
)

COMPOSITIONS = (
    "prefill_only",
    "verify_only",
    "ar_only",
    "prefill_verify",
    "prefill_ar",
    "verify_ar",
    "all_three",
)


def _event_count(result: dict[str, Any], kind: str, reasons: tuple[str, ...]) -> int:
    counts = result.get("target_exclusion_events", {}).get(kind, {})
    return sum(int(counts.get(reason, 0)) for reason in reasons)


def _composition_name(composition: dict[str, int]) -> str:
    active = [
        kind
        for kind in ("prefill", "verify", "ar")
        if int(composition.get(kind, 0)) > 0
    ]
    if len(active) == 1:
        return f"{active[0]}_only"
    if len(active) == 2:
        return "_".join(active)
    if len(active) == 3:
        return "all_three"
    return "empty"


def collect_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[tuple[int, float, int]] = set()

    for path in sorted(root.rglob("*.json")):
        if path.name == "manifest.json":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        workload = payload.get("workload", {})
        parameters = payload.get("scope", {}).get(
            "scheduling_policy_parameters", {}
        )
        chunk = parameters.get("deadline_prefill_chunk_tokens")
        if chunk is None:
            chunk = payload.get("config", {}).get("deadline_prefill_chunk_tokens")
        if chunk is None:
            raise ValueError(f"Cannot determine Prefill chunk size for {path}")

        results = payload.get("results", {})
        if len(results) != 1:
            raise ValueError(
                f"Expected exactly one policy in {path}; got {list(results)}"
            )
        result = next(iter(results.values()))
        if not result.get("target_forward_trace_enabled"):
            raise ValueError(
                f"{path} has no Target Forward trace; rerun with "
                "--include-target-forward-trace"
            )
        trace = result.get("target_forward_trace")
        if not isinstance(trace, list):
            raise ValueError(f"{path} is missing target_forward_trace")

        rate = float(workload["arrival_rate_rps"])
        seed = int(workload["seed"])
        key = (int(chunk), rate, seed)
        if key in seen:
            raise ValueError(f"Duplicate (chunk, rate, seed) input: {key}")
        seen.add(key)

        prefill_trace = [
            item for item in trace
            if int(item.get("composition", {}).get("prefill", {}).get("requests", 0)) > 0
        ]
        if len(prefill_trace) != int(
            result.get("prefill_containing_target_forwards", len(prefill_trace))
        ):
            raise ValueError(
                f"{path}: trace and Prefill-containing Forward count disagree"
            )

        composition_counts = Counter(
            _composition_name(item) for item in result.get("target_compositions", [])
        )
        target_forwards = max(1, int(result.get("target_forwards", 0)))
        row: dict[str, Any] = {
            "chunk_size": int(chunk),
            "rate": rate,
            "seed": seed,
            "composition_counts": dict(composition_counts),
        }
        for metric in (
            "output_tokens_per_s",
            "ttft_p95_ms",
            "ar_wait_p95_ms",
            "ready_wait_p95_ms",
            "commit_gap_p95_ms",
            "target_forwards",
            "prefill_containing_target_forwards",
            "prefill_containing_slot_utilization_mean",
            "prefill_containing_slot_utilization_p95",
            "prefill_containing_token_budget_utilization_mean",
            "prefill_containing_token_budget_utilization_p95",
            "prefill_containing_duration_mean_ms",
            "prefill_containing_duration_p95_ms",
            "prefill_chunk_budget_truncations",
            "prefill_chunk_budget_shortfall_tokens",
            "ar_excluded_by_sequence_slots_events",
            "ar_excluded_by_token_budget_events",
        ):
            row[metric] = float(result.get(metric, 0.0))

        event_groups = {
            "ar_slot_exclusions_per_100_target_forwards": (
                "ar", ("sequence_slots_exhausted", "sequence_and_token_budget_exhausted")
            ),
            "ar_token_exclusions_per_100_target_forwards": (
                "ar", ("token_budget_exhausted", "sequence_and_token_budget_exhausted")
            ),
            "verify_slot_exclusions_per_100_target_forwards": (
                "verify", ("sequence_slots_exhausted", "sequence_and_token_budget_exhausted")
            ),
            "verify_token_exclusions_per_100_target_forwards": (
                "verify", ("token_budget_exhausted", "sequence_and_token_budget_exhausted")
            ),
            "prefill_slot_exclusions_per_100_target_forwards": (
                "prefill", ("sequence_slots_exhausted", "sequence_and_token_budget_exhausted")
            ),
            "prefill_token_exclusions_per_100_target_forwards": (
                "prefill", ("token_budget_exhausted", "sequence_and_token_budget_exhausted")
            ),
            "prefill_policy_deferred_per_100_target_forwards": (
                "prefill", ("prefill_policy_budget_exhausted", "policy_deferred")
            ),
        }
        for metric, (kind, reasons) in event_groups.items():
            row[metric] = 100.0 * _event_count(result, kind, reasons) / target_forwards
        records.append(row)

    if not records:
        raise ValueError(f"No result JSON files found under {root}")
    return records


def aggregate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, float], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(record["chunk_size"], record["rate"])].append(record)

    rows: list[dict[str, Any]] = []
    for (chunk, rate), items in sorted(groups.items()):
        ordered = sorted(items, key=lambda item: item["seed"])
        row: dict[str, Any] = {
            "chunk_size": chunk,
            "arrival_rate": rate,
            "n": len(items),
            "seeds": ",".join(str(item["seed"]) for item in ordered),
        }
        for metric in METRICS:
            values = [float(item[metric]) for item in items]
            row[f"{metric}_mean"] = statistics.mean(values)
            row[f"{metric}_std"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
        rows.append(row)
    return rows


def aggregate_compositions(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, float], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(record["chunk_size"], record["rate"])].append(record)

    rows: list[dict[str, Any]] = []
    for (chunk, rate), items in sorted(groups.items()):
        for composition in COMPOSITIONS:
            values = [
                float(item["composition_counts"].get(composition, 0))
                for item in items
            ]
            rows.append({
                "chunk_size": chunk,
                "arrival_rate": rate,
                "composition": composition,
                "n": len(values),
                "mean_forwards_per_run": statistics.mean(values),
                "std_forwards_per_run": (
                    statistics.stdev(values) if len(values) > 1 else 0.0
                ),
            })
    return rows


def paired_vs_baseline(
    records: list[dict[str, Any]], baseline_chunk: int
) -> list[dict[str, Any]]:
    index = {
        (item["chunk_size"], item["rate"], item["seed"]): item
        for item in records
    }
    chunks = sorted({item["chunk_size"] for item in records})
    rates = sorted({item["rate"] for item in records})
    seeds = sorted({item["seed"] for item in records})
    compare_metrics = (
        "output_tokens_per_s",
        "ttft_p95_ms",
        "ar_wait_p95_ms",
        "prefill_containing_slot_utilization_mean",
        "prefill_containing_token_budget_utilization_mean",
        "ar_slot_exclusions_per_100_target_forwards",
        "ar_token_exclusions_per_100_target_forwards",
    )

    rows: list[dict[str, Any]] = []
    for chunk in chunks:
        if chunk == baseline_chunk:
            continue
        for rate in rates:
            deltas: dict[str, list[float]] = {
                metric: [] for metric in compare_metrics
            }
            for seed in seeds:
                candidate = index.get((chunk, rate, seed))
                baseline = index.get((baseline_chunk, rate, seed))
                if candidate is None or baseline is None:
                    continue
                for metric in compare_metrics:
                    deltas[metric].append(
                        float(candidate[metric]) - float(baseline[metric])
                    )
            row: dict[str, Any] = {
                "chunk_size": chunk,
                "baseline_chunk_size": baseline_chunk,
                "arrival_rate": rate,
                "paired_n": len(deltas["output_tokens_per_s"]),
            }
            for metric, values in deltas.items():
                row[f"{metric}_delta_mean"] = (
                    statistics.mean(values) if values else ""
                )
                row[f"{metric}_delta_std"] = (
                    statistics.stdev(values) if len(values) > 1
                    else (0.0 if values else "")
                )
            rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_report(
    path: Path,
    summary: list[dict[str, Any]],
    compositions: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    baseline_chunk: int,
) -> None:
    lines = [
        "# CDB-AV Target Forward Contention Diagnostics",
        "",
        "CPU-only discrete-event simulation. Execution coefficients are synthetic unless a measured profile is supplied.",
        "",
        "## Per-run mean ± sample standard deviation",
        "",
        "| Chunk | Rate | N | TPS | TTFT P95 | AR Wait P95 | Prefill Slot Util. | Prefill Token Util. | Chunk Truncations | Chunk Token Shortfall | AR Slot Exclusions / 100 TF | AR Token Exclusions / 100 TF |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    def cell(row: dict[str, Any], metric: str, digits: int = 2) -> str:
        return (
            f"{row[f'{metric}_mean']:.{digits}f} ± "
            f"{row[f'{metric}_std']:.{digits}f}"
        )

    for row in summary:
        lines.append(
            f"| {row['chunk_size']} | {row['arrival_rate']:g} | {row['n']} | "
            f"{cell(row, 'output_tokens_per_s')} | "
            f"{cell(row, 'ttft_p95_ms')} | "
            f"{cell(row, 'ar_wait_p95_ms')} | "
            f"{cell(row, 'prefill_containing_slot_utilization_mean', 3)} | "
            f"{cell(row, 'prefill_containing_token_budget_utilization_mean', 3)} | "
            f"{cell(row, 'prefill_chunk_budget_truncations')} | "
            f"{cell(row, 'prefill_chunk_budget_shortfall_tokens')} | "
            f"{cell(row, 'ar_slot_exclusions_per_100_target_forwards')} | "
            f"{cell(row, 'ar_token_exclusions_per_100_target_forwards')} |"
        )

    lines.extend([
        "",
        "## Target batch composition",
        "",
        "Values are mean numbers of forwards per run across seeds.",
        "",
        "| Chunk | Rate | Prefill-only | Verify-only | AR-only | Prefill+Verify | Prefill+AR | Verify+AR | All three |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    composition_groups: dict[tuple[int, float], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in compositions:
        composition_groups[(row["chunk_size"], row["arrival_rate"])][row["composition"]] = row
    for (chunk, rate), values in sorted(composition_groups.items()):
        means = [
            values.get(name, {}).get("mean_forwards_per_run", 0.0)
            for name in COMPOSITIONS
        ]
        lines.append(
            f"| {chunk} | {rate:g} | " + " | ".join(f"{value:.1f}" for value in means) + " |"
        )

    lines.extend([
        "",
        f"## Paired deltas against chunk={baseline_chunk}",
        "",
        "Positive TPS deltas are better. Positive latency and exclusion deltas are worse.",
        "",
        "| Chunk | Rate | Paired N | TPS Δ | TTFT P95 Δ | AR Wait P95 Δ | AR Slot Exclusions Δ | AR Token Exclusions Δ |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in paired:
        def delta(metric: str) -> str:
            value = row[f"{metric}_delta_mean"]
            return "-" if value == "" else f"{value:+.2f}"
        lines.append(
            f"| {row['chunk_size']} | {row['arrival_rate']:g} | {row['paired_n']} | "
            f"{delta('output_tokens_per_s')} | {delta('ttft_p95_ms')} | "
            f"{delta('ar_wait_p95_ms')} | "
            f"{delta('ar_slot_exclusions_per_100_target_forwards')} | "
            f"{delta('ar_token_exclusions_per_100_target_forwards')} |"
        )

    lines.extend([
        "",
        "Interpretation note: exclusion counts are request-forward events, not unique requests. "
        "Prefill-containing utilization is conditioned on forwards that include at least one Prefill chunk. "
        "The trace explains each batcher's candidate-order capacity decisions; it does not measure GPU hardware.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-chunk", type=int, default=1024)
    args = parser.parse_args(argv)

    records = collect_records(args.input_dir)
    summary = aggregate(records)
    compositions = aggregate_compositions(records)
    paired = paired_vs_baseline(records, args.baseline_chunk)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "target_contention_summary.csv", summary)
    _write_csv(args.output_dir / "target_contention_composition.csv", compositions)
    _write_csv(args.output_dir / "target_contention_paired.csv", paired)
    _write_report(
        args.output_dir / "target_contention_report.md",
        summary,
        compositions,
        paired,
        args.baseline_chunk,
    )
    print(
        f"Aggregated {len(records)} runs into {len(summary)} chunk/rate groups; "
        f"wrote results to {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
