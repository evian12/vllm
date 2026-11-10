#!/usr/bin/env python3
"""Aggregate CDB-AV Prefill chunk sweep JSON files into CSV and Markdown."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

METRICS = (
    "output_tokens_per_s",
    "ttft_p50_ms",
    "ttft_p95_ms",
    "prefill_wait_p95_ms",
    "prefill_inter_chunk_wait_p95_ms",
    "prefill_target_forward_exposure_p95_ms",
    "prefill_chunk_count_mean",
    "prefill_decomposition_error_max_ms",
    "ar_wait_p95_ms",
    "commit_gap_p95_ms",
    "target_forwards",
    "mean_target_batch_size",
)


def collect_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.json")):
        if path.name == "manifest.json":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        workload = payload.get("workload", {})
        config = payload.get("scope", {}).get("scheduling_policy_parameters", {})
        chunk_size = config.get("deadline_prefill_chunk_tokens")
        if chunk_size is None:
            chunk_size = payload.get("config", {}).get("deadline_prefill_chunk_tokens")
        if chunk_size is None:
            # Runner artifact directories are named chunk_<N>.
            for parent in path.parents:
                if parent.name.startswith("chunk_"):
                    chunk_size = int(parent.name.removeprefix("chunk_"))
                    break
        if chunk_size is None:
            raise ValueError(f"Cannot determine chunk size for {path}")

        results = payload.get("results", {})
        if len(results) != 1:
            raise ValueError(f"Expected exactly one policy in {path}; got {list(results)}")
        policy_result = next(iter(results.values()))
        records.append(
            {
                "chunk_size": int(chunk_size),
                "rate": float(workload["arrival_rate_rps"]),
                "seed": int(workload["seed"]),
                **policy_result,
            }
        )
    if not records:
        raise ValueError(f"No result JSON files found under {root}")
    return records


def aggregate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, float], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(record["chunk_size"], record["rate"])].append(record)

    rows: list[dict[str, Any]] = []
    for (chunk, rate), items in sorted(groups.items()):
        row: dict[str, Any] = {
            "chunk_size": chunk,
            "arrival_rate": rate,
            "n": len(items),
            "seeds": ",".join(str(x["seed"]) for x in sorted(items, key=lambda x: x["seed"])),
        }
        for metric in METRICS:
            values = [
                float(item[metric])
                for item in items
                if item.get(metric) is not None
            ]
            row[f"{metric}_mean"] = statistics.mean(values) if values else ""
            row[f"{metric}_std"] = (
                statistics.stdev(values) if len(values) > 1 else (0.0 if values else "")
            )
        rows.append(row)
    return rows


def paired_vs_baseline(records: list[dict[str, Any]], baseline_chunk: int = 512) -> list[dict[str, Any]]:
    index = {
        (r["chunk_size"], r["rate"], r["seed"]): r
        for r in records
    }
    chunks = sorted({r["chunk_size"] for r in records})
    rates = sorted({r["rate"] for r in records})
    rows: list[dict[str, Any]] = []
    compare_metrics = (
        "output_tokens_per_s",
        "ttft_p95_ms",
        "prefill_wait_p95_ms",
        "prefill_inter_chunk_wait_p95_ms",
        "prefill_target_forward_exposure_p95_ms",
    )
    for chunk in chunks:
        if chunk == baseline_chunk:
            continue
        for rate in rates:
            deltas: dict[str, list[float]] = {metric: [] for metric in compare_metrics}
            for seed in sorted({r["seed"] for r in records}):
                candidate = index.get((chunk, rate, seed))
                baseline = index.get((baseline_chunk, rate, seed))
                if candidate is None or baseline is None:
                    continue
                for metric in compare_metrics:
                    if candidate.get(metric) is not None and baseline.get(metric) is not None:
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
                row[f"{metric}_delta_mean"] = statistics.mean(values) if values else ""
                row[f"{metric}_delta_std"] = (
                    statistics.stdev(values) if len(values) > 1 else (0.0 if values else "")
                )
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary: list[dict[str, Any]], paired: list[dict[str, Any]]) -> None:
    lines = [
        "# CDB-AV Prefill Chunk Size Sweep",
        "",
        "CPU-only discrete-event simulation. Execution coefficients are synthetic unless a measured profile is explicitly supplied.",
        "",
        "## Mean ± sample standard deviation across seeds",
        "",
        "| Chunk | Rate | N | TPS | TTFT P50 | TTFT P95 | Initial Prefill Wait P95 | Inter-chunk Wait P95 | Target-forward Exposure P95 | Chunks/request |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        def cell(metric: str) -> str:
            mean = row[f"{metric}_mean"]
            std = row[f"{metric}_std"]
            return "-" if mean == "" else f"{mean:.2f} ± {std:.2f}"
        lines.append(
            f"| {row['chunk_size']} | {row['arrival_rate']:g} | {row['n']} | "
            f"{cell('output_tokens_per_s')} | {cell('ttft_p50_ms')} | "
            f"{cell('ttft_p95_ms')} | {cell('prefill_wait_p95_ms')} | "
            f"{cell('prefill_inter_chunk_wait_p95_ms')} | "
            f"{cell('prefill_target_forward_exposure_p95_ms')} | "
            f"{cell('prefill_chunk_count_mean')} |"
        )
    lines.extend(
        [
            "",
            f"## Paired deltas against chunk={512}",
            "",
            "Positive TTFT/Wait deltas are worse; positive TPS deltas are better.",
            "",
            "| Chunk | Rate | Paired N | TPS Δ | TTFT P95 Δ | Initial Wait P95 Δ | Inter-chunk Wait P95 Δ | Target Exposure P95 Δ |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in paired:
        def delta(metric: str) -> str:
            value = row[f"{metric}_delta_mean"]
            return "-" if value == "" else f"{value:+.2f}"
        lines.append(
            f"| {row['chunk_size']} | {row['arrival_rate']:g} | {row['paired_n']} | "
            f"{delta('output_tokens_per_s')} | {delta('ttft_p95_ms')} | "
            f"{delta('prefill_wait_p95_ms')} | "
            f"{delta('prefill_inter_chunk_wait_p95_ms')} | "
            f"{delta('prefill_target_forward_exposure_p95_ms')} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-chunk", type=int, default=512)
    args = parser.parse_args()

    records = collect_records(args.input_dir)
    summary = aggregate(records)
    paired = paired_vs_baseline(records, args.baseline_chunk)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "chunk_sweep_summary.csv", summary)
    write_csv(args.output_dir / "chunk_sweep_paired_vs_512.csv", paired)
    write_markdown(args.output_dir / "chunk_sweep_report.md", summary, paired)
    print(
        f"Aggregated {len(records)} runs into {len(summary)} groups; "
        f"wrote results to {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
