#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare matched Draft Decode samples with Draft CUDA Graph ON vs OFF.

Only compares phase=decode samples, grouped by identical request/token row
counts. This avoids comparing graph replay Decode against eager mixed Prefill.
Profiling is diagnostic; this script does not report production TPS.
"""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

MARKER = "[CDB_DRAFT_STAGE_TIMING] "
METRICS = (
    ("backbone_cpu_enqueue_ms", "backbone", "cpu_enqueue_ms"),
    ("backbone_stream_ms", "backbone", "stream_span_ms"),
    ("logits_cpu_enqueue_ms", "logits", "cpu_enqueue_ms"),
    ("logits_stream_ms", "logits", "stream_span_ms"),
    ("confidence_cpu_enqueue_ms", "confidence_gpu", "cpu_enqueue_ms"),
    ("confidence_stream_ms", "confidence_gpu", "stream_span_ms"),
    ("pre_d2h_gpu_wait_cpu_ms", None, None),
    ("d2h_materialize_cpu_ms", None, None),
    ("d2h_wait_cpu_ms", None, None),
    ("batch_update_cpu_ms", None, None),
)


def load(path, expected_action):
    groups = defaultdict(list)
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if MARKER not in line:
            continue
        try:
            row = json.loads(line.split(MARKER, 1)[1])
        except (ValueError, TypeError):
            continue
        if row.get("phase") != "decode" or row.get("graph_action") != expected_action:
            continue
        shape = (int(row["request_rows"]), int(row["token_rows"]))
        groups[shape].append(row)
    return groups


def value(row, name, stage, field):
    measurements = row.get("measurements", {})
    if stage is None:
        result = measurements.get(name)
    else:
        result = measurements.get(stage, {}).get(field)
    return result if isinstance(result, (int, float)) else None


def median(rows, name, stage, field):
    values = [value(row, name, stage, field) for row in rows]
    values = [float(v) for v in values if v is not None]
    return statistics.median(values) if values else None


def fmt(v):
    return "--" if v is None else f"{v:.4f}"


def main(argv):
    if len(argv) != 3:
        print(
            f"Usage: {argv[0]} GRAPH_ON_LOG GRAPH_OFF_LOG",
            file=sys.stderr,
        )
        return 2
    graph_on = load(argv[1], "replay")
    graph_off = load(argv[2], "eager")
    print("Draft Decode only; matched by (request_rows, token_rows)")
    print("Median milliseconds; diagnostic synchronizations perturb timing.")
    print()
    common = sorted(set(graph_on) & set(graph_off))
    if not common:
        print("No matched Decode shapes.")
        print("Graph ON shapes:", {k: len(v) for k, v in sorted(graph_on.items())})
        print("Graph OFF shapes:", {k: len(v) for k, v in sorted(graph_off.items())})
        return 1
    for shape in common:
        on_rows, off_rows = graph_on[shape], graph_off[shape]
        print(
            f"shape requests={shape[0]} tokens={shape[1]} | "
            f"Graph ON n={len(on_rows)} | Graph OFF n={len(off_rows)}"
        )
        print(f"{'metric':38s} {'Graph ON':>12s} {'Graph OFF':>12s} {'ON/OFF':>10s}")
        for label, stage, field in METRICS:
            on = median(on_rows, label, stage, field)
            off = median(off_rows, label, stage, field)
            ratio = (on / off) if on is not None and off not in (None, 0) else None
            print(f"{label:38s} {fmt(on):>12s} {fmt(off):>12s} {fmt(ratio):>10s}")
        print()
    print("Unmatched Graph ON shapes:", {
        k: len(v) for k, v in sorted(set(graph_on) - set(graph_off))
    })
    print("Unmatched Graph OFF shapes:", {
        k: len(v) for k, v in sorted(set(graph_off) - set(graph_on))
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
