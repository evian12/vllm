#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Summarize opt-in CDB Draft stage timing from a local vLLM log.

CPU enqueue and stream spans are different metrics and must not be added.
This parser cannot infer pure PCIe D2H time from a synchronizing .cpu().tolist().
"""
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

MARKER = "[CDB_DRAFT_STAGE_TIMING] "


def parse_records(text):
    for line in text.splitlines():
        if MARKER not in line:
            continue
        try:
            payload = json.loads(line.split(MARKER, 1)[1].strip())
        except (ValueError, json.JSONDecodeError):
            continue
        if (not isinstance(payload, dict)
                or payload.get("graph_action") not in ("eager", "replay")
                or not isinstance(payload.get("measurements"), dict)):
            continue
        yield payload


def _finite(values):
    return [float(v) for v in values
            if isinstance(v, (int, float)) and math.isfinite(v)]


def _stats(values):
    values = _finite(values)
    if not values:
        return None
    ordered = sorted(values)
    return {
        "mean": round(statistics.mean(values), 4),
        "median": round(statistics.median(values), 4),
        "min": round(ordered[0], 4),
        "max": round(ordered[-1], 4),
    }


def _measurement(record, stage, field):
    value = record["measurements"].get(stage, {})
    if not isinstance(value, dict):
        return None
    return value.get(field)


def summary(records):
    groups = defaultdict(list)
    for item in records:
        groups[(item.get("graph_action"), item.get("phase"))].append(item)

    metrics = (
        ("metadata_pack_cpu_ms", lambda r: r.get("metadata_pack_cpu_ms")),
        ("metadata_cpu_enqueue_ms", lambda r: _measurement(r, "metadata", "cpu_enqueue_ms")),
        ("metadata_stream_ms", lambda r: _measurement(r, "metadata", "stream_span_ms")),
        ("backbone_cpu_enqueue_ms", lambda r: _measurement(r, "backbone", "cpu_enqueue_ms")),
        ("backbone_stream_ms", lambda r: _measurement(r, "backbone", "stream_span_ms")),
        ("logits_cpu_enqueue_ms", lambda r: _measurement(r, "logits", "cpu_enqueue_ms")),
        ("logits_stream_ms", lambda r: _measurement(r, "logits", "stream_span_ms")),
        ("confidence_gpu_cpu_enqueue_ms", lambda r: _measurement(r, "confidence_gpu", "cpu_enqueue_ms")),
        ("confidence_gpu_stream_ms", lambda r: _measurement(r, "confidence_gpu", "stream_span_ms")),
        ("pre_d2h_gpu_wait_cpu_ms", lambda r: r["measurements"].get("pre_d2h_gpu_wait_cpu_ms")),
        ("d2h_materialize_cpu_ms", lambda r: r["measurements"].get("d2h_materialize_cpu_ms")),
        ("d2h_wait_cpu_ms", lambda r: r["measurements"].get("d2h_wait_cpu_ms")),
        ("batch_update_cpu_ms", lambda r: r["measurements"].get("batch_update_cpu_ms")),
    )

    result = []
    for (action, phase), items in sorted(groups.items()):
        row = dict(
            graph_action=action,
            phase=phase,
            samples=len(items),
            mean_request_rows=round(statistics.mean(x["request_rows"] for x in items), 3),
            mean_token_rows=round(statistics.mean(x["token_rows"] for x in items), 3),
        )
        for label, getter in metrics:
            stats = _stats(getter(x) for x in items)
            if stats is not None:
                row[label] = stats

        # Per-sample confidence timings are retained to expose cold-start or
        # host-dispatch outliers hidden by group means.
        row["sample_details"] = [
            {
                "sample_index": x.get("sample_index"),
                "window": x.get("window"),
                "step": x.get("step"),
                "request_rows": x.get("request_rows"),
                "token_rows": x.get("token_rows"),
                "confidence_gpu_cpu_enqueue_ms": _measurement(
                    x, "confidence_gpu", "cpu_enqueue_ms"),
                "confidence_gpu_stream_ms": _measurement(
                    x, "confidence_gpu", "stream_span_ms"),
                "pre_d2h_gpu_wait_cpu_ms": x["measurements"].get("pre_d2h_gpu_wait_cpu_ms"),
                "d2h_materialize_cpu_ms": x["measurements"].get("d2h_materialize_cpu_ms"),
                "d2h_wait_cpu_ms": x["measurements"].get("d2h_wait_cpu_ms"),
            }
            for x in items
        ]
        result.append(row)
    return result


def main(argv):
    if len(argv) != 2:
        print(f"Usage: {argv[0]} /tmp/cdb_draft_stage_profile_split_20261109.log")
        return 2
    path = Path(argv[1])
    if not path.is_file():
        print("Log not found:", path)
        return 2
    records = list(parse_records(path.read_text(encoding="utf-8", errors="replace")))
    print("Draft stage samples:", len(records))
    if not records:
        print("No valid records. Check service source, profile flag and log path.")
        return 1
    print(
        "Note: CPU enqueue, CUDA stream interval and D2H blocking time are "
        "different measurements. They must not be added. CUDA event intervals "
        "can include GPU idle gaps while the host submits kernels."
    )
    print(json.dumps(summary(records), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
