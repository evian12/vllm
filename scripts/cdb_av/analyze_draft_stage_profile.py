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


def summary(records):
    groups = defaultdict(list)
    for item in records:
        groups[(item.get("graph_action"), item.get("phase"))].append(item)
    result = []
    keys = (
        ("metadata_pack_cpu_ms", lambda r: r.get("metadata_pack_cpu_ms")),
        ("metadata_cpu_ms", lambda r: r["measurements"].get("metadata", {}).get("cpu_enqueue_ms")),
        ("metadata_stream_ms", lambda r: r["measurements"].get("metadata", {}).get("stream_span_ms")),
        ("backbone_cpu_ms", lambda r: r["measurements"].get("backbone", {}).get("cpu_enqueue_ms")),
        ("backbone_stream_ms", lambda r: r["measurements"].get("backbone", {}).get("stream_span_ms")),
        ("logits_stream_ms", lambda r: r["measurements"].get("logits", {}).get("stream_span_ms")),
        ("confidence_gpu_stream_ms", lambda r: r["measurements"].get("confidence_gpu", {}).get("stream_span_ms")),
        ("d2h_wait_cpu_ms", lambda r: r["measurements"].get("d2h_wait_cpu_ms")),
        ("batch_update_cpu_ms", lambda r: r["measurements"].get("batch_update_cpu_ms")),
    )
    for (action, phase), items in sorted(groups.items()):
        row = dict(graph_action=action, phase=phase, samples=len(items),
                   mean_request_rows=statistics.mean(x["request_rows"] for x in items),
                   mean_token_rows=statistics.mean(x["token_rows"] for x in items))
        for label, getter in keys:
            values = _finite(getter(x) for x in items)
            if values:
                row[label] = round(statistics.mean(values), 4)
        result.append(row)
    return result


def main(argv):
    if len(argv) != 2:
        print(f"Usage: {argv[0]} /tmp/cdb_draft_stage_profile_20261109.log")
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
    print("Note: sampled CUDA synchronizations change throughput and D2H wait includes prior GPU work.")
    print(json.dumps(summary(records), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
