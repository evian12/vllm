#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Unprofiled 512-request Draft CUDA Graph ON/OFF performance client.

Run against a separately launched service. Use the same --run-prefix for
Graph ON and OFF so request IDs and prompt payloads are paired exactly.
"""
import argparse
import json
import math
import re
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

CLIENT = Path(__file__).with_name("stress_waiting_512.py")


def percentile(values, p):
    values = sorted(values)
    if not values:
        return math.nan
    position = (len(values) - 1) * p
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] * (upper - position) + values[upper] * (position - lower)


def check_server(port):
    url = f"http://127.0.0.1:{port}/health"
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            if response.status != 200:
                raise RuntimeError(f"health endpoint returned HTTP {response.status}")
    except Exception as exc:
        raise RuntimeError(
            f"No healthy vLLM service at 127.0.0.1:{port}: {exc}"
        ) from exc


def run_client(args, out_path, requests, run_id):
    command = [
        sys.executable, str(CLIENT),
        "--port", str(args.port),
        "--requests", str(requests),
        "--concurrency", str(args.concurrency),
        "--max-tokens", str(args.max_tokens),
        "--run-id", run_id,
    ]
    started = time.perf_counter()
    with out_path.open("w", encoding="utf-8") as output:
        result = subprocess.run(
            command, stdout=output, stderr=subprocess.STDOUT, check=False
        )
    elapsed = time.perf_counter() - started
    raw = out_path.read_text(encoding="utf-8", errors="replace")
    records = []
    for line in raw.splitlines():
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(row, dict) and "index" in row and "ok" in row:
            records.append(row)
    completed = sum(bool(row["ok"]) for row in records)
    failed = len(records) - completed
    tokens = sum(int(row.get("output_tokens", 0)) for row in records)
    latencies = [
        float(row["elapsed"]) for row in records
        if row.get("ok") and isinstance(row.get("elapsed"), (int, float))
    ]
    match = re.search(r"wall_time=([0-9.]+)s", raw)
    wall = float(match.group(1)) if match else elapsed
    if result.returncode != 0 or len(records) != requests or completed != requests:
        tail = "\n".join(raw.splitlines()[-30:])
        raise RuntimeError(
            f"{out_path.name}: client_rc={result.returncode}, "
            f"records={len(records)}/{requests}, completed={completed}/{requests}, "
            f"failed={failed}\n{tail}"
        )
    return {
        "run_id": run_id,
        "requests": requests,
        "completed": completed,
        "failed": failed,
        "output_tokens": tokens,
        "wall_time_s": round(wall, 4),
        "output_tps": round(tokens / wall, 4) if wall > 0 else None,
        "request_tps": round(completed / wall, 4) if wall > 0 else None,
        "e2e_p50_s": round(percentile(latencies, 0.50), 4),
        "e2e_p95_s": round(percentile(latencies, 0.95), 4),
        "e2e_p99_s": round(percentile(latencies, 0.99), 4),
        "client_log": str(out_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, choices=("graph_on", "graph_off"))
    parser.add_argument("--port", type=int, default=18084)
    parser.add_argument("--requests", type=int, default=512)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--run-prefix", default="cdb_draft_graph_ab_20261109")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    if min(args.requests, args.concurrency, args.max_tokens, args.repeats) < 1:
        parser.error("requests, concurrency, max-tokens and repeats must be positive")
    if args.warmup < 0:
        parser.error("warmup must be non-negative")

    check_server(args.port)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(
        args.output_dir or f"/tmp/cdb_draft_graph_ab_{args.label}_{stamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    config = vars(args).copy()
    config["output_dir"] = str(output_dir)
    config["client_script"] = str(CLIENT)
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(config, indent=2), flush=True)

    if args.warmup:
        print(f"Warmup: {args.warmup} requests", flush=True)
        warmup = run_client(
            args, output_dir / "warmup.log", args.warmup,
            f"{args.run_prefix}_warmup",
        )
        print(json.dumps(warmup, indent=2), flush=True)

    runs = []
    for index in range(1, args.repeats + 1):
        run_id = f"{args.run_prefix}_run_{index}"
        print(
            f"========== {args.label} RUN {index}/{args.repeats} ==========",
            flush=True,
        )
        row = run_client(
            args, output_dir / f"run_{index}.log", args.requests, run_id
        )
        runs.append(row)
        print(json.dumps(row, indent=2), flush=True)

    tps = [row["output_tps"] for row in runs]
    summary = {
        "label": args.label,
        "requests_per_run": args.requests,
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "warmup_requests": args.warmup,
        "repeats": args.repeats,
        "mean_output_tps": round(statistics.mean(tps), 4),
        "stdev_output_tps": round(
            statistics.stdev(tps) if len(tps) > 1 else 0.0, 4
        ),
        "runs": runs,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print("========== SUMMARY ==========", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
