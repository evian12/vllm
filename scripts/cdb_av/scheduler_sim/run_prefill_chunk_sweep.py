#!/usr/bin/env python3
"""Run one Prefill chunk-size slice of the CDB-AV CPU simulator sweep."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def parse_int_list(value: str, name: str) -> list[int]:
    try:
        values = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be comma-separated integers") from exc
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError(f"{name} values must be positive")
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError(f"{name} values must be unique")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-size", type=int, required=True)
    parser.add_argument("--rates", default="5,15,30,60,100")
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--arrival-mode", choices=("poisson", "uniform", "bursty"), default="poisson")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.chunk_size < 1 or args.requests < 1:
        raise SystemExit("--chunk-size and --requests must be positive")

    rates = parse_int_list(args.rates, "rates")
    seeds = parse_int_list(args.seeds, "seeds")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "chunk_size": args.chunk_size,
        "rates": rates,
        "seeds": seeds,
        "requests": args.requests,
        "arrival_mode": args.arrival_mode,
        "policy": "load_adaptive_deadline",
        "git_sha": os.environ.get("GITHUB_SHA"),
        "runs": [],
    }

    for rate in rates:
        for seed in seeds:
            output = args.output_dir / f"rate_{rate}_seed_{seed}.json"
            command = [
                sys.executable,
                "-m",
                "scripts.cdb_av.scheduler_sim.phase_cli",
                "--requests",
                str(args.requests),
                "--arrival-rate",
                str(rate),
                "--arrival-mode",
                args.arrival_mode,
                "--policy",
                "load_adaptive_deadline",
                "--deadline-prefill-chunk-tokens",
                str(args.chunk_size),
                "--seed",
                str(seed),
                "--output",
                str(output),
            ]
            print(
                f"[chunk={args.chunk_size}] rate={rate} seed={seed}",
                flush=True,
            )
            subprocess.run(command, check=True)

            payload = json.loads(output.read_text(encoding="utf-8"))
            results = payload.get("results", {})
            if len(results) != 1:
                raise RuntimeError(f"Expected one policy result in {output}; got {list(results)}")
            result = next(iter(results.values()))
            required = (
                "output_tokens_per_s",
                "ttft_p50_ms",
                "ttft_p95_ms",
                "prefill_wait_p95_ms",
                "prefill_inter_chunk_wait_p95_ms",
                "prefill_target_forward_exposure_p95_ms",
                "prefill_chunk_count_mean",
                "prefill_decomposition_error_max_ms",
            )
            missing = [key for key in required if key not in result]
            if missing:
                raise RuntimeError(
                    f"{output} is missing required metrics {missing}; "
                    "the simulator revision may not include Prefill decomposition"
                )
            manifest["runs"].append(
                {"rate": rate, "seed": seed, "file": output.name}
            )

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Completed {len(manifest['runs'])} runs; manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
