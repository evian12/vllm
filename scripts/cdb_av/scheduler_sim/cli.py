"""Command-line entry point for CDB-AV scheduler simulation."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from .engine import Simulator
from .model import SimulationConfig
from .policies import (
    FixedHPolicy,
    LookaheadPolicy,
    QueuePressurePolicy,
    QueueThresholdPolicy,
)
from .profiles import ExecutionProfile, load_execution_profile
from .workload import generate_workload


def parse_int_list(value: str) -> tuple[int, ...]:
    try:
        result = tuple(
            int(part.strip()) for part in value.split(",") if part.strip()
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected comma-separated integers"
        ) from exc
    if not result or any(item < 1 for item in result):
        raise argparse.ArgumentTypeError(
            "all values must be positive integers"
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "CPU-only, single-GPU serial Draft/Verify scheduling simulator. "
            "Default execution coefficients are synthetic."
        )
    )
    parser.add_argument("--requests", type=int, default=128)
    parser.add_argument(
        "--arrival-rate", type=float, default=20.0, dest="arrival_rate"
    )
    parser.add_argument(
        "--arrival-mode",
        choices=("poisson", "uniform", "bursty"),
        default="poisson",
    )
    parser.add_argument(
        "--policy",
        choices=("all", "fixed_h", "threshold", "pressure", "lookahead"),
        default="all",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-draft-batch", type=int, default=4)
    parser.add_argument("--max-verify-batch", type=int, default=4)
    parser.add_argument("--fixed-h", type=int, default=8)
    parser.add_argument("--queue-threshold", type=int, default=4)
    parser.add_argument("--ready-wait-budget-ms", type=float, default=8.0)
    parser.add_argument("--confidence-threshold", type=float, default=0.8)
    parser.add_argument("--max-output-tokens", type=int, default=64)
    parser.add_argument(
        "--draft-caps", type=parse_int_list, default=(2, 4, 6, 8)
    )
    parser.add_argument("--acceptance-base", type=float, default=0.72)
    parser.add_argument("--acceptance-decay", type=float, default=0.72)
    parser.add_argument("--lookahead-horizon", type=int, default=5)
    parser.add_argument("--lookahead-rollouts", type=int, default=3)
    parser.add_argument("--profile-json", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-steps", type=int, default=200_000)
    return parser


def build_policies(
    args: argparse.Namespace,
    config: SimulationConfig,
) -> dict[str, object]:
    pressure = QueuePressurePolicy(
        ready_wait_budget_ms=args.ready_wait_budget_ms,
        max_consecutive_verify_forwards=config.max_consecutive_verify_forwards,
    )
    policies: dict[str, object] = {
        "fixed_h": FixedHPolicy(args.fixed_h),
        "threshold": QueueThresholdPolicy(args.queue_threshold),
        "pressure": pressure,
        "lookahead": LookaheadPolicy(
            horizon=args.lookahead_horizon,
            rollouts=args.lookahead_rollouts,
            rollout_policy=pressure,
        ),
    }
    if args.policy == "all":
        return policies
    return {args.policy: policies[args.policy]}


def run_comparison(
    requests: Sequence,
    config: SimulationConfig,
    profile: ExecutionProfile,
    policies: dict[str, object],
) -> dict[str, dict[str, object]]:
    results: dict[str, dict[str, object]] = {}
    for name, policy in policies.items():
        simulator = Simulator(
            requests=requests,
            config=config,
            profile=profile,
            policy=policy,
        )
        results[name] = simulator.run()
    return results


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.requests < 1:
        raise SystemExit("--requests must be positive")

    config = SimulationConfig(
        max_draft_batch_size=args.max_draft_batch,
        max_verify_batch_size=args.max_verify_batch,
        fixed_h=args.fixed_h,
        queue_threshold=args.queue_threshold,
        ready_wait_budget_ms=args.ready_wait_budget_ms,
        confidence_threshold=args.confidence_threshold,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    profile = (
        load_execution_profile(args.profile_json)
        if args.profile_json
        else ExecutionProfile()
    )
    requests = generate_workload(
        request_count=args.requests,
        arrival_rate_rps=args.arrival_rate,
        arrival_mode=args.arrival_mode,
        seed=args.seed,
        max_output_tokens=args.max_output_tokens,
        draft_caps=args.draft_caps,
        acceptance_base=args.acceptance_base,
        acceptance_decay=args.acceptance_decay,
    )

    policies = build_policies(args, config)
    results = run_comparison(requests, config, profile, policies)
    payload = {
        "scope": {
            "gpu_execution": "single-GPU serial forwards; no overlap",
            "prefill": "assumed complete before Draft eligibility",
            "ar": "excluded in phase 1",
            "window_h": (
                "fixed only in the fixed_h baseline; event-driven policies "
                "do not consume H"
            ),
            "confidence": (
                "synthetic confidence trace; low-confidence candidate is "
                "included, then the request becomes Ready"
            ),
            "execution_profile": (
                "synthetic defaults unless --profile-json supplies measured data"
            ),
        },
        "config": asdict(config),
        "execution_profile": profile.to_dict(),
        "workload": {
            "requests": args.requests,
            "arrival_rate_rps": args.arrival_rate,
            "arrival_mode": args.arrival_mode,
            "seed": args.seed,
            "draft_caps": list(args.draft_caps),
            "acceptance_base": args.acceptance_base,
            "acceptance_decay": args.acceptance_decay,
        },
        "results": results,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"saved: {args.output}")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
