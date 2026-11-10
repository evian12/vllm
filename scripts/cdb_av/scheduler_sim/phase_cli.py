"""CLI for phase-aware CDB-AV scheduler simulation."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .phase_aware import (
    DeadlineAwareTargetBatcher,
    DynamicPhasePolicy,
    FixedHPhasePolicy,
    JointPhasePolicy,
    LengthAwareTargetBatcher,
    PhaseAwareSimulator,
    PhaseConfig,
    PhaseExecutionProfile,
    PhaseLookaheadPolicy,
    QueuePressurePhasePolicy,
    SarathiTargetBatcher,
    VllmChunkedTargetBatcher,
    generate_phase_workload,
)


def parse_int_list(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not values or any(value < 1 for value in values):
        raise argparse.ArgumentTypeError("all values must be positive integers")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "CPU-only phase-aware CDB-AV simulation. One GPU executes serial "
            "Draft and mixed Target forwards; default service times are synthetic."
        )
    )
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--arrival-rate", type=float, default=30.0)
    parser.add_argument(
        "--arrival-mode",
        choices=("poisson", "uniform", "bursty"),
        default="bursty",
    )
    parser.add_argument(
        "--policy",
        choices=(
            "all", "fixed_h", "queue_pressure", "lookahead", "sarathi",
            "length_aware", "joint", "fixed_h_deadline", "dynamic",
            "dynamic_deadline",
        ),
        default="all",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--max-draft-batch", type=int, default=4)
    parser.add_argument("--fixed-h", type=int, default=8)
    parser.add_argument("--lookahead-horizon", type=int, default=4)
    parser.add_argument("--lookahead-rollouts", type=int, default=2)
    parser.add_argument("--confidence-threshold", type=float, default=0.8)
    parser.add_argument("--ready-wait-budget-ms", type=float, default=8.0)
    parser.add_argument("--ar-itl-budget-ms", type=float, default=40.0)
    parser.add_argument("--prefill-wait-budget-ms", type=float, default=12.0)
    parser.add_argument("--short-prefill-threshold", type=int, default=512)
    parser.add_argument("--prefill-bucket-width", type=int, default=256)
    parser.add_argument("--prefill-wait-window-ms", type=float, default=2.0)
    parser.add_argument("--prefill-wait-min-ms", type=float, default=0.5)
    parser.add_argument("--prefill-wait-max-ms", type=float, default=5.0)
    parser.add_argument("--prefill-budget-min", type=int, default=256)
    parser.add_argument("--prefill-budget-max", type=int, default=1024)
    parser.add_argument("--long-prefill-chunk-tokens", type=int, default=256)
    parser.add_argument("--sarathi-chunk-tokens", type=int, default=512)
    parser.add_argument("--deadline-prefill-chunk-tokens", type=int, default=512)
    parser.add_argument("--ar-fraction", type=float, default=0.25)
    parser.add_argument("--max-output-tokens", type=int, default=64)
    parser.add_argument("--draft-caps", type=parse_int_list, default=(2, 4, 6, 8))
    parser.add_argument(
        "--prompt-lengths",
        type=parse_int_list,
        default=(128, 256, 384, 512, 768, 1024, 2048, 4096),
    )
    parser.add_argument("--acceptance-base", type=float, default=0.72)
    parser.add_argument("--acceptance-decay", type=float, default=0.72)
    parser.add_argument("--profile-json", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-steps", type=int, default=200_000)
    return parser


def build_policies(args: argparse.Namespace, config: PhaseConfig) -> dict[str, object]:
    vllm_batcher = VllmChunkedTargetBatcher()
    sarathi_batcher = SarathiTargetBatcher(args.sarathi_chunk_tokens)
    length_batcher = LengthAwareTargetBatcher(
        short_threshold=args.short_prefill_threshold,
        bucket_width=args.prefill_bucket_width,
        base_wait_ms=args.prefill_wait_window_ms,
        min_wait_ms=args.prefill_wait_min_ms,
        max_wait_ms=args.prefill_wait_max_ms,
        min_prefill_budget=args.prefill_budget_min,
        max_prefill_budget=args.prefill_budget_max,
        long_chunk_tokens=args.long_prefill_chunk_tokens,
    )
    deadline_batcher = DeadlineAwareTargetBatcher(
        prefill_chunk_tokens=args.deadline_prefill_chunk_tokens,
    )
    policies = {
        "fixed_h": FixedHPhasePolicy(args.fixed_h, vllm_batcher),
        "queue_pressure": QueuePressurePhasePolicy(vllm_batcher),
        "lookahead": PhaseLookaheadPolicy(
            horizon=args.lookahead_horizon,
            rollouts=args.lookahead_rollouts,
            rollout_policy=QueuePressurePhasePolicy(vllm_batcher),
        ),
        "sarathi": QueuePressurePhasePolicy(sarathi_batcher),
        "length_aware": QueuePressurePhasePolicy(length_batcher),
        "joint": JointPhasePolicy(length_batcher),
        # E2: isolate Target batch organization under the original Fixed-H stage policy.
        "fixed_h_deadline": FixedHPhasePolicy(args.fixed_h, deadline_batcher),
        # E3: isolate dynamic stage selection with the original FCFS Target batcher.
        "dynamic": DynamicPhasePolicy(vllm_batcher),
        # E4: combine dynamic stage selection and deadline-aware Target batching.
        "dynamic_deadline": DynamicPhasePolicy(deadline_batcher),
    }
    # Give each comparison row an explicit, stable strategy label.
    policies["fixed_h"].name = "fixed_h+vllm"
    policies["queue_pressure"].name = "queue_pressure+vllm"
    policies["sarathi"].name = "queue_pressure+sarathi"
    policies["length_aware"].name = "queue_pressure+length_aware"
    policies["fixed_h_deadline"].name = "fixed_h+deadline_batcher"
    policies["dynamic"].name = "dynamic_stage+vllm"
    policies["dynamic_deadline"].name = "dynamic_stage+deadline_batcher"
    if args.policy == "all":
        return policies
    return {args.policy: policies[args.policy]}


def run_comparison(
    requests,
    config: PhaseConfig,
    profile: PhaseExecutionProfile,
    policies: dict[str, object],
) -> dict[str, dict[str, object]]:
    results = {}
    for name, policy in policies.items():
        simulator = PhaseAwareSimulator(requests, config, profile, policy)
        results[name] = simulator.run()
    return results


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.requests < 1:
        raise SystemExit("--requests must be positive")
    if not 0.0 <= args.ar_fraction <= 1.0:
        raise SystemExit("--ar-fraction must be in [0, 1]")

    config = PhaseConfig(
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_draft_batch_size=args.max_draft_batch,
        fixed_h=args.fixed_h,
        confidence_threshold=args.confidence_threshold,
        ready_wait_budget_ms=args.ready_wait_budget_ms,
        ar_itl_budget_ms=args.ar_itl_budget_ms,
        prefill_wait_budget_ms=args.prefill_wait_budget_ms,
        max_steps=args.max_steps,
        seed=args.seed,
        short_prefill_threshold=args.short_prefill_threshold,
    )
    profile = (
        PhaseExecutionProfile.from_json(args.profile_json)
        if args.profile_json
        else PhaseExecutionProfile()
    )
    requests = generate_phase_workload(
        request_count=args.requests,
        arrival_rate_rps=args.arrival_rate,
        arrival_mode=args.arrival_mode,
        seed=args.seed,
        max_output_tokens=args.max_output_tokens,
        draft_caps=args.draft_caps,
        prompt_lengths=args.prompt_lengths,
        ar_fraction=args.ar_fraction,
        acceptance_base=args.acceptance_base,
        acceptance_decay=args.acceptance_decay,
    )
    policies = build_policies(args, config)
    results = run_comparison(requests, config, profile, policies)

    payload = {
        "scope": {
            "gpu_execution": "one GPU; Draft and Target forwards are serial; no overlap",
            "target_batch": "Verify, AR and chunked Prefill may share one Target forward",
            "prefill": "Target-first; first output token is committed when Prefill completes",
            "ar": "non-speculative requests use one-token Target AR steps",
            "verify": "each request's candidate verification is atomic",
            "length_aware": "LAPS-inspired approximation, not a reproduction of LAPS",
            "sarathi": "Sarathi-inspired chunked Prefill baseline",
            "execution_profile": (
                "synthetic defaults unless --profile-json supplies measured coefficients"
            ),
            "metrics_note": (
                "commit-gap is the interval between per-request commit events, "
                "not token-level ITL within a multi-token Verify commit"
            ),
        },
        "config": asdict(config),
        "execution_profile": profile.to_dict(),
        "workload": {
            "requests": args.requests,
            "arrival_rate_rps": args.arrival_rate,
            "arrival_mode": args.arrival_mode,
            "seed": args.seed,
            "ar_fraction": args.ar_fraction,
            "draft_caps": list(args.draft_caps),
            "prompt_lengths": list(args.prompt_lengths),
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
