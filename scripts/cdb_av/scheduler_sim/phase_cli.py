"""CLI for phase-aware CDB-AV scheduler simulation."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .marginal_yield_pressure import PressureCorrectedMarginalYieldPhasePolicy
from .phase_aware import (
    DeadlineAwareTargetBatcher,
    DynamicPhasePolicy,
    FixedHPhasePolicy,
    JointPhasePolicy,
    LengthAwareTargetBatcher,
    LoadAdaptivePhasePolicy,
    MarginalYieldPhasePolicy,
    PhaseAwareSimulator,
    PhaseConfig,
    PhaseExecutionProfile,
    PhaseLookaheadPolicy,
    QueuePressurePhasePolicy,
    SarathiTargetBatcher,
    VllmChunkedTargetBatcher,
    WindowPredictivePhasePolicy,
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
            "dynamic_deadline", "load_adaptive", "load_adaptive_deadline",
            "window_predictive", "e5_e6", "load_adaptive_deadline_oracle",
            "marginal_yield", "marginal_yield_pressure", "marginal_yield_oracle",
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
    parser.add_argument("--window-predictive-draft-lookahead", type=int, default=8)
    parser.add_argument("--marginal-yield-min-relative-gain", type=float, default=0.01)
    parser.add_argument("--marginal-yield-ready-backlog-weight", type=float, default=0.50)
    parser.add_argument("--marginal-yield-ready-age-weight", type=float, default=0.50)
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
    parser.add_argument(
        "--include-target-forward-trace",
        action="store_true",
        help="include per-Target-Forward composition, capacity, and exclusion details",
    )
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
        # Load-adaptive: only delay Verify when another Draft increases Target batch size.
        "load_adaptive": LoadAdaptivePhasePolicy(vllm_batcher),
        # E5: keep load-adaptive stage decisions, change only Target batch composition.
        "load_adaptive_deadline": LoadAdaptivePhasePolicy(deadline_batcher),
        # Explicit oracle ablation: preserves the historical full-request-list projection.
        "load_adaptive_deadline_oracle": LoadAdaptivePhasePolicy(
            deadline_batcher, allow_future_arrivals=True
        ),
        # E7: one-step marginal useful-output gain per added Draft+Target time.
        "marginal_yield": MarginalYieldPhasePolicy(
            deadline_batcher,
            min_relative_gain=args.marginal_yield_min_relative_gain,
        ),
        # Diagnostic upper bound only: the one-step projection sees arrivals
        # that occur during the projected Draft; not a deployable online policy.
        "marginal_yield_oracle": MarginalYieldPhasePolicy(
            deadline_batcher,
            min_relative_gain=args.marginal_yield_min_relative_gain,
            allow_future_arrivals=True,
        ),
        # E7.1-A: preserve E7 yield comparison, discount extra Draft under Ready pressure.
        "marginal_yield_pressure": PressureCorrectedMarginalYieldPhasePolicy(
            deadline_batcher,
            min_relative_gain=args.marginal_yield_min_relative_gain,
            ready_backlog_weight=args.marginal_yield_ready_backlog_weight,
            ready_age_weight=args.marginal_yield_ready_age_weight,
        ),
        # E6: predict heterogeneous Draft completion over a bounded horizon;
        # the stopping point is re-evaluated after every actual forward.
        "window_predictive": WindowPredictivePhasePolicy(
            deadline_batcher,
            max_draft_lookahead=args.window_predictive_draft_lookahead,
        ),
    }
    # Give each comparison row an explicit, stable strategy label.
    policies["fixed_h"].name = "fixed_h+vllm"
    policies["queue_pressure"].name = "queue_pressure+vllm"
    policies["sarathi"].name = "queue_pressure+sarathi"
    policies["length_aware"].name = "queue_pressure+length_aware"
    policies["fixed_h_deadline"].name = "fixed_h+deadline_batcher"
    policies["dynamic"].name = "dynamic_stage+vllm"
    policies["dynamic_deadline"].name = "dynamic_stage+deadline_batcher"
    policies["load_adaptive"].name = "load_adaptive+vllm"
    policies["load_adaptive_deadline"].name = "load_adaptive_online+deadline_batcher"
    policies["load_adaptive_deadline_oracle"].name = "load_adaptive_oracle+deadline_batcher"
    policies["window_predictive"].name = "window_predictive+deadline_batcher"
    policies["marginal_yield"].name = "marginal_yield+deadline_batcher"
    policies["marginal_yield_oracle"].name = "marginal_yield_oracle+deadline_batcher"
    policies["marginal_yield_pressure"].name = "marginal_yield_pressure+deadline_batcher"
    if args.policy == "all":
        return policies
    if args.policy == "e5_e6":
        return {
            key: policies[key]
            for key in (
                "load_adaptive_deadline",
                "load_adaptive_deadline_oracle",
                "window_predictive",
            )
        }
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
        include_target_forward_trace=args.include_target_forward_trace,
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
            "marginal_yield_stage": (
                "one-step online marginal expected committed-output yield per "
                "Draft-plus-Target time; uses request-level expected Verify yield "
                "and the same hard Ready/AR/Prefill wait guards as E5"
            ),
            "window_predictive_stage": (
                "CDB-specific bounded forecast of currently active heterogeneous Draft "
                "requests; predicts their exact confidence/cap stop points, carries "
                "unfinished requests, excludes not-yet-arrived requests from forecasts, "
                "and replans after each real forward"
            ),
            "load_adaptive_stage": (
                "one-step marginal Target-batch gain; the default online policy projects "
                "only requests already visible at the decision time. The explicit "
                "load_adaptive_deadline_oracle policy is a clairvoyant ablation."
            ),
            "dynamic_stage": (
                "no configured H; compares expected Target yield and batch fill "
                "against marginal Draft yield, with queue-deadline guards"
            ),
            "deadline_aware_target_batcher": (
                "earliest absolute service deadline across Verify, AR, and Prefill; "
                "Prefill deadline remains anchored to arrival until TTFT is emitted"
            ),
            "scheduling_policy_parameters": {
                "dynamic_fill_gain_weight": 0.25,
                "dynamic_urgency_weight": 0.20,
                "deadline_prefill_chunk_tokens": args.deadline_prefill_chunk_tokens,
                "window_predictive_draft_lookahead": args.window_predictive_draft_lookahead,
                "marginal_yield_min_relative_gain": args.marginal_yield_min_relative_gain,
                "marginal_yield_ready_backlog_weight": args.marginal_yield_ready_backlog_weight,
                "marginal_yield_ready_age_weight": args.marginal_yield_ready_age_weight,
                "load_adaptive_default_projection_mode": "visible_only",
                "load_adaptive_oracle_policy": "load_adaptive_deadline_oracle",
            },
            "execution_profile": (
                "synthetic defaults unless --profile-json supplies measured coefficients"
            ),
            "metrics_note": (
                "commit-gap is the interval between per-request commit events, "
                "not token-level ITL within a multi-token Verify commit"
            ),
            "target_forward_diagnostics": (
                "per-forward trace is emitted only with --include-target-forward-trace; "
                "exclusion events distinguish sequence-slot, token-budget, and "
                "policy-deferred work. Exclusion counts are repeated request-forward "
                "events, not unique requests."
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
