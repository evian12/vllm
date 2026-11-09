"""Deterministic heterogeneous request-trace generation."""

from __future__ import annotations

import random

from .model import Request


def generate_workload(
    request_count: int,
    arrival_rate_rps: float,
    arrival_mode: str = "poisson",
    seed: int = 42,
    max_output_tokens: int = 64,
    draft_caps: tuple[int, ...] = (2, 4, 6, 8),
    acceptance_base: float = 0.72,
    acceptance_decay: float = 0.72,
) -> list[Request]:
    if request_count < 0:
        raise ValueError("request_count must be non-negative")
    if arrival_rate_rps <= 0:
        raise ValueError("arrival_rate_rps must be positive")
    if arrival_mode not in {"poisson", "uniform", "bursty"}:
        raise ValueError("arrival_mode must be poisson, uniform, or bursty")
    if not draft_caps or any(cap < 1 for cap in draft_caps):
        raise ValueError("draft_caps must contain positive values")
    if not 0.0 <= acceptance_base <= 1.0:
        raise ValueError("acceptance_base must be in [0, 1]")
    if not 0.0 <= acceptance_decay <= 1.0:
        raise ValueError("acceptance_decay must be in [0, 1]")

    rng = random.Random(seed)
    mean_interarrival_ms = 1000.0 / arrival_rate_rps
    arrival_time_ms = 0.0
    requests: list[Request] = []
    max_cap = max(draft_caps)

    for index in range(request_count):
        if index > 0:
            if arrival_mode == "poisson":
                arrival_time_ms += rng.expovariate(
                    1.0 / mean_interarrival_ms
                )
            elif arrival_mode == "uniform":
                arrival_time_ms += mean_interarrival_ms
            elif index % 8 == 0:
                arrival_time_ms += 7.65 * mean_interarrival_ms
            else:
                arrival_time_ms += 0.05 * mean_interarrival_ms

        draft_cap = rng.choice(draft_caps)
        request_acceptance_base = min(
            1.0,
            max(0.0, acceptance_base * rng.uniform(0.85, 1.15)),
        )
        acceptance = tuple(
            min(
                1.0,
                max(
                    0.0,
                    request_acceptance_base
                    * (acceptance_decay**position)
                    * rng.uniform(0.97, 1.03),
                ),
            )
            for position in range(max_cap)
        )

        confidence_base = rng.uniform(0.76, 0.98)
        confidence = tuple(
            min(
                1.0,
                max(
                    0.0,
                    confidence_base
                    - 0.006 * position
                    + rng.uniform(-0.015, 0.015),
                ),
            )
            for position in range(max_cap)
        )

        requests.append(
            Request(
                request_id=f"req-{index:05d}",
                arrival_time_ms=arrival_time_ms,
                prompt_tokens=rng.choice((128, 512, 1024, 2048)),
                max_output_tokens=max_output_tokens,
                draft_cap=draft_cap,
                acceptance_probs=acceptance,
                confidence_by_position=confidence,
            )
        )

    return requests
