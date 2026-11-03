# SPDX-License-Identifier: Apache-2.0
"""Validated, fixed-at-startup CDB-AV runtime configuration."""

import math
import os


DEFAULT_WINDOW_H = 8
MIN_WINDOW_H = 1
MAX_WINDOW_H = 64


def get_window_h() -> int:
    """Return the configured fixed Draft window horizon.

    H is a per-window upper bound, not an adaptive runtime control variable.
    It is read from the environment when the EngineCore creates its CDB queue.
    """
    raw = os.getenv("CDB_AV_WINDOW_H", str(DEFAULT_WINDOW_H))
    try:
        horizon = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"CDB_AV_WINDOW_H must be an integer in "
            f"[{MIN_WINDOW_H}, {MAX_WINDOW_H}], got {raw!r}"
        ) from exc
    if not MIN_WINDOW_H <= horizon <= MAX_WINDOW_H:
        raise ValueError(
            f"CDB_AV_WINDOW_H must be in "
            f"[{MIN_WINDOW_H}, {MAX_WINDOW_H}], got {horizon}"
        )
    return horizon


def is_enabled(name: str, default: bool = False) -> bool:
    """Parse a strict boolean environment variable."""
    fallback = "1" if default else "0"
    raw = os.getenv(name, fallback).strip()
    if raw not in ("0", "1"):
        raise ValueError(f"{name} must be '0' or '1', got {raw!r}")
    return raw == "1"


def get_initial_admit_limit(capacity: int) -> int:
    """Return the maximum number of new requests admitted at a window start."""
    if capacity < 1:
        raise ValueError(f"Invalid Draft capacity: {capacity}")
    raw = os.getenv("CDB_AV_INITIAL_ADMIT", str(capacity))
    try:
        limit = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"CDB_AV_INITIAL_ADMIT must be an integer in [1, {capacity}], "
            f"got {raw!r}"
        ) from exc
    if not 1 <= limit <= capacity:
        raise ValueError(
            f"CDB_AV_INITIAL_ADMIT must be in [1, {capacity}], got {limit}"
        )
    return limit



def get_request_k_cap(extra_args, max_k: int) -> int:
    """Resolve a request's maximum Draft length, not its actual generated K.

    In HF-confidence mode, confidence may stop a request before this cap.
    Omitting cdb_draft_k therefore means the full server-side max_k.
    """
    if type(max_k) is not int or max_k < 1:
        raise ValueError(f"Invalid server maximum Draft K: {max_k!r}")
    if extra_args is None:
        extra_args = {}
    if not hasattr(extra_args, "get"):
        raise ValueError("SamplingParams.extra_args must be a mapping")

    policy = os.getenv("CDB_AV_K_POLICY", "manual").strip().lower()
    if policy not in ("manual", "hf_confidence"):
        raise ValueError(
            "CDB_AV_K_POLICY must be 'manual' or 'hf_confidence', "
            f"got {policy!r}"
        )

    # HF-confidence mode owns the request-level stopping decision. A client
    # may still send stale cdb_draft_k values (e.g. the old 1/2/3/4 cycle);
    # ignore them and let confidence choose actual K up to the server limit.
    if policy == "hf_confidence":
        return max_k

    cap = extra_args.get("cdb_draft_k", max_k)
    if type(cap) is not int or not 1 <= cap <= max_k:
        raise ValueError(
            f"cdb_draft_k must be an integer in [1, {max_k}], got {cap!r}"
        )
    return cap


def confidence_stop_flags(confidences, threshold: float) -> list[bool]:
    """Stop only when selected-token confidence is strictly below threshold."""
    try:
        threshold = float(threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError("CDB_AV_HF_THRESHOLD must be a finite value in [0, 1]") from exc
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("CDB_AV_HF_THRESHOLD must be a finite value in [0, 1]")

    stop_flags = []
    for confidence in confidences:
        try:
            probability = float(confidence)
        except (TypeError, ValueError) as exc:
            raise ValueError("Draft confidence must be a finite probability") from exc
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(
                f"Draft confidence must be a finite probability, got {confidence!r}"
            )
        stop_flags.append(probability < threshold)
    return stop_flags


def get_k_stop_reason(
    actual_k: int, k_cap: int, low_confidence: bool
) -> str | None:
    """Return the request stop reason after retaining the current Draft token."""
    if (
        type(actual_k) is not int
        or type(k_cap) is not int
        or not 0 <= actual_k <= k_cap
    ):
        raise ValueError(
            f"Invalid Draft K progress: actual_k={actual_k!r}, k_cap={k_cap!r}"
        )
    if actual_k == k_cap:
        return "max_k"
    if low_confidence:
        return "low_confidence"
    return None



def get_frozen_request_ids(lease_ids, ready_ids, selected_ready_ids):
    """Return current Draft leases and published Ready work not selected by Target.

    Ready entries are published only after their originating window reaches H.
    They therefore remain eligible during later Draft windows; only unselected
    entries stay frozen for the current Target forward.
    """
    return set(lease_ids) | (set(ready_ids) - set(selected_ready_ids))


def select_ready_request_ids(
    running_ids,
    ready_ids,
    protected_ids,
    max_num_running_reqs: int,
    ready_token_costs=None,
    token_budget: int | None = None,
) -> tuple[str, ...]:
    """Select a FIFO Ready subset within Target request and token capacity."""
    if type(max_num_running_reqs) is not int or max_num_running_reqs < 0:
        raise ValueError(
            f"Invalid Target running capacity: {max_num_running_reqs!r}"
        )
    if token_budget is not None and (
        type(token_budget) is not int or token_budget < 0
    ):
        raise ValueError(f"Invalid Target token budget: {token_budget!r}")

    running_ids = list(running_ids)
    ready_order = list(ready_ids)
    ready_set = set(ready_order)
    protected_ids = set(protected_ids)
    running_set = set(running_ids)
    non_ready_running = running_set - ready_set - protected_ids
    available = max_num_running_reqs - len(non_ready_running)
    if available <= 0:
        return ()

    costs = ready_token_costs or {}
    selected = []
    seen = set()
    remaining_tokens = token_budget
    # The Ready dictionary preserves publication order across H boundaries.
    # Select FIFO from that queue, not from Scheduler's unrelated running order.
    # A candidate that cannot fit the current token budget is skipped so it
    # cannot head-of-line block a shorter Ready request behind it.
    for req_id in ready_order:
        if req_id not in running_set or req_id in seen:
            continue
        cost = costs.get(req_id, 1)
        if type(cost) is not int or cost < 1:
            raise ValueError(f"Invalid Ready token cost for {req_id}: {cost!r}")
        if remaining_tokens is not None and cost > remaining_tokens:
            continue
        selected.append(req_id)
        seen.add(req_id)
        if remaining_tokens is not None:
            remaining_tokens -= cost
        if len(selected) == available:
            break
    return tuple(selected)



def can_verify_ready_tokens(k_cap: int, target_token_budget: int) -> bool:
    """Return whether Target can verify K candidates plus the current input token.

    Speculative verification is atomic: one Target forward consumes the
    current input token and all K candidates. A request that cannot fit even
    when scheduled alone must remain on native Target-only generation.
    """
    if type(k_cap) is not int or k_cap < 1:
        raise ValueError(f"Invalid Ready K cap: {k_cap!r}")
    if type(target_token_budget) is not int or target_token_budget < 1:
        raise ValueError(
            f"Invalid Target token budget: {target_token_budget!r}"
        )
    return k_cap + 1 <= target_token_budget


def can_admit_prefill_tokens(
    planned_tokens: int,
    request_tokens: int,
    token_limit: int | None,
    max_token_budget: int,
) -> bool:
    """Check a mixed Draft forward's packed prefill/decode token budget."""
    values = {
        "planned_tokens": planned_tokens,
        "request_tokens": request_tokens,
        "max_token_budget": max_token_budget,
    }
    if token_limit is not None:
        values["token_limit"] = token_limit
    for name, value in values.items():
        if type(value) is not int or value < 0:
            raise ValueError(f"Invalid {name}: {value!r}")
    if request_tokens > max_token_budget:
        return False
    return token_limit is None or planned_tokens + request_tokens <= token_limit



def validate_exclusive_kv_ownership(
    block_states, formal_block_ids
) -> None:
    """Require every CDB-owned KV page to be unique and exclusively owned."""
    states = list(block_states)
    block_ids = [block_id for block_id, _, _ in states]
    if len(set(block_ids)) != len(block_ids):
        raise ValueError("CDB KV ownership contains duplicate block IDs")
    overlap = set(block_ids).intersection(formal_block_ids)
    if overlap:
        raise ValueError(
            f"CDB KV ownership overlaps formal Target blocks: {sorted(overlap)}"
        )
    for block_id, ref_cnt, is_null in states:
        if is_null or ref_cnt != 1:
            raise ValueError(
                "CDB KV block is not exclusively owned: "
                f"block={block_id}, ref_cnt={ref_cnt}, is_null={is_null}"
            )


def validate_worker_reset_ids(expected_ids, reported_ids) -> None:
    """Require Worker reset ACKs to cover exactly the CPU-owned Worker IDs."""
    expected = set(expected_ids)
    reported = list(reported_ids)
    reported_set = set(reported)
    if len(reported_set) != len(reported):
        raise ValueError("Worker reset acknowledgement contains duplicate IDs")
    if reported_set != expected:
        missing = sorted(expected - reported_set)
        unexpected = sorted(reported_set - expected)
        raise ValueError(
            "Worker reset membership mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )


def validate_draft_worker_status(
    lease_ids,
    active_ids,
    ready_ids,
    progress_ids,
    capacity: int,
    slots: int,
) -> None:
    """Validate that Worker Active/Ready state exactly matches CPU leases."""
    if type(capacity) is not int or capacity < 1:
        raise ValueError(f"Invalid Draft/Target capacity: {capacity!r}")
    if type(slots) is not int or slots < 0:
        raise ValueError(f"Invalid Worker Active slots: {slots!r}")

    leases = set(lease_ids)
    active = list(active_ids)
    ready = list(ready_ids)
    progress = set(progress_ids)
    active_set = set(active)
    ready_set = set(ready)

    if len(active_set) != len(active) or len(ready_set) != len(ready):
        raise ValueError("Worker Active/Ready IDs contain duplicates")
    if active_set.intersection(ready_set):
        raise ValueError("Worker request cannot be both Active and Ready")
    if active_set.union(ready_set) != leases or progress != leases:
        raise ValueError(
            "Worker Active/Ready/progress membership differs from CPU leases"
        )
    if len(active_set) > capacity:
        raise ValueError(
            f"Worker Active count {len(active_set)} exceeds capacity {capacity}"
        )
    expected_slots = capacity - len(active_set)
    if slots != expected_slots:
        raise ValueError(
            f"Worker reports {slots} free slots; expected {expected_slots}"
        )



def validate_server_k_cap(cap, max_k: int) -> int:
    """Validate a Worker plan's cap against the configured server maximum."""
    if type(max_k) is not int or max_k < 1:
        raise ValueError(f"Invalid server maximum Draft K: {max_k!r}")
    if type(cap) is not int or not 1 <= cap <= max_k:
        raise ValueError(
            f"Draft K cap must be an integer in [1, {max_k}], got {cap!r}"
        )
    return cap
