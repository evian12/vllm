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
    running_ids, ready_ids, protected_ids, max_num_running_reqs: int
) -> tuple[str, ...]:
    """Select a queue-ordered Ready subset within Target running capacity."""
    if type(max_num_running_reqs) is not int or max_num_running_reqs < 0:
        raise ValueError(
            f"Invalid Target running capacity: {max_num_running_reqs!r}"
        )

    running_ids = list(running_ids)
    ready_ids = set(ready_ids)
    protected_ids = set(protected_ids)
    non_ready_running = set(running_ids) - ready_ids - protected_ids
    available = max_num_running_reqs - len(non_ready_running)
    if available <= 0:
        return ()

    selected = []
    seen = set()
    for req_id in running_ids:
        if req_id in ready_ids and req_id not in seen:
            selected.append(req_id)
            seen.add(req_id)
            if len(selected) == available:
                break
    return tuple(selected)
