# SPDX-License-Identifier: Apache-2.0
"""Validated, fixed-at-startup CDB-AV runtime configuration."""

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
