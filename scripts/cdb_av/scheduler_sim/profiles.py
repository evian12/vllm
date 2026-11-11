"""Execution-time profiles used by the discrete-event simulator."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from .model import Request


@dataclass(frozen=True, slots=True)
class ExecutionProfile:
    """Simple shape-aware latency model.

    Defaults are synthetic placeholders for algorithm development. They are
    deliberately not presented as RTX 4090 measurements. Replace them with
    coefficients fitted from local GPU profiles before making performance
    claims.
    """

    draft_base_ms: float = 0.60
    draft_per_request_ms: float = 0.22
    draft_per_context_token_ms: float = 0.000015
    verify_base_ms: float = 1.00
    verify_per_request_ms: float = 0.32
    verify_per_candidate_ms: float = 0.055
    verify_per_context_token_ms: float = 0.000012

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.draft_base_ms == 0 and self.draft_per_request_ms == 0:
            raise ValueError("Draft profile must have positive execution cost")
        if self.verify_base_ms == 0 and self.verify_per_request_ms == 0:
            raise ValueError("Verify profile must have positive execution cost")

    @staticmethod
    def _context_tokens(request: Request) -> int:
        return request.prompt_tokens + request.committed_tokens

    def draft_duration_ms(self, requests: Sequence[Request]) -> float:
        if not requests:
            raise ValueError("Draft duration requires a non-empty batch")
        context_tokens = sum(
            self._context_tokens(request) + request.draft_depth
            for request in requests
        )
        duration = (
            self.draft_base_ms
            + self.draft_per_request_ms * len(requests)
            + self.draft_per_context_token_ms * context_tokens
        )
        return max(duration, 1e-6)

    def verify_duration_ms(self, requests: Sequence[Request]) -> float:
        if not requests:
            raise ValueError("Verify duration requires a non-empty batch")
        context_tokens = sum(
            self._context_tokens(request) + request.draft_depth
            for request in requests
        )
        candidate_tokens = sum(request.draft_depth for request in requests)
        duration = (
            self.verify_base_ms
            + self.verify_per_request_ms * len(requests)
            + self.verify_per_candidate_ms * candidate_tokens
            + self.verify_per_context_token_ms * context_tokens
        )
        return max(duration, 1e-6)

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def load_execution_profile(path: str | Path) -> ExecutionProfile:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("profile JSON must contain an object")
    allowed = set(ExecutionProfile.__dataclass_fields__)
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"unknown profile fields: {sorted(unknown)}")
    return ExecutionProfile(**data)
