# SPDX-License-Identifier: Apache-2.0
"""Per-request historical Target acceptance mean for diagnostic K selection."""

import json
import math
from dataclasses import dataclass
from pathlib import Path


def mean_to_k(accepted_total: int, verified_count: int, max_k: int) -> int:
    """Use the ceiling of the observed mean; K=0 is not in this protocol."""
    if (type(accepted_total) is not int or type(verified_count) is not int
            or type(max_k) is not int or verified_count < 1 or max_k < 1
            or not 0 <= accepted_total <= verified_count * max_k):
        raise ValueError("Invalid historical acceptance totals")
    return max(1, min(max_k, math.ceil(accepted_total / verified_count)))


@dataclass(frozen=True)
class AverageDecision:
    cap: int
    source: str
    mean_accepted: float | None = None
    verified_count: int = 0


class RequestAverageK:
    def __init__(self, records: dict, max_k: int):
        self.records = records
        self.max_k = max_k

    @classmethod
    def from_jsonl(cls, path: str | Path, max_k: int):
        if type(max_k) is not int or max_k < 1:
            raise ValueError("Invalid server maximum K")
        records = {}
        for line_no, line in enumerate(Path(path).read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                key = row["trace_key"]
                prompt_fp = row["prompt_fingerprint"]
                accepted_total = row["accepted_total"]
                verified_count = row["verified_count"]
                chosen_k = row["chosen_k"]
                if (not isinstance(key, str) or not key
                        or not isinstance(prompt_fp, str) or len(prompt_fp) != 32
                        or any(c not in "0123456789abcdef" for c in prompt_fp)
                        or row["server_max_k"] != max_k
                        or type(chosen_k) is not int
                        or chosen_k != mean_to_k(
                            accepted_total, verified_count, max_k)):
                    raise ValueError("Invalid request-average observation")
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid request-average row {path}:{line_no}") from exc
            if key in records:
                raise ValueError(f"Duplicate request-average trace key: {key}")
            records[key] = (prompt_fp, chosen_k,
                            accepted_total / verified_count, verified_count)
        if not records:
            raise ValueError("Request-average trace contains no observations")
        return cls(records, max_k)

    def resolve(self, trace_key: str | None, prompt_fingerprint: str):
        if not trace_key or trace_key not in self.records:
            return AverageDecision(self.max_k, "missing_fallback")
        expected, cap, mean, count = self.records[trace_key]
        if expected != prompt_fingerprint:
            return AverageDecision(self.max_k, "prompt_mismatch_fallback")
        return AverageDecision(cap, "request_average", mean, count)
