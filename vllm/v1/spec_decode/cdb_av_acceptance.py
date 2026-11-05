# SPDX-License-Identifier: Apache-2.0
"""Hindsight Draft-length replay for diagnostic experiments.

A reference observation is usable only for the same logical request and exact
Target-confirmed token prefix. This changes Draft length, never Target commit.
Full acceptance at the reference K is right-censored, so replay never claims to
know whether a longer Draft would have succeeded.
"""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReplayDecision:
    cap: int
    source: str
    censored: bool = False


def conservative_replay_cap(accepted: int, observed_k: int, max_k: int) -> int:
    """Keep through the first rejection; trim only certainly wasted suffix.

    If all observed tokens were accepted, keep the observed K: the result is
    censored and says nothing about the next candidate. K=0 is unsupported by
    the current Draft protocol, hence a rejection at position zero uses K=1.
    """
    if (type(accepted) is not int or type(observed_k) is not int
            or type(max_k) is not int or max_k < 1
            or not 0 <= accepted <= observed_k <= max_k):
        raise ValueError("Invalid acceptance observation or server K")
    return observed_k if accepted == observed_k else max(1, accepted + 1)


def should_stop(policy: str, k_source: str, confidence_stop: bool) -> bool:
    """A matched replay uses its fixed cap; unmatched contexts use HF."""
    if policy == "replay_acceptance" and k_source == "replay":
        return False
    return confidence_stop


class AcceptanceReplay:
    def __init__(self, records: dict, ambiguous: set):
        self.records = records
        self.ambiguous = ambiguous

    @classmethod
    def from_jsonl(cls, path: str | Path, max_k: int):
        if type(max_k) is not int or max_k < 1:
            raise ValueError("Invalid replay maximum K")
        records, ambiguous = {}, set()
        path = Path(path)
        for line_no, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                key = row["trace_key"]
                fingerprint = row["fingerprint"]
                context_tokens = row["context_tokens"]
                observed_k = row["observed_k"]
                accepted = row["accepted"]
                if (not isinstance(key, str) or not key
                        or not isinstance(fingerprint, str)
                        or len(fingerprint) != 32
                        or any(c not in "0123456789abcdef" for c in fingerprint)
                        or type(context_tokens) is not int or context_tokens < 1):
                    raise ValueError("Invalid replay identity")
                cap = conservative_replay_cap(accepted, observed_k, max_k)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid replay record at {path}:{line_no}") from exc
            identity = (key, fingerprint, context_tokens)
            observation = (cap, accepted == observed_k)
            if identity in records and records[identity] != observation:
                ambiguous.add(identity)
            else:
                records[identity] = observation
        if not records:
            raise ValueError("Replay trace contains no observations")
        return cls(records, ambiguous)

    def resolve(self, trace_key, fingerprint, context_tokens, max_k: int):
        if type(max_k) is not int or max_k < 1:
            raise ValueError("Invalid server maximum K")
        if not isinstance(trace_key, str) or not trace_key:
            return ReplayDecision(max_k, "missing_key_fallback")
        identity = (trace_key, fingerprint, context_tokens)
        if identity in self.ambiguous:
            return ReplayDecision(max_k, "ambiguous_fallback")
        observation = self.records.get(identity)
        if observation is None:
            return ReplayDecision(max_k, "missing_fallback")
        cap, censored = observation
        if cap > max_k:
            return ReplayDecision(max_k, "cap_mismatch_fallback")
        return ReplayDecision(cap, "replay", censored)
