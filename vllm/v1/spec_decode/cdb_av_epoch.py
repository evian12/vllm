# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CDB-AV committed-context epoch tracking.

Logical epochs only. Does not own GPU KV or perform verification.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DraftEpoch:
    req_id: str
    epoch: int
    committed_tokens: int


class DraftEpochRegistry:
    def __init__(self):
        self._epochs: dict[str, DraftEpoch] = {}

    def observe(self, req_id: str, committed_tokens: int) -> DraftEpoch:
        if committed_tokens < 0:
            raise ValueError("Negative committed token count")

        old = self._epochs.get(req_id)

        if old is None:
            current = DraftEpoch(req_id, 0, committed_tokens)
        elif committed_tokens < old.committed_tokens:
            raise RuntimeError(
                f"{req_id}: committed context unexpectedly shortened"
            )
        elif committed_tokens > old.committed_tokens:
            current = DraftEpoch(
                req_id, old.epoch + 1, committed_tokens
            )
        else:
            return old

        self._epochs[req_id] = current
        return current

    def valid(
        self,
        req_id: str,
        epoch: int,
        committed_tokens: int,
    ) -> bool:
        current = self._epochs.get(req_id)
        return (
            current is not None
            and current.epoch == epoch
            and current.committed_tokens == committed_tokens
        )

    def remove(self, req_id: str):
        self._epochs.pop(req_id, None)
