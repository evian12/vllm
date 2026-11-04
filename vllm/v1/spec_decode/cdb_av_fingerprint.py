# SPDX-License-Identifier: Apache-2.0
"""Deterministic token-prefix fingerprint for CDB-AV diagnostics."""

import hashlib
import struct
from collections.abc import Iterable


def token_fingerprint(token_ids: Iterable[int]) -> str:
    h = hashlib.blake2b(digest_size=16, person=b"CDB-AV-prefix")

    count = 0
    for token in token_ids:
        if not isinstance(token, int) or not 0 <= token < 2**32:
            raise ValueError(f"Invalid token ID: {token!r}")
        h.update(struct.pack("<I", token))
        count += 1

    # Include the number of tokens to define the encoding unambiguously.
    h.update(struct.pack("<Q", count))
    return h.hexdigest()
