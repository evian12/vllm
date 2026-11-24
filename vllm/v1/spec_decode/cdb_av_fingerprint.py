"""Deterministic token-prefix fingerprint for CDB-AV diagnostics."""

import hashlib
import os
import struct
from collections.abc import Iterable


def _legacy_token_fingerprint(token_ids: Iterable[int]) -> str:
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


# Bounded source encoding; do not cache fingerprints by length or request ID.
_CHUNK_SIZE = 256
_CHUNK_STRUCT = struct.Struct("<256I")
_mode = os.getenv("CDB_AV_FINGERPRINT_BATCH", "0").strip()
if _mode not in ("0", "1"):
    raise ValueError("CDB_AV_FINGERPRINT_BATCH must be '0' or '1'")
BATCH_ENABLED = _mode == "1"

_native_mode = os.getenv("CDB_AV_FINGERPRINT_NATIVE_PACK", "0").strip()
if _native_mode not in ("0", "1"):
    raise ValueError("CDB_AV_FINGERPRINT_NATIVE_PACK must be '0' or '1'")
_native_pack = None
if _native_mode == "1":
    if not BATCH_ENABLED:
        raise ValueError("Native fingerprint packing requires CDB_AV_FINGERPRINT_BATCH=1")
    from vllm.v1.spec_decode import _cdb_av_prefix_pack

    _native_pack = _cdb_av_prefix_pack


def _batched_token_fingerprint(token_ids: Iterable[int]) -> str:
    if _native_pack is not None:
        payload = _native_pack.encode(token_ids)
        if payload is not None:
            return hashlib.blake2b(
                payload, digest_size=16, person=b"CDB-AV-prefix"
            ).hexdigest()
    h = hashlib.blake2b(digest_size=16, person=b"CDB-AV-prefix")
    values = []
    count = 0
    for token in token_ids:
        if not isinstance(token, int) or not 0 <= token < 2**32:
            raise ValueError(f"Invalid token ID: {token!r}")
        values.append(token)
        count += 1
        if len(values) == _CHUNK_SIZE:
            h.update(_CHUNK_STRUCT.pack(*values))
            values.clear()
    if values:
        h.update(struct.pack(f"<{len(values)}I", *values))
    h.update(struct.pack("<Q", count))
    return h.hexdigest()


def token_fingerprint(token_ids: Iterable[int]) -> str:
    """The same uint32-LE token stream/count digest in either startup mode."""
    if BATCH_ENABLED:
        return _batched_token_fingerprint(token_ids)
    return _legacy_token_fingerprint(token_ids)
