# SPDX-License-Identifier: Apache-2.0
import hashlib
import importlib.util
import random
import struct

import pytest

from vllm.v1.spec_decode import cdb_av_fingerprint as fp


def reference(values):
    data = b"".join(struct.pack("<I", x) for x in values)
    data += struct.pack("<Q", len(values))
    return hashlib.blake2b(data, digest_size=16, person=b"CDB-AV-prefix").hexdigest()


@pytest.mark.parametrize("n", [0, 1, 2, 63, 128, 255, 256, 257, 511, 512, 513, 1024, 4097])
@pytest.mark.parametrize("enabled", [False, True])
def test_digest_reference_and_chunk_boundaries(n, enabled, monkeypatch):
    rng = random.Random(42+n)
    values = [rng.randrange(2**32) for _ in range(n)]
    monkeypatch.setattr(fp, "BATCH_ENABLED", enabled)
    assert fp.token_fingerprint(values) == reference(values)
    assert fp.token_fingerprint(iter(values)) == reference(values)


@pytest.mark.parametrize("enabled", [False, True])
def test_boundaries_bool_and_no_stale_length_cache(enabled, monkeypatch):
    monkeypatch.setattr(fp, "BATCH_ENABLED", enabled)
    values = [0, True, False, 2**32-1]
    first = fp.token_fingerprint(values)
    assert first == reference(values)
    values[0] = 1
    assert fp.token_fingerprint(values) == reference(values) != first


@pytest.mark.parametrize("bad", [-1, 2**32, 1.5, "7", None])
@pytest.mark.parametrize("enabled", [False, True])
def test_invalid_token_keeps_early_iterator_failure(bad, enabled, monkeypatch):
    monkeypatch.setattr(fp, "BATCH_ENABLED", enabled)
    seen = []
    def tokens():
        for value in [2, 3, bad]:
            seen.append(value)
            yield value
        pytest.fail("Consumed beyond first invalid token")
    with pytest.raises(ValueError, match="Invalid token ID"):
        fp.token_fingerprint(tokens())
    assert len(seen) == 3


@pytest.mark.parametrize("enabled", [False, True])
def test_generator_error_propagates(enabled, monkeypatch):
    monkeypatch.setattr(fp, "BATCH_ENABLED", enabled)
    def tokens():
        yield from range(260)
        raise RuntimeError("source failed")
    with pytest.raises(RuntimeError, match="source failed"):
        fp.token_fingerprint(tokens())


def test_encoding_chunk_memory_is_bounded(monkeypatch):
    sizes = []
    actual = fp._CHUNK_STRUCT
    class Observe:
        def pack(self, *values):
            sizes.append(len(values))
            return actual.pack(*values)
    monkeypatch.setattr(fp, "BATCH_ENABLED", True)
    monkeypatch.setattr(fp, "_CHUNK_STRUCT", Observe())
    values = list(range(4097))
    assert fp.token_fingerprint(values) == reference(values)
    assert sizes == [256]*16


@pytest.mark.parametrize("value,expected", [("0", False), ("1", True), (" 1 ", True)])
def test_flag_is_fixed_at_module_startup(value, expected, monkeypatch):
    monkeypatch.setenv("CDB_AV_FINGERPRINT_BATCH", value)
    spec = importlib.util.spec_from_file_location("fp_startup_test", fp.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.BATCH_ENABLED is expected
    monkeypatch.setenv("CDB_AV_FINGERPRINT_BATCH", "0" if expected else "1")
    assert module.BATCH_ENABLED is expected


@pytest.mark.parametrize("value", ["", "2", "true"])
def test_invalid_startup_flag_fails(value, monkeypatch):
    monkeypatch.setenv("CDB_AV_FINGERPRINT_BATCH", value)
    spec = importlib.util.spec_from_file_location("fp_bad_flag_test", fp.__file__)
    with pytest.raises(ValueError, match="CDB_AV_FINGERPRINT_BATCH"):
        spec.loader.exec_module(importlib.util.module_from_spec(spec))
