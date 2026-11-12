# SPDX-License-Identifier: Apache-2.0
"""No-GPU bit/layout checks; actual FA2/Graph acceptance is a separate gate."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.spec_decode.cdb_av_metadata import packed_metadata_tensors
from vllm.v1.spec_decode.cdb_av_refill import RefillSession


CASES = [
    ([[3]], [1], [1], [48]),
    ([[7], [9, 10]], [4, 19], [2, 5], list(range(7))),
    ([[1, 2, 3], [8], [9, 12], [22, 23, 24]],
     [40, 8, 28, 48], [8, 1, 5, 8], list(range(22))),
    ([[30, 31], [40, 41]], [32, 31], [8, 8],
     [2**40 + i for i in range(16)]),
]


@pytest.mark.parametrize('blocks,seqs,qlens,slots', CASES)
def test_packed_fields_match_legacy_bit_for_bit(monkeypatch, blocks, seqs, qlens, slots):
    session = RefillSession.__new__(RefillSession)
    session.p = SimpleNamespace(device='cpu')
    inputs = deepcopy((blocks, seqs, qlens, slots))
    monkeypatch.setenv('CDB_AV_PACKED_METADATA', '0')
    original = session._metadata(blocks, seqs, qlens, slots)
    monkeypatch.setenv('CDB_AV_PACKED_METADATA', '1')
    packed = session._metadata(blocks, seqs, qlens, slots)
    for name in ('query_start_loc', 'query_start_loc_cpu', 'seq_lens',
                 'block_table_tensor', 'slot_mapping'):
        left, right = getattr(original, name), getattr(packed, name)
        assert left.dtype == right.dtype
        assert left.shape == right.shape
        assert left.stride() == right.stride()
        assert torch.equal(left.contiguous().view(torch.uint8),
                           right.contiguous().view(torch.uint8))
    for name in ('num_reqs', 'num_actual_tokens', 'max_query_len', 'max_seq_len',
                 'causal', '_seq_lens_cpu', '_num_computed_tokens_cpu'):
        assert getattr(original, name) == getattr(packed, name)
    assert (blocks, seqs, qlens, slots) == inputs


@pytest.mark.parametrize('blocks,seqs,qlens,slots', CASES)
def test_one_transfer_and_nonoverlapping_typed_views(monkeypatch, blocks, seqs, qlens, slots):
    calls = []
    original_to = torch.Tensor.to

    def record(tensor, *args, **kwargs):
        calls.append((tensor.dtype, tensor.numel(), args, kwargs))
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, 'to', record)
    fields = packed_metadata_tensors(blocks, seqs, qlens, slots, 'cpu')
    assert len(calls) == 1
    assert calls[0][0] == torch.int64
    names = ('query_start_loc', 'seq_lens', 'block_table_tensor', 'slot_mapping')
    assert len({fields[k].untyped_storage().data_ptr() for k in names}) == 1
    assert all(fields[k].data_ptr() % 16 == 0 for k in names)
    before = {k: fields[k].clone() for k in names if k != 'seq_lens'}
    cpu_query = fields['query_start_loc_cpu'].clone()
    fields['seq_lens'].fill_(111)
    for name, value in before.items():
        assert torch.equal(fields[name], value)
    assert torch.equal(fields['query_start_loc_cpu'], cpu_query)
    assert fields['slot_mapping'].dtype == torch.int64
    assert fields['slot_mapping'].tolist() == slots


@pytest.mark.parametrize('blocks,seqs,qlens,slots', CASES)
def test_calls_own_independent_metadata_lifetimes(blocks, seqs, qlens, slots):
    first = packed_metadata_tensors(blocks, seqs, qlens, slots, 'cpu')
    second = packed_metadata_tensors(blocks, seqs, qlens, slots, 'cpu')
    assert first['query_start_loc'].data_ptr() != second['query_start_loc'].data_ptr()
    assert first['query_start_loc_cpu'].data_ptr() != second['query_start_loc_cpu'].data_ptr()
    saved = {k: t.clone() for k, t in first.items()}
    for t in second.values():
        t.fill_(0)
    for k, t in first.items():
        assert torch.equal(t, saved[k])


@pytest.mark.parametrize('args', [
    ([], [], [], []),
    ([[1]], [], [1], [16]),
    ([[1]], [1], [], [16]),
    ([[]], [1], [1], [16]),
    ([[1]], [1], [0], []),
    ([[1]], [1], [2], [16]),
])
def test_invalid_counts_fail_before_copy(args):
    with pytest.raises(ValueError):
        packed_metadata_tensors(*args, 'cpu')


def test_int32_overflow_is_not_silently_truncated():
    with pytest.raises(RuntimeError):
        packed_metadata_tensors([[2**40]], [1], [1], [2**40], 'cpu')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='isolated GPU transfer gate')
@pytest.mark.parametrize('blocks,seqs,qlens,slots', CASES)
def test_gpu_packed_fields_match_legacy_bytes(monkeypatch, blocks, seqs, qlens, slots):
    session = RefillSession.__new__(RefillSession)
    session.p = SimpleNamespace(device='cuda:0')
    monkeypatch.setenv('CDB_AV_PACKED_METADATA', '0')
    original = session._metadata(blocks, seqs, qlens, slots)
    monkeypatch.setenv('CDB_AV_PACKED_METADATA', '1')
    packed = session._metadata(blocks, seqs, qlens, slots)
    for name in ('query_start_loc', 'seq_lens', 'block_table_tensor', 'slot_mapping'):
        left, right = getattr(original, name), getattr(packed, name)
        assert left.shape == right.shape
        assert left.stride() == right.stride()
        assert left.dtype == right.dtype
        assert left.device == right.device
        assert torch.equal(left.contiguous().view(torch.uint8),
                           right.contiguous().view(torch.uint8))
    assert torch.equal(original.query_start_loc_cpu, packed.query_start_loc_cpu)
    torch.cuda.synchronize()


def test_packed_runtime_evidence_records_actual_factory_once(monkeypatch):
    import json
    from vllm.v1.spec_decode import cdb_av_refill
    records = []
    monkeypatch.setenv('CDB_AV_PACKED_METADATA', '1')
    monkeypatch.setattr(cdb_av_refill.logger, 'info',
                        lambda fmt, raw: records.append((fmt, json.loads(raw))))
    session = RefillSession.__new__(RefillSession)
    session.p = SimpleNamespace(device='cpu')
    session._metadata([[1]], [1], [1], [16])
    session._metadata([[2]], [2], [1], [33])
    assert len(records) == 1
    marker, evidence = records[0]
    assert marker == '[CDB_METADATA_RUNTIME] %s'
    assert evidence['caller_source'] == cdb_av_refill.__file__
    assert evidence['factory_source'] == packed_metadata_tensors.__code__.co_filename
    assert evidence['pid'] > 0
    assert evidence['device'] == 'cpu'
    assert evidence['single_transfer'] is True
