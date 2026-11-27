"""Exercise real Worker prefill state transitions with a CPU forward stub."""

import os
from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm.v1.spec_decode.cdb_av_fingerprint import token_fingerprint
from vllm.v1.spec_decode.cdb_av_kv_reuse import ReuseDraftSession


def test_chunked_prefill_carry_lineage_and_failure(monkeypatch):
    monkeypatch.setenv('CDB_AV_CHUNKED_DRAFT_PREFILL', '1')
    prefix = [i % 31 for i in range(3073)]
    blocks = list(range(1, 194))
    session = ReuseDraftSession.__new__(ReuseDraftSession)
    session.p = SimpleNamespace(
        device='cpu', vllm_config=SimpleNamespace(scheduler_config=SimpleNamespace(max_num_batched_tokens=1024)))
    session.capacity, session.block_size = 4, 16
    session.window, session.horizon, session.steps_completed = 1, 1, 0
    session.target_epoch, session.initialized, session.done = 0, True, False
    session.suspended, session.kv_snapshots, session.cached = {}, {}, {}
    session.pending_prefill = [(prefix, 0, len(prefix), blocks, 'r', 'reuse_full')]
    session.active, session.ready_records = [], []
    session._draft_stage_pending = None
    session.policy = 'fixed'
    member = dict(req_id='r', context_tokens=len(prefix), fingerprint=token_fingerprint(prefix),
                  cap=1, progress=0, next_token=prefix[-1], next_position=len(prefix)-1,
                  blocks=blocks, tokens=[], origin_window=1, ready=False, kv_inputs=[],
                  prefix=prefix, stale_kv_end=0, source='target_first')
    session.members = {'r': member}
    forwards = []
    session._metadata = lambda block_rows, seqs, qlens, slots: (seqs, qlens, slots)

    def forward(ids, positions, metadata, indices, req_ids, phase):
        seqs, qlens, slots = metadata
        assert len(ids) == qlens[0] <= 1024 and len(req_ids) == 1
        assert ids.tolist() == [prefix[pos] for pos in positions.tolist()]
        assert seqs == [int(positions[-1])+1]
        assert slots == [blocks[pos // 16] * 16 + pos % 16 for pos in positions.tolist()]
        forwards.append((len(ids), phase))
        return None if indices is None else torch.tensor([[0., 1.]])

    session._forward = forward
    request = SimpleNamespace(prompt_token_ids=prefix[:-1], output_token_ids=prefix[-1:])
    with patch('torch.cuda.synchronize'):
        with patch.object(session, '_forward', side_effect=RuntimeError('failed forward')):
            try:
                session.step()
            except RuntimeError as exc:
                assert str(exc) == 'failed forward'
            else:
                raise AssertionError('Forward failure was ignored')
        assert session.steps_completed == 0 and not member['kv_inputs']
        assert session.pending_prefill[0][1] == 0 and 'prefill_cursor' not in member
        for index in range(1, 4):
            status = session.step()
            assert status['prefill_pending'] and status['slots'] == 3
            assert status['active_req_ids'] == ['r'] and status['progress'] == {'r': 0}
            assert not member['tokens'] and member['kv_inputs'] == prefix[:1024*index]
            carry = session.boundary()['carry']
            assert carry['r']['prefill_cursor'] == 1024*index
            session.open_window(dict(window=index+1, H=1, target_epoch=index, capacity=4, max_k=1), {'r':request})
        assert session.step()['ready_req_ids'] == ['r']
        assert member['tokens'] == [1] and member['kv_inputs'] == prefix
        assert not session.pending_prefill and 'prefill_cursor' not in member
        assert session.boundary()['records'][0]['tokens'] == [1]
        assert session.drop(['r']) == {'dropped':['r']}
        assert not session.members and not session.cached
    assert forwards == [(1024, 'chunked_draft_prefill')]*3 + [(1, 'mixed_prefill')]
    print('3073 Draft inputs crossed three H boundaries without early sampling or a budget overrun')
