# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest

from vllm.v1.request import RequestStatus
from vllm.v1.spec_decode.cdb_av_coalesce import can_defer_ready
from vllm.v1.spec_decode.cdb_av_persistent import PersistentDraftQueue


def config():
    return dict(window_open=True,steps=2,horizon=8,capacity=4,
        ready_ids=['a'],worker_ready_ids=['b','c','d'],leased_ids=['b','c','d'],
        draft_wait_ids=['e'],inflight_ids=[],waiting_count=0,
        running_rows=[(x,True) for x in 'abcde'])


@pytest.mark.parametrize('step',range(1,8))
def test_only_current_h_boundary_can_end_wait(step):
    cfg=config();cfg['steps']=step
    assert can_defer_ready(**cfg)
    cfg['steps']=8
    assert not can_defer_ready(**cfg)


@pytest.mark.parametrize('change',[
    dict(window_open=False),dict(steps=0),dict(steps=8),dict(horizon=7),
    dict(capacity=8),dict(waiting_count=1),dict(inflight_ids=['a']),
    dict(ready_ids=[]),dict(ready_ids=['a','e','f','g']),
    dict(worker_ready_ids=['b','c']),dict(worker_ready_ids=['a','b','c']),
    dict(leased_ids=['b','c']),dict(leased_ids=['b','c','d','stale']),
    dict(running_rows=[(x,x!='a') for x in 'abcde']),
    dict(running_rows=[(x,True) for x in 'abcdeX']),
    dict(running_rows=[(x,True) for x in 'abcdee']),
])
def test_prefill_unfinished_candidates_stale_state_and_capacity_never_wait(change):
    cfg=config();cfg.update(change)
    assert not can_defer_ready(**cfg)


def queue():
    q=PersistentDraftQueue.__new__(PersistentDraftQueue)
    q.ready_coalesce_enabled=True;q._p3_window_open=True;q._p3_steps=2
    q.horizon=8;q.ready={'a':dict(tokens=[10,11])};q.leases={x:[] for x in 'bcd'}
    q.draft_wait={'e':None};q.inflight={};q._worker_step_ready_ids=frozenset('bcd')
    q.s=SimpleNamespace(max_num_running_reqs=4,waiting=[],running=[])
    for rid in 'abcde':
        q.s.running.append(SimpleNamespace(request_id=rid,status=RequestStatus.RUNNING,
            is_finished=lambda:False,num_output_tokens=1,num_computed_tokens=2,
            num_tokens=3,spec_token_ids=[10,11] if rid=='a' else []))
    return q


def test_queue_validates_official_state_without_mutation():
    q=queue();before=(q.ready.copy(),q.leases.copy(),q.draft_wait.copy())
    assert q.can_defer_ready_target()
    assert before==(q.ready,q.leases,q.draft_wait)
    q.s.running[0].spec_token_ids=[99]
    assert not q.can_defer_ready_target()
    q.s.running[0].spec_token_ids=[10,11]
    q.s.running[1].num_computed_tokens=0
    assert not q.can_defer_ready_target()


def test_default_off_does_not_inspect_queue():
    q=PersistentDraftQueue.__new__(PersistentDraftQueue)
    assert not q.can_defer_ready_target()
