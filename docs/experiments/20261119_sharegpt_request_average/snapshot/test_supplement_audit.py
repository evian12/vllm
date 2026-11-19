import copy

import pytest

from audit_supplement import audit_events


def events():
    return [
        dict(event='reuse_pending_prefill', req_id='A', blocks=[1, 2]),
        dict(event='ready', req_id='A', k_actual=8, k_cap=8, reason='max_k'),
        dict(event='p2_handoff', req_id='A', fingerprint='fp', window=1),
        dict(event='p2_boundary', carry=[], leases=0, leased_pages=0),
        dict(event='p2_verified', req_id='A', fingerprint='fp', window=1, trace_key='measurement:A', drafts=list(range(8)), accepted=0),
        dict(event='reuse_lease_release', req_id='A', blocks=[1, 2]),
    ]


def test_no_reset_is_reported_as_unknown_not_zero():
    result = audit_events(events(), {'measurement:A'})
    assert result['cleanup']['reset'] is None
    assert result['cleanup']['outstanding_logged_owners'] == 0
    assert result['cleanup']['unverified_handoffs'] == 0
    assert result['verified'] == result['ready'] == result['release_events'] == 1


@pytest.mark.parametrize('case', ['missing_release', 'wrong_release', 'duplicate_verify', 'missing_handoff', 'collision', 'missing_request'])
def test_broken_lifecycle_is_rejected(case):
    rows = copy.deepcopy(events())
    expected = {'measurement:A'}
    if case == 'missing_release':
        rows.pop()
    elif case == 'wrong_release':
        rows[-1]['blocks'] = [2, 3]
    elif case == 'duplicate_verify':
        rows.insert(5, copy.deepcopy(rows[4]))
    elif case == 'missing_handoff':
        rows.pop(2)
    elif case == 'collision':
        rows.insert(1, dict(event='reuse_pending_prefill', req_id='B', blocks=[2, 3]))
    elif case == 'missing_request':
        expected.add('measurement:B')
    with pytest.raises(AssertionError):
        audit_events(rows, expected)
