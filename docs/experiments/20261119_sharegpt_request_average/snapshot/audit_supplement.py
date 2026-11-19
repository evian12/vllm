"""Audit completed C1 collection without inventing an unlogged reset result."""
from collections import Counter


def audit_events(events, expected):
    owners, handed, verified, observed = {}, set(), set(), set()
    ready_count = release_count = 0
    for event in events:
        kind = event['event']
        rid = event.get('req_id')
        if kind == 'reuse_pending_prefill':
            blocks = event['blocks']
            assert len(blocks) == len(set(blocks)), 'Duplicate owned page'
            others = {b for owner, bs in owners.items() if owner != rid for b in bs}
            assert not others.intersection(blocks), 'Cross-request page ownership overlap'
            owners[rid] = blocks
        elif kind == 'ready':
            assert event['k_actual'] == event['k_cap'] == 8
            assert event['reason'] != 'low_confidence'
            ready_count += 1
        elif kind == 'p2_handoff':
            key = (rid, event['fingerprint'], event['window'])
            assert rid in owners, 'Handoff without owned context'
            assert key not in handed, 'Duplicate handoff'
            handed.add(key)
        elif kind == 'p2_verified':
            key = (rid, event['fingerprint'], event['window'])
            assert key in handed and key not in verified, 'Missing or duplicate verification'
            assert event['trace_key'] in expected
            assert len(event['drafts']) == 8 and 0 <= event['accepted'] <= 8
            verified.add(key)
            observed.add(event['trace_key'])
        elif kind == 'reuse_lease_release':
            assert rid in owners, 'Release without owner'
            assert owners.pop(rid) == event['blocks'], 'Released pages differ from last owned block table'
            release_count += 1
        elif kind == 'p2_boundary':
            assert not event['carry'] and event['leases'] == event['leased_pages'] == 0
    assert observed == set(expected), 'Incomplete request acceptance coverage'
    assert handed == verified and len(verified) == ready_count, 'Unclosed candidate lifecycle'
    assert not owners, 'Unreleased request lease ledger'
    resets = [x for x in events if x['event'] == 'p2_reset']
    for reset in resets:
        assert reset['live_leases'] == reset['ready'] == reset['inflight'] == reset['released_blocks'] == 0
    return dict(requests=len(expected), verified=len(verified), ready=ready_count,
                K_values=[8], release_events=release_count,
                cleanup=dict(outstanding_logged_owners=len(owners), unverified_handoffs=len(handed - verified),
                             reset=resets[-1] if resets else None,
                             reset_note='No reset event is inferred when absent; release ledger and final service metrics are checked separately.'),
                events=dict(Counter(x['event'] for x in events)))
