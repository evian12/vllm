import json

from build_average import build


def test_only_formal_verifications_contribute_and_missing_is_not_zero(tmp_path):
    reference = tmp_path / 'reference'
    (reference / 'measurement').mkdir(parents=True)
    (reference / 'warmup').mkdir()
    measurement = [dict(trace_key='measurement:A', prompt_token_ids=[1, 2]),
                   dict(trace_key='measurement:B', prompt_token_ids=[3, 4])]
    warmup = [dict(trace_key='warmup:A', prompt_token_ids=[1, 2])]
    for name, rows in [('measurement', measurement), ('warmup', warmup)]:
        (reference / name / 'identities.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in rows))
    events = [dict(event='p2_verified', trace_key='measurement:A', req_id='formal-A', accepted=accepted, drafts=list(range(8))) for accepted in [1, 2]]
    events.append(dict(event='p2_verified', trace_key='warmup:A', req_id='warm-A', accepted=8, drafts=list(range(8))))
    (reference / 'server.log').write_text(''.join('[CDB_REFILL] ' + json.dumps(x) + '\n' for x in events))
    output = tmp_path / 'average.jsonl'
    summary = build(reference, output)
    records = [json.loads(x) for x in output.read_text().splitlines()]
    assert summary['observed_requests'] == 1
    assert summary['requests_without_verification'] == ['measurement:B']
    assert summary['observations'] == 2 and summary['accepted_draft_tokens'] == 3
    assert records[0]['chosen_k'] == 2 and records[0]['mean_accepted'] == 1.5
    assert records[1]['trace_key'] == 'warmup:A' and records[1]['phase_alias']
    assert records[1]['accepted_total'] == 3 and records[1]['verified_count'] == 2
    assert all(x['trace_key'] != 'measurement:B' for x in records)
