import json

import pytest

from build_complete import build_complete


def reference(path, identities, values):
    (path / 'measurement').mkdir(parents=True)
    (path / 'measurement/identities.jsonl').write_text(''.join(json.dumps(dict(trace_key='measurement:' + key, prompt_token_ids=prompt)) + '\n' for key, prompt in identities.items()))
    (path / 'warmup').mkdir()
    (path / 'warmup/identities.jsonl').write_text(json.dumps(dict(trace_key='warmup:A', prompt_token_ids=[1])) + '\n')
    events = [dict(event='p2_verified', trace_key='measurement:' + key, req_id=key, accepted=n, drafts=list(range(8))) for key, ns in values.items() for n in ns]
    events.append(dict(event='p2_verified', trace_key='warmup:A', req_id='warm-A', accepted=8, drafts=list(range(8))))
    (path / 'server.log').write_text(''.join('[CDB_REFILL] ' + json.dumps(x) + '\n' for x in events))
    return path


def test_complete_preserves_original_mean_and_real_zero(tmp_path):
    base = reference(tmp_path / 'base', {'A': [1], 'B': [2]}, {'A': [1, 2]})
    extra = reference(tmp_path / 'extra', {'B': [2]}, {'B': [0, 0]})
    output = tmp_path / 'complete.jsonl'
    summary = build_complete(base, extra, output)
    assert summary['observed_requests'] == 2
    assert not summary['requests_without_verification']
    assert summary['original_c16_observed'] == summary['coverage_c1_observed'] == 1
    assert summary['observations'] == 4 and summary['accepted_draft_tokens'] == 3
    rows = [json.loads(x) for x in output.read_text().splitlines()][:2]
    assert rows[0]['chosen_k'] == 2 and rows[0]['mean_accepted'] == 1.5
    assert rows[0]['reference_collection'] == 'original_c16'
    assert rows[1]['chosen_k'] == 1 and rows[1]['mean_accepted'] == 0
    assert rows[1]['verified_count'] == 2
    assert rows[1]['reference_collection'] == 'coverage_c1'


@pytest.mark.parametrize('identities,values,message', [
    ({'A': [1], 'B': [2]}, {'A': [8], 'B': [2]}, 'exactly'),
    ({'B': [99]}, {'B': [2]}, 'Prompt'),
    ({'B': [2]}, {}, 'lacks'),
])
def test_incomplete_or_wrong_supplement_is_rejected(tmp_path, identities, values, message):
    base = reference(tmp_path / 'base', {'A': [1], 'B': [2]}, {'A': [1, 2]})
    extra = reference(tmp_path / 'extra', identities, values)
    with pytest.raises(AssertionError, match=message):
        build_complete(base, extra, tmp_path / 'complete.jsonl')
