"""Complete missing request means from separate, explicitly labelled K8 runs."""
import argparse
import hashlib
import json
from pathlib import Path

from build_average import build_rows, RequestAverageK


def read_reference(directory):
    identities = (directory / 'measurement/identities.jsonl').read_text().splitlines()
    identity_map = {x['trace_key']: x for x in map(json.loads, identities)}
    assert len(identity_map) == len(identities), 'Duplicate request identity'
    warmup = directory / 'warmup/identities.jsonl'
    all_identities = identities + (warmup.read_text().splitlines() if warmup.exists() else [])
    with (directory / 'server.log').open(errors='replace') as log:
        rows, _ = build_rows(log, all_identities, 8)
    rows = [x for x in rows if x['trace_key'].startswith('measurement:')]
    assert len(rows) == len({x['trace_key'] for x in rows})
    assert all(x['trace_key'] in identity_map for x in rows)
    return identity_map, {x['trace_key']: x for x in rows}


def build_complete(reference, supplement, output):
    expected, original = read_reference(reference)
    extra_ids, extra = read_reference(supplement)
    missing = expected.keys() - original.keys()
    assert missing, 'Original collection already has complete coverage'
    assert extra_ids.keys() == missing, 'Supplement must contain exactly the unobserved requests'
    assert extra.keys() == missing, 'Supplement still lacks verified acceptance observations'
    for key, identity in extra_ids.items():
        assert identity['prompt_token_ids'] == expected[key]['prompt_token_ids'], 'Prompt identity changed'
    measured = []
    for key in sorted(expected):
        is_extra = key in extra
        row = dict((extra if is_extra else original)[key])
        row.update(reference_directory=str(supplement if is_extra else reference),
                   reference_collection='coverage_c1' if is_extra else 'original_c16')
        assert row['verified_count'] > 0
        assert 0 <= row['accepted_total'] <= 8 * row['verified_count']
        measured.append(row)
    records = list(measured)
    for row in measured:
        records.append(dict(row, trace_key='warmup:' + row['trace_key'].split(':', 1)[1],
                            reference_trace_key=row['trace_key'], phase_alias=True))
    assert not output.exists()
    output.write_text(''.join(json.dumps(x, separators=(',', ':')) + '\n' for x in records))
    RequestAverageK.from_jsonl(output, 8)
    summary = dict(measured_requests=len(expected), observed_requests=len(measured),
                   requests_without_verification=[], original_c16_observed=len(original),
                   coverage_c1_observed=len(extra), alias_records=len(records) - len(measured),
                   observations=sum(x['verified_count'] for x in measured),
                   accepted_draft_tokens=sum(x['accepted_total'] for x in measured),
                   k_distribution={str(k): sum(x['chosen_k'] == k for x in measured) for k in range(1, 9)},
                   references=[str(reference), str(supplement)],
                   sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                   definition='Per-request ceil(mean accepted Draft tokens) clamped [1,8]; bonus/correction excluded. Missing C16 observations are collected separately at C1 with original prompt/output budget and fixed K8. Collection provenance is retained; neither collection is a performance arm or a per-round optimal oracle.')
    output.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--supplement', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build_complete(args.reference, args.supplement, args.output), indent=2))
