"""Use real Target verification only; warmup never contributes to the mean."""
import argparse
import json
from pathlib import Path
import sys

W = Path('/home/yypan/CDB-AV/worktrees/cdb-request-avg-sharegpt-20261119')
sys.path.insert(0, str(W))
from scripts.cdb_av.build_request_average import build_rows
from vllm.v1.spec_decode.cdb_av_request_average import RequestAverageK


def build(directory, output):
    identities = []
    for name in ['warmup', 'measurement']:
        path = directory / name / 'identities.jsonl'
        if path.exists():
            identities.extend(path.read_text().splitlines())
    with (directory / 'server.log').open(errors='replace') as log:
        all_rows, _ = build_rows(log, identities, 8)
    measured = [x for x in all_rows if x['trace_key'].startswith('measurement:')]
    expected = {json.loads(x)['trace_key'] for x in (directory / 'measurement/identities.jsonl').read_text().splitlines()}
    assert measured and all(x['trace_key'] in expected for x in measured)
    assert len(measured) == len({x['trace_key'] for x in measured})
    assert not output.exists()
    # All evaluated phases use the mean from the formal reference run only.
    # Aliases do not add observations or change any historical totals.
    records = list(measured)
    for row in measured:
        records.append(dict(row, trace_key='warmup:' + row['trace_key'].split(':', 1)[1], reference_trace_key=row['trace_key'], phase_alias=True))
    output.write_text(''.join(json.dumps(x, separators=(',', ':')) + '\n' for x in records))
    RequestAverageK.from_jsonl(output, 8)
    missing = sorted(expected - {x['trace_key'] for x in measured})
    result = dict(reference_directory=str(directory), average_file=str(output), measured_requests=len(expected), observed_requests=len(measured), requests_without_verification=missing, alias_records=len(records) - len(measured), observations=sum(x['verified_count'] for x in measured), accepted_draft_tokens=sum(x['accepted_total'] for x in measured), k_distribution={str(k): sum(x['chosen_k'] == k for x in measured) for k in range(1, 9)}, definition='ceil(mean accepted Draft tokens), clamped to [1,8]; bonus/correction excluded; means use formal reference only; unobserved does not mean zero acceptance')
    output.with_suffix('.summary.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.reference, args.output), indent=2))
