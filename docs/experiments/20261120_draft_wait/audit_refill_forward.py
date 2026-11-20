"""Link an actual Draft forward to an intra-window admission from the same log."""
import argparse
import json
from pathlib import Path


def audit(path):
    events = [json.loads(line.split('[CDB_REFILL] ', 1)[1])
              for line in path.read_text().splitlines() if '[CDB_REFILL] ' in line]
    steps = {(e['window'], e['step']): e for e in events if e['event'] == 'p2_draft_step'}
    forward = {}
    for event in events:
        if event['event'] == 'forward':
            key = event['window'], event['step']
            if event['phase'] in ('mixed_prefill', 'mixed_prefill_decode'):
                if key in forward:
                    raise RuntimeError(f'Multiple mixed Draft forwards for {key}')
                forward[key] = event
    admissions = [(e['window'], e['after_step'], rid)
                  for e in events if e['event'] == 'p2_intra_admit' and e['after_step'] > 0
                  for rid in e['req_ids']]
    if len(set(admissions)) != len(admissions):
        raise RuntimeError('Duplicate refill admission in diagnostic log')
    matches = []
    for window, previous, rid in admissions:
        parent = steps[(window, previous)]
        output = forward.get((window, previous + 1))
        if output is None or rid not in output['req_ids']:
            continue
        if rid not in parent['admitted'] or parent['slots'] < 1:
            raise RuntimeError(f'Invalid intra-window admission for {rid}')
        if output['graph_padding'] != 0 or output['forward_rows'] < len(output['req_ids']):
            raise RuntimeError(f'Unexpected Draft row accounting for {rid}')
        matches.append(dict(window=window, admitted_after=previous, req_id=rid,
                            preexisting_ready=len(parent['ready']),
                            survivor_rows=len(set(parent['active']) & set(output['req_ids'])),
                            forward_phase=output['phase'], actual_token_rows=output['forward_rows']))
    strict = [m for m in matches if m['preexisting_ready'] and m['survivor_rows']]
    return dict(log=str(path), positive_step_admissions=len(admissions),
                next_forward_same_window=len(matches),
                with_ready_and_surviving_active_rows=len(strict),
                distinct_windows=len({m['window'] for m in strict}),
                examples=strict[:6],
                scope='Request IDs were linked across scheduler and actual Draft forward events. Counts measure admission events, not unique client requests.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('server_log', type=Path)
    args = parser.parse_args()
    result = audit(args.server_log)
    if not result['with_ready_and_surviving_active_rows']:
        raise RuntimeError('Diagnostic lacks same-window refill evidence')
    print(json.dumps(result, indent=2))
