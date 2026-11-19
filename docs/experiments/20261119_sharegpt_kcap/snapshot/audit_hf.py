"""Read-only HF/Target joins; distinguish proven waste and bounded reference evidence."""
from collections import Counter
import json
from pathlib import Path

R = Path(__file__).resolve().parent
REF = Path('/home/yypan/CDB-AV/results/request_avg_sharegpt_gpu3_20261119/gate_reference_retry1')


def events(directory):
    with (directory / 'server.log').open() as source:
        return [json.loads(line.split('[CDB_REFILL] ', 1)[1]) for line in source if '[CDB_REFILL] ' in line]


def identities(directory):
    rows = [json.loads(x) for x in (directory / 'measurement/identities.jsonl').read_text().splitlines()]
    budgets = {x['source_id']: x['requested_tokens'] for x in map(json.loads, (directory / 'measurement/requests.jsonl').read_text().splitlines())}
    return {x['trace_key']: dict(prompt=x['prompt_token_ids'], budget=budgets[x['source_id']]) for x in rows}


reference = { (e['trace_key'], e['fingerprint'], e['context_tokens']): e
              for e in events(REF) if e['event'] == 'p2_verified' }
ref_ids = identities(REF)
results = {}
for cap in [4, 8]:
    directory = R / ('gate_cap' + str(cap))
    observed = events(directory)
    ids = identities(directory)
    assert all(ids[k] == ref_ids[k] for k in ids), 'Different prompt or output budget'
    ready = {(x['req_id'], x['fingerprint']): x for x in observed if x['event'] == 'ready'}
    verified = [x for x in observed if x['event'] == 'p2_verified']
    assert len(ready) == len(verified)
    counts = Counter()
    examples = {}
    for value in verified:
        row = ready[(value['req_id'], value['fingerprint'])]
        k, a = len(value['drafts']), value['accepted']
        assert row['tokens'] == value['drafts'] and row['k_actual'] == k
        assert 0 <= a <= k <= cap
        meta = ids[value['trace_key']]
        remaining = meta['budget'] - (value['context_tokens'] - len(meta['prompt']))
        assert remaining > 0
        sample = dict(trace_key=value['trace_key'], fingerprint=value['fingerprint'],
                      context_tokens=value['context_tokens'], remaining_output_tokens=remaining,
                      actual_K=k, accepted=a, stop_reason=row['reason'], confidence=row['confidence'])
        counts['verified'] += 1
        counts['draft_candidates'] += k
        counts['accepted'] += a
        counts['unaccepted_candidates'] += k - a
        # First rejection itself is not counted as an avoidable later proposal.
        extra = max(0, k - a - 1)
        counts['tokens_after_first_rejection'] += extra
        if extra:
            counts['rounds_continued_past_first_rejection'] += 1
            if remaining > k + 1:
                counts['nonterminal_rounds_continued_past_first_rejection'] += 1
            examples.setdefault('continued_past_first_rejection', sample)
        if k >= 2 and a == 0:
            counts['K_ge2_first_candidate_rejected'] += 1
            examples.setdefault('high_confidence_first_rejection', sample)
        low = row['reason'] == 'low_confidence'
        if low:
            assert row['confidence'] < row['threshold'] == .8
            counts['low_confidence_stops'] += 1
            if a == k:
                counts['low_confidence_stops_all_generated_candidates_accepted'] += 1
                examples.setdefault('low_confidence_all_accepted', sample)
        key = (value['trace_key'], value['fingerprint'], value['context_tokens'])
        ref = reference.get(key)
        if ref is None:
            counts['no_exact_reference_context'] += 1
            continue
        counts['exact_reference_context'] += 1
        if ref['drafts'][:k] != value['drafts']:
            counts['reference_draft_prefix_differs'] += 1
            continue
        if min(ref['accepted'], k) != a:
            counts['reference_acceptance_within_HF_span_differs'] += 1
            continue
        counts['reference_agrees_on_generated_span'] += 1
        if low and ref['accepted'] > k:
            gain = min(ref['accepted'] + 1, remaining) - min(k + 1, remaining)
            bounded_gain = min(min(ref['accepted'], cap) + 1, remaining) - min(k + 1, remaining)
            if bounded_gain > 0:
                counts['low_confidence_stops_with_longer_accepted_fixed8_reference'] += 1
                counts['additional_output_opportunity_in_aligned_reference'] += gain
                counts['additional_output_opportunity_bounded_by_current_cap'] += bounded_gain
                examples.setdefault('longer_aligned_reference', dict(sample, reference_accepted=ref['accepted'], additional_output=gain, current_cap_additional_output=bounded_gain))
            else:
                counts['longer_reference_but_output_budget_already_satisfied'] += 1
    results[str(cap)] = dict(counts=dict(counts), examples=examples)

output = dict(reference=str(REF), results=results, notes=[
    'Sixteen short GPU functional requests per cap, not the complete 512-request workload; no population-wide misclassification rate or TPS attribution is inferred.',
    'HF ready tokens are joined to actual Target verification using request ID and exact context fingerprint, with token identity assertions.',
    'K>=2 and first-candidate rejection proves HF continued after a high-confidence incorrect candidate: prior tokens had to pass the threshold to continue.',
    'All generated candidates accepted after low-confidence stop is only a possible early-stop signal. Without a longer reference its ungenerated suffix is unknown.',
    'Longer-reference evidence additionally requires the same trace key, complete official-context fingerprint/length, prompt/output budget, identical generated Draft prefix and agreement on acceptance within the HF span. Terminal output clipping is excluded.',
    'The fixed8 reference is a separate execution with different batch/query shapes. Aligned observations are conditional evidence, not proof of identical counterfactual execution or an end-to-end speedup.',
    'The current-cap opportunity clamps the reference accepted depth to the HF service cap, separating HF early stops from the maximum-K limit.',
    'Counts of extra request token rows do not equal saved GPU Forward calls because other active requests may still require those steps.',
])
(R / 'HF_MISMATCH_AUDIT.json').write_text(json.dumps(output, indent=2) + '\n')
print(json.dumps(output, indent=2))
