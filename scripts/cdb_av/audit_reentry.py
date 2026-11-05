"""Audit formal-prefix growth between verification and Draft re-entry.

Consumes full CDB_REFILL traces, never aggregate counters or client labels.
Growth is a token-count association, not proof of wasted work or GPU timing.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path


def audit(lines):
    verified = {}
    counts = collections.Counter()
    lengths = collections.Counter()
    growth = collections.Counter()
    mixed = collections.Counter()
    for line in lines:
        if '[CDB_REFILL] ' not in line:
            continue
        try:
            event = json.loads(line.split('[CDB_REFILL] ', 1)[1])
            if not isinstance(event, dict):
                raise ValueError('event must be an object')
        except ValueError:
            counts['malformed_events'] += 1
            continue
        kind = event.get('event')
        if kind == 'p2_verified':
            rid = event.get('req_id')
            generated = event.get('generated')
            context = event.get('context_tokens')
            if (not isinstance(rid, str) or not isinstance(generated, list)
                    or not generated or type(context) is not int or context < 0):
                counts['invalid_verification_records'] += 1
                if isinstance(rid, str):
                    verified.pop(rid, None)
                continue
            verified[rid] = context + len(generated)
        elif kind == 'reuse_pending_prefill':
            rid = event.get('req_id')
            context = event.get('context_tokens')
            pending = event.get('pending_tokens')
            reused = event.get('reused')
            if (not isinstance(rid, str) or any(type(x) is not int for x in
                    (context, pending, reused)) or min(context, reused) < 0
                    or pending <= 0 or reused + pending != context):
                counts['invalid_preparation_records'] += 1
                continue
            counts['preparations'] += 1
            lengths[pending] += 1
            if reused == 0:
                counts['uncached_or_fallback'] += 1
                continue
            if rid not in verified:
                counts['cached_without_prior_verify_evidence'] += 1
                continue
            delta = context - verified[rid]
            if delta < 0:
                counts['negative_context_growth'] += 1
                continue
            counts['comparable_cached_preparations'] += 1
            growth[delta] += 1
            counts['formal_growth_tokens'] += delta
            counts['cached_pending_tokens'] += pending
            if delta > 0:
                counts['cached_with_formal_growth'] += 1
            if pending > 2:
                counts['query_gt2_with_growth' if delta > 0
                       else 'query_gt2_without_growth'] += 1
        elif kind == 'mixed_batch':
            q = event.get('query_lengths')
            total = event.get('total_tokens')
            if (not isinstance(q, list) or not q or any(
                    type(x) is not int or x <= 0 for x in q)
                    or sum(q) != total):
                counts['invalid_mixed_batch_records'] += 1
                continue
            mixed['q1' if max(q) == 1 else 'q2' if max(q) == 2
                  else 'q3plus'] += 1
    return dict(counts=dict(counts), pending_query_lengths=dict(sorted(lengths.items())),
                formal_growth_after_verify=dict(sorted(growth.items())),
                mixed_forward_query_classes=dict(mixed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    args = parser.parse_args()
    raw = args.log.read_bytes()
    result = audit(raw.decode(errors='replace').splitlines())
    result.update(log=str(args.log.resolve()), sha256=hashlib.sha256(raw).hexdigest())
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
