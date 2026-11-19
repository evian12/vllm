"""Formal paired HF K-cap intervals; functional gates excluded from TPS."""
from datetime import datetime
import json
from pathlib import Path
import re
import statistics

R = Path(__file__).resolve().parent


def metrics(path):
    result = {}
    for line in path.read_text().splitlines():
        if line and not line.startswith('#'):
            key, value = line.rsplit(' ', 1)
            result[key] = float(value)
    return result


def interior(path, start, end):
    buckets = {'CDB_AGG': [], 'CDB_TARGET_FULL_GRAPH_AGG': []}
    for line in path.read_text(errors='replace').splitlines():
        found = re.search(r'(\d\d-\d\d) (\d\d:\d\d:\d\d)', line)
        if not found:
            continue
        ts = datetime.strptime('2026-' + found[1] + ' ' + found[2], '%Y-%m-%d %H:%M:%S').timestamp()
        if not start <= ts <= end:
            continue
        for key in buckets:
            if '[' + key + '] ' in line:
                buckets[key].append(json.loads(line.split('[' + key + '] ', 1)[1]))
    result = {}
    for key, values in buckets.items():
        if len(values) < 2:
            continue
        before, after = values[0], values[-1]
        if key == 'CDB_TARGET_FULL_GRAPH_AGG':
            result['target_graph'] = {k: v - before.get(k, 0) for k, v in after.items()}
        else:
            for group in ['events', 'k_actual', 'k_sources', 'stop_reasons', 'token_counts', 'replay']:
                result[group] = {k: v - before.get(group, {}).get(k, 0) for k, v in after.get(group, {}).items()}
    return result


def token_sequences(path):
    records = [json.loads(x) for x in path.read_text().splitlines()]
    assert len({x['source_id'] for x in records}) == len(records)
    return {x['source_id']: [token for event in x['token_events'] for token in event['tokens']] for x in records}


runs = []
for directory in sorted(R.glob('pair*')):
    if not directory.is_dir() or not (directory / 'DONE.json').exists() or not (directory / 'released.json').exists():
        continue
    summary = json.loads((directory / 'summary.json').read_text())
    manifest = json.loads((directory / 'manifest.json').read_text())
    released = json.loads((directory / 'released.json').read_text())
    assert summary['completed'] == 512 and summary['failed'] == 0 and summary['output_tokens'] == 101003
    assert released['port_closed'] and not released['our_gpu_pids']
    assert not manifest['diagnostic'] and manifest['environment']['CDB_AV_LOW_LOG'] == '1'
    assert manifest['environment']['CDB_AV_CARRY_VALIDATE'] == '0'
    assert manifest['environment']['CDB_AV_VALIDATE_METADATA'] == '0'
    before = metrics(directory / 'metrics_before.prom')
    after = metrics(directory / 'metrics_after.prom')

    def delta(metric):
        return sum(value - before.get(key, 0) for key, value in after.items() if key.split('{', 1)[0] == 'vllm:' + metric)

    record = dict(directory=str(directory), mode=manifest['mode'], cap=manifest['k_cap'], seed=manifest['seed'], source_commit=manifest['source_commit'], summary=summary)
    for name, metric in [('verify_events', 'spec_decode_num_drafts_total'), ('candidates', 'spec_decode_num_draft_tokens_total'), ('accepted', 'spec_decode_num_accepted_tokens_total'), ('target_iterations', 'iteration_tokens_total_count'), ('preemptions', 'num_preemptions_total')]:
        record[name] = delta(metric)
    record['mean_verified_K'] = record['candidates'] / record['verify_events']
    record['acceptance_rate'] = record['accepted'] / record['candidates']
    record['verify_events_per_target_iteration'] = record['verify_events'] / record['target_iterations']
    old = {x['pid']: x for x in json.loads((directory / 'observer_before.json').read_text())}
    new = {x['pid']: x for x in json.loads((directory / 'observer_after.json').read_text())}
    names = {key for value in new.values() for key in value if (key.startswith('cdb_') or key == 'target_forward_calls') and not key.endswith('_max_ms')}
    observed = {key: sum(value.get(key, 0) - old.get(pid, {}).get(key, 0) for pid, value in new.items()) for key in names}
    assert observed['target_forward_calls'] == record['target_iterations']
    assert observed['cdb_target_schedules'] == record['target_iterations']
    assert observed['cdb_target_verify_rows'] == record['verify_events']
    assert observed['cdb_draft_dispatches'] > 0
    record['observed'] = observed
    record['mean_draft_request_rows'] = observed['cdb_draft_request_rows'] / observed['cdb_draft_dispatches']
    record['mean_draft_token_rows'] = observed['cdb_draft_token_rows'] / observed['cdb_draft_dispatches']
    waits = observed.get('cdb_ready_wait_count', 0)
    record['ready_to_schedule_mean_ms'] = observed.get('cdb_ready_wait_sum_ms', 0) / waits if waits else None
    bins = sorted((float(key.removeprefix('cdb_ready_wait_bucket_').removesuffix('_ms')), value) for key, value in observed.items() if key.startswith('cdb_ready_wait_bucket_'))
    assert sum(value for _, value in bins) == waits
    accumulated = 0
    record['ready_to_schedule_p95_upper_bound_ms'] = None
    for bound, count in bins:
        accumulated += count
        if waits and accumulated >= waits * .95:
            record['ready_to_schedule_p95_upper_bound_ms'] = bound if bound != float('inf') else '>1000'
            break
    record['interior'] = interior(directory / 'server.log', summary['first_dispatch_wall'], summary['last_completion_wall'])
    gpu = []
    for line in (directory / 'gpu.csv').read_text().splitlines():
        parts = line.split(',')
        try:
            ts = datetime.strptime(parts[0].strip(), '%Y/%m/%d %H:%M:%S.%f').timestamp()
            if summary['first_dispatch_wall'] <= ts <= summary['last_completion_wall']:
                gpu.append([float(x) for x in parts[1:]])
        except ValueError:
            continue
    record['gpu_mean'] = {key: statistics.mean(row[i] for row in gpu) for i, key in enumerate(['utilization_pct', 'memory_mib', 'power_w', 'temperature_c', 'sm_clock_mhz'])}
    runs.append(record)

groups = {}
for cap in [4, 8]:
    rows = [x for x in runs if x['cap'] == cap]
    if not rows:
        continue
    tps = [x['summary']['throughput_tok_s'] for x in rows]
    group = dict(n=len(rows), tps=tps, mean_tps=statistics.mean(tps), sd_tps=statistics.stdev(tps) if len(tps) > 1 else None, mean_run_e2e_p95=statistics.mean(x['summary']['latency_s']['e2e']['p95'] for x in rows), mean_run_ttft_p95=statistics.mean(x['summary']['latency_s']['ttft']['p95'] for x in rows))
    for key in ['verify_events', 'candidates', 'accepted', 'target_iterations']:
        group[key] = sum(x[key] for x in rows)
    group['acceptance_rate'] = group['accepted'] / group['candidates']
    group['mean_verified_K'] = group['candidates'] / group['verify_events']
    group['verify_events_per_target_iteration'] = group['verify_events'] / group['target_iterations']
    group['draft_dispatches'] = sum(x['observed']['cdb_draft_dispatches'] for x in rows)
    group['mean_draft_request_rows'] = sum(x['observed']['cdb_draft_request_rows'] for x in rows) / group['draft_dispatches']
    groups[str(cap)] = group

paired = []
for seed in [42, 43, 44]:
    base = next((x for x in runs if x['cap'] == 8 and x['seed'] == seed), None)
    arm = next((x for x in runs if x['cap'] == 4 and x['seed'] == seed), None)
    if not base or not arm:
        continue
    assert base['source_commit'] == arm['source_commit']
    assert base['summary']['plan_sha256'] == arm['summary']['plan_sha256']
    a = token_sequences(Path(arm['directory']) / 'measurement/requests.jsonl')
    b = token_sequences(Path(base['directory']) / 'measurement/requests.jsonl')
    assert a.keys() == b.keys()
    differences = []
    for source, tokens in a.items():
        if tokens == b[source]:
            continue
        lcp = next((i for i, (x, y) in enumerate(zip(tokens, b[source])) if x != y), min(len(tokens), len(b[source])))
        differences.append(dict(source_id=source, common_prefix=lcp, total_tokens=len(tokens)))
    paired.append(dict(seed=seed, tps_change_pct=100 * (arm['summary']['throughput_tok_s'] / base['summary']['throughput_tok_s'] - 1), e2e_p95_change_pct=100 * (arm['summary']['latency_s']['e2e']['p95'] / base['summary']['latency_s']['e2e']['p95'] - 1), exact_output_requests=512 - len(differences), output_differences=differences))

result = dict(runs=runs, groups=groups, paired=paired, notes=['Same HF0.8, H8 and low-level flags. Only server speculative cap changes 8 to 4; Target maximum query domain also changes 9 to 5, so this is a complete configuration comparison, not a pure algorithm-cost decomposition.', 'Diagnostic gates and warmup are excluded; no native baseline reruns.', 'Draft dispatch counters include logical Graph replays but not additional work internal to capture.', 'Interior snapshots exclude boundary tails.', 'Reported P95 summary is a mean of run percentiles, not a pooled percentile.'])
(R / 'ANALYSIS.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(dict(groups=groups, paired=[{k: v for k, v in x.items() if k != 'output_differences'} for x in paired]), indent=2))
