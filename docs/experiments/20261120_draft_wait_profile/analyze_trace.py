"""Summarize observed private Kineto events without calling gaps GPU idle."""
import argparse
from collections import Counter, defaultdict
import gzip
from itertools import chain
import json
import math
from pathlib import Path
import statistics


def union_us(intervals):
    ordered = sorted(intervals)
    total = 0.0
    left = right = None
    for start, end in ordered:
        if not (math.isfinite(start) and math.isfinite(end) and end >= start):
            raise ValueError('Invalid trace interval')
        if left is None:
            left, right = start, end
        elif start > right:
            total += right - left
            left, right = start, end
        else:
            right = max(right, end)
    return total + (right - left if left is not None else 0)


def correlation(event):
    args = event.get('args', {})
    for name in ['correlation', 'Correlation ID', 'correlation_id', 'correlationId']:
        if name in args:
            return str(args[name])
    return None


def scope_name(name):
    for prefix in ['target_schedule','draft_wait_refresh','draft_wait_reason','draft_wait_candidates','draft_wait_admit','engine_draft_advance','engine_reconcile','draft_graph_forward','target_dispatch','worker_rpc','target_input_update_states', 'target_input_prepare_input_ids', 'target_input_prepare_inputs', 'target_input_build_attention_metadata', 'target_input_calc_spec_decode_metadata', 'target_input_preprocess', 'draft_logits', 'target_logits', 'clear_slots', 'draft_metadata_pack',
                   'target_storage_audit', 'target_sampling', 'draft_forward',
                   'target_forward', 'draft_step_with_sampling', 'draft_prepare',
                   'kv_ownership_before_target', 'draft_plans']:
        if name.startswith('CDB::' + prefix):
            return prefix
    return 'unscoped'


def summarize(events):
    complete = [e for e in events if e.get('ph') == 'X' and 'dur' in e]
    if any(not math.isfinite(e['ts']) or not math.isfinite(e['dur']) or e['dur'] < 0 for e in complete):
        raise ValueError('Nonfinite or negative trace duration')
    kernels = [e for e in complete if e.get('cat') == 'kernel']
    memory = [e for e in complete if e.get('cat') in ('gpu_memcpy', 'gpu_memset')]
    apis = [e for e in complete if e.get('cat') in ('cuda_runtime', 'cuda_driver')]
    ranges = [e for e in complete if e.get('cat') == 'user_annotation'
              and e.get('name', '').startswith('CDB::')]
    by_thread = defaultdict(list)
    cpu_groups = defaultdict(list)
    for event in ranges:
        by_thread[(event.get('pid'), event.get('tid'))].append(event)
        cpu_groups[scope_name(event['name'])].append(event['dur'])
    api_map = defaultdict(list)
    for event in apis:
        if (key := correlation(event)) is not None:
            api_map[key].append(event)

    def attribution(event):
        matches = api_map.get(correlation(event), [])
        if len(matches) != 1:
            return 'unattributed', None
        api = matches[0]
        point = api['ts'] + api['dur'] / 2
        parents = [e for e in by_thread[(api.get('pid'), api.get('tid'))]
                   if e['ts'] <= point <= e['ts'] + e['dur']]
        if not parents:
            return 'unscoped', api
        parent = min(parents, key=lambda e: e['dur'])
        return scope_name(parent['name']), api

    gpu_groups = defaultdict(lambda: dict(kernel_count=0, kernel_sum_us=0.0, memory_count=0, memory_sum_us=0.0))
    graph_linked_kernels = 0
    kernel_names = Counter()
    kernel_times = Counter()
    tagged = chain((('kernel', e) for e in kernels), (('memory', e) for e in memory))
    for kind, event in tagged:
        group, api = attribution(event)
        gpu_groups[group][kind + '_count'] += 1
        gpu_groups[group][kind + '_sum_us'] += event['dur']
        if kind == 'kernel':
            graph_linked_kernels += api is not None and 'GraphLaunch' in api['name']
            kernel_names[event['name']] += 1
            kernel_times[event['name']] += event['dur']
    gpu = kernels + memory
    if gpu:
        start = min(e['ts'] for e in gpu)
        end = max(e['ts'] + e['dur'] for e in gpu)
        active = union_us([(e['ts'], e['ts'] + e['dur']) for e in gpu])
        span = end - start
    else:
        active = span = None
    return dict(
        gpu_kernel_count=len(kernels), gpu_memory_event_count=len(memory), cdb_ranges_count=len(ranges),
        ignored_gpu_annotation_ranges=sum(e.get('cat') == 'gpu_user_annotation'
                                         and e.get('name', '').startswith('CDB::')
                                         for e in complete),
        observed_gpu_activity_union_us=active, observed_gpu_span_us=span,
        uncovered_trace_gap_us=span - active if span is not None else None,
        cuda_graph_launches=sum('GraphLaunch' in e['name'] for e in apis),
        graph_linked_kernel_records=graph_linked_kernels,
        gpu_by_cpu_launch_scope=dict(gpu_groups),
        cpu_range_inclusive_us={name: dict(count=len(values), sum=sum(values), median=statistics.median(values), mean=statistics.mean(values)) for name, values in cpu_groups.items()},
        draft_live_shapes=dict(Counter(e['name'] for e in ranges if e['name'].startswith('CDB::draft_forward'))),
        target_live_shapes=dict(Counter(e['name'] for e in ranges if e['name'].startswith('CDB::target_forward'))),
        top_kernels=[dict(name=name, count=kernel_names[name], sum_us=elapsed) for name, elapsed in kernel_times.most_common(40)],
        cuda_api_inclusive_us={name: sum(e['dur'] for e in apis if e['name'] == name) for name in sorted({e['name'] for e in apis})},
        notes=[
            'Times are Chrome/Kineto microseconds. CPU ranges are inclusive wall spans; overlapping scopes must not be summed as CPU overhead.',
            'GPU scope attribution follows unique CUDA launch correlation to the innermost CDB CPU range. Missing or ambiguous links stay unattributed.',
            'Observed activity gaps are not proven physical GPU idle; trace/CUDA Graph coverage may be incomplete.',
            'Graph kernel sums and memory copies may overlap; union avoids double counting. Neither is DRAM throughput.',
            'D2H API wait includes preceding GPU work. It is not wholly removable copy overhead.',
            'Bounded profiling perturbs execution; these durations cannot replace paired unprofiled throughput.',
        ])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('trace', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    opener = gzip.open if args.trace.suffix == '.gz' else open
    with opener(args.trace, 'rt') as source:
        data = json.load(source)
    result = summarize(data['traceEvents'])
    result['trace'] = str(args.trace)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['top_kernels', 'cuda_api_inclusive_us']}, indent=2))
