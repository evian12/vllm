import pytest

from analyze_trace import summarize, union_us


def event(name, ts, dur, category='user_annotation', *, correlation=None):
    return dict(name=name, ph='X', ts=ts, dur=dur, cat=category, pid=1, tid=2,
                args={} if correlation is None else dict(correlation=correlation))


def test_union_does_not_double_count_nested_or_overlapping_activity():
    assert union_us([(0, 10), (2, 4), (9, 15), (20, 25)]) == 20
    assert union_us([]) == 0
    with pytest.raises(ValueError):
        union_us([(5, 4)])
    with pytest.raises(ValueError):
        union_us([(0, float('nan'))])


def test_gpu_kernel_uses_launch_correlation_not_gpu_timestamp_containment():
    rows = [event('CDB::draft_step_with_sampling', 0, 30),
            event('CDB::draft_forward::decode::r4::t4', 1, 15),
            event('CDB::draft_logits', 8, 5),
            event('cudaLaunchKernel', 9, 1, 'cuda_runtime', correlation=1),
            event('head_kernel', 40, 20, 'kernel', correlation=1)]
    result = summarize(rows)
    assert result['gpu_by_cpu_launch_scope']['draft_logits']['kernel_sum_us'] == 20
    assert result['observed_gpu_activity_union_us'] == 20


def test_missing_and_ambiguous_correlations_remain_unknown():
    rows = [event('cudaLaunchKernel', 1, 1, 'cuda_runtime', correlation=1),
            event('cudaLaunchKernel', 2, 1, 'cuda_runtime', correlation=1),
            event('kernel_a', 10, 5, 'kernel', correlation=1),
            event('kernel_b', 14, 6, 'kernel')]
    result = summarize(rows)
    assert result['gpu_by_cpu_launch_scope']['unattributed']['kernel_count'] == 2
    assert result['observed_gpu_activity_union_us'] == 10
    assert result['uncovered_trace_gap_us'] == 0


def test_no_gpu_events_are_not_reported_as_zero_gpu_cost():
    result = summarize([event('CDB::draft_plans', 1, 2)])
    assert result['gpu_kernel_count'] == 0
    assert result['observed_gpu_activity_union_us'] is None
    assert result['uncovered_trace_gap_us'] is None


def test_graph_launch_can_own_multiple_kernel_records():
    result = summarize([
        event('CDB::target_forward::t8', 0, 10),
        event('cudaGraphLaunch', 2, 1, 'cuda_runtime', correlation=5),
        event('kernel_a', 20, 5, 'kernel', correlation=5),
        event('kernel_b', 30, 5, 'kernel', correlation=5)])
    assert result['cuda_graph_launches'] == 1
    assert result['graph_linked_kernel_records'] == 2
    assert result['observed_gpu_activity_union_us'] == 10
    assert result['uncovered_trace_gap_us'] == 5


def test_gpu_mirror_annotation_is_not_a_second_cpu_forward():
    result = summarize([
        event('CDB::target_forward::t8', 0, 10),
        event('CDB::target_forward::t8', 20, 70, 'gpu_user_annotation'),
        event('cudaGraphLaunch', 2, 1, 'cuda_runtime', correlation=5),
        event('kernel', 25, 5, 'kernel', correlation=5)])
    assert result['cdb_ranges_count'] == 1
    assert result['ignored_gpu_annotation_ranges'] == 1
    assert result['cpu_range_inclusive_us']['target_forward']['count'] == 1
    assert result['cpu_range_inclusive_us']['target_forward']['sum'] == 10
    assert result['target_live_shapes'] == {'CDB::target_forward::t8': 1}
    assert result['gpu_by_cpu_launch_scope']['target_forward']['kernel_sum_us'] == 5
