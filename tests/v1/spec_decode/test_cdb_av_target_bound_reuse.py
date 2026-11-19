"""CPU-only cache-domain tests; GPU exactness is a separate mandatory gate."""
from dataclasses import replace

import pytest
import torch

from vllm.v1.spec_decode.cdb_av_target_graph import (
    bound_reuse_opt_in, clone_metadata, compatible_graph_key, update_metadata,
)
from tests.v1.spec_decode.test_cdb_av_target_graph import metadata


KEY = (8, 4, 9, 128, 64, 0, 'bounded')


def test_opt_in_defaults_off():
    assert not bound_reuse_opt_in({})
    assert not bound_reuse_opt_in({'CDB_AV_TARGET_GRAPH_REUSE_BOUNDS': '0'})
    assert bound_reuse_opt_in({'CDB_AV_TARGET_GRAPH_REUSE_BOUNDS': '1'})


@pytest.mark.parametrize('value', ['true', 'yes', '2', '', ' 1'])
def test_invalid_switch_is_rejected(value):
    with pytest.raises(ValueError, match='REUSE_BOUNDS must be 0 or 1'):
        bound_reuse_opt_in({'CDB_AV_TARGET_GRAPH_REUSE_BOUNDS': value})


@pytest.mark.parametrize('bounds,expected', [
    ([], None), ([64], None), ([64, 128, 512], 128),
    ([1024, 512, 256], 256), ([256], 256),
])
def test_smallest_existing_safe_bound_is_selected(bounds, expected):
    keys = [KEY[:3] + (bound,) + KEY[4:] for bound in bounds]
    original = list(keys)
    result = compatible_graph_key(KEY, keys)
    assert (None if result is None else result[3]) == expected
    assert keys == original  # Selection has no cache lifetime side effects.


@pytest.mark.parametrize('index,value', [
    (0, 9), (1, 3), (2, 1), (2, 8), (4, 65), (5, 1), (6, 'exact'),
])
def test_every_other_capture_domain_must_match(index, value):
    other = list(KEY)
    other[3] = 1024
    other[index] = value
    assert compatible_graph_key(KEY, [tuple(other)]) is None


def test_packed_gqa_query_one_never_shares_mixed_query_graph():
    query_one = (4, 4, 1, 128, 64, 0, 'bounded')
    mixed_query = (4, 4, 9, 1024, 64, 0, 'bounded')
    assert compatible_graph_key(query_one, [mixed_query]) is None


def test_larger_bound_refreshes_all_live_values_without_padding():
    initial = metadata((2, 2, 2, 2))
    static = clone_metadata(initial, 1024, query_bound=9)
    pointers = {f: getattr(static, f).data_ptr() for f in
                ('query_start_loc', 'seq_lens', 'block_table', 'slot_mapping')}
    live = metadata((1, 3, 1, 3))
    live.seq_lens.copy_(torch.tensor([65, 78, 91, 123], dtype=torch.int32))
    live.max_seq_len = 123
    live.block_table.add_(200)
    live.slot_mapping.add_(300)
    before = {f: getattr(live, f).clone() for f in pointers}
    update_metadata(static, live, mixed_bound=9)
    assert static.max_seq_len == 1024 and static.num_actual_tokens == 8
    for f in pointers:
        assert getattr(static, f).data_ptr() == pointers[f]
        assert torch.equal(getattr(static, f), before[f])
        assert torch.equal(getattr(live, f), before[f])


def test_existing_overflow_and_tensor_shape_checks_remain_strict():
    live = metadata((2, 2, 2, 2))
    static = clone_metadata(live, 128, query_bound=9)
    with pytest.raises(RuntimeError, match='scalar domain'):
        update_metadata(static, replace(live, max_seq_len=129), mixed_bound=9)
    with pytest.raises(RuntimeError, match='storage mismatch'):
        update_metadata(static, replace(live, block_table=torch.zeros((4, 5), dtype=torch.int32)), mixed_bound=9)
