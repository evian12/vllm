# SPDX-License-Identifier: Apache-2.0

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


# Load this dependency-free config module directly. Importing the vllm package
# would also import torch and unrelated GPU runtime dependencies.
_CONFIG_PATH = (
    Path(__file__).resolve().parents[3]
    / "vllm"
    / "v1"
    / "spec_decode"
    / "cdb_av_config.py"
)
_CONFIG_SPEC = spec_from_file_location("cdb_av_config_under_test", _CONFIG_PATH)
assert _CONFIG_SPEC is not None and _CONFIG_SPEC.loader is not None
_config = module_from_spec(_CONFIG_SPEC)
_CONFIG_SPEC.loader.exec_module(_config)

DEFAULT_WINDOW_H = _config.DEFAULT_WINDOW_H
confidence_stop_flags = _config.confidence_stop_flags
get_frozen_request_ids = _config.get_frozen_request_ids
get_initial_admit_limit = _config.get_initial_admit_limit
get_k_stop_reason = _config.get_k_stop_reason
get_request_k_cap = _config.get_request_k_cap
get_window_h = _config.get_window_h
is_enabled = _config.is_enabled
select_ready_request_ids = _config.select_ready_request_ids
can_admit_prefill_tokens = _config.can_admit_prefill_tokens
can_verify_ready_tokens = _config.can_verify_ready_tokens
validate_draft_worker_status = _config.validate_draft_worker_status
validate_draft_open_ack = _config.validate_draft_open_ack
validate_draft_step_status = _config.validate_draft_step_status
validate_server_k_cap = _config.validate_server_k_cap
validate_worker_reset_ids = _config.validate_worker_reset_ids
get_expected_worker_request_ids = _config.get_expected_worker_request_ids
validate_verification_commit = _config.validate_verification_commit
get_token_only_input_unsupported_reason = _config.get_token_only_input_unsupported_reason
validate_exclusive_kv_ownership = _config.validate_exclusive_kv_ownership


def test_default_window_is_fixed_and_configurable(monkeypatch):
    monkeypatch.delenv("CDB_AV_WINDOW_H", raising=False)
    assert DEFAULT_WINDOW_H == 8
    assert get_window_h() == 8

    monkeypatch.setenv("CDB_AV_WINDOW_H", "16")
    assert get_window_h() == 16


@pytest.mark.parametrize("value", ["0", "-1", "65", "not-an-int", ""])
def test_invalid_window_h_is_rejected(monkeypatch, value):
    monkeypatch.setenv("CDB_AV_WINDOW_H", value)
    with pytest.raises(ValueError, match="CDB_AV_WINDOW_H"):
        get_window_h()


def test_initial_admit_defaults_to_available_capacity(monkeypatch):
    monkeypatch.delenv("CDB_AV_INITIAL_ADMIT", raising=False)
    assert get_initial_admit_limit(4) == 4


def test_initial_admit_can_be_limited(monkeypatch):
    monkeypatch.setenv("CDB_AV_INITIAL_ADMIT", "2")
    assert get_initial_admit_limit(4) == 2


@pytest.mark.parametrize("value", ["0", "-1", "5", "x"])
def test_invalid_initial_admit_is_rejected(monkeypatch, value):
    monkeypatch.setenv("CDB_AV_INITIAL_ADMIT", value)
    with pytest.raises(ValueError, match="CDB_AV_INITIAL_ADMIT"):
        get_initial_admit_limit(4)


def test_boolean_environment_is_strict(monkeypatch):
    monkeypatch.setenv("CDB_AV_INTRA_WINDOW", "1")
    assert is_enabled("CDB_AV_INTRA_WINDOW")
    monkeypatch.setenv("CDB_AV_INTRA_WINDOW", "0")
    assert not is_enabled("CDB_AV_INTRA_WINDOW")
    monkeypatch.setenv("CDB_AV_INTRA_WINDOW", "true")
    with pytest.raises(ValueError, match="CDB_AV_INTRA_WINDOW"):
        is_enabled("CDB_AV_INTRA_WINDOW")



@pytest.mark.parametrize(
    ("extra_args", "expected"),
    [
        (None, 8),
        ({}, 8),
        ({"cdb_draft_k": 8}, 8),
        ({"cdb_draft_k": 3}, 3),
    ],
)
def test_request_k_cap_is_an_upper_bound(extra_args, expected):
    assert get_request_k_cap(extra_args, 8) == expected


@pytest.mark.parametrize(
    "extra_args",
    [
        {"cdb_draft_k": 0},
        {"cdb_draft_k": 9},
        {"cdb_draft_k": True},
        {"cdb_draft_k": "8"},
    ],
)
def test_invalid_request_k_cap_is_rejected(extra_args):
    with pytest.raises(ValueError, match="cdb_draft_k"):
        get_request_k_cap(extra_args, 8)


def test_hf_confidence_stops_only_below_threshold():
    assert confidence_stop_flags([0.2, 0.399, 0.4, 0.8], 0.4) == [
        True,
        True,
        False,
        False,
    ]


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan")])
def test_invalid_hf_threshold_is_rejected(threshold):
    with pytest.raises(ValueError, match="CDB_AV_HF_THRESHOLD"):
        confidence_stop_flags([0.5], threshold)


@pytest.mark.parametrize("confidence", [-0.1, 1.1, float("nan")])
def test_invalid_draft_confidence_is_rejected(confidence):
    with pytest.raises(ValueError, match="Draft confidence"):
        confidence_stop_flags([confidence], 0.4)


def test_k_stop_reason_uses_confidence_until_max_cap():
    assert get_k_stop_reason(1, 8, False) is None
    assert get_k_stop_reason(3, 8, True) == "low_confidence"
    assert get_k_stop_reason(8, 8, False) == "max_k"
    # Reaching the cap is the primary reason if both conditions coincide.
    assert get_k_stop_reason(8, 8, True) == "max_k"



@pytest.mark.parametrize("client_k", [1, 2, 3, 4])
def test_hf_confidence_ignores_legacy_client_k_cycle(monkeypatch, client_k):
    monkeypatch.setenv("CDB_AV_K_POLICY", "hf_confidence")
    assert get_request_k_cap({"cdb_draft_k": client_k}, 8) == 8


def test_hf_confidence_uses_server_limit_without_client_override(monkeypatch):
    monkeypatch.setenv("CDB_AV_K_POLICY", "hf_confidence")
    assert get_request_k_cap(None, 8) == 8
    assert get_request_k_cap({}, 8) == 8


def test_manual_policy_preserves_explicit_request_cap(monkeypatch):
    monkeypatch.setenv("CDB_AV_K_POLICY", "manual")
    assert get_request_k_cap({"cdb_draft_k": 3}, 8) == 3


def test_invalid_k_policy_is_rejected(monkeypatch):
    monkeypatch.setenv("CDB_AV_K_POLICY", "cycle_1234")
    with pytest.raises(ValueError, match="CDB_AV_K_POLICY"):
        get_request_k_cap({}, 8)



def test_published_ready_is_not_frozen_by_a_later_draft_window():
    assert get_frozen_request_ids(
        {"carry", "current_window_member"},
        {"old_ready_1", "old_ready_2"},
        {"old_ready_1"},
    ) == {"carry", "current_window_member", "old_ready_2"}


def test_target_ready_selection_respects_running_capacity_and_queue_order():
    selected = select_ready_request_ids(
        running_ids=["carry", "r0", "ar0", "r1", "r2"],
        ready_ids=["r2", "r0", "r1"],
        protected_ids={"carry"},
        max_num_running_reqs=3,
    )
    # AR already occupies one Target slot, leaving two slots for Ready.
    # FIFO Ready order wins over the unrelated Scheduler running order.
    assert selected == ("r2", "r0")


def test_target_ready_selection_returns_empty_when_ar_uses_capacity():
    selected = select_ready_request_ids(
        running_ids=["carry", "ar0", "ar1", "r0"],
        ready_ids={"r0"},
        protected_ids={"carry"},
        max_num_running_reqs=2,
    )
    assert selected == ()


def test_target_ready_selection_rejects_invalid_capacity():
    with pytest.raises(ValueError, match="Target running capacity"):
        select_ready_request_ids(["r0"], {"r0"}, set(), -1)



@pytest.mark.parametrize(
    ("planned", "request_tokens", "limit", "maximum", "expected"),
    [
        (0, 8, 8, 8, True),
        (3, 5, 8, 8, True),
        (3, 6, 8, 8, False),
        (0, 9, 9, 8, False),
        (3, 5, None, 8, True),
    ],
)
def test_mixed_prefill_token_budget(
    planned, request_tokens, limit, maximum, expected
):
    assert can_admit_prefill_tokens(
        planned, request_tokens, limit, maximum
    ) is expected


def test_mixed_prefill_token_budget_rejects_negative_values():
    with pytest.raises(ValueError, match="request_tokens"):
        can_admit_prefill_tokens(0, -1, 8, 8)



def test_worker_status_matches_cpu_lease_partition():
    validate_draft_worker_status(
        lease_ids={"a", "b", "c"},
        active_ids=["a", "b"],
        ready_ids=["c"],
        progress_ids={"a", "b", "c"},
        capacity=4,
        slots=2,
    )


@pytest.mark.parametrize(
    ("active", "ready", "progress", "capacity", "slots", "message"),
    [
        (["a"], ["b"], {"a"}, 4, 3, "membership"),
        (["a"], ["a"], {"a"}, 4, 3, "both Active and Ready"),
        (["a", "b"], [], {"a", "b"}, 1, 0, "exceeds capacity"),
        (["a"], [], {"a"}, 4, 1, "free slots"),
        (["a", "a"], [], {"a"}, 4, 3, "duplicates"),
    ],
)
def test_worker_status_rejects_inconsistent_partition(
    active, ready, progress, capacity, slots, message
):
    with pytest.raises(ValueError, match=message):
        validate_draft_worker_status(
            {"a", "b"} if "b" in progress else {"a"},
            active,
            ready,
            progress,
            capacity,
            slots,
        )



@pytest.mark.parametrize("cap", [1, 4, 8])
def test_worker_plan_k_cap_within_server_limit(cap):
    assert validate_server_k_cap(cap, 8) == cap


@pytest.mark.parametrize("cap", [0, 9, True, "8"])
def test_worker_plan_k_cap_rejects_invalid_values(cap):
    with pytest.raises(ValueError, match="Draft K cap"):
        validate_server_k_cap(cap, 8)



def test_ready_selection_skips_head_that_cannot_fit_token_budget():
    selected = select_ready_request_ids(
        running_ids=["r0", "r1", "r2"],
        ready_ids=["r0", "r1", "r2"],
        protected_ids=set(),
        max_num_running_reqs=1,
        ready_token_costs={"r0": 9, "r1": 3, "r2": 2},
        token_budget=4,
    )
    assert selected == ("r1",)


def test_ready_selection_respects_aggregate_token_budget():
    selected = select_ready_request_ids(
        running_ids=["r0", "r1", "r2"],
        ready_ids=["r0", "r1", "r2"],
        protected_ids=set(),
        max_num_running_reqs=3,
        ready_token_costs={"r0": 3, "r1": 3, "r2": 2},
        token_budget=5,
    )
    assert selected == ("r0", "r2")



@pytest.mark.parametrize(
    ("k_cap", "target_token_budget", "expected"),
    [
        (8, 9, True),
        (8, 8, False),
        (1, 2, True),
        (1, 1, False),
    ],
)
def test_atomic_ready_verification_must_fit_alone(
    k_cap, target_token_budget, expected
):
    assert can_verify_ready_tokens(k_cap, target_token_budget) is expected


@pytest.mark.parametrize(
    ("k_cap", "target_token_budget"),
    [(0, 8), (1, 0), (-1, 8)],
)
def test_atomic_ready_verification_rejects_invalid_budgets(
    k_cap, target_token_budget
):
    with pytest.raises(ValueError):
        can_verify_ready_tokens(k_cap, target_token_budget)



def test_worker_reset_ack_must_match_cpu_owned_ids():
    validate_worker_reset_ids({"a", "b"}, ["b", "a"])


@pytest.mark.parametrize(
    ("expected", "reported", "message"),
    [
        ({"a", "b"}, ["a"], "missing"),
        ({"a"}, ["a", "orphan"], "unexpected"),
        ({"a"}, ["a", "a"], "duplicate"),
    ],
)
def test_worker_reset_ack_rejects_ownership_mismatch(
    expected, reported, message
):
    with pytest.raises(ValueError, match=message):
        validate_worker_reset_ids(expected, reported)



def test_cdb_kv_ownership_accepts_distinct_exclusive_pages():
    validate_exclusive_kv_ownership(
        [(1, 1, False), (2, 1, False), (3, 1, False)],
        {10, 11},
    )


@pytest.mark.parametrize(
    ("states", "formal", "message"),
    [
        ([(1, 1, False), (1, 1, False)], set(), "duplicate"),
        ([(1, 1, False)], {1}, "formal Target"),
        ([(1, 2, False)], set(), "exclusively owned"),
        ([(1, 1, True)], set(), "exclusively owned"),
    ],
)
def test_cdb_kv_ownership_rejects_overlap_and_shared_pages(
    states, formal, message
):
    with pytest.raises(ValueError, match=message):
        validate_exclusive_kv_ownership(states, formal)



def test_worker_open_ack_matches_window_capacity_and_k():
    validate_draft_open_ack(
        {
            "opened": True,
            "window": 3,
            "horizon": 8,
            "capacity": 4,
            "max_k": 8,
        },
        expected_window=3,
        horizon=8,
        capacity=4,
        max_k=8,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("opened", False),
        ("window", 2),
        ("horizon", 4),
        ("capacity", 2),
        ("max_k", 4),
    ],
)
def test_worker_open_ack_rejects_protocol_mismatch(field, value):
    ack = {
        "opened": True,
        "window": 3,
        "horizon": 8,
        "capacity": 4,
        "max_k": 8,
    }
    ack[field] = value
    with pytest.raises(ValueError, match=field):
        validate_draft_open_ack(ack, 3, 8, 4, 8)


def test_worker_draft_step_status_matches_fixed_h_window():
    validate_draft_step_status(
        {
            "window": 3,
            "steps_completed": 4,
            "remaining_steps": 4,
            "pending": True,
            "done": False,
        },
        expected_window=3,
        expected_step=4,
        horizon=8,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("window", 2),
        ("steps_completed", 3),
        ("remaining_steps", 5),
        ("pending", False),
        ("done", True),
    ],
)
def test_worker_draft_step_status_rejects_stale_or_incomplete_state(
    field, value
):
    status = {
        "window": 3,
        "steps_completed": 4,
        "remaining_steps": 4,
        "pending": True,
        "done": False,
    }
    status[field] = value
    with pytest.raises(ValueError, match=field):
        validate_draft_step_status(status, 3, 4, 8)



def test_aborted_request_ignores_late_verification_result():
    record = {"tokens": [11, 12], "window": 3}
    assert validate_verification_commit(
        record, True, [11, 12], [11, 99]
    ) is None


def test_finished_request_without_inflight_record_ignores_late_drafts():
    assert validate_verification_commit(
        None, True, [11, 12], [11, 99]
    ) is None


def test_live_request_without_inflight_record_rejects_drafts():
    with pytest.raises(ValueError, match="unexpected verification"):
        validate_verification_commit(
            None, False, [11, 12], [11, 99]
        )


def test_live_request_validates_accepted_prefix():
    record = {"tokens": [11, 12, 13]}
    assert validate_verification_commit(
        record, False, [11, 12, 13], [11, 12, 13, 99]
    ) == 3


@pytest.mark.parametrize(
    ("drafts", "generated"),
    [
        ([11, 12], [12, 99]),
        ([11, 12], [99, 11]),
        ([11, 12], []),
    ],
)
def test_live_request_rejects_invalid_verification_prefix(drafts, generated):
    with pytest.raises(ValueError, match="accepted-prefix"):
        validate_verification_commit(
            {"tokens": [11, 12]}, False, drafts, generated
        )



def test_token_only_input_accepts_text_request():
    assert get_token_only_input_unsupported_reason(
        [1, 2, 3], None, []
    ) is None


@pytest.mark.parametrize(
    ("prompt_token_ids", "prompt_embeds", "mm_features", "expected"),
    [
        (None, None, [], "missing_prompt_token_ids"),
        ([1, 2], object(), [], "prompt_embeds"),
        ([1, 2], None, [object()], "multimodal_inputs"),
    ],
)
def test_token_only_input_skips_unsupported_request_types(
    prompt_token_ids, prompt_embeds, mm_features, expected
):
    assert get_token_only_input_unsupported_reason(
        prompt_token_ids, prompt_embeds, mm_features
    ) == expected



def test_expected_worker_ids_excludes_boundary_ready_without_kv_reuse():
    assert get_expected_worker_request_ids(
        {"carry", "ready"},
        set(),
        {"ready"},
        kv_reuse=False,
    ) == {"carry"}


def test_expected_worker_ids_keeps_boundary_ready_with_kv_reuse():
    assert get_expected_worker_request_ids(
        {"carry", "ready"},
        set(),
        {"ready"},
        kv_reuse=True,
    ) == {"carry", "ready"}


def test_expected_worker_ids_includes_parked_cached_ready():
    assert get_expected_worker_request_ids(
        {"carry"},
        {"ready"},
        {"ready"},
        kv_reuse=True,
    ) == {"carry", "ready"}


@pytest.mark.parametrize('changes,expected', [
    ({}, True), ({'window_open': False}, False), ({'steps': 0}, False),
    ({'steps': 8}, False), ({'leased_ids': []}, False),
    ({'ready_ids': ['a']}, False), ({'inflight_ids': ['a']}, False),
    ({'waiting_count': 1}, False),
    ({'running_rows': [('a', False), ('b', True)]}, False),
    ({'running_rows': [('a', True), ('a', True)]}, False),
    ({'leased_ids': ['missing']}, False), ({'running_rows': []}, False),
])
def test_ar_only_target_stage_bypass(changes, expected):
    args = dict(window_open=True, steps=1, horizon=8, leased_ids=['a'],
                ready_ids=[], inflight_ids=[], waiting_count=0,
                running_rows=[('a', True), ('b', True)])
    args.update(changes)
    assert _config.can_bypass_ar_target(**args) is expected


@pytest.mark.parametrize('value,expected', [(None, 2), ('2', 2), ('4', 4)])
def test_ragged_graph_query_bound_config(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv('CDB_AV_DRAFT_RAGGED_MAX_QUERY', raising=False)
    else:
        monkeypatch.setenv('CDB_AV_DRAFT_RAGGED_MAX_QUERY', value)
    assert _config.get_ragged_graph_query_bound() == expected


@pytest.mark.parametrize('value', ['0', '1', '3', '5', '8', '2.0', 'bad'])
def test_ragged_graph_query_bound_rejects_unsupported(monkeypatch, value):
    monkeypatch.setenv('CDB_AV_DRAFT_RAGGED_MAX_QUERY', value)
    with pytest.raises(ValueError, match='2 or 4'):
        _config.get_ragged_graph_query_bound()


@pytest.mark.parametrize('name,module,aot,allowed', [
    ('FlashAttentionMetadataBuilder', 'vllm.v1.attention.backends.flash_attn', False, True),
    ('FlashAttentionMetadataBuilder', 'vllm.v1.attention.backends.flash_attn', True, False),
    ('FlashAttentionMetadataBuilder', 'vllm.v1.attention.backends.flash_attn', None, False),
    ('FlashInferMetadataBuilder', 'vllm.v1.attention.backends.flashinfer', False, False),
])
def test_query4_builder_scope(name, module, aot, allowed):
    builder = type(name, (), {'__module__': module})()
    builder.aot_schedule = aot
    if allowed:
        _config.validate_ragged_graph_backend(4, builder)
    else:
        with pytest.raises(ValueError, match='FA2'):
            _config.validate_ragged_graph_backend(4, builder)


def test_default_query2_builder_scope_unchanged():
    _config.validate_ragged_graph_backend(2, object())


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('persistent,refill', [(False, False), (False, True),
                                              (True, False), (True, True)])
def test_native_target_graph_probe_scope(enabled, persistent, refill):
    if enabled and not (persistent and refill):
        with pytest.raises(ValueError, match='persistent refill'):
            _config.validate_target_graph_probe(enabled, persistent=persistent,
                                                refill=refill)
    else:
        _config.validate_target_graph_probe(enabled, persistent=persistent,
                                            refill=refill)
