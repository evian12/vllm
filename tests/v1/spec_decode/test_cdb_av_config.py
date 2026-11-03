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
get_initial_admit_limit = _config.get_initial_admit_limit
get_k_stop_reason = _config.get_k_stop_reason
get_request_k_cap = _config.get_request_k_cap
get_window_h = _config.get_window_h
is_enabled = _config.is_enabled


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
