# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm.v1.spec_decode.cdb_av_config import (
    DEFAULT_WINDOW_H,
    get_initial_admit_limit,
    get_window_h,
    is_enabled,
)


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
