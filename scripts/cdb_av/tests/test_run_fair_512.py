# SPDX-License-Identifier: Apache-2.0
"""Offline checks for paired serving experiment validation."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.cdb_av.run_fair_512 import (
    parse_client_log,
    server_command,
    server_environment,
)


def test_parse_client_log_requires_unique_valid_completions(tmp_path):
    path = tmp_path / "run.log"
    rows = [
        {"index": 0, "ok": True, "output_tokens": 8, "elapsed": 1.0,
         "started_at": 10.0},
        {"index": 1, "ok": True, "output_tokens": 8, "elapsed": 1.5,
         "started_at": 10.1},
    ]
    path.write_text("\n".join(map(json.dumps, rows)) + "\nwall_time=2.000s\n")
    result = parse_client_log(path, 2)
    assert result["completed"] == 2
    assert result["output_tokens"] == 16
    assert result["output_tps"] == 8.0
    assert result["e2e_p50_s"] == 1.25

    rows[1]["index"] = 0
    path.write_text("\n".join(map(json.dumps, rows)) + "\nwall_time=2.000s\n")
    with pytest.raises(RuntimeError, match="duplicate"):
        parse_client_log(path, 2)

    rows[1]["index"] = 1
    rows[1]["ok"] = False
    path.write_text("\n".join(map(json.dumps, rows)) + "\nwall_time=2.000s\n")
    with pytest.raises(RuntimeError, match="failed"):
        parse_client_log(path, 2)


def test_modes_keep_server_limits_and_isolate_cdb_flags(monkeypatch):
    monkeypatch.setenv("CDB_AV_PERSISTENT", "stale")
    args = SimpleNamespace(
        source=Path("/tmp/source"), gpu=1, port=18084,
        target="/target", draft="/draft", horizon=8, kv_reuse=1, low_log=1,
        hf_threshold=0.8,
    )
    for mode in ("cdb", "native-spec", "ar"):
        args.mode = mode
        env = server_environment(args)
        command = server_command(args)
        assert env["CDB_AV_PERSISTENT"] == ("1" if mode == "cdb" else "0")
        assert env["CDB_AV_LOW_LOG"] == ("1" if mode == "cdb" else "0")
        assert env["CUDA_VISIBLE_DEVICES"] == "1"
        assert env["VLLM_ATTENTION_BACKEND"] == "FLASH_ATTN"
        assert env["CDB_AV_HF_THRESHOLD"] == (
            "0.8" if mode == "cdb" else "0.4"
        )
        assert command[command.index("--max-num-seqs") + 1] == "4"
        assert command[command.index("--attention-backend") + 1] == "FLASH_ATTN"
        assert ("--speculative-config" in command) == (mode != "ar")
