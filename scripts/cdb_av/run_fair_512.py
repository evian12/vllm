# SPDX-License-Identifier: Apache-2.0
"""Run a paired CDB/native 512-request serving experiment from a clean worktree.

Each invocation owns one new service process group and a unique results directory.
Run once per mode (cdb, native-spec, ar) with the same CLI arguments.
"""

import argparse
import hashlib
import json
import os
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def percentile(values: list[float], percent: float) -> float:
    values = sorted(values)
    index = (len(values) - 1) * percent / 100
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def parse_client_log(path: Path, count: int) -> dict:
    rows = []
    lines = path.read_text().splitlines()
    for line in lines:
        if line.startswith("{"):
            rows.append(json.loads(line))
    if len(rows) != count:
        raise RuntimeError(f"Expected {count} client records, found {len(rows)}")
    if any(not row.get("ok") for row in rows):
        failures = [row for row in rows if not row.get("ok")]
        raise RuntimeError(f"{len(failures)} requests failed: {failures[:3]}")
    if any(row.get("output_tokens", 0) <= 0 for row in rows):
        raise RuntimeError("A successful HTTP response had no output tokens")
    expected_indices = set(range(count))
    if {row["index"] for row in rows} != expected_indices:
        raise RuntimeError("Client records have missing or duplicate request indices")
    wall_line = next(
        (line for line in lines if line.startswith("wall_time=")), None
    )
    if wall_line is None:
        raise RuntimeError("Client wall time missing")
    wall = float(wall_line.split("=", 1)[1].removesuffix("s"))
    total_tokens = sum(row["output_tokens"] for row in rows)
    latencies = [float(row["elapsed"]) for row in rows]
    return {
        "completed": len(rows),
        "failed": 0,
        "output_tokens": total_tokens,
        "wall_time_s": wall,
        "output_tps": total_tokens / wall,
        "request_tps": len(rows) / wall,
        "e2e_p50_s": percentile(latencies, 50),
        "e2e_p95_s": percentile(latencies, 95),
        "e2e_p99_s": percentile(latencies, 99),
        "first_request_started_at": min(row["started_at"] for row in rows),
        "last_request_started_at": max(row["started_at"] for row in rows),
    }


def git_output(source: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(source), *args], text=True
    ).strip()


def port_is_free(host: str, port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(1)
        return sock.connect_ex((host, port)) != 0


def gpu_used_mib(gpu: int) -> int:
    result = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={gpu}",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return int(result.strip())


def save_metrics(port: int, path: Path) -> None:
    """Snapshot vLLM histograms between client runs, outside measured wall time."""
    with urllib.request.urlopen(
        f"http://127.0.0.1:{port}/metrics", timeout=5
    ) as response:
        path.write_bytes(response.read())


def server_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        args.target,
        "--served-model-name",
        "qwen3-8b",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--tensor-parallel-size",
        "1",
        "--max-num-seqs",
        "4",
        "--max-model-len",
        "1024",
        "--gpu-memory-utilization",
        "0.90",
        "--enforce-eager",
        "--no-async-scheduling",
        "--no-enable-prefix-caching",
        "--seed",
        "42",
    ]
    if args.mode != "ar":
        command += [
            "--speculative-config",
            json.dumps(
                {
                    "method": "draft_model",
                    "model": args.draft,
                    "num_speculative_tokens": 8,
                }
            ),
        ]
    return command


def server_environment(args: argparse.Namespace) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if not key.startswith("CDB_")}
    env.update(
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        PYTHONPATH=str(args.source),
        VLLM_ATTENTION_BACKEND="FLASH_ATTN",
        CDB_AV_REFILL="1" if args.mode == "cdb" else "0",
        CDB_AV_PERSISTENT="1" if args.mode == "cdb" else "0",
        CDB_AV_KV_REUSE=str(args.kv_reuse if args.mode == "cdb" else 0),
        CDB_AV_INTRA_WINDOW="1" if args.mode == "cdb" else "0",
        CDB_AV_LOW_LOG=str(args.low_log if args.mode == "cdb" else 0),
        CDB_AV_K_POLICY="hf_confidence",
        CDB_AV_HF_THRESHOLD="0.4",
        CDB_AV_WINDOW_H=str(args.horizon),
        CDB_AV_STEP_MODE="1" if args.mode == "cdb" else "0",
        CDB_AV_VARLEN="1" if args.mode == "cdb" else "0",
        CDB_AV_SPLIT_RPC="1" if args.mode == "cdb" else "0",
        CDB_AV_WORK_QUEUE="1" if args.mode == "cdb" else "0",
        CDB_AV_INITIAL_ADMIT="3",
        CDB_AV_ALLOW_FRESH_WAITING="0",
        CDB_P1_VIRTUAL_H="0",
        CDB_AV_TRACE_STEPS="0",
        CDB_AV_REFILL_TRACE="0",
        CDB_AV_REUSE_REPLAY="0",
        CDB_AV_REUSE_VALIDATE="0",
        CDB_AV_VALIDATE_METADATA="0",
    )
    return env


def wait_healthy(proc: subprocess.Popen, port: int) -> None:
    for _ in range(120):
        if proc.poll() is not None:
            raise RuntimeError(f"Server exited during startup: {proc.returncode}")
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=2
            ) as response:
                if response.status == 200:
                    return
        except Exception:
            pass
        time.sleep(5)
    raise TimeoutError("Server health check timed out")


def run_client(args: argparse.Namespace, path: Path, run_id: str, count: int) -> dict:
    command = [
        sys.executable,
        str(args.source / "scripts/cdb_av/stress_waiting_512.py"),
        "--port",
        str(args.port),
        "--requests",
        str(count),
        "--concurrency",
        str(args.concurrency),
        "--max-tokens",
        str(args.max_tokens),
        "--run-id",
        run_id,
    ]
    with path.open("w") as output:
        status = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT).returncode
    if status:
        raise RuntimeError(f"Client failed with exit code {status}: {path}")
    return parse_client_log(path, count)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cdb", "native-spec", "ar"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--port", type=int, default=18084)
    parser.add_argument("--requests", type=int, default=512)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--horizon", type=int, choices=(4, 8), default=8)
    parser.add_argument("--kv-reuse", type=int, choices=(0, 1), default=1)
    parser.add_argument("--low-log", type=int, choices=(0, 1), default=0)
    parser.add_argument("--target", default="/disk14T/yypan/models/Qwen3-8B")
    parser.add_argument("--draft", default="/disk14T/yypan/models/Qwen3-1.7B")
    args = parser.parse_args()
    args.source = args.source.resolve()
    if min(args.requests, args.concurrency, args.max_tokens, args.repeats) < 1:
        parser.error("request, concurrency, token and repeat counts must be positive")
    if git_output(args.source, "status", "--porcelain"):
        raise RuntimeError("Source worktree must be clean")
    if not port_is_free("127.0.0.1", args.port):
        raise RuntimeError(f"Port {args.port} is occupied; no service was changed")
    if gpu_used_mib(args.gpu) > 2048:
        raise RuntimeError(f"GPU {args.gpu} is occupied; no service was changed")
    if args.result_dir.exists():
        raise FileExistsError(args.result_dir)
    if not (args.source / "vllm/_C.abi3.so").exists():
        raise RuntimeError("Compiled vLLM extension is missing from source worktree")
    if not (args.source / "vllm/vllm_flash_attn/flash_attn_interface.py").exists():
        raise RuntimeError("FLASH_ATTN extension interface is missing")

    env = server_environment(args)
    command = server_command(args)
    imported = subprocess.check_output(
        [sys.executable, "-c", "import vllm; print(vllm.__file__)"],
        cwd=args.source,
        env=env,
        text=True,
    ).strip()
    if not Path(imported).is_relative_to(args.source):
        raise RuntimeError(f"Runtime imports the wrong source: {imported}")

    args.result_dir.mkdir(parents=True)
    client_path = args.source / "scripts/cdb_av/stress_waiting_512.py"
    manifest = {
        "mode": args.mode,
        "source": str(args.source),
        "git_commit": git_output(args.source, "rev-parse", "HEAD"),
        "vllm_import": imported,
        "client_sha256": hashlib.sha256(client_path.read_bytes()).hexdigest(),
        "gpu": args.gpu,
        "port": args.port,
        "backend": "FLASH_ATTN",
        "target": args.target,
        "draft": args.draft if args.mode != "ar" else None,
        "horizon": args.horizon if args.mode == "cdb" else None,
        "kv_reuse": args.kv_reuse if args.mode == "cdb" else None,
        "low_log": args.low_log if args.mode == "cdb" else None,
        "k_policy": "hf_confidence" if args.mode == "cdb" else None,
        "k_cap": 8 if args.mode != "ar" else None,
        "requests": args.requests,
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "server_command": command,
        "server_environment": {
            key: value for key, value in env.items()
            if key.startswith("CDB_") or key in ("CUDA_VISIBLE_DEVICES", "PYTHONPATH", "VLLM_ATTENTION_BACKEND")
        },
    }
    (args.result_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    proc = None
    try:
        with (args.result_dir / "server.log").open("w") as output:
            proc = subprocess.Popen(
                command,
                cwd=args.source,
                env=env,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            (args.result_dir / "server.pid").write_text(str(proc.pid) + "\n")
            wait_healthy(proc, args.port)
            if args.warmup:
                run_client(
                    args,
                    args.result_dir / "warmup.log",
                    "warmup",
                    args.warmup,
                )
                save_metrics(args.port, args.result_dir / "metrics_after_warmup.prom")
            runs = []
            for index in range(1, args.repeats + 1):
                result = run_client(
                    args,
                    args.result_dir / f"run_{index}.log",
                    f"paired_{index}",
                    args.requests,
                )
                result["run"] = index
                runs.append(result)
                save_metrics(
                    args.port,
                    args.result_dir / f"metrics_after_run_{index}.prom",
                )
                (args.result_dir / "progress.json").write_text(
                    json.dumps(runs, indent=2) + "\n"
                )
            tps = [run["output_tps"] for run in runs]
            summary = {
                "runs": runs,
                "mean_output_tps": statistics.mean(tps),
                "sample_stdev_output_tps": statistics.stdev(tps) if len(tps) > 1 else None,
            }
            (args.result_dir / "summary.json").write_text(
                json.dumps(summary, indent=2) + "\n"
            )
            print(json.dumps(summary, indent=2))
    finally:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=10)


if __name__ == "__main__":
    main()
