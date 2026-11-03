#!/usr/bin/env python3
"""Small, fail-closed Git and experiment checks for CDB-AV."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parents[2]
MAX_STAGED_BYTES = 20 * 1024 * 1024
PRIVATE_CONTENT = re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----", re.I)
ASSIGNMENT_SECRET = re.compile(
    r"(?i)(?:api[_-]?key|access[_-]?token|password|client[_-]?secret)"
    r"\s*[:=]\s*['\"]?[A-Za-z0-9_./+\-=]{16,}"
)
WEIGHT_SUFFIXES = {".safetensors", ".ckpt", ".gguf", ".pt", ".pth", ".bin"}


def command(args: list[str], *, check: bool = True, timeout: int = 90) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    proc = subprocess.run(args, cwd=ROOT, text=True, capture_output=True,
                          timeout=timeout, env=env)
    if check and proc.returncode:
        raise RuntimeError(f"{args[0]} {args[1:]} failed ({proc.returncode}): {proc.stderr.strip()}")
    return proc


def git(*args: str, check: bool = True, timeout: int = 90) -> str:
    return command(["git", *args], check=check, timeout=timeout).stdout.strip()


def redact_url(url: str) -> str:
    if "://" not in url:
        return url
    parsed = urlsplit(url)
    host = parsed.hostname or ""
    if parsed.port:
        host += f":{parsed.port}"
    return urlunsplit((parsed.scheme, host, parsed.path, parsed.query, parsed.fragment))


def remote_sha(remote: str, branch: str) -> str | None:
    ref = f"refs/heads/{branch}"
    output = git("ls-remote", "--heads", remote, ref)
    if not output:
        return None
    lines = output.splitlines()
    if len(lines) != 1 or lines[0].split()[-1] != ref:
        raise RuntimeError(f"Unexpected remote result for {ref}: {output}")
    return lines[0].split()[0]


def unsafe_path(name: str) -> str | None:
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        return "path leaves repository"
    parts = {p.lower() for p in path.parts}
    if parts & {"models", "checkpoints", "datasets", ".venv", "venv", "__pycache__"}:
        return "model, dataset, or generated directory"
    if path.suffix.lower() in WEIGHT_SUFFIXES:
        return "model or checkpoint file"
    if path.suffix.lower() == ".log" or path.name.lower().endswith(".log.gz"):
        return "raw log"
    if path.name == ".env" or (path.name.startswith(".env.") and path.name != ".env.example"):
        return "environment secrets file"
    if "raw" in parts and "results" in parts:
        return "raw experiment artifact"
    return None


def staged_paths() -> list[str]:
    return [p for p in git("diff", "--cached", "--name-only", "--diff-filter=ACMR").splitlines() if p]


def scan_staged() -> list[str]:
    errors = []
    for name in staged_paths():
        reason = unsafe_path(name)
        if reason:
            errors.append(f"{name}: {reason}")
            continue
        path = ROOT / name
        if path.is_symlink():
            errors.append(f"{name}: symlink requires explicit review")
            continue
        staged = subprocess.run(["git", "show", f":{name}"], cwd=ROOT,
                                capture_output=True, timeout=30)
        if staged.returncode:
            errors.append(f"{name}: unable to read staged blob")
            continue
        blob = staged.stdout
        if b"\0" in blob:
            errors.append(f"{name}: binary staged content requires explicit review")
            continue
        if len(blob) > MAX_STAGED_BYTES:
            errors.append(f"{name}: staged file exceeds {MAX_STAGED_BYTES} bytes")
        text = blob.decode("utf-8", "replace")
        if PRIVATE_CONTENT.search(text) or ASSIGNMENT_SECRET.search(text):
            errors.append(f"{name}: possible secret in staged content")
    return errors


def change_paths() -> list[str]:
    tracked = git("diff", "HEAD", "--name-only").splitlines()
    untracked = git("ls-files", "--others", "--exclude-standard").splitlines()
    return sorted(set(tracked + untracked))


def count_tests(output: str) -> int:
    match = re.search(r"Ran (\d+) tests?", output)
    if match:
        return int(match.group(1))
    match = re.search(r"(\d+) passed", output)
    return int(match.group(1)) if match else 0


def validate(pytest_paths: list[str]) -> dict[str, object]:
    git("diff", "--check")
    git("diff", "--cached", "--check")
    errors = scan_staged()
    if errors:
        raise RuntimeError("Staged content rejected: " + "; ".join(errors))
    changed = change_paths()
    py_files = [str(ROOT / p) for p in changed if p.endswith(".py") and (ROOT / p).is_file()]
    sh_files = [str(ROOT / p) for p in changed if p.endswith(".sh") and (ROOT / p).is_file()]
    for path in py_files:
        command([sys.executable, "-m", "py_compile", path])
    for path in sh_files:
        command(["bash", "-n", path])
    unit = command([sys.executable, "-m", "unittest", "discover", "-s",
                    "scripts/cdb_av/tests", "-p", "test_*.py", "-v"])
    unit_count = count_tests(unit.stdout + unit.stderr)
    if unit_count == 0:
        raise RuntimeError("Workflow unit tests did not run")
    core_changed = any(p.startswith("vllm/v1/") for p in changed)
    if core_changed and not pytest_paths:
        raise RuntimeError("vllm/v1 changes require --pytest with relevant tests")
    extra = []
    for path in pytest_paths:
        result = command([sys.executable, "-m", "pytest", "-q", path], timeout=600)
        extra.append({"path": path, "tests": count_tests(result.stdout + result.stderr)})
    return {"status": "pass", "static_python_files": len(py_files),
            "static_shell_files": len(sh_files), "workflow_tests": unit_count,
            "extra_tests": extra, "staged_paths": staged_paths()}


def preflight(remote: str) -> dict[str, object]:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    url = redact_url(git("remote", "get-url", remote))
    remote_head = remote_sha(remote, branch)
    status = git("status", "--porcelain=v1", "--untracked-files=normal").splitlines()
    return {"repo": str(ROOT), "branch": branch, "head": head,
            "remote": remote, "remote_url": url, "remote_head": remote_head,
            "remote_matches_local": remote_head == head,
            "worktree_entries": len(status), "worktree_status": status}


def checkpoint(args: argparse.Namespace) -> dict[str, object]:
    branch = git("branch", "--show-current")
    if branch == "main" or not branch:
        raise RuntimeError("Use a named development branch; main is protected")
    if not args.path:
        raise RuntimeError("Pass each intended path explicitly with --path")
    intended = set(args.path)
    if any((ROOT / name).is_dir() for name in intended):
        raise RuntimeError("--path must name individual files, not directories")
    previously_staged = set(git("diff", "--cached", "--name-only").splitlines())
    if previously_staged - intended:
        raise RuntimeError("Unrelated paths are already staged: " + ", ".join(sorted(previously_staged - intended)))
    before = remote_sha(args.remote, branch)
    if before:
        git("fetch", "--no-tags", args.remote, branch, timeout=120)
        if command(["git", "merge-base", "--is-ancestor", before, "HEAD"], check=False).returncode:
            raise RuntimeError("Remote branch has changes missing locally; inspect before committing")
    for path in args.path:
        if unsafe_path(path):
            raise RuntimeError(f"Unsafe path {path}: {unsafe_path(path)}")
    git("add", "--", *args.path)
    staged = set(git("diff", "--cached", "--name-only").splitlines())
    if not staged:
        raise RuntimeError("No staged changes")
    if staged - intended:
        raise RuntimeError("Unexpected staged paths: " + ", ".join(sorted(staged - intended)))
    report = validate(args.pytest)
    git("commit", "-m", args.message, timeout=120)
    head = git("rev-parse", "HEAD")
    git("push", args.remote, f"HEAD:refs/heads/{branch}", timeout=180)
    after = remote_sha(args.remote, branch)
    if after != head:
        raise RuntimeError(f"Push verification failed: local={head}, remote={after}")
    return {"commit": head, "branch": branch, "remote": args.remote,
            "remote_head": after, "validation": report}


def manifest(args: argparse.Namespace) -> dict[str, object]:
    dirty = bool(git("status", "--porcelain=v1", "--untracked-files=normal"))
    if dirty and not args.allow_dirty:
        raise RuntimeError("Experiment source is dirty; commit or use --allow-dirty with documented hashes")
    env = {k: v for k, v in os.environ.items()
           if (k.startswith("CDB_") or k in {"CUDA_VISIBLE_DEVICES", "VLLM_ATTENTION_BACKEND"})
           and not re.search(r"KEY|TOKEN|PASSWORD|SECRET|CREDENTIAL", k, re.I)}
    result = {
        "experiment_name": args.experiment_name,
        "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("branch", "--show-current"),
        "git_dirty": dirty,
        "tracked_diff_sha256": hashlib.sha256(git("diff", "HEAD", "--binary").encode()).hexdigest() if dirty else None,
        "untracked_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in git("ls-files", "--others", "--exclude-standard").splitlines()
            if (ROOT / name).is_file() and not (ROOT / name).is_symlink()
        } if dirty else {},
        "vllm_version": args.vllm_version,
        "target_model": args.target_model,
        "draft_model": args.draft_model,
        "gpu": args.gpu,
        "window_h": args.window_h,
        "dynamic_k_policy": args.dynamic_k_policy,
        "request_count": args.request_count,
        "dataset": args.dataset,
        "source_import_path": args.source_import_path,
        "environment": env,
        "command": args.command,
        "test_status": args.test_status,
        "result_dir": str(Path(args.run_dir).resolve()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    output = Path(args.run_dir) / "experiment_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.overwrite:
        raise RuntimeError(f"Manifest already exists: {output}")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return {"manifest": str(output), "git_commit": result["git_commit"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("preflight")
    p.add_argument("--remote", default="myfork")
    p = sub.add_parser("validate")
    p.add_argument("--pytest", action="append", default=[])
    p = sub.add_parser("checkpoint")
    p.add_argument("--remote", default="myfork")
    p.add_argument("--message", required=True)
    p.add_argument("--path", action="append", default=[])
    p.add_argument("--pytest", action="append", default=[])
    p = sub.add_parser("manifest")
    for name in ("experiment-name", "vllm-version", "target-model", "draft-model",
                 "gpu", "dynamic-k-policy", "dataset", "command", "run-dir",
                 "source-import-path"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--window-h", type=int, required=True)
    p.add_argument("--request-count", type=int, required=True)
    p.add_argument("--test-status", choices=["not_run", "pass", "fail", "partial"], required=True)
    p.add_argument("--allow-dirty", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        if args.action == "preflight":
            result = preflight(args.remote)
        elif args.action == "validate":
            result = validate(args.pytest)
        elif args.action == "checkpoint":
            result = checkpoint(args)
        else:
            result = manifest(args)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (RuntimeError, subprocess.TimeoutExpired) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
