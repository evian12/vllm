#!/usr/bin/env python3
"""Apply the post-forward worker-cache handoff fix for CDB mixed Prefill/Decode."""
from pathlib import Path
import argparse, ast, hashlib, shutil, time

REL = Path("vllm/v1/spec_decode/cdb_av_kv_reuse.py")
ANCHOR = "        status = super().step()\n"
INSERT = (
    "        status = super().step()\n"
    "\n"
    "        # The old worker cache remains visible during Prepare/Forward so its\n"
    "        # KV can be reused. Retire its metadata only after the mixed Forward\n"
    "        # succeeds; otherwise boundary() would reject a duplicate Ready cache.\n"
    "        for _, _, _, _, rid, _ in pending:\n"
    "            old_cache = self.cached.pop(rid, None)\n"
    "            if old_cache is not None:\n"
    "                emit('reuse_cache_consumed', window=self.window, req_id=rid)\n"
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo", nargs="?", default=".")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    path = Path(args.repo).resolve() / REL
    if not path.is_file():
        raise SystemExit(f"Missing {path}")
    src = path.read_text()
    ast.parse(src, filename=str(path))
    tree = ast.parse(src)
    method = None
    lines = src.splitlines(keepends=True)
    for cls in tree.body:
        if isinstance(cls, ast.ClassDef) and cls.name == "ReuseDraftSession":
            for node in cls.body:
                if isinstance(node, ast.FunctionDef) and node.name == "step":
                    method = (node.lineno - 1, node.end_lineno)
                    break
    if method is None:
        raise SystemExit("ReuseDraftSession.step not found")
    start, end = method
    body = "".join(lines[start:end])
    if "reuse_cache_consumed" in body or "old_cache = self.cached.pop(rid, None)" in body:
        print("ALREADY_APPLIED")
        return
    if body.count(ANCHOR) != 1:
        raise SystemExit("Expected exactly one successful parent-step anchor; no change made")
    if "pending = list(getattr(self, 'pending_prefill', ()))" not in body:
        raise SystemExit("Expected pending Prefill snapshot missing; no change made")
    patched_body = body.replace(ANCHOR, INSERT, 1)
    patched = "".join(lines[:start]) + patched_body + "".join(lines[end:])
    ast.parse(patched, filename=str(path))
    if args.dry_run:
        print("DRY_RUN_OK", path, hashlib.sha256(patched.encode()).hexdigest())
        print("The old cache entry will be retired only after super().step() succeeds.")
        return
    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(path.name + f".before_cache_handoff_{stamp}")
    shutil.copy2(path, backup)
    tmp = path.with_name(path.name + ".cache_handoff.tmp")
    try:
        tmp.write_text(patched)
        tmp.replace(path)
    except BaseException:
        if backup.exists():
            shutil.copy2(backup, path)
        raise
    print("BACKUP", backup)
    print("PATCHED", path)
    print("SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    print("DONE")

if __name__ == "__main__":
    main()
