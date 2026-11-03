#!/usr/bin/env bash
# Run the current persistent CDB-AV implementation against 512 requests.
# Execute on the RTX 4090 host; this script performs real GPU inference.
set -Eeuo pipefail

ROOT="$HOME/CDB-AV"
VLLM_DIR="$ROOT/cdb-av-512-test"
CLIENT="$ROOT/cdb_av_stepper_patch/stress_waiting.py"
TARGET="/disk14T/yypan/models/Qwen3-8B"
DRAFT="/disk14T/yypan/models/Qwen3-1.7B"
GPU="${CDB_AV_512_GPU:-1}"
PORT="${CDB_AV_512_PORT:-18083}"
N_REQUESTS=512
CONCURRENCY=16
MAX_TOKENS=128
REPEATS=3
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$ROOT/results/cdb_av_512/$STAMP"
SERVER_PID=""

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

port_pid() {
  ss -lntp 2>/dev/null \
    | sed -n "s/.*:$PORT .*pid=\([0-9]\+\).*/\1/p" \
    | head -1
}

stop_server() {
  if [[ -n "$SERVER_PID" ]]; then
    if kill -0 "$SERVER_PID" 2>/dev/null; then
      kill -TERM "$SERVER_PID" 2>/dev/null || true
    fi
    for ((i=0; i<120; i++)); do
      [[ -z "$(port_pid)" ]] && break
      sleep 1
    done
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
  fi
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  stop_server
  echo "[INFO] Results: $RUN_DIR"
  exit "$status"
}
trap cleanup EXIT INT TERM

source "$ROOT/activate-cdb-av.sh"
# Clear stale experiment switches inherited from an interactive shell.
for key in $(compgen -e | grep '^CDB_' || true); do
  unset "$key"
done
cd "$VLLM_DIR"

[[ -d "$TARGET" ]] || die "Target model not found: $TARGET"
[[ -d "$DRAFT" ]] || die "Draft model not found: $DRAFT"
[[ -f "$CLIENT" ]] || die "Missing load generator: $CLIENT"
grep -q '"cdb_draft_k": 8' "$CLIENT" \
  || die "stress_waiting.py must send vllm_xargs.cdb_draft_k=8"

# This extracted source tree is not a Git worktree. Never fetch, switch
# branches, or inspect/modify the original ~/CDB-AV/vllm checkout.
[[ -f "$VLLM_DIR/vllm/__init__.py" ]] || die "Missing isolated vLLM source: $VLLM_DIR"
[[ "$VLLM_DIR" == "$ROOT/cdb-av-512-test" ]] || die "Unexpected VLLM_DIR: $VLLM_DIR"

# Require an otherwise idle GPU for the independent Qwen3-8B + Qwen3-1.7B run.
GPU_USED_MIB="$(nvidia-smi --id="$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
[[ "$GPU_USED_MIB" =~ ^[0-9]+$ ]] || die "Cannot read memory usage for GPU $GPU"
(( GPU_USED_MIB <= 2048 )) || die "GPU $GPU already has ${GPU_USED_MIB} MiB allocated; refusing to start another model"

if [[ -n "$(port_pid)" ]]; then
  die "Port $PORT is already occupied by PID $(port_pid); refusing to stop or replace any existing service."
fi

python -m compileall -q \
  vllm/v1/core/sched/scheduler.py \
  vllm/v1/engine/core.py \
  vllm/v1/worker/gpu_model_runner.py \
  vllm/v1/spec_decode/cdb_av_config.py \
  vllm/v1/spec_decode/cdb_av_varlen.py \
  vllm/v1/spec_decode/cdb_av_persistent.py \
  vllm/v1/spec_decode/cdb_av_kv_reuse.py

mkdir -p "$RUN_DIR"
{
  echo "timestamp=$STAMP"
  echo "source_branch=cdb-av-complete-runtime-20261103"
  echo "source_commit=4a8500ab6799ae3f8f10931df9596e7fa78f8118"
  echo "source_dir=$VLLM_DIR"
  echo "gpu=$GPU"
  echo "port=$PORT"
  echo "target=$TARGET"
  echo "draft=$DRAFT"
  echo "requests=$N_REQUESTS"
  echo "concurrency=$CONCURRENCY"
  echo "max_tokens=$MAX_TOKENS"
  echo "repeats=$REPEATS"
  echo "H=8"
  echo "initial_admit=3"
  echo "K_max=8"
  echo "K_policy=hf_confidence"
  echo "HF_threshold=0.4"
  echo "KV_reuse=1"
  python -c 'import importlib.metadata as m, vllm; print("vllm_version=" + m.version("vllm")); print("vllm_file=" + str(vllm.__file__))'
} | tee "$RUN_DIR/run_config.txt"

python "$CLIENT" \
  --dry-run \
  --requests "$N_REQUESTS" \
  --concurrency "$CONCURRENCY" \
  --max-tokens "$MAX_TOKENS" \
  > "$RUN_DIR/request_payload_check.jsonl"

python - "$RUN_DIR/request_payload_check.jsonl" "$N_REQUESTS" <<'PY'
import json
import sys
from pathlib import Path

rows = [
    json.loads(line)
    for line in Path(sys.argv[1]).read_text().splitlines()
    if line.strip()
]
expected = int(sys.argv[2])
assert len(rows) == expected, (len(rows), expected)
assert all(row["requested_K"] == 8 for row in rows)
assert all(row["payload"]["vllm_xargs"]["cdb_draft_k"] == 8 for row in rows)
print(f"[OK] payloads={len(rows)}, client_K_cap=8 for every request")
PY

# Never stop or replace an existing process. This runner owns only the
# process it starts below and exits if another process acquired the port.
if [[ -n "$(port_pid)" ]]; then
  die "Port $PORT became occupied before startup; refusing to stop any service."
fi

start_server() {
  local trace="$1"
  local logfile="$2"
  env \
    CUDA_VISIBLE_DEVICES="$GPU" \
    CDB_AV_REFILL=1 \
    CDB_AV_PERSISTENT=1 \
    CDB_AV_K_POLICY=hf_confidence \
    CDB_AV_HF_THRESHOLD=0.4 \
    CDB_AV_WINDOW_H=8 \
    CDB_AV_STEP_MODE=1 \
    CDB_AV_VARLEN=1 \
    CDB_AV_SPLIT_RPC=1 \
    CDB_AV_WORK_QUEUE=1 \
    CDB_AV_KV_REUSE=1 \
    CDB_AV_INTRA_WINDOW=1 \
    CDB_AV_INITIAL_ADMIT=3 \
    CDB_AV_ALLOW_FRESH_WAITING=0 \
    CDB_P1_VIRTUAL_H=0 \
    CDB_AV_TRACE_STEPS="$trace" \
    CDB_AV_REUSE_REPLAY=0 \
    CDB_AV_REUSE_VALIDATE=0 \
    CDB_AV_VALIDATE_METADATA=0 \
    vllm serve "$TARGET" \
      --served-model-name qwen3-8b \
      --host 127.0.0.1 \
      --port "$PORT" \
      --tensor-parallel-size 1 \
      --max-num-seqs 4 \
      --max-model-len 1024 \
      --gpu-memory-utilization 0.90 \
      --enforce-eager \
      --no-async-scheduling \
      --no-enable-prefix-caching \
      --seed 42 \
      --speculative-config "{\"method\":\"draft_model\",\"model\":\"$DRAFT\",\"num_speculative_tokens\":8}" \
      > "$logfile" 2>&1 &
  SERVER_PID=$!

  local ready=0
  for ((i=0; i<600; i++)); do
    if curl -fsS --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
      ready=1
      break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      tail -n 100 "$logfile" >&2 || true
      die "vLLM server exited during startup"
    fi
    sleep 1
  done
  [[ "$ready" -eq 1 ]] || {
    tail -n 100 "$logfile" >&2 || true
    die "vLLM health check timed out"
  }
  echo "[OK] server ready on port $PORT (TRACE_STEPS=$trace)"
}

echo
echo "========== HF DYNAMIC-K DIAGNOSTIC =========="
start_server 1 "$RUN_DIR/diagnostic_server.log"
python "$CLIENT" \
  --port "$PORT" \
  --requests 32 \
  --concurrency "$CONCURRENCY" \
  --max-tokens "$MAX_TOKENS" \
  2>&1 | tee "$RUN_DIR/diagnostic_client.log"
grep '\[CDB_HF_CONF\]' "$RUN_DIR/diagnostic_server.log" \
  > "$RUN_DIR/hf_confidence_trace.log" || true
python - "$RUN_DIR/hf_confidence_trace.log" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

path = Path(sys.argv[1])
caps, actuals = set(), set()
reasons = Counter()
records = 0
low_confidence_stops = 0
sub_cap_observations = 0
for line in path.read_text().splitlines():
    marker = "[CDB_HF_CONF]"
    if marker not in line:
        continue
    payload = line.split(marker, 1)[1].strip()
    try:
        event = json.loads(payload)
    except json.JSONDecodeError:
        continue
    records += 1
    caps.update(int(v) for v in event.get("K_cap", {}).values())
    actuals.update(int(v) for v in event.get("K_actual", {}).values())
    event_reasons = list(event.get("stop_reasons", {}).values())
    reasons.update(event_reasons)
    low_confidence_stops += sum(reason == "low_confidence" for reason in event_reasons)
    sub_cap_observations += sum(
        int(value) < 8 for value in event.get("K_actual", {}).values()
    )

print(f"[HF-K] records={records}, K_cap={sorted(caps)}, "
      f"K_actual={sorted(actuals)}, stop_reasons={dict(reasons)}, "
      f"low_confidence_stops={low_confidence_stops}, "
      f"sub_cap_observations={sub_cap_observations}")
if records == 0:
    raise SystemExit("No CDB_HF_CONF telemetry; dynamic K was not verified.")
if caps != {8}:
    raise SystemExit(f"Expected every request to use K_cap=8, observed {caps}")
if not actuals or any(k < 1 or k > 8 for k in actuals):
    raise SystemExit(f"Invalid actual Draft K values: {actuals}")
if low_confidence_stops == 0 or sub_cap_observations == 0:
    raise SystemExit(
        "HF dynamic stopping was not demonstrated: expected at least one "
        "low_confidence stop and one K_actual < K_cap observation."
    )
PY
echo "CDB_HF_CONF records: $(wc -l < "$RUN_DIR/hf_confidence_trace.log")"
stop_server

echo
echo "========== PERFORMANCE RUNS (TRACE DISABLED) =========="
start_server 0 "$RUN_DIR/performance_server.log"

echo "[INFO] Warmup: 16 requests"
python "$CLIENT" \
  --port "$PORT" \
  --requests 16 \
  --concurrency 16 \
  --max-tokens "$MAX_TOKENS" \
  2>&1 | tee "$RUN_DIR/warmup.log"

for ((run=1; run<=REPEATS; run++)); do
  echo
  echo "========== RUN $run / $REPEATS =========="
  python "$CLIENT" \
    --port "$PORT" \
    --requests "$N_REQUESTS" \
    --concurrency "$CONCURRENCY" \
    --max-tokens "$MAX_TOKENS" \
    2>&1 | tee "$RUN_DIR/run_$run.log"
done

python - "$RUN_DIR" "$N_REQUESTS" "$REPEATS" <<'PY'
import json
import re
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected = int(sys.argv[2])
repeats = int(sys.argv[3])

def percentile(values, p):
    values = sorted(values)
    if not values:
        return float("nan")
    x = (len(values) - 1) * p
    lo = int(x)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] * (hi - x) + values[hi] * (x - lo)

summaries = []
for run in range(1, repeats + 1):
    path = root / f"run_{run}.log"
    raw = path.read_text()
    rows = []
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "index" in item and "ok" in item:
            rows.append(item)

    completed = sum(bool(x["ok"]) for x in rows)
    tokens = sum(int(x.get("output_tokens", 0)) for x in rows)
    wall_match = re.search(r"wall_time=([0-9.]+)s", raw)
    wall = float(wall_match.group(1)) if wall_match else float("nan")
    latencies = [float(x["elapsed"]) for x in rows if "elapsed" in x]
    tps = tokens / wall if wall > 0 else float("nan")

    summary = {
        "run": run,
        "requests_seen": len(rows),
        "completed": completed,
        "failed": len(rows) - completed,
        "output_tokens": tokens,
        "wall_time_s": wall,
        "output_tps": tps,
        "request_tps": completed / wall if wall > 0 else float("nan"),
        "e2e_p50_s": percentile(latencies, 0.50),
        "e2e_p95_s": percentile(latencies, 0.95),
        "e2e_p99_s": percentile(latencies, 0.99),
    }
    summaries.append(summary)
    print(
        f"RUN {run}: {completed}/{expected}, "
        f"tokens={tokens}, wall={wall:.3f}s, "
        f"output_tps={tps:.3f}, "
        f"E2E p50/p95/p99="
        f"{summary['e2e_p50_s']:.3f}/"
        f"{summary['e2e_p95_s']:.3f}/"
        f"{summary['e2e_p99_s']:.3f}s"
    )

if any(x["completed"] != expected or x["requests_seen"] != expected for x in summaries):
    raise SystemExit("At least one run did not complete all requested requests.")

tps_values = [x["output_tps"] for x in summaries]
result = {
    "expected_requests_per_run": expected,
    "repeats": repeats,
    "mean_output_tps": statistics.mean(tps_values),
    "stdev_output_tps": statistics.stdev(tps_values) if len(tps_values) > 1 else 0.0,
    "runs": summaries,
}
(root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
print(
    f"MEAN output_tps={result['mean_output_tps']:.3f} "
    f"+/- {result['stdev_output_tps']:.3f}"
)
print(f"Saved: {root / 'summary.json'}")
PY

echo
echo "========== RUNTIME EVIDENCE =========="
echo "p2_runtime_config:"
grep -m 2 'p2_runtime_config' "$RUN_DIR/performance_server.log" || true
echo "Target Ready selections: $(grep -c 'target_ready_select' "$RUN_DIR/performance_server.log" || true)"
echo "Prefill budget deferrals: $(grep -c 'p2_defer_prefill_budget' "$RUN_DIR/performance_server.log" || true)"
echo "H-boundary events: $(grep -Ec 'p2_.*boundary|window_boundary' "$RUN_DIR/performance_server.log" || true)"
grep -Ei 'Traceback|RuntimeError|lease ownership|stale context|input buffer capacity|ERROR' \
  "$RUN_DIR/performance_server.log" \
  > "$RUN_DIR/runtime_errors.log" || true
echo "Runtime error matches: $(wc -l < "$RUN_DIR/runtime_errors.log")"
echo "HF K trace: $RUN_DIR/hf_confidence_trace.log"
echo "Performance server log: $RUN_DIR/performance_server.log"
echo "Run summaries: $RUN_DIR/summary.json"

stop_server
echo "[OK] 512-request CDB-AV benchmark finished."
