#!/usr/bin/env bash
# Opt-in Draft stage diagnostic. Run via: bash scripts/cdb_av/start_draft_stage_profile_20261109.sh
# No process termination; never source this file into an interactive shell.
set -uo pipefail

main() {
  local ROOT=/home/yypan/CDB-AV
  local WT="$ROOT/worktrees/cdb-draft-stage-profile-20261109"
  local PY="$ROOT/.venv/bin/python"
  local VLLM="$ROOT/.venv/bin/vllm"
  local MODEL=/disk14T/yypan/models/Qwen3-8B
  local DRAFT=/disk14T/yypan/models/Qwen3-1.7B
  local LOG=/tmp/cdb_draft_stage_profile_20261109.log
  local USED RC

  if [ ! -x "$PY" ] || [ ! -x "$VLLM" ] || [ ! -d "$WT" ]; then
    echo "Missing Python, CLI, or isolated Draft profile worktree"; return 1
  fi
  if [ ! -e "$WT/vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so" ]; then
    echo "Missing FA2 extension link in isolated worktree"; return 1
  fi
  if [ -e "$LOG" ]; then
    echo "Log already exists, refusing to overwrite: $LOG"; return 1
  fi
  if ss -lntp | grep -q ':18084'; then
    echo "Port 18084 occupied; stop only the owned old service first"; return 1
  fi
  USED=$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  if ! [[ "$USED" =~ ^[0-9]+$ ]] || [ "$USED" -gt 1024 ]; then
    echo "GPU 1 not free or status unavailable, used MiB: $USED"; return 1
  fi

  cd "$WT" || return 1
  echo "Source: $(git rev-parse HEAD)"
  echo "GPU1 / 127.0.0.1:18084; log: $LOG"
  CUDA_VISIBLE_DEVICES=1 \
  PYTHONPATH="$WT" \
  PYTHONUNBUFFERED=1 \
  CDB_AV_PERSISTENT=1 \
  CDB_AV_REFILL=1 \
  CDB_AV_VARLEN=1 \
  CDB_AV_STEP_MODE=1 \
  CDB_AV_SPLIT_RPC=1 \
  CDB_AV_WORK_QUEUE=1 \
  CDB_AV_WINDOW_H=8 \
  CDB_AV_KV_REUSE=1 \
  CDB_AV_K_POLICY=hf_confidence \
  CDB_AV_HF_THRESHOLD=0.8 \
  CDB_AV_INTRA_WINDOW=1 \
  CDB_AV_INITIAL_ADMIT=3 \
  CDB_AV_ALLOW_FRESH_WAITING=0 \
  CDB_P1_VIRTUAL_H=0 \
  CDB_AV_REFILL_TRACE=0 \
  CDB_AV_TRACE_STEPS=0 \
  CDB_AV_REUSE_REPLAY=0 \
  CDB_AV_REUSE_VALIDATE=0 \
  CDB_AV_VALIDATE_METADATA=0 \
  CDB_AV_DRAFT_DECODE_GRAPH=1 \
  CDB_AV_DRAFT_RAGGED_GRAPH=1 \
  CDB_AV_DRAFT_RAGGED_DYNAMIC=1 \
  CDB_AV_DRAFT_RAGGED_MAX_QUERY=4 \
  CDB_AV_RAGGED_GRAPH_MAX_KEYS=12 \
  CDB_AV_RAGGED_GRAPH_MIN_HITS=8 \
  CDB_AV_DRAFT_GRAPH_VALIDATE=0 \
  CDB_AV_TRACE_GRAPH=0 \
  CDB_AV_TARGET_FULL_GRAPH=1 \
  CDB_AV_TARGET_GRAPH_VALIDATE=0 \
  CDB_AV_TARGET_GRAPH_TIMING=0 \
  CDB_AV_TARGET_GRAPH_DYNAMIC=1 \
  CDB_AV_TARGET_GRAPH_SEQ_FLOOR=0 \
  CDB_AV_TARGET_GRAPH_MIN_HITS=8 \
  CDB_AV_TARGET_GRAPH_MAX_KEYS=16 \
  CDB_AV_TRACE_TARGET_GRAPH=0 \
  CDB_AV_DRAFT_STAGE_PROFILE=1 \
  CDB_AV_DRAFT_STAGE_PROFILE_MAX_SAMPLES=20 \
  "$PY" "$VLLM" serve "$MODEL" \
    --served-model-name qwen3-8b \
    --host 127.0.0.1 --port 18084 \
    --tensor-parallel-size 1 --max-num-seqs 4 --max-model-len 1024 \
    --gpu-memory-utilization 0.90 --dtype bfloat16 \
    --enforce-eager --no-async-scheduling --no-enable-prefix-caching \
    --attention-backend FLASH_ATTN --seed 42 \
    --speculative-config "{\"method\":\"draft_model\",\"model\":\"$DRAFT\",\"num_speculative_tokens\":8}" \
    2>&1 | tee "$LOG"
  RC=${PIPESTATUS[0]}
  echo "vLLM stopped: status=$RC"
  return "$RC"
}

main "$@"
