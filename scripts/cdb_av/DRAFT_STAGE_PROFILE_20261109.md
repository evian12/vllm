# CDB-AV bounded Draft stage profiler (2026-11-09)

Experimental, **off by default**. Branch: `codex/cdb-draft-stage-profile-20261109`.
Based on `f7213cb78f0f55a3463547e3d6b851233b884213`.
It only instruments `vllm/v1/spec_decode/cdb_av_refill.py` and adds
`vllm/v1/spec_decode/cdb_av_draft_stage_profile.py`.
H=8, HF threshold=0.8, Draft K, target scheduling, lease ownership, KV reuse,
sampling, and commit rules are unchanged.

## Environment

- `CDB_AV_DRAFT_STAGE_PROFILE=1` enables the one-time profiler.
- `CDB_AV_DRAFT_STAGE_PROFILE_MAX_SAMPLES=20` limits samples **per proposer lifetime**
  (valid 1..100). A fresh service is required to reset.
  For limit>=8, the sampler reserves approximately half for warmed Graph
  replays; extra eager warmups cannot consume replay slots. It may emit fewer
  than the configured maximum if replay never occurs.
- Disabled mode creates no CUDA timing events and does not synchronize.
- Do not combine with `CDB_AV_DRAFT_GRAPH_VALIDATE=1` or Target timing/validation
  when measuring stage costs; those perform extra forwards.
- Graph Capture calls are excluded: capturing synchronizes and has a very
  different cost from steady-state Graph Replay.

## Log schema

`[CDB_DRAFT_STAGE_TIMING] {...}` contains `window`, `step`,
`phase`, `request_rows`, `token_rows`, `graph_action`
(`eager` or `replay`), `graph_key`, `sample_index`, `sample_limit`,
`metadata_pack_cpu_ms`, and `measurements`.

`measurements` has `metadata`, `backbone` and `logits`, each with
`cpu_enqueue_ms` and `stream_span_ms`; when logits are postprocessed
there are also `confidence_gpu` (both fields),
`d2h_wait_cpu_ms` and `batch_update_cpu_ms` (CPU only).

- `metadata_pack_cpu_ms` measures `RefillSession._metadata()`, including
  CPU tensor construction and asynchronous GPU allocations/copy submission.
  It does **not** include earlier upstream scheduling/lease lookup.
- `metadata` is the downstream attention metadata build for eager, or
  the static CUDA buffer update for graph replay. These are not identical
  operations and should be compared by phase rather than blindly equated.
- `backbone` is the model call or CUDA Graph Replay.
- `logits` includes output row selection and LM-head execution.
- `confidence_gpu` includes confidence/logsumexp, greedy selection, packing
  the tiny result buffer; on manual policy it measures argmax only.
- `d2h_wait_cpu_ms` measures the full blocking path from immediately
  before the explicit current-stream synchronization through
  `.cpu().tolist()`.
- `pre_d2h_gpu_wait_cpu_ms` isolates the CPU time spent waiting for
  outstanding work on the tensor's producer stream. This includes prior
  metadata/backbone/logits/confidence work that has not completed yet.
- `d2h_materialize_cpu_ms` measures `.cpu().tolist()` after that stream
  has completed: host allocation/copy plus Python-list materialization.
  It is not a pure PCIe bandwidth metric.
- The explicit synchronization is only inserted for sampled diagnostic calls;
  profiling-disabled execution retains the original single `.cpu().tolist()`
  path. Moving the wait earlier decomposes the existing barrier; it does not
  remove the dependency or make the step faster.
- `batch_update_cpu_ms` covers per-request stop-reason/status bookkeeping
  and final active/ready list updates.
- CUDA stream spans are interval timings (which may include idle gaps);
  **not** kernel-active times or isolated FLOP calculations.
- The field named `cpu_enqueue_ms` is the wall-clock duration of the Python
  stage call. It is **not** pure CPU compute or a pure CUDA enqueue duration:
  eager model calls can include substantial host dispatch and implicit waits,
  while Graph Replay often returns quickly and leaves GPU work outstanding.
  The later D2H wait may therefore contain work that eager execution already
  waited for inside the model call.
- Sampled calls synchronize their event endpoints in the diagnostic logging;
  TPS/TTFT/TPOT/ITL measured with the profiler enabled are invalid for
  production comparison. Keep it off for Graph ON/OFF performance runs.

## Smoke test (GPU 1 and port 18084 only)

First stop any *owned* server on port 18084 and confirm GPU 1 free. Do not
kill other users' processes. To create a separate worktree, from the main
Git repository run:

```bash
ROOT=/home/yypan/CDB-AV
git -C "$ROOT/vllm" fetch myfork codex/cdb-draft-stage-profile-20261109
git -C "$ROOT/vllm" worktree add -b cdb-draft-stage-profile-20261109 \
  "$ROOT/worktrees/cdb-draft-stage-profile-20261109" \
  myfork/codex/cdb-draft-stage-profile-20261109
```

The new worktree needs the same ignored FlashAttention Python shim files and
FA2/FA3 native extension **symlinks** as `$ROOT/cdb-av-512-test`; do not
overwrite other worktrees.

CPU tests (no GPU needed):

```bash
cd /home/yypan/CDB-AV/worktrees/cdb-draft-stage-profile-20261109
CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PWD" /home/yypan/CDB-AV/.venv/bin/python -m pytest -q \
  tests/v1/spec_decode/test_cdb_av_draft_stage_profile.py \
  tests/v1/spec_decode/test_cdb_av_draft_decode_graph.py
```

Launch using a separate Bash process, rather than sourcing commands with
`exit` into an interactive terminal:

```bash
cd /home/yypan/CDB-AV/worktrees/cdb-draft-stage-profile-20261109
bash scripts/cdb_av/start_draft_stage_profile_20261109.sh
```

Run the checked-in diagnostic client, then summarize its log:

```bash
cd /home/yypan/CDB-AV/worktrees/cdb-draft-stage-profile-20261109
/home/yypan/CDB-AV/.venv/bin/python scripts/cdb_av/smoke_draft_stage_profile.py
/home/yypan/CDB-AV/.venv/bin/python scripts/cdb_av/analyze_draft_stage_profile.py \
  /tmp/cdb_draft_stage_profile_split_20261109.log
```

The client sends 32 identical greedy prompts, concurrency4 and output128.
Check that 1..20 valid timing records appear, with both eager and replay
records *if* Graph keys are warmed. No errors, all requests complete, and
probe never records a Graph Capture.
Then stop the owned foreground service via Ctrl+C and verify GPU/port release.

## Matched Decode Graph ON/OFF control

The initial Graph-ON run samples mostly replay Decode, while eager samples
include mixed Prefill. Those are different token-row workloads and must not
be compared directly. Use a second fresh service with only Draft graphs
disabled; Target Full Graph remains enabled.

After stopping the owned Graph-ON service and confirming port 18084/GPU 1 are
free, update the worktree to the latest branch and run:

```bash
cd /home/yypan/CDB-AV/worktrees/cdb-draft-stage-profile-20261109
git fetch myfork codex/cdb-draft-stage-profile-20261109
git merge --ff-only FETCH_HEAD
bash scripts/cdb_av/start_draft_stage_profile_eager_20261109.sh
```

The Graph-OFF run writes
`/tmp/cdb_draft_stage_profile_eager_20261109.log`. Run the same smoke client:

```bash
/home/yypan/CDB-AV/.venv/bin/python scripts/cdb_av/smoke_draft_stage_profile.py
```

Then, after stopping the owned Graph-OFF service, compare only identical
Decode shapes (request rows and token rows), excluding mixed Prefill:

```bash
PY=/home/yypan/CDB-AV/.venv/bin/python
$PY scripts/cdb_av/compare_draft_decode_graph_profile.py \
  /tmp/cdb_draft_stage_profile_split_20261109.log \
  /tmp/cdb_draft_stage_profile_eager_20261109.log
```

The comparison reports medians for matched shapes. It is still a diagnostic
profile, not a production throughput comparison. If the Graph-ON Decode path
has lower Backbone stream time at the same shape, proceed to a separate
profile-disabled TPS/TTFT/TPOT A/B run. Do not infer a production speedup
from the profiled samples.

**Status (2026-11-09):** the user ran the Graph-ON GPU 1 smoke on the
RTX 4090: 32/32 requests succeeded, 4,096 completion tokens, and 20 timing
samples were emitted. In replay Decode (8 samples; mean 3.5 request/token
rows), `pre_d2h_gpu_wait_cpu_ms` averaged 4.432 ms and
`d2h_materialize_cpu_ms` averaged 0.0399 ms. This confirms the host wait is
for outstanding GPU work, not a costly D2H materialization. The 24.192 s
smoke wall time is diagnostic only, not production TPS. CPU pytest output has
not yet been supplied. The matched-shape Draft Graph-OFF control is now
provided by `start_draft_stage_profile_eager_20261109.sh`; its run remains
pending.
