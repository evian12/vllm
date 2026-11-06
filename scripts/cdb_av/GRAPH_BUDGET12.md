# Combined H8 / HF0.8 / dynamic Draft Graph configuration

The code tested here is f68848cf7c41f5c7bb3d63fe96ec90f622f47950 (runtime
change674c390;f68848c only adds its earlier report). No strategy,H,KV or model
code was changed for these Graph cache experiments. H8,maxK8,HF0.8,exclusive
KV reuse and AR-stage bypass remain fixed.12 is a Graph cache upper bound,
not a batch size,K,orH. Target still runs enforce-eager in this configuration.

32 requests,concurrency16,128 output,4 warmups,three balanced orders:

| Draft Graph configuration | Mean TPS | Sample SD |
|---|---:|---:|
|single-query mixed only|180.628|4.087|
|dynamic1/2-query,4 keys|188.310|4.842|
|dynamic1/2-query,12 keys|197.158|1.060|

All9 passes completed32/32.12-key configuration improves mean TPS9.15% over
mixed-only and4.70% over4 keys. Functional32-request validation:414 Graph/eager
checks,all touched-layer KV and greedy tokens exact against both static and
live metadata;6 dynamic captures,153 dynamic replays,119 cross-layout replays.
Requested limit12 is not evidence that all12 keys were captured in validation.

512 requests,same concurrent16/128 output/4 warmup,one mixed->dynamic12 pair:

| Config | TPS | P50(s) | P95(s) | Drafts | Drafted | Accepted |
|---|---:|---:|---:|---:|---:|---:|
|mixed only|185.641|11.139|13.448|17884|43124|30237|
|dynamic12|202.872|10.202|12.229|17874|42937|30382|

Both512/512,65536 output tokens,throughput+9.28%.Prometheus work totals include
warmup. Work counts differ by less than1%;a reverse512 pair is still required
before claiming stable long-load benefit. This is not a comparison against all
native AR/SD configurations. Heavy KV/metadata/Graph validation was disabled for
performance runs and preserved in separate correctness runs.

Prior AR-stage bypass reverse512 pairing also remained positive:OFF179.760,
ON184.443 TPS. Across two pairs OFF180.488±1.030 vsON185.100±0.928 TPS,+2.56%.
Do not multiply different experiment gains into a single measured speedup.

The tested high-throughput profile is opt-in. Code defaults for persistent/
ragged/dynamic/AR bypass remain unchanged. Production18082 was not restarted.
Raw artifacts: /home/yypan/CDB-AV/results/ar_graph_budget_20261106/ and
/home/yypan/CDB-AV/results/ar_bypass_20261106/. The configs JSON and commands
below reproduce the measured source,GPU1,port18084,models and flags. Verify
GPU1/18084 are free first;clear other CDB_* variables as the harness does.
Use the test harness for automatic child-service cleanup;manual launch stays
foreground and must be stopped after the experiment.

```bash
cd /home/yypan/CDB-AV/worktrees/cdb-ar-bypass-20261106
env PYTHONPATH=/home/yypan/CDB-AV/worktrees/cdb-ar-bypass-20261106 \
    CUDA_VISIBLE_DEVICES=1 \
    VLLM_ATTENTION_BACKEND=FLASH_ATTN \
    CDB_AV_REFILL=1 \
    CDB_AV_PERSISTENT=1 \
    CDB_AV_KV_REUSE=1 \
    CDB_AV_INTRA_WINDOW=1 \
    CDB_AV_LOW_LOG=1 \
    CDB_AV_K_POLICY=hf_confidence \
    CDB_AV_HF_THRESHOLD=0.8 \
    CDB_AV_AR_TARGET_BYPASS=1 \
    CDB_AV_WINDOW_H=8 \
    CDB_AV_STEP_MODE=1 \
    CDB_AV_VARLEN=1 \
    CDB_AV_SPLIT_RPC=1 \
    CDB_AV_WORK_QUEUE=1 \
    CDB_AV_INITIAL_ADMIT=3 \
    CDB_AV_ALLOW_FRESH_WAITING=0 \
    CDB_P1_VIRTUAL_H=0 \
    CDB_AV_TRACE_STEPS=0 \
    CDB_AV_REFILL_TRACE=0 \
    CDB_AV_REUSE_REPLAY=0 \
    CDB_AV_REUSE_VALIDATE=0 \
    CDB_AV_VALIDATE_METADATA=0 \
    CDB_AV_DRAFT_DECODE_GRAPH=1 \
    CDB_AV_TRACE_GRAPH=0 \
    CDB_AV_DRAFT_GRAPH_VALIDATE=0 \
    CDB_AV_DRAFT_RAGGED_GRAPH=1 \
    CDB_AV_RAGGED_GRAPH_MIN_HITS=8 \
    CDB_AV_RAGGED_GRAPH_MAX_KEYS=12 \
    CDB_AV_DRAFT_RAGGED_DYNAMIC=1 \
    /home/yypan/CDB-AV/.venv/bin/python -m vllm.entrypoints.cli.main serve /disk14T/yypan/models/Qwen3-8B --served-model-name qwen3-8b --host 127.0.0.1 --port 18084 --tensor-parallel-size 1 --max-num-seqs 4 --max-model-len 1024 --gpu-memory-utilization 0.90 --enforce-eager --no-async-scheduling --no-enable-prefix-caching --seed 42 --attention-backend FLASH_ATTN --speculative-config '{"method": "draft_model", "model": "/disk14T/yypan/models/Qwen3-1.7B", "num_speculative_tokens": 8}'
```

The next bottleneck audit is in PROFILE_DRAFT_TARGET.md and the isolated
codex/cdb-target-modes-20261106 branch. Native FULL graphs expect query length9
at cap8;logged mixed-layout dispatch used PIECEWISE and39% padded token rows.
Dense PIECEWISE captures removed padding in a32-request diagnostic;this is
not yet the recommended performance profile. Keep H fixed. GPU preemption
pressure,strict end-to-end output equivalence,and final native baselines remain
separate acceptance gates;do not call the full system complete yet.
