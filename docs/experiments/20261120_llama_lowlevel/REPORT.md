# Llama C16 low-level optimization: completed decisions and next GPU comparison

This is a bounded research result, **Partial Implementation** under strict output
equivalence; it does not meet the native SD throughput target. Original evidence,
baselines, old reports, worktrees and unrelated processes are preserved.

## Common workload and scope

GPU3 RTX4090; Target Llama-3.1-8B-Instruct, Draft Llama-3.2-1B-Instruct;
frozen ShareGPT 512 requests, closed-loop client concurrency 16; variable output
lengths 16–384, 101,003 measured output tokens per run. Service max_num_seqs=4,
max_model_len=1024, max_num_batched_tokens=1024, BF16/FlashAttention2/TP1,
eager outer execution with opt-in manual CUDA Graphs, async scheduling disabled,
prefix caching disabled; HF threshold0.8, cap8, fixed H8. Each fresh service has
128/C16/128-token warmup. Seeds42/43/44, alternating OFF/ON order.
Native baselines were not rerun. Synthetic shape probes and profiled slices are
not substituted for paired end-to-end measurements.

## Draft dynamic-ragged Graph sequence floor

Source: `008c04f89f319b51b26187e81f93406b2322b9c9`; independent switch `CDB_AV_RAGGED_GRAPH_SEQ_FLOOR=0/1024`.
Decision: **retain_opt_in**. Each arm has three fresh-service runs, 512 requests per run,
101,003 measured output tokens per run, no failures; warmup is excluded.

| Seed | OFF tok/s | ON tok/s | Paired change | Complete output IDs equal |
| --- | ---: | ---: | ---: | ---: |
| 42 | 270.182134 | 275.664005 | +2.029% | 199/512 |
| 43 | 273.085112 | 276.836790 | +1.374% | 202/512 |
| 44 | 271.154807 | 276.370222 | +1.923% | 280/512 |

OFF **271.474018 ± 1.477580**, ON
**276.290339 ± 0.590459** output tok/s (sample SD).
Ratio of means +1.774%; paired changes
+1.775 ± 0.352 percentage points.
Mean of run-level E2E P95: 21.859044 → 21.820850 s;
TTFT P95 0.363666 → 0.347553 s.
These are means of three P95 statistics, not a pooled percentile.

Every owned service was released after its run (`released.json`); raw output
differences are retained in `ANALYSIS.json` and remain unclassified. Mechanism
checks passing does not establish strict generation equivalence.

## Draft down-projection FP32 GEMM row padding

Source: `d17d4ec22d4df0b2a48c019b50e8f9810989bfbd`; independent switch `CDB_AV_DRAFT_DOWN_PAD32=0/1`.
Decision: **keep_disabled**. Each arm has three fresh-service runs, 512 requests per run,
101,003 measured output tokens per run, no failures; warmup is excluded.

| Seed | OFF tok/s | ON tok/s | Paired change | Complete output IDs equal |
| --- | ---: | ---: | ---: | ---: |
| 42 | 269.122010 | 269.008903 | -0.042% | 185/512 |
| 43 | 270.979150 | 270.951246 | -0.010% | 179/512 |
| 44 | 270.470307 | 272.726485 | +0.834% | 190/512 |

OFF **270.190489 ± 0.959670**, ON
**270.895545 ± 1.859417** output tok/s (sample SD).
Ratio of means +0.261%; paired changes
+0.261 ± 0.497 percentage points.
Mean of run-level E2E P95: 21.943003 → 22.165979 s;
TTFT P95 0.358094 → 0.351730 s.
These are means of three P95 statistics, not a pooled percentile.

Every owned service was released after its run (`released.json`); raw output
differences are retained in `ANALYSIS.json` and remain unclassified. Mechanism
checks passing does not establish strict generation equivalence.

## Why these decisions

The sequence-floor change only merges sequence-bound keys for dynamic ragged
Draft captures. Actual token/request rows, positions, per-request sequence lengths,
query offsets, slots and leased blocks remain live and unchanged. It does not
generate padding Draft tokens. Three positive pairs support retaining this
opt-in optimization for this workload, not universal speedup. Its GPU gate passed
703 Draft Graph checks (365 floor-bound checks), 193 Target Graph checks,
210 cross-Target Carry KV byte checks, and 1,223 unique handoffs/verifications;
final lease/Ready/inflight counts were zero.

GEMM row padding is internal to 16 Draft down projections ([2048,8192] weights),
not extra model/attention requests. Local speedups did not survive repeated
end-to-end measurement; retain OFF. Projection arithmetic is not byte identical
to native and those differences are recorded rather than waived.

Compatible Target Graph bound reuse was also previously neutral
(272.987649±1.241109 →273.505562±1.956248 tok/s), despite more replay coverage;
it remains OFF. Do not combine independent noisy means to claim cumulative gain.

## Packed speculative Metadata: implemented, GPU gate passed, performance pending

Runtime commit **a27d8b157b5dd6955c8899119606bbb1897d6dfd**, worktree
`/home/yypan/CDB-AV/worktrees/cdb-spec-metadata-joint-20261120`.
Independent flag `CDB_AV_PACKED_SPEC_METADATA=1`, default0; safety/audit flag
`CDB_AV_SPEC_METADATA_VALIDATE=1` only for diagnostic runs. The paired test keeps
the positive Draft sequence floor1024 in both arms. It does not include the GEMM
padding patch and keeps compatible Target bound reuse OFF.

The CPU constructs the exact native integer vectors and direct Draft-input
indices, preserves each vector's dtype, packs six aligned views into one pinned
payload, and performs one payload-copy API call plus one Draft-token gather.
This replaces five native transfer API calls and the two-stage GPU gather/offset
chain. These are API call counts; physical DMA reduction was not independently
profiled. One pinned source is protected by a CUDA event before overwrite/growth;
each call has independent GPU storage so earlier metadata remains valid.
Unsupported scopes fail before the optimized path; default OFF retains native
execution. The optimization changes no scheduling, positions, attention/KV
ownership, token commit, HF policy or window logic.

30 feature CPU cases plus regressions pass; the joint CPU group has234 cases.
The standalone CUDA primitive has128 cases with independent native-reference
comparisons, mixed dtypes, allocation growth, pending-copy waits and live earlier
outputs. This primitive is not a model acceptance test.

Actual Llama functional gate:64 warmup+16 measured requests completed;
692 Draft Graph checks, 195 Target Graph checks,
208 cross-Target Carry byte checks, 1218 unique handoffs
and 1218 verifications; zero final leases/Ready/inflight.
All32 native-versus-packed metadata audits match six GPU outputs and dtypes,
with actual Worker request IDs and mixed K/AR rows (first [5,0,2,1]). This proves
the exercised metadata path, not whole-generation equivalence or performance.

Six formal 512-request runs are in progress under the existing isolated
controller. Do not launch duplicates or infer a performance benefit before all
three pairs complete. Heavy validation is OFF in every formal run.
Each driver releases only its own service process group in `finally` and verifies
port18084 closed/no owned GPU PIDs. Foreign jobs are never terminated.

## Remaining evidence and next iteration

The positive floor result276.290339 is still about19.0% below historical native
SD4 341.073279±5.496631, and below SD1 316.873769±0.569042. The goal is unmet.
Bounded steady profiling showed Target Forward514.69 ms versus Draft105.02 ms
(32 Target and37 Draft calls); summed GPU work is not a per-call like-for-like
comparison. Five metadata copies and GPU indexing were a measured CPU/launch
candidate (~0.383 ms inclusive per metadata call), not a complete explanation
of the throughput deficit. Sampling wait scopes include GPU wait and cannot
be counted again as pure CPU work. Gaps inside the same captured Graph were
only0.504 ms of that profiled slice; unattributed trace gaps are not proven idle.

After the metadata pairs, retain only measured winners, review remaining eager
fallback shapes, logits/synchronization and RPC costs against new bounded
profiles. Ready capacity/AR reservation also has read-only evidence of low
verification density, but changing request selection is outside this low-level
iteration. H and HF are fixed. No global optimum or strict output equality is
claimed; no safety assertion is disabled to pass tests.

## Provenance and reproduction

`SOURCE.json` and `PLAN.json` retain their original snapshots. Some inherited
`source`/`remote_head` fields and nested `PLAN.source` describe the earlier base
ccfd worktree, not the runtime feature branch. Do not treat them as actual runtime
evidence. Authoritative fields are each run's `manifest.json` source/source_commit,
its hash-checked preflight, `runtime_imports.json`, tracking manifest, exact
`launch.sh`, and the new `PROVENANCE_CHECK.json`. Original files are not rewritten
after execution. No performance attribution relies on a stale base field.

All six completed launch commands for each completed comparison are archived
in the matching run subdirectory, including environment, model paths, port and
vLLM arguments. Use only on an idle GPU3/18084 with the pinned worktree and
dependencies verified. Rollback is selecting the corresponding OFF launch
(`CDB_AV_RAGGED_GRAPH_SEQ_FLOOR=0` or `CDB_AV_DRAFT_DOWN_PAD32=0`),
or metadata flag0 in the joint code; no reset/checkout of another worktree.
The exact metadata diagnostic command is `spec_metadata_joint/gate_packed/launch.sh`.
Raw logs/outputs/traces stay under `/home/yypan/CDB-AV/results`; no historical
evidence or cache is deleted. The running service belongs only to the current
metadata comparison; completed experiments released all their own services.
