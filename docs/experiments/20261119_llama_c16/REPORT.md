# Llama C16 CDB reproduction and low-level audit — 2026-11-19

Status: GPU mechanism checks passed; performance target not met; strict generated-output equivalence remains unestablished. The subsequent compatible-bound Graph optimization is a separate experiment and is not included in these results.

## Reproducible source and workload

Runtime: `/home/yypan/CDB-AV/worktrees/cdb-llama-sharegpt-20261119`, branch `codex/cdb-llama-sharegpt-20261119`, commit `ccfd19da75969e79c242e860f18403eadad0e589`, pushed and remote SHA checked. Separate docs branches are not runtime versions.

Target Llama-3.1-8B-Instruct, Draft Llama-3.2-1B-Instruct; GPU3 UUID `GPU-b493bb52-b935-1cf4-47fe-7869125775b9`; BF16, FA2, TP1, max_num_seqs=4, max_model_len=1024, batched-token budget1024, no async scheduling or prefix caching. H8, HF confidence0.8, server K cap8, no client K override. Generic Draft/Target manual Graph, pinned staging, batched fingerprints and ownership-check reuse retained. Qwen-specific QK normalization is inapplicable and disabled.

Frozen ShareGPT512, closed-loop client concurrency16, identical Llama token IDs/output budgets to historical native references, seeds42/43/44; output lengths16–384, total101003 per run. This is not the older uniform128-output benchmark. Each fresh service gets128 warmup requests/C16/128 tokens, excluded from measurement. Formal runs disable heavy Graph/KV/metadata validation and detailed trace.

Complete service launch: `launch_llama_seed42.sh`; substitute seed43/44 for the other runs. On the server: `/home/yypan/CDB-AV/results/sharegpt_llama_cdb_gpu3_20261119/llama_seed42/launch.sh`. After each run the isolated service is terminated and port18084/processes/memory checked. All three runs and the later diagnostic trace have release records: port closed, no owned GPU PID, GPU memory15MiB at cleanup. Other services were not terminated.

## Formal performance

| Method | Runs | Output tok/s, mean ± sample SD | Mean run E2E P95, s | Mean run TTFT P95, s |
|---|---:|---:|---:|---:|
| CDB HF0.8/H8/cap8 | 3 | 271.859 ±0.712 | 22.094 | 0.356 |
| Historical native AR | 3 | 220.482 ±0.386 | 18.388 | 13.395 |
| Historical native SD K1 | 3 | 316.874 ±0.569 | 12.705 | 9.374 |
| Historical native SD K2 | 3 | 339.526 ±4.373 | 11.890 | 8.729 |
| Historical native SD K4 | 3 | 341.073 ±5.497 | 11.832 | 8.766 |

CDB runs:271.060488,272.089150,272.426369. All1536/1536 requests completed,303009 output tokens, zero HTTP/request failures. Native numbers were read from preserved results, not rerun. Same GPU UUID, workload plan hashes, public limits; native uses its original Torch compilation/Graph path and CDB the manual Graph path. The compilation-path difference is an implementation factor under investigation, not evidence of an isolated kernel speedup. Percentiles in the table are means of run percentiles, not pooled percentiles.

CDB versus native SD1 generated-token exact matches:161/165/181 out of512 per seed; versus SD4:163/176/170. These differences are retained and unclassified. Graph diagnostic checks do not establish strict full-request output equivalence.

## Mechanism and actual work

Functional gate:64 short warmup plus16 measured requests, all80 complete. Draft Graph576 checks, Target Graph195 checks,217 Carry byte checks all across real Target phases;1225 unique candidate handoffs matched1225 verifications; final live lease/Ready/inflight/released-block counters zero. K_actual1–8 and both confidence/max-K stops observed. A previous insufficient-warmup gate did not capture Target Graphs and was retained as an incomplete gate; the corrected gate uses more diagnostic input, not weaker validation.

Across the three formal runs:40416 Target calls,48639 Draft logical dispatches,68195 verified request events,183697 candidates,147332 accepted. Mean verified K2.6937, candidate acceptance80.2038%,1.6873 verified requests/Target call,3.7767 effective request rows/Draft dispatch. Draft counters include logical Graph replays and exclude extra capture-internal model execution. Prometheus iteration counts match the independent Target observer; verified rows match native speculative-request counts.

First seed:13487 Target calls versus historical native SD1 14102 and SD4 7577. Thus CDB does not simply issue more Target calls than every native configuration; it issues approximately78% more than native SD4 while having fewer Draft dispatches. Total Target rows52732 include22835 verify,29385 AR and512 prefill; these are actual scheduled rows, not output-token counters. Ready-to-schedule mean135.90ms, histogram P95 upper bound250ms. This is a CPU scheduling wait, not a GPU kernel duration.

Target Graph interior-interval replay fractions:9173/13312,8779/13312,8985/13312. Budget fallbacks3679/4065/3855, layout fallbacks460/468/472. The fixed cache filled during warmup; formal interior intervals captured no new keys. Interior counters omit boundary tails. High fallback count motivates examining compatible capture domains; increasing graph limits alone was not adopted.

## Bounded GPU trace, diagnostic only

Trace root `/home/yypan/CDB-AV/results/llama_stage_profile_20261119`, same source/model/config,128-request warmup followed by16 short requests; Torch/Kineto max32 Target iterations. No throughput comparison uses this profiled interval. GPU scope attribution follows unique launch correlation, including Graph kernels; inclusive CPU scopes are not added together.

Observed32 Target backbones:609.526ms kernel sum;35 Draft backbones:111.154ms; Target logits35.735ms; Draft logits19.637ms; Draft sampling/confidence2.199ms; KV clear0.041ms. These are different live shapes, including prompt work, not a controlled per-forward model latency ratio. Target/Draft matmuls dominate observed GPU work. CUDA memcpy/event API durations include waiting for earlier GPU work and must not be described as wholly removable copy overhead. Trace coverage gaps are not proof of hardware idle.

The initial parser counted CPU user annotations and their GPU mirrors twice in CPU range/shape summaries. It was corrected to restrict CPU ranges to `user_annotation`, with a regression test; the original summary remains archived. GPU kernel scope totals were unaffected. Corrected summaries and9 passing parser/hook CPU tests are retained. No new GPU run was needed for that parser fix.

## Next low-level candidate and safety boundary

Independent opt-in compatible-bound Graph reuse is under test in `codex/cdb-target-bound-reuse-20261119`, commit `8ed540f6bcf3204bd298dd12765d4c673f781efa`. It can reuse an existing larger sequence launch bound only when token rows, request rows, query domain, block-table width, splits and capture mode match exactly. It does not pad rows, mutate live positions/lengths, share KV ownership or change HF/H/scheduling. Default remains OFF. Strict GPU and paired end-to-end evidence is required before recommending it; implementation/CPU tests alone do not imply benefit.

An additional read-only short-gate finding shows non-Ready Running rows reserving Target capacity while Ready work waits (`READY_CAPACITY_AUDIT.json`). This is a scheduler-policy candidate, not a current low-level change or a formal512 frequency estimate. No selection strategy was changed. HF tuning, historical oracle K and dynamic H are out of scope for the ongoing low-level iteration.

All raw logs, responses, launches, environment/import snapshots, hashes, failed gates and release records are retained on the server. Results are soft-linked from the source directories. Revert experiment configuration by leaving the new opt-in switch unset/0 and launching the preserved source/config; no destructive Git rollback is needed.
