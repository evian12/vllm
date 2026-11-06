# Opt-in query-four Draft diagnosis

Based on e44942d486aed49ef8acc399d37a98745573c9b5. The previously checked
Draft/Target host profiler1948e355 is integrated without changing model inputs,
selection, H, HF thresholds, KV ownership or invalidation semantics.
CDB_AV_PROFILE_DRAFT=1 is diagnostic only; default0 preserves execution.
The query bound remains default2; the GPU diagnostic explicitly selects4.

New counters distinguish eager fallback from cache_budget, frequency_warmup,
query_gt_bound and unsupported_layout/disabled_or_phase; these are the gates
actually applied to the current forward. Actions also count actual logical
request rows, actual token rows and maximum query length.
They are GPU forward batch counts, not the native request-level spec proposal
counter. Do not mistake input-query length for Draft K.

Optional invalidation timing covers the existing physical-slot clearing
operation; all original tensor writes, block selection and empty-range
behavior remain. Three CPU tests check profiling OFF/ON preserve those writes
and an empty range neither accesses model state nor adds a sample.

Timings are host elapsed intervals, not isolated CUDA kernel durations.
gpu_wait_copy_ns includes preceding pending GPU model work. RPC is the
synchronous same-process UniProcExecutor path and includes Worker work, not
network latency. Periodic Target, Draft and RPC counters cover different
prefixes, not aligned whole-run totals. Invalidation outside Draft step is not
to be blindly added to/compared against step_total_ns. A separate diagnostic
run has nonzero timing overhead and cannot be treated as a throughput test.

GPU results and paired query2/query4 results will be recorded after validation.
Raw evidence: /home/yypan/CDB-AV/results/draft_q4_20261106/.

## Actual GPU results

201 CPU functionality tests and12 workflow tests pass,including3 new physical
invalidation semantics tests. A final4/4 GPU smoke has51 Graph/eager checks;
all static/live metadata KV/token checks andlogits_close pass. Initial probes
failed FLASH_ATTN import because the fresh worktree lacked existing Python
wrappers;the same baseline wrappers were linked,not installed or reconfigured.
profile4/ andprofile4_retry/ are retained failures;profile4_retry2/ is successful.

profile32 completed32/32. Last512 effective Draft-step prefix:
Forward3.240696s;190 eager2.749558s,21 captures0.408282s,
301 replays0.082857s;112 warmup fallbacks1.614667s;
78 query>4 fallbacks1.134891s;no budget fallback in this prefix.
Metadata0.131184s,confidence submit0.203409s,GPU wait/copy1.559171s;
step total5.272324s.29 suffix invalidations,70 tokens,0.024443s host.
Actual logical Draft average3.781 requests/forward. Target512-stage prefix
10.861129s,2011 request rows,maximum4;different endpoint from Draft prefix.

profile128 completed128/128. Last2816 effective Draft-step prefix:
643 eager9.497262s,25 captures0.494313s,2148 replays0.581298s.
446 query>4 fallbacks6.647884s;127 frequency fallbacks1.841006s;
70 cache-budget fallbacks1.008372s. Input max-query5..8 accounts256 calls,
all eager;longer initial Prefills andsuffixes also occur. Graph replay fraction
76.28%,capture0.89%,eager22.83%. Metadata0.725990s,confidence submit0.746304s,
GPU wait/copy10.348641s,step total23.073803s.167 invalidations,406 tokens,
0.148366s host;this is much smaller than eager-forward intervals,not the
primary observed hot path. Average3.769 Draft requests peractual forward.
Target2304-stage prefix49.007800s,8975 request rows(average3.895,max4);
4394 verify,4453 AR,128 preparation rows. Oneempty host stage0.190ms is not
evidence of an empty GPU model forward. Do not sum these unequal prefixes.

Interpretation:query>4 eager processing remains the main measured Draft host
interval. Warmup is conspicuous inshort runs;cache-budget misses emerge in
longer runs. Larger static domains andbudget changes need separate correctness
andpaired tests. Target capacity is already well filled;runtime native Target
graph coverage andunnecessary AR stage execution remain candidates. No such
new optimization is included here. Full-layer invalidation is retained.
This is diagnostic evidence,not an isolated GPU-time decomposition or a claim
that no other bottlenecks exist. Strict final-output equivalence andnatural
memory-pressure lifecycle coverage remain incomplete.
