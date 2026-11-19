# Llama low-level iteration — interim GPU evidence

This report separates completed mechanisms from pending performance acceptance. HF remains0.8, H8, server K cap8, max_num_seqs4. No HF algorithm, dynamic H, request-selection change, native rerun or browser-algorithm integration. Neither candidate below is yet recommended ON; all flags default OFF.

## Completed negative/neutral experiment

Compatible Target Graph bounds, commit8ed540f6bcf3204bd298dd12765d4c673f781efa:3 OFF/ON pairs completed,3072 requests606018 tokens. OFF272.987649±1.241109 versus ON273.505562±1.956248 tok/s. Paired mean+0.190079±0.642373 percentage points, mixed signs. Higher replay counts did not establish end-to-end benefit; leave OFF. Detailed report in the separate TargetBoundReuse delivery and server results.

## Candidate1 — FP32-accumulation Draft down GEMM

Commitd17d4ec22d4df0b2a48c019b50e8f9810989bfbd on `codex/cdb-llama-linear-20261119`, normal push/remote SHA checked. Files: `vllm/v1/spec_decode/cdb_av_linear.py`, three-line opt-in hook in `draft_model.py`, `tests/v1/spec_decode/test_cdb_av_linear.py`.48 final feature tests and15 workflow checks passed; earlier combined regression156 tests passed before adding six extra dispatch-boundary cases.

Flag `CDB_AV_DRAFT_DOWN_PAD32=1`. Restricts Llama1B16-layer BF16 eager/TP1, no async/LoRA/quantization/sleep. Only16 down projections of shape[2048,8192], live token rows2–10,12,16. Other shapes/projections use original methods. Temporary projection rows are padded to32 and sliced back; attention/model request rows, IDs, positions, block tables, KV slots, request progress and weights are unchanged. FP32 accumulation is selected during the call/capture; process reduction flag restored in `finally`; owner-thread check; full-model validation before installing any methods; reinstallation/training rejected. This is internal GEMM padding, not completed requests participating in later model/attention forwards.

Independent synthetic tests stream≥256MiB of weights, alternate eight trials and include padding GPU work. Original reduced-precision padding had approximately0.0026 relative L2 error; disabling reduced accumulation preserves approximately21% component speedup with relative L2 approximately0.0002. Physical-column variants were also checked and not globally adopted. Component gains alone are not a performance claim.

Real model gate64+16 requests all80 completed;556 Draft Graph checks,190 Target Graph checks,210 Carry byte checks all across real Target phases;1188 unique candidate handoffs/verified arrays match. Ending live leases/Ready/inflight/released blocks all zero, service/port released, memory15MiB at cleanup. Four actual projection audits compare the same real input/weights against original native projection: max_abs0.000244–0.000488, relative L2 .000135–.000345,4/4 projected argmax rows equal each; not byte-exact. Projection argmax is not a model token. These comparisons retain numerical differences; they do not prove full-generation equivalence. Existing Graph checks compare new Graph against new eager execution and preserve their original assertions.

Three same-commit OFF/ON pairs are running, fresh128/C16/128 warmup and identical ShareGPT512/C16/101003-output-token load. First pair269.122010 OFF versus269.008903 ON (−0.0420%), E2EP95 21.907053/22.301977s,185/512 full outputs exact. No stable speedup established. Validation/audit disabled for performance. Full launch files `launch_linear_off.sh`/`launch_linear_on.sh` show actual environment, models, source, GPU3 and port18084.

Server root `/home/yypan/CDB-AV/results/llama_linear_20261119`; runtime `/home/yypan/CDB-AV/worktrees/cdb-llama-linear-20261119`. Read STATUS/RESULTS_PARTIAL/COMPLETE before resuming; do not duplicate running jobs. Release/rollback by closing only the owned service and using flag0/unset on the preserved launch; no Git reset is needed.

## Candidate2 — normalize only ragged Draft sequence bound

Commit008c04f89f319b51b26187e81f93406b2322b9c9 on `codex/cdb-draft-seq-bound-20261119`, normal push/remote SHA checked. Files: `cdb_av_refill.py`, `cdb_av_config.py`,18 new sequence-bound tests.204 CPU regression tests plus15 workflow checks passed.

Flag `CDB_AV_RAGGED_GRAPH_SEQ_FLOOR=1024`, default0. Source inspection ruled out raw block-table width as a key dimension: Draft already buckets/preallocates its table. The experiment instead merges sequence buckets for dynamic ragged Graphs, retaining exact token/request counts/query domain and copying actual lengths, query offsets, slots and block IDs. Static unused table columns remain zero. Query1 decode and exact-layout modes keep original keys. Full model-length/block-table capacity checks remain. Dynamic Llama/FA2/BF16/eager synchronous TP1 scope is required; no KV lease or scheduling mutation. Linear candidate is absent from this branch and compatible Target bound reuse is OFF, so the comparisons isolate this factor.

Real gate64+16 requests all80 completed;703 Draft Graph checks including365 actual dynamic checks at the new1024 bound;193 Target Graph checks;210 cross-Target Carry byte checks;1223 handoffs/verified arrays match; reset counters zero, owned service/port closed and GPU released. Existing byte/token/logit checks and collision/context guards remain. `launch_floor_functional.sh` is the actual heavy-validation launch, not a performance configuration.

Formal three-pair OFF/ON testing is queued after candidate1 finishes; no performance result yet. Server root `/home/yypan/CDB-AV/results/draft_seq_bound_20261119`, runtime `/home/yypan/CDB-AV/worktrees/cdb-draft-seq-bound-20261119`. The early gate was inserted only after the preceding measured service finished and released GPU3; the CPU sequence coordinator was paused/resumed, not its live service. Historical files and other users' services remain intact.

## Actual short decode trace and next audit

Separate diagnostic root `/home/yypan/CDB-AV/results/llama_steady_profile_20261119`; same baseline commitccfd19da, HF/H/limits/models unchanged.128-request warmup,16 short requests max128 output; profiler starts1.5s after requests begin, max32 Target iterations, stack/shapes/memory tracing OFF. Thirty-two observed Target inputs all4–17 token rows; thus initial large-prefill dominance from the earlier trace is avoided, but this is still a diagnostic sample, not formal512 timing.

Observed GPU kernel sums: Target backbone514.692ms/32, Target logits35.710ms/32, Draft backbone105.024ms/37, Draft logits20.642ms/37. CPU inclusive spans: state update .150ms/call, input preparation .914ms (includes nested work), speculative metadata .383ms, attention metadata .193ms, ownership checks .355ms, Draft staging .381ms. Do not add nested spans or call sampling wait time pure CPU sampler work.

GPU activity union681.916ms over1006.871ms observed span; uncovered intervals324.955ms. Correlation/union analysis: approximately96.438ms within Target Forward CPU scopes,69.995ms within Draft Forward CPU scopes,137.736ms between scopes/unattributed, and only.504ms between kernel records of the same Graph launch. This motivates Graph fallback/host-submission review; it does not prove physical GPU idle or fully removable CPU overhead, because profiler effects/incomplete coverage can contribute. Parser excludes GPU mirror annotations, uses unique launch correlation, handles overlapping intervals and missing correlations;4 new interval tests pass. All traces/services released.

Additional source evidence: `GPUModelRunner._calc_spec_decode_metadata()` still has an explicit CPU→GPU-copy TODO and five separate NumPy tensor transfers, followed by multiple index/gather kernels. The sampled CPU span is approximately.383ms. A future packed transfer or equivalent direct-index preparation is a concrete low-level candidate, but not implemented/tested here. Preserve dtype, index mapping, host-source lifetime, stream ordering and GPU output lifetime; validate against unchanged native metadata field-by-field before any performance test.

The overall strict-output status remains Partial Implementation. No below-native result is called a success; no local kernel speedup substitutes for replicated end-to-end throughput/P95. The target remains historical native Llama SD4 341.073±5.497 tok/s on the same frozen workload/physical GPU3. Research logs, responses, manifests, hashes, imports and release records are retained and results soft-linked from the code trees. Two-hour iteration heartbeat is ACTIVE and should inspect current controllers before taking action.
