# Experimental AR-only Target stage bypass

`CDB_AV_AR_TARGET_BYPASS=1` enables an opt-in EngineCore.step optimization.
Default 0 preserves the existing path. Tested scope: single-GPU UniProc,
no async scheduling, fixed H=8, maxK=8, HF confidence0.8, Draft Graph enabled.

A bypass requires an open Draft window with 0<step<H, at least one owned
Draft lease, no published Ready or inflight verification, an EMPTY Waiting
queue, and every Running request at its ordinary AR input boundary with no
speculative candidates. It runs before Scheduler.schedule, so it discards no
Scheduler output and mutates no token/block accounting. Abort processing is
retained; post_step still advances one Draft step and publishes Ready at H.
Some unleased Running AR work can be deferred up to the remaining fixed
window. This is a scheduling tradeoff, not a guarantee of speedup on all loads.
It does not change Draft/KV operations, H, or overlap Target/Draft execution.

Do not infer Prefill eligibility from len(running): this CDB Scheduler excludes
frozen requests when enforcing Target capacity. An initial experimental guard
made that mistake; its runs under `pair32/` were stopped and are EXCLUDED.
The corrected gate unconditionally rejects any nonempty Waiting queue.

Validation: 149 relevant CPU tests (115 config,23 queue,11 KV ownership), plus
12 workflow tests. GPU smoke4_safe completed4/4,83 bypasses,47 Graph/eager
checks with exact tokens/KV;12 windows each performed exactly8 steps, and40
verifications followed their associated boundaries. New-gate GPU cancellation
and native preemption pressure are not yet separately exercised.

Three 32x128 paired repetitions, concurrency16,max_num_seqs4,seed42,FLASH_ATTN,
4 warmup requests, GPU1/18084, low logs, H8/maxK8/HF0.8, mixed Graph ON,
ragged Graph OFF, order OFF/ON then ON/OFF then OFF/ON:

| Pair | OFF TPS | ON TPS | OFF P95(s) | ON P95(s) |
|---|---:|---:|---:|---:|
|1|170.552|181.945|14.542|12.856|
|2|176.611|185.124|13.963|12.256|
|3|173.252|185.150|13.911|13.369|

All six passed32/32,4096 output tokens per measured run. Mean/sample SD TPS:
OFF173.472/3.035,ON184.073/1.843,+6.11%. This is small-load evidence only;
no 512 gain is claimed here. Candidate/acceptance counts differ; do not equate
Target stage bypass counts to removed GPU Forward calls (many stages were
empty already). End-to-end correctness across batch-shape variations remains
a separate acceptance requirement. Keep this experiment default OFF.

Artifacts: `/home/yypan/CDB-AV/results/ar_bypass_20261106/` includes full commands,
environment JSON, backend link hashes, source diffs, original failure logs,
smoke4_safe and pair32_safe/summary.json. Worktree FLASH_ATTN artifacts link to
the same binaries as the prior worktree; no dependencies were installed.
First startup lacked these links; that failure is preserved under smoke4/.
GPU was released after each suite, and the original18082 service was untouched.


## 512-request follow-up (same code commit 674c39034d941ad3569813c1229e1d5d3e03b58c)

One OFF->ON paired run,512 requests,16 concurrency,128 output tokens each,
other settings identical to the32-request suite. Both512/512,65536 outputs.
OFF181.216 TPS,P50 11.528s,P95 13.797s;ON185.756 TPS,P50 11.123s,P95 13.406s.
Gain+2.51% is one pair only; reverse repetitions remain required. OFF/ON
candidate verification counts17165/17757,drafted tokens41302/42942,accepted
29401/30255 (Prometheus totals include warmup). Different work/acceptance
trajectories prevent attributing the whole gain to CPU overhead removal.
Full config and metrics reside in pair512/;512_commit.txt records the tested
code,512_source.patch is empty. GPU1 returned to15MiB/0%,18084 closed,
original18082 process1682417 remained running. Candidate remains default OFF.
