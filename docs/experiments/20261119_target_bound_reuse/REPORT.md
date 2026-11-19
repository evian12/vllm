# Compatible Target Graph sequence bounds — completed experiment

Runtime `/home/yypan/CDB-AV/worktrees/cdb-target-bound-reuse-20261119`, branch `codex/cdb-target-bound-reuse-20261119`, commit8ed540f6bcf3204bd298dd12765d4c673f781efa; GitHub remote SHA verified. Flag `CDB_AV_TARGET_GRAPH_REUSE_BOUNDS=1`, default0. Only an existing larger sequence upper bound can be reused; token/request counts, query domain, block-table width, splits and capture mode must match. Model/KV storage identity checks remain. No row padding, KV ownership, HF/H or scheduling change.

Llama-3.1-8B/Llama-3.2-1B, GPU3, BF16/FA2/TP1, no async or prefix caching, max_num_seqs4, max_model_len1024, token budget1024, HF0.8/H8/cap8. Frozen ShareGPT512, closed-loop C16, output16–384/101003 total, seeds42/43/44. Each fresh service gets128/C16/128 warmup, excluded. OFF/ON order alternates42:[0,1],43:[1,0],44:[0,1]. Heavy validation/trace OFF for all six performance runs. Public config, actual commands, client/data/observer/driver hashes and source are checked by analysis; only the feature flag and private observer directory differ between arms. Shared-host load/clocks are not fixed; these are GPU-isolated runs, not an otherwise idle machine.

| Seed | OFF tok/s | ON tok/s | Paired change | OFF E2E P95 s | ON E2E P95 s |
|---|---:|---:|---:|---:|---:|
|42|273.090697|271.597128|−0.5469%|21.995552|21.921064|
|43|274.174021|275.506353|+0.4859%|22.073567|22.359992|
|44|271.698228|273.413205|+0.6312%|22.182390|21.801370|

OFF272.987649±1.241109 versus ON273.505562±1.956248 tok/s, sample SD across three repetitions. Ratio of means≈+0.190%; mixed paired signs and run variance do not establish a reproducible speedup. Mean run E2E P95 22.083836/22.027475s; TTFT P95 .352027/.354737s. All3072/3072 requests completed,606018 tokens, zero request failures. Native was not rerun. Recommendation: leave OFF; this does not close the gap to historical native Llama SD4 341.073±5.497 tok/s.

First seed interior replay increased9238→11009 of13312 calls and budget fallbacks fell3602→1847, yet TPS did not improve. This is evidence against treating replay count as an end-to-end optimization result. Interior counters omit interval boundary tails.

CPU tests114 plus15 workflow tests passed. GPU gate64+16 short requests completed;99 bound-reuse replays,241 Target Graph checks,577 Draft Graph checks,218 cross-Target Carry byte checks passed;1223 unique handoffs matched1223 verification candidate arrays; ending lease/Ready/inflight/released-block counters zero. Existing collision, position, context and progress checks remain.

Strict whole-request output identity OFF/ON:219/193/214 of512 across seeds; differences remain unclassified. Graph/static/eager KV checks are local and do not prove full-generation equivalence. Overall system remains Partial Implementation under the strict-output criterion.

All six services closed on completion and owned GPU PIDs/port18084 were checked. Two independent short synthetic component tests were inserted between fresh services after the preceding service had completed and released its resources; no measured service was paused or overlapped with another GPU job. CPU coordinators alone were paused/resumed, with logs retained. Other users' GPU services were untouched. CPU development/validation activity and shared-host load were not fully excluded; sub-percent effects need stronger evidence than these results.

Raw results, full launches, hashes/import paths, responses, counters, GPU samples and release records: `/home/yypan/CDB-AV/results/target_bound_reuse_20261119`. Complete startup commands are in each `pair_seed*_bound*/launch.sh`. A fresh OFF run uses `/home/yypan/CDB-AV/.venv/bin/python /home/yypan/CDB-AV/results/target_bound_reuse_20261119/driver.py --tag new_off_seed42 --mode hf --k-cap 8 --seed 42 --reuse-bounds 0` after checking GPU3/18084 are free and no experiment controller is using them. Rollback: use flag0/unset with the preserved launch; no destructive Git command is required.

Next independent candidates: FP32-accumulation Draft down-projection experiment (separate source commitd17d4ec2), and ragged Draft sequence-bound normalization (separate source commit008c04f8). They are not included in the results above, remain opt-in and require their own GPU and paired performance gates.
