# Formal-prefix growth audit (2026-11-06)

`audit_reentry.py` joins actual `p2_verified` and `reuse_pending_prefill` events chronologically by request ID:

```
growth = new_context_tokens - (verified_context_tokens + len(generated))
```

`generated` includes accepted candidates and the Target correction/bonus. Growth counts formal tokens appended after verification, excluding its own output. Missing evidence, malformed events, inconsistent preparation lengths and negative growth are reported separately. `uncached_or_fallback` is not necessarily a fresh request. Empty mixed-batch statistics mean no such trace evidence, not zero Forward calls.

CPU-only reproduction:

```bash
/home/yypan/CDB-AV/.venv/bin/python scripts/cdb_av/audit_reentry.py LOG
/home/yypan/CDB-AV/.venv/bin/python -m pytest -q --confcutdir=scripts/cdb_av/tests scripts/cdb_av/tests/test_audit_reentry.py
```

Existing GPU traces under `/home/yypan/CDB-AV/results/request_average_20261105/`:

| Trace | Cached preparations | With later growth | Growth tokens | Cached preparation input tokens | Query >2 with growth |
| --- | ---: | ---: | ---: | ---: | ---: |
| phase_trace32/hf_graph_phase32_server.log | 1068 | 393 | 1100 | 2676 | 303 |
| ragged_dynamic_validate128/dynamic_validate128_server.log | 1031 | 393 | 1132 | 2670 | 300 |
| ragged_dynamic_trace32/dynamic_trace32_server.log | 1017 | 405 | 1048 | 2572 | 317 |

These are 32-request / 128-output diagnostic traces, not 512 performance replicates. `validate128` also performed Graph/eager checks. The first trace predates dynamic-layout; the latter two use its development worktree. No new GPU workload was run for this analysis.

Log SHA256, in row order:

```
35f076845f88d9253d6adda94357a1fe1fe2d5831328cac46aade07a507fbc43
a71defb701dc52894904d829591934a0831c96ab6c69e172f92392466c7f24f6
eb77bb84352c4af922309ba431d93d3271b957ec5be824f2d1bce698d66450d7
```

No invalid/missing cached associations were observed. Every noninitial preparation longer than two tokens coincided with formal growth. The latter two logs show growth in 38.1%/39.8% of cached re-entries; growth accounts for 42.4%/40.7% of their preparation input tokens. This association is not a measured speedup or proof that these tokens were wasted.

`PersistentDraftQueue.advance()` executes one Draft step after each Engine iteration. The next Target stage may perform AR/Prefill; current-window Ready work remains protected until H. `ReuseDraftQueue.plans()` can reuse a certified cache even when AR has appended tokens, and must prepare the missing suffix. This legitimately produces query lengths above two and takes the eager path of the current bounded Graph implementation.

Next measure Target AR rows and time between Verify and re-admission alongside Draft suffix lengths and Graph misses. A bounded AR-only bypass is a candidate only when no Ready verification/Prefill is schedulable and Draft work can advance. Keep H and KV assertions fixed; validate lifecycle, admission fairness and latency before performance claims. This tool changes no runtime scheduling, KV lifetime or model computation.
