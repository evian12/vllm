# Packed Target speculative metadata: final paired decision

Supersedes the performance-pending status in the earlier REPORT.md snapshot.
Implementation and GPU functional acceptance passed; throughput improvement
was **not established**, so keep `CDB_AV_PACKED_SPEC_METADATA=0`.

Source `a27d8b157b5dd6955c8899119606bbb1897d6dfd`, independent OFF/ON flag,
same pinned runtime and frozen client/request plans. Both arms retain the prior
positive Draft sequence-floor1024 optimization; compatible Target bounds and
GEMM padding remain OFF. GPU3 RTX4090, Llama8B/1B, ShareGPT512/C16,
max_num_seqs4, max_model_len1024, budget1024, BF16/FA2/TP1/eager outer execution,
manual Graphs enabled, no async scheduling or prefix cache, HF0.8/cap8/H8.
Fresh services;128/C16/128-token warmup excluded. Per-run output101003 tokens,
3072/3072 completed and606018 measured output tokens over all six runs.

| Seed | OFF output tok/s | ON output tok/s | Paired change | Full output ID equality |
| --- | ---: | ---: | ---: | ---: |
| 42 | 275.201828 | 274.943968 | -0.094% | 209/512 |
| 43 | 276.275598 | 276.399420 | +0.045% | 204/512 |
| 44 | 277.312761 | 276.227748 | -0.391% | 186/512 |

OFF **276.263396 ± 1.055519**, ON
**275.857045 ± 0.795393** output tok/s (sample SD,n3).
Ratio of means -0.147%; paired change
-0.147 ± 0.223 percentage points.
Mean of run-level E2E P95 21.889735 →21.448548 s;
TTFT P95 0.354609 →0.339253 s.
Latency means are not pooled percentiles. Some latency measures improved,
but signs and throughput magnitude do not justify enabling this as a throughput
optimization. No claim of statistically established slowdown either.

The six native-versus-packed GPU metadata outputs/dtypes match in32 actual
model audit samples, including mixed K/AR rows. The primitive128-case test and
joint234 CPU cases remain valid correctness evidence. Kernel/copy API counts
and bounded CPU timing did not predict end-to-end benefit; physical DMA counts
were not independently profiled. Possible host packing/event/allocation costs
offsetting saved submissions remain a hypothesis requiring instrumentation.

Full output IDs are not equal for all requests; differences are retained in
ANALYSIS.json and are unclassified. No assertion or test expectation was relaxed.
The mechanism gate does not establish strict whole-generation equivalence.

Target iterations OFF40405/ON40422; Draft dispatches OFF48592/ON48683.
Mean verified K 2.688644/2.690683;
candidate acceptance 0.802134/0.802093.
These are actual formal-interval observer/Prometheus deltas, checked against
each other, not native SD counters relabeled as Draft GPU work.

All six owned model services were released (`released.json`, port18084 closed,
no owned GPU PID). The next isolated candidate may subsequently occupy GPU3
only during its tests. No foreign process is stopped. Raw logs/results are kept
under /home/yypan/CDB-AV/results/spec_metadata_joint_20261120; exact launch,
runtime import, manifest and release records are archived per run. Rollback is
the packed0 launch in this same source version, no reset or checkout needed.

The recommended measured low-level change remains Draft sequence-floor1024
(276.290339±0.590459 versus271.474018±1.477580, three separate paired runs).
These two experiments are not combined to manufacture an additive speedup.
The target native SD4 historical341.073279±5.496631 remains unmet; overall status
is Partial Implementation under strict output equivalence. Native tests were
not rerun, H/HF/request selection were not changed.
