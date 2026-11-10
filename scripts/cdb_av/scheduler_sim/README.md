# CDB-AV Event-Driven Scheduler Simulator

This is a CPU-only discrete-event simulator for studying the outer scheduling decision:
**after each completed Forward, should the single GPU execute another Draft Forward or a Target Verify Forward?**

## Research scope (Phase 1)

- One GPU executes exactly one Forward at a time. Draft and Verify never overlap.
- Requests are assumed to have completed Target Prefill before becoming Draft Eligible.
- AR and Target Prefill scheduling are intentionally excluded. They will be added only after the outer Draft/Verify policy is understood.
- Draft is request-ragged at the simulator level: each active request advances one candidate position per Draft Forward and exits independently at its request cap, confidence stop, or output limit.
- Verify selects at most max_verify_batch_size Ready requests in FIFO order. Unselected requests remain Ready.
- A Verify commits the accepted candidate prefix plus one correction/bonus token, capped by the request's remaining output budget.
- acceptance_probs[j] is the conditional probability that candidate j is accepted given that all earlier candidates were accepted.
- The confidence abstraction checks the confidence attached to a generated candidate. The candidate is retained, and the request becomes Ready if that confidence is below the threshold. This is a simulation convention, not a claim about the exact Worker implementation.

The event-driven policies do not receive a window H. The simulator records the realized number of Draft Forwards between consecutive Verify Forwards as an output metric. Only fixed_h consumes H.

## Policies

| Policy | Decision |
|---|---|
| fixed_h | Fixed-H reference baseline; after H Draft Forwards, execute a Verify batch when Ready work exists. |
| threshold | Verify when Ready Queue cardinality reaches a threshold, or when Draft cannot run. |
| pressure | Compare expected useful Draft/Verify work per estimated millisecond, adjusted by Ready backlog and age. Includes a consecutive-Verify guard. |
| lookahead | Evaluate each legal first action using short rollouts controlled by the queue-pressure policy. |

### Queue-pressure decision

For the currently active Draft batch and the FIFO Verify batch that fits the Target capacity:

- Draft rate: sum of each active request's next-position conditional acceptance probability divided by estimated Draft duration.
- Verify rate: sum of each Ready request's expected accepted-prefix-plus-correction/bonus tokens divided by estimated Verify duration.
- Verify score multiplies Verify rate by a Ready-backlog factor and adds normalized oldest-Ready age.
- Draft score multiplies Draft rate by a Draft-Eligible supply factor.
- Verify is selected when its score is at least the Draft score. An oldest-Ready wait-budget guard and a consecutive-Verify guard provide additional starvation protection.

The next-candidate Draft contribution is the unconditional marginal probability: the product of conditional acceptance probabilities through the current depth and the next position's conditional probability. The exact weights are exposed in QueuePressurePolicy. They are heuristic parameters to be swept in simulation, not analytically optimal constants.

### Lookahead decision

For each legal first action, LookaheadPolicy clones the current state, executes that action, and rolls forward for a bounded number of additional Forwards using QueuePressurePolicy. It compares:

```
(committed_tokens + discount * terminal_candidate_potential) / elapsed_simulated_ms
```

The rollout is an approximate model-predictive heuristic, not an exact solution of the SMDP Bellman equation.

## Run

From the repository root:

```bash
python -m scripts.cdb_av.scheduler_sim --requests 128 --arrival-rate 20 --arrival-mode poisson --policy all
```

Bursty workload and a smaller lookahead run:

```bash
python -m scripts.cdb_av.scheduler_sim --requests 64 --arrival-rate 30 --arrival-mode bursty --policy all --lookahead-horizon 4 --lookahead-rollouts 2
```

Save machine-readable results:

```bash
python -m scripts.cdb_av.scheduler_sim --requests 512 --arrival-rate 50 --policy all --output /tmp/cdb_av_sim.json
```

## Measured execution profile

Pass a JSON file with any subset of the following non-negative coefficients:

```json
{
  "draft_base_ms": 0.6,
  "draft_per_request_ms": 0.22,
  "draft_per_context_token_ms": 0.000015,
  "verify_base_ms": 1.0,
  "verify_per_request_ms": 0.32,
  "verify_per_candidate_ms": 0.055,
  "verify_per_context_token_ms": 0.000012
}
```

These values are examples only. For GPU evaluation, fit the coefficients from measured Draft/Verify batch-shape profiles on the target host. Keep the same profile, request trace, and seed across policy comparisons.

## Metrics

The simulator reports committed output tokens/s, first-commit latency, Ready queue wait sampled at Verify dispatch (excluding Verify service time), inter-Verify commit-gap (a phase-1 proxy, not full ITL), realized H distribution, batch sizes, slot utilization, GPU busy ratio, and candidate acceptance rate.

## Validation

The simulator uses only the Python standard library. Run:

```bash
python -m compileall -q scripts/cdb_av/scheduler_sim
python -m unittest discover -s scripts/cdb_av/tests -p 'test_scheduler_sim.py' -v
```

No vLLM runtime code, CUDA kernels, KV cache implementation, or GPU settings are modified by this research branch.


## Phase-aware Target scheduling (Phase 2)

Phase 1 above remains available as the isolated Draft/Verify reference. The
phase-aware simulator is a separate model and entry point:

```bash
python -m scripts.cdb_av.scheduler_sim.phase_cli \
  --requests 64 --arrival-rate 30 --arrival-mode bursty \
  --policy all --seed 42
```

Phase 2 adds the Target-side request lifecycle:

- Every new request enters Target Prefill first. A prompt can span multiple
  Target forwards; the first output token is committed only after its final
  Prefill chunk completes.
- A speculative request then enters Draft; a non-speculative request enters
  one-token-at-a-time Target AR.
- One Target forward may contain a mixture of atomic Verify requests, AR
  requests, and Prefill chunks. The combined batch must respect both
  `max_num_seqs` and `max_num_batched_tokens`.
- Draft and Target still execute serially on one GPU. The simulator does not
  model GPU overlap.

### Phase-aware comparison policies

| CLI policy | Stage decision | Target batch organization |
|---|---|---|
| `fixed_h` | Fixed number of Draft forwards between Verify-bearing Target forwards; Prefill/AR-only Target steps do not reset H | FCFS, chunked Prefill |
| `queue_pressure` | Queue pressure and waiting-time guards | FCFS, chunked Prefill |
| `lookahead` | Short-horizon rollout over Draft/Target actions | Queue-pressure rollout with FCFS Target batches |
| `sarathi` | Queue pressure | Bounded Prefill chunks, Decode-like work first |
| `length_aware` | Queue pressure | Short/long queues, similar-length buckets, adaptive Prefill budget |
| `joint` | Queue pressure plus a short-Prefill batching wait | Length-aware Target batching |

The length-aware policy is an **LAPS-inspired algorithmic approximation**, not
a reproduction of LAPS. Its adaptive waiting window \(W\) is estimated from
recent short-Prefill arrival intervals and bounded by configured minimum and
maximum values. Its Prefill budget \(D\) grows with queued Prefill tokens and
shrinks when Ready/AR work is urgent. Long prompts are chunked independently
of the short-request bucket.

### Phase-2 metrics and interpretation

The phase-aware output includes committed output tok/s, TTFT (measured when the
initial Target Prefill completes), Prefill queue wait/latency, Ready wait, AR
wait, commit-gap, Target token/slot utilization, mixed-Target-forward ratio,
Prefill chunk sizes, Target batch composition, and the realized adaptive
Prefill wait-window (W) and token-budget (D) averages.

The reported commit-gap is the interval between per-request commit events. It
is **not** token-level ITL inside a multi-token Verify commit. Default execution
coefficients are synthetic and are only useful for testing scheduling logic;
fit them from GPU batch-shape profiles before drawing hardware-performance
conclusions.


## Load-adaptive verification

`load_adaptive` is an additional stage policy that makes verification delay conditional on predicted batch growth:

- If one more Draft Forward does not increase the number of requests that can enter the next Target batch, verify immediately. This is the sparse-load behavior: waiting cannot reduce Target Forward count through batching, so it only adds request latency.
- If one more Draft Forward is predicted to add at least one Target request, it may continue Draft to form a larger batch.
- Ready, AR, and Prefill wait budgets remain hard guards; a projected deadline violation forces Target.
- A full current Target batch is verified immediately.

This is a first deterministic marginal-batch-gain policy, not yet an arrival-rate classifier or a calibrated optimal policy. Evaluate it across low, medium, high, and saturation arrival rates before drawing performance conclusions.


### E5 — Load-Adaptive + Deadline-aware Target batching

E5 keeps the exact same `LoadAdaptivePhasePolicy` stage decision as the existing `load_adaptive` baseline, but swaps its Target batcher from FCFS/VLLM-like ordering to `DeadlineAwareTargetBatcher`.

This isolates whether the high-load Prefill/TTFT tail observed with load-adaptive stage control is caused by Target batch composition rather than the decision to delay Verify. Compare `load_adaptive` and `load_adaptive_deadline` on identical generated workloads and seeds. No GPU-performance claim follows from the CPU simulator.


### Target Forward contention diagnostics

The phase-aware simulator records resource use specifically for Prefill-containing
Target forwards, rather than relying only on utilization averaged across all
Target forwards. The summary includes:

- \`prefill_containing_target_forwards\`: number of Target forwards containing at
  least one Prefill chunk.
- \`prefill_containing_slot_utilization_{mean,p50,p95}\` and
  \`prefill_containing_token_budget_utilization_{mean,p50,p95}\`: per-forward
  utilization conditioned on a Prefill-containing forward.
- \`prefill_containing_duration_{mean,p50,p95}_ms\`: wall-clock duration of those
  forwards under the configured execution profile.
- \`prefill_chunk_requested_tokens\`, \`prefill_chunk_allocated_tokens\`,
  \`prefill_chunk_budget_shortfall_tokens\`, and
  \`prefill_chunk_budget_truncations\`: distinguish a Prefill chunk's requested
  cap from the amount actually admitted to the Target forward. Per-item traces
  include the requested/allocated counts and whether a shortfall came from the
  Target token budget or an adaptive Prefill-budget/token-budget constraint.
- \`target_exclusion_events\`: counts by request class and exclusion reason.
  Reasons distinguish \`sequence_slots_exhausted\`,
  \`token_budget_exhausted\`, their joint exhaustion,
  \`prefill_policy_budget_exhausted\`, and \`policy_deferred\` (for example, a
  length-bucket policy not selecting a Prefill request in the current batch).
- \`ar_excluded_by_sequence_slots_events\` and
  \`ar_excluded_by_token_budget_events\`: convenient AR-specific exclusion
  counters. These count request-forward exclusion events, not unique requests.

To include the complete per-Target-Forward trace in the output JSON, pass
\`--include-target-forward-trace\`:

\`\`\`bash
python -m scripts.cdb_av.scheduler_sim.phase_cli \\
  --requests 32 \\
  --arrival-rate 60 \\
  --arrival-mode poisson \\
  --policy load_adaptive_deadline \\
  --deadline-prefill-chunk-tokens 1024 \\
  --seed 42 \\
  --include-target-forward-trace \\
  --output /tmp/cdb_target_trace.json
\`\`\`

Each trace entry records start/end/duration, batch size, slot and token-budget
capacity/use/remaining, per-class request and token counts, selected request
items, and queued candidates excluded from that forward with the remaining
capacity at the point of exclusion. The trace is opt-in to keep large workload
outputs compact. This instrumentation does not change batch selection or
execution timing. All values remain outputs of the CPU simulator's configured
service-time model; they are not GPU measurements.

#### Run the contention-diagnostic sweep

The focused diagnostic matrix uses three Prefill chunk sizes, three medium/high
arrival rates, and five paired seeds (45 CPU-only runs). It keeps the stage
policy fixed at \`load_adaptive_deadline\` and changes only the configured
Prefill chunk cap:

\`\`\`bash
cd ~/CDB-AV/vllm-phase-aware-target-scheduler-20261110
mkdir -p /tmp/cdb_target_contention_raw

for CHUNK in 256 1024 2048; do
  for RATE in 30 60 100; do
    for SEED in 42 43 44 45 46; do
      python -m scripts.cdb_av.scheduler_sim.phase_cli \\
        --requests 32 \\
        --arrival-rate "$RATE" \\
        --arrival-mode poisson \\
        --policy load_adaptive_deadline \\
        --deadline-prefill-chunk-tokens "$CHUNK" \\
        --seed "$SEED" \\
        --include-target-forward-trace \\
        --output "/tmp/cdb_target_contention_raw/cdb_diag_c\${CHUNK}_r\${RATE}_s\${SEED}.json"
    done
  done
done

python -m scripts.cdb_av.scheduler_sim.analyze_target_contention \\
  --input-dir /tmp/cdb_target_contention_raw \\
  --output-dir /tmp/cdb_target_contention_report \\
  --baseline-chunk 1024
\`\`\`

The analyzer validates unique (chunk, rate, seed) combinations and the presence
of a trace, then writes:
- \`target_contention_summary.csv\`: mean/sample standard deviation across seeds.
- \`target_contention_composition.csv\`: Prefill-only, Verify-only, AR-only, and
  mixed-forward counts.
- \`target_contention_paired.csv\`: same-seed deltas against chunk 1024.
- \`target_contention_report.md\`: compact tables for interpretation.

Rate 15 is intentionally omitted from this focused pass: the immediate
question is whether Target capacity constraints change batch composition under
medium/high contention. A later sparse-load run can check that the same policy
does not introduce avoidable delay when the queues are lightly populated.

### Prefill latency metric semantics

The simulator emits the first output token only when the final Prefill chunk completes. Therefore `prefill_latency_ms` is measured from request arrival to Prefill completion and is intentionally identical, request by request, to TTFT. It is not Prefill GPU service time.

The original `prefill_wait_ms` records only arrival-to-first-Prefill-chunk-start. It does not include gaps between later chunks.

The simulator now additionally reports:
- `prefill_inter_chunk_wait_p50/p95_ms`: cumulative queue/phase waiting between successive Prefill chunks for each completed request.
- `prefill_target_forward_exposure_p50/p95_ms`: sum of wall-clock durations of Target forwards that carried this request's Prefill chunks. This includes shared batch execution and is not exclusive per-request GPU compute.
- `prefill_chunk_count_mean`: mean number of Prefill chunks per completed request.
- `prefill_decomposition_error_max_ms`: maximum absolute residual for the per-request identity:

  `TTFT = initial Prefill queue wait + inter-chunk wait + Target-forward exposure`.

This decomposition separates waiting from the wall-clock Target execution intervals experienced by a Prefill request; it does not claim to isolate marginal GPU compute cost.


## GitHub Actions: Prefill chunk-size sweep

The 4-by-rate-by-seed CPU-only Prefill chunk-size experiment can run on GitHub-hosted CPU runners without using the development GPU server.

1. Open **Actions → CDB-AV Prefill chunk-size sweep → Run workflow**.
2. Select the feature branch `cdb-av/phase-aware-target-scheduler-20261110`.
3. Keep the default workload inputs for the current E6 experiment: 32 requests, Poisson arrivals, rates `5,15,30,60,100`, seeds `42,43,44,45,46`.
4. Start the workflow. Four matrix jobs independently evaluate chunk sizes 256, 512, 1024, and 2048 tokens. Each job runs one chunk size across all selected rates and seeds.
5. The aggregate job downloads all raw JSON files, computes mean and sample standard deviation across seeds, performs paired per-seed deltas against the 512-token baseline, and publishes a Markdown report in the workflow summary.

Artifacts:
- `cdb-prefill-chunk-256`, `512`, `1024`, and `2048`: raw per-run JSON and manifest.
- `cdb-prefill-chunk-sweep-report`: summary CSV, paired-comparison CSV, and Markdown report.

The workflow is manual-only; ordinary pushes do not launch the 100-run sweep. The existing lightweight simulator CI continues to compile the code and run unit/smoke tests. Results are CPU simulation outputs using the simulator's synthetic execution profile unless a measured profile is explicitly supplied. They are not GPU performance measurements.
