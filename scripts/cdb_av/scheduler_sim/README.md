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

\`\`\`bash
python -m scripts.cdb_av.scheduler_sim.phase_cli \\
  --requests 64 --arrival-rate 30 --arrival-mode bursty \\
  --policy all --seed 42
\`\`\`

Phase 2 adds the Target-side request lifecycle:

- Every new request enters Target Prefill first. A prompt can span multiple
  Target forwards; the first output token is committed only after its final
  Prefill chunk completes.
- A speculative request then enters Draft; a non-speculative request enters
  one-token-at-a-time Target AR.
- One Target forward may contain a mixture of atomic Verify requests, AR
  requests, and Prefill chunks. The combined batch must respect both
  \`max_num_seqs\` and \`max_num_batched_tokens\`.
- Draft and Target still execute serially on one GPU. The simulator does not
  model GPU overlap.

### Phase-aware comparison policies

| CLI policy | Stage decision | Target batch organization |
|---|---|---|
| \`fixed_h\` | Fixed number of Draft forwards between Target forwards | FCFS, chunked Prefill |
| \`queue_pressure\` | Queue pressure and waiting-time guards | FCFS, chunked Prefill |
| \`sarathi\` | Queue pressure | Bounded Prefill chunks, Decode-like work first |
| \`length_aware\` | Queue pressure | Short/long queues, similar-length buckets, adaptive Prefill budget |
| \`joint\` | Queue pressure plus a short-Prefill batching wait | Length-aware Target batching |

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
Prefill chunk sizes, and Target batch composition.

The reported commit-gap is the interval between per-request commit events. It
is **not** token-level ITL inside a multi-token Verify commit. Default execution
coefficients are synthetic and are only useful for testing scheduling logic;
fit them from GPU batch-shape profiles before drawing hardware-performance
conclusions.
