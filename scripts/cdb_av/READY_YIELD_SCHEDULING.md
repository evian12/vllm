# E7: Acceptance-Aware Ready Selection

Status: isolated runtime candidate; default remains FIFO. No GPU performance claim is made until the paired benchmark is complete.

## Motivation

The current Target selector publishes completed Draft requests into an ordered Ready queue and selects FIFO under Target running-slot and token-budget constraints. CDB-AV requests are heterogeneous: each Ready item carries a different number of Draft candidates. When Target capacity is occupied by AR/other running requests, or the Ready queue exceeds available slots, FIFO does not account for the expected committed output associated with each candidate length.

The E5 stage policy is retained. E6's window-predictive policy reduced Ready wait but increased Target-forward count and reduced throughput at medium/high offered load in the CPU study. E7 therefore does not shorten H, change Draft admission, or alter the Verify boundary. It changes only the order in which already-published Ready requests are selected for a Target forward.

## Online expected-yield estimate

For a verified request with K candidates and accepted prefix A:

- Candidate positions 1..A are observed successes.
- Position A+1 is observed as a failure when A < K.
- Positions after the first mismatch are censored, not counted as failures.

A Beta(1, 1)-smoothed per-position acceptance estimate p_i gives expected committed output:

    E[K] = 1 + sum(i=1..K, product(j=1..i, p_j))

The leading 1 accounts for the correction/bonus token. The estimator updates only from completed real Target verification outcomes. It uses no future request information and performs no GPU operation or synchronization.

## Selection rule

The feature is enabled with CDB_AV_READY_SELECT_POLICY=yield_age; default fifo preserves the previous behavior exactly.

- Requests whose Ready age reaches 8 ms are ordered oldest-first.
- Other Ready requests are ranked by expected committed tokens plus a bounded age bonus: yield + 0.5 * min(age_ms / 8, 2).
- Existing running-slot capacity and aggregate token-budget checks remain authoritative.
- A candidate that does not fit the current token budget is skipped; verification remains atomic and the complete candidate list is unchanged.

The policy does not modify H, K generation, Draft/Target execution order, KV ownership, Carry, leases, acceptance/commit semantics, AR selection, or Prefill scheduling.

## Correctness and evaluation gates

1. Pure CPU tests cover censored-prefix estimation, priority ordering, overdue ageing, FIFO compatibility, and token-budget skip behavior.
2. Run the existing runtime correctness gates on the isolated branch.
3. Compare FIFO and yield_age on the same code commit, same request trace, same seed, same H=8/HF=0.8/maxK=8, same Target Graph setting, and identical concurrency/output limits. Alternate run order for at least three paired 512-request runs.
4. Report output tok/s, request P95, TTFT, TPOT/ITL, Target forwards, Ready wait P95/max, selected/deferred Ready counts, and accepted output tokens per Target forward.
5. Keep the feature disabled unless the throughput gain is repeatable and latency/fairness remain within the existing SLO.

## Reproduction

Set exactly one of the following in the otherwise identical service launch environment:

    CDB_AV_READY_SELECT_POLICY=fifo
    CDB_AV_READY_SELECT_POLICY=yield_age

The current default is fifo. The branch is a research candidate, not a new baseline.
