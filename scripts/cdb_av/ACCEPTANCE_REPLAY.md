# Prior-acceptance Draft length replay

`CDB_AV_K_POLICY=replay_acceptance` reads a JSONL file produced by
`build_acceptance_replay.py` from a **previous** max-K CDB run. It is an
experimental hindsight policy, not an online K predictor. The replay file is
loaded when the service starts, before any replay request arrives.

Use the same models, tokenizer, request prompts, greedy sampling, service
configuration, and maximum K in both runs. Set `CDB_AV_LOW_LOG=0` in the
reference run so that `p2_verified` events carry individual observations.
The client sends `vllm_xargs.cdb_trace_key` (the stress client uses the stable
key `stress512:<request-index>`). Build the trace with:

```bash
python scripts/cdb_av/build_acceptance_replay.py \
  --server-log /path/to/reference_server.log \
  --output /path/to/new_acceptance_replay.jsonl \
  --max-k 8
```

Start a separate CDB service with the usual P2/KV reuse settings and:

```bash
CDB_AV_K_POLICY=replay_acceptance
CDB_AV_ACCEPT_REPLAY_FILE=/path/to/new_acceptance_replay.jsonl
CDB_AV_HF_THRESHOLD=0.8
```

At each new Draft admission, the Scheduler matches the request's stable key,
the fingerprint of its **Target-committed** token prefix, and the prefix
length. A match chooses `K=accepted+1` when the reference rejected a candidate,
bounded below by 1; this retains the first known rejection and removes only
the suffix known to be useless in that reference execution. Full acceptance
keeps the observed K because longer acceptance is unknown. A missing or
ambiguous context falls back to the online HF-confidence stop rule. The
Worker's `ready` event reports `k_source`, `k_cap`, `k_actual`, and stop reason;
`k_replay_decision` reports match/fallback counts. HTTP request labels alone
are not evidence that a cap was used by the Worker.

The policy only changes Draft length. The current Target must still verify
every candidate before commit. Changing K can change batch shapes and future
token prefixes, so a replay hit does **not** guarantee the reference acceptance
will recur. Even identical formal prefixes may yield different Draft or Target
tokens across runs; compare output token hashes and record exact-context
divergences. Report match rate and fallback rate with every result. This trace
is neither a deployable predictor nor a rigorous end-to-end throughput upper
bound. It can be used as a common input schema for future K-policy adapters,
but other serving systems do not yet consume it.
