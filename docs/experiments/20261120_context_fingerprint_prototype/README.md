# Exact-context fingerprint CPU prototype — NOT integrated or GPU accepted

Motivation: the recommended-config bounded profile recorded 240 `reconcile`
calls, 51.339 ms inclusive CPU time. Actual source recomputes the full prefix
digest for frozen contexts on every reconcile. This does not prove all of the
time belongs to hashing; no end-to-end improvement is claimed.

Implemented an isolated, pure-stdlib helper, not connected to `RefillQueue`:

- Cache hit requires same live Request object, same live hasher function and
  equality of the COMPLETE immutable token tuple. Only builtin ints can hit.
  It never caches on request ID or length alone.
- First/mutated prefixes run the unchanged original validator/hasher. Bool,
  int subclasses, equal-valued non-ints and unsupported keys use the native
  path; native errors propagate. Same-length changes cannot reuse a digest.
- Bounded64 weak Request keys, max1024-token cached prefix; weak hasher refs
  cannot retain a Request through callbacks. Returned context dicts are fresh.
- Optional first32 hit audits recompute the native digest and fail on mismatch;
  no silent healing. Cross-thread calls use native hashing without cache writes.
- Defaults OFF. No production source, queue state, ownership assertion, Carry,
  token commit or scheduling changed. Current GPU comparisons do not import it.

`python3 work/context_fingerprint_cache_20261120/test_standalone.py`:
**19 tests passed**, including mutation, invalid types/ranges, hasher replacement,
weak lifetime, bounds, audit corruption, dict aliasing and cross-thread fallback.
`cdb_av_fingerprint_reference.py` is an unchanged copy of the actual server
fingerprint helper; tests call its batched implementation (the current mode).

Required before adoption: read the complete caller chain again; wire only the
existing formal `prompt_token_ids + output_token_ids` snapshot behind an
independent default-OFF flag in an isolated worktree; exercise actual frozen
Carry/progress assertions, same-length mutation, cancellation/reset/GC and
original CPU regressions. Then run a real GPU functional gate with native digest
audits and same-config repeated OFF/ON512 tests after the current Target-bound
experiment finishes. Select the Target-bound configuration from complete paired
evidence first. Do not start parallel GPU work or claim this CPU prototype is an
implemented GPU/CDB optimization.
