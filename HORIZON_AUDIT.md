# A01: H/publication contract audit (2026-11-20)

Base ccf96aac0064aebebc2077fb47e36929d65aa06a; independent algorithm branch.
No algorithm change or performance run in this audit.

The accepted Ready coalescing helper in cdb_av_coalesce.py explicitly requires
horizon == 8 and capacity == 4. CDB_AV_WINDOW_H accepts 1..64 elsewhere. Merely
changing H therefore disables the accepted coalescing opportunity even with
CDB_AV_READY_COALESCE=1. Such a sweep would change both cadence and effective
coalescing policy, and cannot be described as an isolated H optimization.

Existing causal_audit_20261120/REVIEW.md was collected before coalescing: 2310
of 2448 underfull Target calls overlapped completed unpublished Worker drafts;
1693 calls met the later accepted coalescing predicate. Those are opportunities,
not savings, and do not establish a new optimum after integration.

Keep HF0.8/cap8/H8 as system control. Before an H candidate, specify the
coalescing contract for that H, verify exact-H Worker status, lease ownership,
unique handoff, candidate-prefix mapping and all termination paths. Collect
post-coalescing timing evidence or replay the saved event trace with explicit
counterfactual limits. Do not extrapolate pre-coalescing counts or repeat the
already rejected tail budget cap. No GPU tuning sweep is authorized by this
record alone; use the standing task protocol and register a concrete mechanism.

Next: examine post-coalescing Ready/Worker timelines for a publication rule
whose expected work reduction survives state and quality constraints.
