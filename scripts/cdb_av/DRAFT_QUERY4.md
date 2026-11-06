# Experimental bounded query1..4 Draft Graph

`CDB_AV_DRAFT_RAGGED_MAX_QUERY=4` extends dynamic mixed-input graphs to small
query layouts1..4. Default2 preserves the previously GPU-validated domain.
Enable with DRAFT_DECODE_GRAPH=1,DRAFT_RAGGED_GRAPH=1,DRAFT_RAGGED_DYNAMIC=1.
This bound describes INPUT tokens per request in one forward,not Draft K.
H8,maxK8 andHF confidence0.8 remain unchanged. Query>4 and unsupported layouts
retain eager execution. Exact-layout mode retains its old1/2 scope.

The key fixes token count,request count,sequence bucket anddynamic4. A first
capture with(2,2,2) may later replay(1,1,4),so its static max_query_len must be4
from the outset. It must not be max(first_capture_qlens). GPU query offsets,
CPU shadow offsets,sequence lengths,block tables andslots are still copied
before replay;last-logit row indices are gathered from the current layout.
Input rows are exact;no request-mask shrink or invented padding was added.

Query4 is restricted to FlashAttentionMetadataBuilder withFA2/noAOT. Other
builders may retain planner state not refreshed by these tensor copies and
are explicitly rejected for this new bound. Query2 behavior is unchanged.
No KV ownership,release,commit,reuse_span,confidence algorithm orscheduler
code was modified. Old mode and old graph cache remain selectable.

CPU checks:130 config,34 graph,23 persistent queue,11 ownership tests passed
(across pre/post scope-guard runs);12 workflow checks. New tests cover default
and invalid bounds,shared keys across maxq2/3/4,max16 input rows,invalid offsets
or metadata counts,capacity violations,and backend/AOT scope.

GPU3 initial tests:smoke4 completed4/4 with64 exact Graph/eager checks;32-request
validation completed32/32 with430 checks,all touched-layer KV values andgreedy
tokens exact against both static andoriginal live metadata;all logits_close.
150 dynamic checks include55 maxq>2 and15 maxq4. All dynamic entries used static
bound4. Observed model token row counts were1..8;max16 is CPU-covered but not
observed in this trace. No capture-max2->replay-max4 transition was observed;
the upper-bound rule is CPU-tested andactual maxq4 replay was GPU-tested.
Final backend scope guard then passed another4/4 with64 exact GPU checks.
Do not inflate this into coverage of every layout or full generation equivalence.

GPU1 was occupied by another user. The owner authorized stopping the idle
GPU3/18082 baseline;running/waiting were both0. SIGTERM1682417 cleanly stopped
API andEngineCore1684042;GPU3 returned22245->15MiB. Full stop audit andold command
are retained,not deleted. All newGPU tests use3/18084 andcleanup their own
service. Paired throughput must compare query2/query4 onGPU3;do not compare
absolute numbers with previousGPU1 runs.

Artifacts:/home/yypan/CDB-AV/results/draft_q4_20261106/ including source hashes,
initial/scoped patches,backend links,full commands,logs,requests/responses and
stopped_baseline.json. Performance is pending at this checkpoint. Keep default2
until paired evidence andbroader lifecycle gates are complete.
