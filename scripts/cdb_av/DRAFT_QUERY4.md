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

## GPU3 paired performance (same e44942d runtime)

FLASH_ATTN FA2/BF16,Target8B,Draft1.7B,H8,HF0.8,maxK8,max_num_seqs4,
max_model_len1024,C16,128 output tokens,4 HTTP warmups. AR bypass,KV reuse,
mixed/ragged/dynamic Graph ON;12 ragged cache entries,minhits8. Validation,
per-step logging and diagnostic timing OFF. Only the input-query Graph bound
changes. Commands and environments are retained per run;all runs complete.

| Workload | Query2 TPS | Query4 TPS |
|---|---:|---:|
|32 requests,rep1|194.414|199.258|
|32 requests,rep2|192.951|205.755|
|32 requests,rep3|197.210|205.069|
|512 requests,query4 then query2|207.084|211.526|
|512 requests,query2 then query4|205.335|214.420|

32 mean+/-sampleSD:194.858+/-2.164 vs203.361+/-3.570(+4.36%).
512 mean+/-sampleSD:206.210+/-1.237 vs212.973+/-2.046(+3.28%).
Only two large pairs;this is promising paired evidence,not a final stable
performance ceiling or comparison against AR/native SD. Both512 pairs
generate65536 tokens per side. Do not compare absolute GPU3 TPS to olderGPU1.

512 latency(mean/P50/P95,s):
query2 first9.8387/9.9929/12.0742;query4 first9.6394/9.7169/11.9692;
query2 reverse9.9239/9.9856/12.0776;query4 reverse9.5131/9.6870/11.6551.
Latency starts when a client thread sends its request,excluding the executor
queue. This is closed-loop concurrency16,not constant-rate arrivals orGPU B16.

Full-text hashes match139/512 and160/512 across bounds. Strict final generation
equivalence is not established. Even same-bound32 repeats vary(query2 pairs
16/32,13/32,15/32;query4 pairs12/32,9/32,11/32). Existing non-equivalence alone
does not prove a new Graph bug,nor prove all differences are harmless floating
point effects. Keep touched-KV/token checks and explicit equivalence limits.
Proposal/candidate counts also vary slightly;TPS includes those trajectories.

Real HTTP cancellation:4 streams disconnected after3 chunks;active leases
released at live counts3,2,1,0;4 recovery requests complete. Final Lease,
Ready,inflight0,Worker reset dropped[]. Bound4 selected,but this cancellation
trace only observes mixed query2;do not claim query3/4 cancellation coverage.
No collision,Cuda error or Graph validation mismatch in the paired logs.

Current Scheduler excludes frozen requests from the active Target capacity.
Thus its resident running metric may reach15,while actual Target forwards
remain bounded by4 requests. Neither resident running nor client concurrency
is the GPU batch size. KV reuse andHF dynamic K are Worker aggregates,not
client requested_K echoes;aggregate logs cover cumulative prefixes with tails
unflushed. Native num_drafts counts request-level proposal rounds,not forwards.

Results are also visible via code-directory symlink cdb_av_draft_q4_results.
Default bound2 remains selectable;query4 stays opt-in until broader lifecycle
and final-equivalence gates. Stop independent services after each experiment.
