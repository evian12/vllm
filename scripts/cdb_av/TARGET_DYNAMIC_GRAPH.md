# Opt-in bounded mixed-query whole Target graph

Based on2523fdd75d75fbd9223c8eb46172a58b1f9ed511. Exact Target graphs were
functional butpaired32 throughput was-0.52%,with only141 replays/640 calls
and458 frequency fallbacks inone recorded prefix. CDB_AV_TARGET_GRAPH_DYNAMIC=1
reduces that fragmentation. Default0 retains exact-maximum query keys.

Input token/request counts remain exact. Query1 hasstatic maximum1;actual
mixed query2..9 hasstatic maximum9 frominitialcapture. GPU offsets,positions,
sequence lengths,blocks andslots areupdated onreplay. This isnot input padding
anddoesnot change K,H,HF threshold,ownership,scheduling,sampling orcommit.
All original byte/token assertions remain. A separate domain matcher forbids
query1/mixed crossover andany query beyondthe captured bound. Other CPU
launch scalars remain keyed. This stillcaptures only _model_forward backbone;
LM head/logits/sampling stay outside andglobal enforce-eager staysON.

248 functionality tests pass(50 Target tests including11 new bounded-domain
cases,130 config,34 Draft Graph,23 queue,11 ownership),plus12 workflow checks.
GPU4/4 has18 Target byte/token comparisons;32/32 has162 comparisons across
all36 Target layers,eachboth static andoriginal-live eager references exact.
Actual maxima1..9 areobserved withstatic maxima1/9. Samekey sees99 layout,
55 actual-maximum and137 request-member changes. No observed max<=4->9
transition;the explicit CPU3,3,3,3->1,1,1,9 test covers its key/buffer protocol.
Observed Target input rows3..11,13..15,17,21 inthat validation trace.
Do not inflate this into every possible n<=36 layout beingGPU-certified.

Independent16/16 Carry gate:48 byte-identical cross-Target resumes,28 Draft
layers,442236928 cumulative bytes;all48 epochs truly advance. Final Lease,
Ready,inflight0. Final4-stream cancellation+4 recovery also passes withstrict
Target graph byte/token checks;details inrawaudit.json. No safetyrules removed.

Rawsource/commands/logs underresults/target_dynamic_20261106/.
Paired32x3:200.934+/-1.196 OFF vs202.554+/-1.932 ON tok/s(+0.806%).
Then paired512x3 onthe same runtime14b225d:214.438+/-2.978 OFF vs
221.427+/-2.471 ON tok/s(+3.259%;sample SD). All3072 measuredrequests
complete;all3 paired gains positive butvary5.17/3.48/1.17%,not proof of
universal/significant gain. PooledclientP95:11.701->11.389 seconds.
H8/HF0.8/maxK8/query4 Draft/cache12,Target cache24/minhits8,warmup4,
heavychecks OFF; orderOFF/ON,ON/OFF,OFF/ON. Each ownservice closed.
GPU3 ends15MiB,ports18084/18082 closed. SeeTARGET_DYNAMIC_RESULTS_20261106.md
andJSON for exactruns,scope,sourcehashes,limits andcomplete startupfiles.
No newAR/nativeSD comparison andno requestselection/H/KV policychange.
Overall Partial status remains untilstrict final generation equivalence and
broader natural-memory/preemption lifecycle gates areestablished.
