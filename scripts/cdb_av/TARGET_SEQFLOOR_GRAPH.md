# Target Graph sequence launch floor (experimental)

Base2169190259212722f04a0fbf977c18ee38047964; Target runtime14b225d.
CDB_AV_TARGET_GRAPH_SEQ_FLOOR=0(default) preserves the existing power-of-two
sequence buckets. Opt-in256 maps actual maxima<=256 to launch upper bound256;
larger maxima still use their ordinary bucket. The value must be0 or a positive
power of two within max_model_len. A bucket beyond max_model_len still falls back.

This changes only the cached FA2 max_seqlen_k launch bound. Actual GPU seq_lens,
input token/request rows,query starts,positions,blocks andslot mapping remain
exact. No padding,KV writes,lease/commit/reuse logic,H,HF threshold,request
selection orrefill strategy ischanged. Existing model/backend/synchronous scope,
query1-vs-mixed guard,metadata checks,collision andlifecycle assertions remain.
A wider bound could choose a slower FA2 kernel; fewer keys do not prove speedup.

CPU261 functionality tests pass(63 Target,130 config,34 Draft Graph,23 queue,
11 ownership),plus12 workflow checks. The13 added cases cover invalid floors,
short/long/over-limit domains,live-buffer invariance,key merging anddefault0.
An initialnewfixture incorrectly cloned query1 withmixed-query maximum9;
the existing domain guard rejected it. The fixture now follows the actual key's
query bound; no guard orcomparison wasrelaxed. Failure evidence retained.

GPU gates pass:4/4 smoke20 comparisons;32/32 functional519 comparisons with
137 real same-key sequence-bucket changes. All36 Target layers byte-exact and
Token-exact against both static andoriginal-live eager references;actual query
maxima1..9 observed. Boundary prompts1 and301 tokens pass10 more comparisons,
including positions0 and302..320 withcapture bounds256/512. Larger candidate
positions are native speculative lookahead,not proof ofextra committed output.
Carry16/16 passes44 true cross-Target byte checks,28 Draft layers,407371776 bytes;
final Lease/Ready/inflight0. Four stream disconnects and4 recovery completions
pass138 Target comparisons andfinal0. These heavy checks areNOT timing results.
No safety assertion changed. Three paired512 runs:222.394+/-0.926->223.140+/-1.539 tok/s(+0.335%);
P95 mean11.192->11.205s. No demonstrated throughput benefit;keep default0.
Target replay coverage improved72.7->90.7% inone9216-call prefix,not a timing
claim. SeeTARGET_SEQFLOOR_RESULTS_20261106.md/JSON forfull results andlimits.
Formal timing compares floor0 vs256 onthe same cleancommit withTarget GraphON
andmixed-query domainON inboth modes,H8/HF0.8/maxK8,maxseq4,maxlen1024,
Draft query4/cache12/minhits8,Target cache24/minhits8,4 equal HTTPwarmups,
C16,output128. Heavy KV/metadata/graph checks areOFF for timing only.
Raw source,commands,logs androllback evidence:results/target_seqfloor_20261106/.
Rollback:unset CDB_AV_TARGET_GRAPH_SEQ_FLOOR or set0 before a fresh service.
Do not restart other services;use own GPU3/18084 only while testing andclose it.
Overall system remainsPartial; pointwise GPU checks cannot certify strict
end-to-end output equivalence orall natural-memory/preemption conditions.
