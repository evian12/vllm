# Opt-in remaining-output Draft cap

Baseabb7256db6837a680b7e6f37152b5906f7654f00. Target sequence floor stays0;
floor256 had no demonstrated throughput benefit. H8/HF0.8/maxK8 unchanged.
CDB_AV_OUTPUT_BUDGET_CAP=1(default0) bounds new admissions to
min(policy_cap,max(0,max_tokens-num_output_tokens-1)). One native Target
bonus/correction isreserved. Zero isCPU-only AR-tail eligibility,not a Worker
Draft length. Both PersistentDraftQueue.plans andReuseDraftQueue.plans filter
zero before any allocation/verification-budget check orWorker handoff.

Policy source andHF stopping stay unchanged. Existing Carry keeps its cap:
its official prefix/budget isfrozen until verification. A cached lease skipped
forAR-only tail remains under existing ownership andfinish/preempt cleanup;
no fill,reuse_span,commit certificate,invalidation/free orScheduler code changed.
Budget decisions arelogged separately as p2_output_budget_cap. They areNOT
GPU evidence. Worker Ready/progress,actual Target Graph rows andp2_verified
records provide execution evidence;zero isnever logged as a Worker cap.

301 CPU functionality tests+12 workflow pass,including26 new bound/config,
default-off andboth queue admission tests. GPU paired boundary cases pass:
HF long/short/fresh-AR plus controlled manualK1 cached-AR,4 request pairs with
identical returned Token IDs. Long tail changes realWorker K8->3 andTarget
9->4 outputs atremaining4. Fresh andcached tails bothhave a realnative AR
query1 atlast official input position,not a dummy draft withK0. All36-layer
Target byte/token comparisons pass;Lease/Ready/inflight end0.

GPU32/32:223 Target comparisons;1107 verified records acrossall32 requests
respect remaining budget (zero acceptance315,partial261,all531). EveryWorker
cap ispositive. Carry16/16:38 true cross-Target byte checks,28 Draft layers,
343146496 bytes unchanged;final0. Four stream cancels+4 recovery pass138 Target
comparisons andfinal0. No collision/progress/context safety assertion relaxed.

Original native Scheduler CPU subset isnot certified:17 failures initiallydue
missing publicOPT/LLaVA configs;original configs were restored onlyinto an
isolated test cache andGit blob hashes verified,then missing tokenizers still
blockthat subset. No test expectation ormodel/tokenizer wasmodified to pass.
Actual Qwen3 Scheduler runs inall GPU gates. Overall remainsPartial pending
broader preemption/memory pressure andstrict full-generation equivalence.

Three paired512 runs:221.215+/-2.256 OFF vs222.664+/-0.627 ON tok/s(+0.655%).
P95 mean11.230->11.217s;not a demonstrated stable speedup,keep default0.
SeeOUTPUT_BUDGET_RESULTS_20261107.md/JSON. Timing usedthe same cleancommit,
featureOFF/ON,bothTarget GraphON,
dynamic queryON,sequence floor0,Draftquery4/cache12/minhits8,Targetcache24/8,
4 equal HTTPwarmups,HF0.8/H8/maxK8,maxseq4/maxlen1024,C16,output128,diagnosticsOFF.
Raw commands/evidence:results/output_budget_20261106/. Rollbacksetflag0 and
start a fresh ownedservice;do not reset/overwrite oldbranches orstopotherjobs.
