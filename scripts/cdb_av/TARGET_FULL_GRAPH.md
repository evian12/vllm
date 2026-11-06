# Experimental whole-Target-backbone CUDA Graph

Based on32a62bf34a1d3ac023f7896e6f79405a2dd109d7 (query4 Draft Graph).
CDB_AV_TARGET_FULL_GRAPH=1 adds a manual CUDA Graph around _model_forward.
Default0 preserves eager execution. This captures embedding,all36 Qwen3-8B
transformer layers andfinal norm. Logits gathering/LM head,sampling,rejection
handling,formal commit andthe scheduler remain on their original paths.
--enforce-eager staysON;there isno global torch.compile orfusion change.
No native uniform/padding capability isforged andno invalid candidates added.

Supported scope:Qwen3 BF16,FA2,no planner/cascade/prefix/LoRA/quantization,
synchronous TP/PP/DP/DCP1,max_num_seqs4,max_model_len<=1024,maxspecK<=8.
H8 andHF0.8 are unchanged. Unsupported layouts fall back before model work.
Exact token/request count,CPU maximum query,sequence bucket,block-table width
andsplit count form the key. In particular query1 packed-GQA never shares
a key withquery>1. GPU offsets,sequence lengths,block tables,slots,input IDs
andpositions arecopied to persistent buffers for each replay. No extra rows.
Private graph pools avoid cross-key temporary/output lifetime assumptions.
Capture admission uses a frequency gate andbounded key count;free device
memory<512MiB skips capture,not safety assertions.

Target layers are discovered from actual model modules andchecked against the
HF layer count andFA2 implementation. Model replacement recreates the graph
owner;changed KV addresses/layouts invalidate captures. Attention object
replacement fails closed;model sleep isoutside the supported scope.
Programs keep no allocator Block lease orrequest object;old IDs inbuffers
areoverwritten before use. No KV reuse,lease,release,invalidation,Carry,
Ready,verification orrequest-selection code waschanged.

237 CPU functionality tests pass(39 new Target schema/scope/buffer/lifecycle,
130 config,34 Draft Graph,23 queue,11 ownership),plus12 workflow tests.
The firstGPU smoke completed4/4 butall Target calls fellback:the actual FA2
builder uses inactive max_dcp_context_kv_len=0,notNone. This wasnot counted
as Graph validation. The guard now explicitly accepts inactive0/None with
DCP1 andno DCP tensor;active/planned state remains rejected,with8 tests.

After that protocol correction:4/4 smoke,18 Target graph checks;32/32 stress
functional,63 checks over36 Target layers. Static/live eager references both
havebyte-exact touched Target KV,exact greedy Token IDs andallclose logits.
Observed maxqueries1,2,3,4,5,6,7,9 andtoken row counts3,4,5,6,7,8,9,10,13,14,16,17
inthe32 trace. It exercises18 cross-layout and37 cross-request changes inside
cached keys. q8 andall boundary layouts are not GPU-certified bythat trace.

Independent16/16 Carry validation:40 byte-identical resumes,all40 witha real
Target epoch advance,28 Draft layers,383516672 cumulative bytes. Final Lease,
Ready,inflight0. This run used the pre-model-guard version,whoseexact source
hashes/backups areretained;the guard onlychanges lifecycle detection.
Final guard code thenpassed real4-stream disconnect+4 recovery completion,
138 Target graph checks(all36 layers exact),leasecounts3,2,1,0 andfinal0.
Validation verifiesactual CPU/GPU query starts/seq lengths,Token positions,
block-table-derived slots,unique physical slots andreal InputBatch request IDs.
It restores tested KV values after its two diagnostic eager forwards.
Those extra forwards areexplicitly logged anddisabled inperformance runs.

Raw commands,configs,source snapshots,old fallback evidence andGPU audits:
results/target_full_20261106/. Performance pending at this checkpoint.
Compare the same cleancommit,query4/cache12 Draft,H8/HF0.8,Target graphOFF/ON,
4 identical HTTPwarmups,C16,output128,balanced repeats. Target cache24,minhits8.
No512 test until the paired32 gate ispromising. Alwayscloseownedservices.
Overall system remainsPartial until broader memory pressure,lifecycle and
strict final generation equivalence areestablished.
