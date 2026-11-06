# Experimental dense native Target compilation + query4 Draft Graph

Based on32a62bf34a1d3ac023f7896e6f79405a2dd109d7. Integrates the already scoped
native Target probe d4ebb650 into the query4 worktree without changing its
execution guards. CDB_AV_TARGET_GRAPH_PROBE=1 permits the checked persistent
refill route to remove enforce_eager;default0 retains its original path.
Custom Draft uses exact rows andForwardContext.NONE,so it does not consume
the native Draft dispatcher padding descriptor. No ownership/progress/context
assertion is removed. Other synchronous,singleGPU,noLoRA,noPrefix guards stay.
No H,threshold,selection,weights,tokenizer,reuse/commit/release change.

The candidate uses native compilation andPIECEWISE capture sizes1..36.
Bothcandidate/control use query4 Dynamic Draft Graph,cache12,minhits8,
H8,HF0.8,maxK8,max_num_seqs4,max_model_len1024,BF16,FLASH_ATTN FA2,
CUDA_VISIBLE_DEVICES3,API18084,Target8B/Draft1.7B. Native compilation affects
Draft as well as Target,andnative default custom_ops/fusion differs from eager;
do not call a throughput delta a pure Target-only Graph effect. No irregular
batch is falsely marked uniform. This does not implement irregular FULL mode.

206 functionality tests(138 config,34 Graph,23 queue,11 ownership) and12
workflow tests passed. GPU4/4 smoke has62 exact static/live Draft Graph/eager
KV/token checks;32/32 validation has337 checks,51 max-query>2 and14 query4.
Logits are allclose;KV values are element-equal at touched slots,not whole-cache
byte equivalence or strict end-to-end generation equivalence.

Native Target dispatch validation prefix:477 PIECEWISE,14 NONE,0 FULL;
5285 actual/padded token rows,no padding. Metrics are interval prefixes with
unflushed tail,not whole-run final totals. Prefill fallback remains available.

Independent16/16 Carry test(CARRY_VALIDATE1,heavy checks ON) proves51 resumes
with a real Target epoch increase;all28 Draft layers remained byte-identical,
489947136 cumulative bytes checked. Final Lease/Ready/inflight0.
Native joint4-stream cancellation releases active leases at3,2,1,0 and4
recovery completions pass;final Lease/Ready/inflight0.
GPU lifecycle pressure,all layout cancellation branches andstrict output
equivalence remain incomplete;overall system status stays Partial.

All raw commands,configs,source patch/hash,requests/responses andfailure-safe
owned-service cleanup live inresults/target_q4_20261106/. Performance pending.
Formal paired32x3 will disable validation/trace,keep4 identical HTTPwarmups,
record source commit,service startup polling time andready-time GPU memory.
Startup is separate from client wall/TPS. Only if that gate is promising will
a512 pair run. Each isolated child service is closed after its pass.

## Actual paired performance gate

Clean runtime59f2690e930f95d34a24206124523a59aa60522d,GPU3.
32 requests,C16,128 output tokens,4 identical short HTTPwarmups each pass;
H8,HF0.8,maxK8,query4,cache12,minhits8. Validation,carry byte audit and
per-step Graph trace OFF. --cudagraph-metrics ON both sides. Baseline retains
--enforce-eager;candidate uses the dense PIECEWISE config above.
Orders:control/candidate,candidate/control,control/candidate.

| Repeat | control TPS | candidate TPS |
|---|---:|---:|
|1|203.497|205.030|
|2|205.267|203.001|
|3|199.042|200.275|

Control202.602+/-3.208 sampleSD vscandidate202.769+/-2.386;
mean difference+0.082%,within run variation andone pairnegative.
All6 complete32/32 and4096 tokens each. Mean ofper-run P95 latency:
12.9485s control,12.6237s candidate;this small-sample change isnot certified
as a stable latency benefit. Health-poll startup interval averages30.086s
vs65.073s(includes up to5s polling granularity),separate fromclient TPS.
The candidate isnot promoted andno512 test isstarted because the gate
doesnot show a meaningful throughput benefit. Keep the existing query4
enforce-eager configuration as the testedbest choice for now.

This rejects this combined PIECEWISE/compilation configuration as a measured
throughput optimization;it doesnot show that every Target Graph design is
useless. FULL0 isexpected because this candidate explicitly selects PIECEWISE.
The FA2 builder comment(flash_attn.py234ff) explains its special query=1
packed-GQA branch andconservative UNIFORM_BATCH capability. Do not simply
advertise ALWAYS orsetuniform=True forvarlen batches.

A next separate candidate can investigate a scoped manual whole-Target graph
atGPUModelRunner._model_forward(),with persistent input/metadata buffers and
correct current GPU/CPU query starts,sequence lengths,slots andblock tables.
Keep pure query1 capture separate from mixed query>1;fix every CPU scalar
kernel-launch domain(max query/sequence bounds) in the key andvalidate against
both static-metadata andoriginal eager execution. Sampling,verification and
commit must remain outside,as mustall ownership/assertion logic. Start only
on the checked Qwen3/FA2/eager/synchronous TP1 path,with an independent OFF
switch andactual full-Target KV/token evidence. This isnot implemented here.

Version audit:primarybb3a8c isnot an ancestor ofthis snapshot lineage.
An initial overly strict ancestry preflight interrupted its owned runner
before anyclients;pair32/ isretained andexcluded. Actual source hascomplete
Drop,Prepare/Drop/Reset ACK andstaged audit-lease guards;11 ownership tests
pass with additional newer ACK cases. Formal results areonlypair32_formal/.
Never equate commit ancestry with presence/absence ofcherry-picked fixes,
andnever blindly overwrite the primary tree to force an ancestry relation.

Raw summaries,configs,startup measurements andall logs are under
/home/yypan/CDB-AV/results/target_q4_20261106/. Primary/source code directories
havecdb_av_target_q4_results symlinks. Every owned service isclosed after
experiments;GPU3 returns15MiB and18084 closes. Original18082 staysstopped
perthe owner's earlier authorization. Other users' services areuntouched.
Overall status remainsPartial:strict final output equivalence andbroader
natural memory/preemption pressure are not established.
