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
