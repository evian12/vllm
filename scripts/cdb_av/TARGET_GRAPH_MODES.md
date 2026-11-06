# Native Target Graph dispatch audit (diagnostic only)

Branch codex/cdb-target-modes-20261106 is based on f68848c. Its opt-in
CDB_AV_TARGET_GRAPH_PROBE=1 permits native Target graphs only with persistent
refill;all single-GPU,synchronous,no-prefix-cache,no-LoRA constraints remain.
The custom Draft path still passes exact input rows with ForwardContext mode
NONE;it does not consume the native Draft dispatcher/padding descriptor.
Default0 preserves the original guard. This is a separately gated execution
route,not a suppression of KV,progress,context or numeric assertions.

The8 scope combinations are CPU-tested;bad enabled configurations fail before
model execution.145 relevant tests(123 config+22 DraftGraph) and12 workflow
checks passed. The initial mode probes used the same allowed P2 configuration
before the stricter scope check;their exact initial patch is preserved. A
separate4-request native dense smoke validates the final guarded route.

Qwen3-8B /1.7B,GPU1/18084,H8,maxK8,HF0.8,max_num_seqs4,32 requests,client
concurrency16,max output128,warmup4. FLASH_ATTN,Metadata validationON,Draft mixed
GraphON,raggedOFF,AR bypassON. No dependencies/backend/H were changed. The
supported --cudagraph-metrics flag records actual Target dispatch,padding and
frequencies. Logger tables reset each interval;figures below include HTTP
warmups and omit the unflushed tail,not the entire run.

| Native capture config | PIECEWISE calls | NONE calls | FULL calls | Effective rows | Padded rows |
|---|---:|---:|---:|---:|---:|
|default FULL_AND_PIECEWISE|614|5|0|6063|8428|
|PIECEWISE,sizes1..36|636|15|0|6220|6220|

Default padding overhead39.0% is additional model token rows,not39% additional
latency. GPUModelRunner sets uniform_decode_query_len=1+num_spec_tokens=9;
_is_uniform_decode requires all requests exactly9 tokens. Its dedicated FULL
routine rounds capture sizes to multiples of9,also affecting mixed PIECEWISE
sizes. Variable K/AR usually violates the FULL predicate. Dense PIECEWISE skips
that dedicated-FULL adjustment and captures exact1..36 rows;larger Prefills
fall back to NONE safely. No fake uniform metadata or invalid candidates were
introduced. Padding tokens in the standard vLLM route are ignored by native
valid-length/sampling logic;the padding itself is not an output-correctness bug.

Both initial diagnostic passes completed32/32. Client TPS166.898(default) vs
180.261(dense) is one diagnostic comparison with Metadata checks,not a repeated
production performance claim. Native compilation also affects Draft model
compilation,so these runs cannot isolate a Target-only Graph effect against an
enforce-eager CDB baseline. Do not mix their TPS with the202.872 TPS eager-Target
joint Graph configuration in GRAPH_BUDGET12.md.

Raw source/config/requests/responses/logs: /home/yypan/CDB-AV/results/target_modes_20261106/.
source.patch/json describe the initial probes;guarded_source.patch/json describe
the final scope guard. The final smoke uses guard_smoke4/. All services are
isolated and the harness shuts down only its own child process group.

Next:validate native dense capture with dynamic Draft Graph12 under identical
flags,then paired repetitions against the current eager-Target best. Retain a
bounded full Target Graph for irregular queries as a separate candidate;never
force uniform=True without correct query lengths,positions,block tables and
slot mapping. Hold H and the HF policy fixed. Current implementation remains
experimental;native preemption pressure and strict final output equivalence
are not certified by this dispatch audit.
