# Opt-in Draft/Target stage profile after AR bypass

Diagnostic worktree `cdb-ar-profile-20261106`, based on f68848cf7c41f5c7bb3d63fe96ec90f622f47950.
`CDB_AV_PROFILE_DRAFT=1` collects Counter/perf_counter_ns measurements and logs
summaries every256 effective Draft steps /128 Target calls. It changes no
model inputs, token selection, metadata, KV ownership, window or scheduler.
All timings are host elapsed intervals, not pure GPU kernel time. Copy/wait
includes pending model work; it must not all be charged to confidence.

The32-request/128-output diagnostic with H8,maxK8,HF0.8,bypassON,mixed Draft
GraphON,raggedOFF,FLASH_ATTN,GPU1/18084 completed32/32. Separate4-request
Graph/eager validation completed4/4.171 relevant CPU tests and12 workflow tests
passed. Raw logs/config/patch hashes: results/ar_bypass_20261106/profile32/,
profile_smoke4/,profile_source.json on the server. Diagnostic TPS is not a
performance comparison. Periodic summaries cover different cumulative prefixes;
do not sum Target/Draft endpoints as a complete run time decomposition.

Last recorded Draft prefix (512 effective steps):

| Counter | Value |
|---|---:|
|Total step elapsed|7.424667s|
|Forward host interval|6.030267s|
|Eager forwards|349;5.743025s|
|Graph replays|153;0.044786s|
|Graph captures|10;0.242456s|
|Metadata|0.160448s|
|Confidence submit interval|0.236960s|
|GPU wait + copy|0.853630s|
|Eager max-query-length2|175|
|Eager max-query-length>2|174|

Forward host interval is81.2% of step elapsed; confidence submit3.2%,metadata2.2%.
The largest removable interval remains eager mixed Forward, including query2
shapes supported by the experimental dynamic graph. Profile record intervals
include copies/allocator stalls and logits work; they are not isolated launch
costs or GPU-only times. Increasing graph coverage should be measured before
micro-optimizing scalar packing or weakening any ownership/fingerprint asserts.

Target prefix (512 calls) elapsed12.026274s,2000 request rows /5462 token rows:
90 pure verify,355 mixed,61 pureAR,6 prepare. Average3.906 request rows/call is
close to max_num_seqs4. Verify rows975,AR rows993,prepare rows32. PureAR elapsed
1.508661s;most Target time belongs to mixed stages(8.203026s). These numbers
include sampling/CPU wait and cannot be read as pure model Forward GPU time.

## Remaining Target Graph evidence gap

Old native-graph probe logs captured both mixed PIECEWISE and uniform FULL
graphs, but disabled cudagraph_metrics and did not record runtime hit frequencies.
It also removed enforce_eager globally; draft config inherits target model config
and compilation settings. Its TPS delta is not a clean measurement of Target-only
Graph benefit.

Current gpu_model_runner.py sets uniform_decode_query_len=1+num_spec_tokens,
which is9 at cap8. `_is_uniform_decode` requires max scheduled query9 and all
requests exactly that length. HF-variable K and AR mixtures generally violate
this predicate. FlashAttention builder grants ALWAYS support only for FA3;
otherwise UNIFORM_BATCH. FlashInfer does not automatically resolve the issue:
its builder grants UNIFORM_BATCH or UNIFORM_SINGLE_TOKEN_DECODE. Never force
an irregular batch to uniform without matching query offsets, positions and
KV slot metadata.

The next isolated native Target probe should enable the supported
`--cudagraph-metrics` flag (arg_utils.py and observability.py), recording mode
frequencies and padding. Only then evaluate bounded full-graph handling for
small irregular Target query layouts, retaining sampling outside graph and all
KV/cancellation assertions. Keep H fixed and preserve backend/model comparability.
