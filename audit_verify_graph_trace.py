"""Offline structural cache/copy-cost audit. No CUDA imports or runtime changes."""
from collections import Counter
from pathlib import Path
import hashlib,json,subprocess

VOCAB=128256


def estimate(key):
    nr,nt,mk=key
    return (nr+nt)*VOCAB*4*8+4096


def simulate(entries,min_hits=8,max_keys=8,max_bytes=512<<20):
    cache=set();hits={};used=0;calls=Counter(dict(plain_sampler_calls=0,spec_sampler_calls=0,graph_calls=0,captures=0));payload=0
    for phase,key in entries:
        if key is None:
            if phase=='measurement':calls['plain_sampler_calls']+=1
            continue
        if phase=='measurement':calls['spec_sampler_calls']+=1
        capture=False
        if key not in cache:
            if key not in hits and len(hits)>=256:hits.pop(next(iter(hits)))
            hits[key]=hits.get(key,0)+1
            if hits[key]>=min_hits and len(cache)<max_keys and used+estimate(key)<=max_bytes:
                cache.add(key);used+=estimate(key);capture=True
        if phase=='measurement' and key in cache:
            calls['graph_calls']+=1;calls['captures']+=capture
            nr,nt,mk=key
            # BF16 logits + source-expected metadata dtypes (logits_indices
            # int64, remaining integer tensors int32) + independent tiny clone.
            payload+=(nr+nt)*VOCAB*2 + 16*nt+20*nr + nr*(mk+1)*4
    n=calls['spec_sampler_calls']
    return dict(counts=dict(calls),conditional_spec_coverage=calls['graph_calls']/n if n else 0,
        cache_keys=sorted(cache),estimated_cache_bytes=used,
        added_copy_payload_bytes=payload,added_read_plus_write_bytes=2*payload,
        note='Structural simulation assumes dtype/contiguity/processor/stream scope passes; not actual Graph hit rate or TPS')


def main():
    root=Path('/home/yypan/CDB-AV/results')
    old=root/'causal_audit_20261120/full512_causal'
    entries=[];sources=[];cpu_rows=[]
    for p in sorted((old/'observer').glob('*_requests.jsonl')):
        h=hashlib.sha256();found=0
        with p.open('rb') as f:
            for raw in f:
                h.update(raw)
                if b'"target_batch"' not in raw:continue
                x=json.loads(raw)
                if x.get('kind')!='target_batch':continue
                counts=[r['draft_count'] for r in x['rows'].values()]
                assert counts and len(counts)<=4 and all(0<=k<=8 for k in counts)
                key=(len(counts),sum(counts),max(counts)) if any(counts) else None
                entries.append((x['epoch'],key));found+=1
                if x['epoch']=='measurement' and key is not None and len(cpu_rows)<128:
                    cpu_rows.append(list(x['rows'].values()))
        if found:sources.append(dict(path=str(p),sha256=h.hexdigest(),target_batches=found))
    assert len(sources)==1,'Cannot concatenate independent Engine timelines'
    measured=[x for x in entries if x[0]=='measurement']
    assert len(measured)==9707
    freq=Counter(k for _,k in measured if k is not None)
    upper=sum(n for k,n in freq.most_common(8))/sum(freq.values())
    summary=json.loads((old/'summary.json').read_text())
    assert summary['completed']==512 and summary['failed']==0 and summary['output_tokens']==101003
    current=json.loads((root/'ready_coalesce_20261120/MECHANISM_AUDIT.json').read_text())
    bounds=[]
    for index,row in enumerate(current['on_runs']):
        spec=row['cdb_target_calls_with_verify'];plain=row['cdb_target_schedules']-spec
        verify=row['cdb_target_verify_rows'];prefill=row['cdb_target_prefill_rows']
        assert row['cdb_target_ar_rows']==0
        draft=int(row['draft_candidates']);assert draft==row['draft_candidates']
        # Every plain call consumes1..4 Prefill rows; remaining Prefill rows
        # may join speculative calls. This bounds nr-sum without inventing K partitions.
        mixed_lo=max(0,prefill-4*plain);mixed_hi=max(0,prefill-plain)
        logits_lo=draft+verify+mixed_lo;logits_hi=draft+verify+mixed_hi
        bounds.append(dict(seed=42+index,spec_calls=spec,
            total_raw_logits_rows_bounds=[logits_lo,logits_hi],
            raw_BF16_copy_payload_all_spec_upper_bytes=logits_hi*VOCAB*2,
            read_plus_write_raw_copy_upper_bytes=logits_hi*VOCAB*4,
            maximum_extra_live_copy_API_calls=spec*7,
            maximum_extra_output_clone_calls=spec,
            scope='All spec calls covered is a traffic upper bound, not coverage forecast'))
    # Execute only the two unchanged metadata-construction methods on CPU.
    # AST extraction avoids importing/initializing the full GPU runner.
    import ast
    import numpy as np
    import torch
    from types import SimpleNamespace as NS
    from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
    from vllm.v1.sample.cdb_av_verify_graph import verify_graph_key
    worker=Path(__file__).parent/'vllm/v1/worker/gpu_model_runner.py'
    tree=ast.parse(worker.read_text())
    runner=next(z for z in tree.body if isinstance(z,ast.ClassDef) and z.name=='GPUModelRunner')
    methods=[z for z in runner.body if isinstance(z,ast.FunctionDef) and z.name in
             {'_get_cumsum_and_arange','_calc_spec_decode_metadata'}]
    assert len(methods)==2
    module=ast.Module(body=[ast.ClassDef(name='CPUReplay',bases=[],keywords=[],body=methods,decorator_list=[])],type_ignores=[])
    ns=dict(np=np,torch=torch,SpecDecodeMetadata=SpecDecodeMetadata)
    exec(compile(ast.fix_missing_locations(module),str(worker),'exec'),ns)
    obj=ns['CPUReplay']();obj.device=torch.device('cpu')
    obj.arange_np=np.arange(8192,dtype=np.int64)
    obj.input_ids=NS(gpu=torch.arange(8192,dtype=torch.int32,device='cpu'))
    vectors=Counter();fields=('draft_token_ids','cu_num_draft_tokens','cu_num_sampled_tokens',
                             'target_logits_indices','bonus_logits_indices','logits_indices')
    for batch in cpu_rows:
        counts=[r['draft_count'] for r in batch]
        scheduled=np.cumsum(np.array([r['n'] for r in batch],dtype=np.int32))
        meta=obj._calc_spec_decode_metadata(np.array(counts,dtype=np.int32),scheduled)
        expected=[i for end,k in zip(scheduled,counts) for i in range(int(end)-k,int(end))]
        assert meta.draft_token_ids.tolist()==expected
        assert meta.cu_num_draft_tokens.tolist()==np.cumsum(counts).tolist()
        vectors[str(tuple(str(getattr(meta,n).dtype) for n in fields))]+=1
        device=torch.device('cuda:0')
        tensors={n:NS(device=device,dtype=getattr(meta,n).dtype,shape=getattr(meta,n).shape,
                      is_contiguous=getattr(meta,n).is_contiguous) for n in fields}
        descriptor=NS(**tensors,num_draft_tokens=meta.num_draft_tokens,max_spec_len=meta.max_spec_len)
        logits=NS(device=device,dtype=torch.bfloat16,ndim=2,shape=(sum(counts)+len(counts),VOCAB),is_contiguous=lambda:True)
        sampling=NS(all_greedy=True,all_random=False,max_num_logprobs=None,no_penalties=True,
                    bad_words_token_ids={},allowed_token_ids_mask=None,logitsprocs=NS(non_argmax_invariant=[]))
        assert verify_graph_key(descriptor,None,logits,sampling) is not None
    metadata_replay=dict(samples=len(cpu_rows),worker_sha256=hashlib.sha256(worker.read_bytes()).hexdigest(),
        metadata_dtype_vectors=dict(vectors),CPU_value_checks=True,BF16_descriptor_scope_checks=True,
        scope='Unchanged source methods executed with CPU tensors; CUDA descriptors only for scope checks. No GPU pointer/capture/numerical test.')
    out=dict(candidate='S05',scope='Saved same-load formal512 causal audit BEFORE Ready coalescing; conditional shape-only simulation, no new GPU run',
        source_sha='73d37c98fb3ad44709a5a4d343c78d6944aa8490',sources=sources,CPU_source_metadata_replay=metadata_replay,
        prototype_git_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).parent,text=True).strip(),
        prototype_source_sha256=hashlib.sha256((Path(__file__).parent/'vllm/v1/sample/cdb_av_verify_graph.py').read_bytes()).hexdigest(),
        measurement_target_calls=len(measured),measurement_spec_calls=sum(freq.values()),unique_structural_keys=len(freq),
        old_FP32_only_prototype=dict(source_expected_raw_dtype='BF16',conditional_coverage_if_BF16=0,
            defect='Prototype would always fall back for raw BF16 Llama LM-head output'),
        fixed_dtype_first8_warmed=simulate(entries),fixed_dtype_first8_cold=simulate(measured),
        bounded_output8_hypothesis=dict(
            mechanism='Keep full probability math and actual per-row K; use fixed nine-column output workspace and crop independent returned clone to live maxK+1. Requires dedicated GPU equivalence gate.',
            warmed=simulate([(p,(k[0],k[1],8) if k is not None else None) for p,k in entries]),
            copy_cost_note='Tiny output clone charged at full nine-column upper bound in this simulation; real crop would copy less.'),
        any8keys_shape_frequency_upper_bound_ignoring_memory=upper,
        top8_structural_keys=[dict(key=k,count=n) for k,n in freq.most_common(8)],
        accepted_current_three_run_copy_bounds=bounds,
        screening_requirement=dict(nominal_reference_completion_s=101003/322.7994798208757,
            nominal_seconds_to_save_for_0_5pct_TPS=(101003/322.7994798208757)*(1-1/1.005),
            required_net_us_per_cached_call_by_assumed_count={str(n):(101003/322.7994798208757)*(1-1/1.005)*1e6/n for n in [1000,2500,4000,8727]},
            scope='Hypothetical net savings requirement only; current cached-call count unknown. No overlapping GPU/CPU time added or claimed saved.'),
        source_cost_review={'original_full_greedy_GPU_ops_BF16_approx':12,'original_full_greedy_GPU_ops_FP32_approx':10,
            'prototype_per_hit_added_live_copies':7,'prototype_per_hit_graph_launches':1,'prototype_per_hit_output_clones':1,
            'unchanged_GPU_math':'All original operations remain; extra copies add work. Source operation counts are not CUDA kernel/API timings.'},
        limits=['Current accepted coalescing formal512 logs provide marginals, not per-batch K partitions: its exact structural coverage cannot be recovered from those summaries.',
            'Historical before-coalescing trace coverage cannot be reported as current322.799 coverage.',
            'Estimated cache bytes are a policy estimate, not measured peak allocation or reserved-memory bound.',
            'Copy byte/API counts do not bound latency without GPU measurements; no saved TPS or critical-path savings are inferred.',
            'A favorable cache hit count alone is not a reason to run full performance; exact GPU gate and useful primitive wall timing remain required.'])
    dest=root/'verify_graph_20261120';dest.mkdir(exist_ok=True)
    (dest/'S05_TRACE_COVERAGE.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
