# SPDX-License-Identifier: Apache-2.0
"""Opt-in exact-row whole-Target graphs for synchronous Qwen3/FA2 CDB.

Sampling and commit stay in GPUModelRunner. No native uniform/padding descriptor
is forged. Query1 is always separate; mixed maximum query can use an explicit
opt-in bound while the exact-maximum mode remains available.
"""
import os
import json
import weakref
import time
from collections import Counter
from dataclasses import replace

import torch

from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.logger import init_logger

logger = init_logger(__name__)


def validate_scope(*, persistent, refill, eager, async_scheduling,
                   prefix_caching, lora, model, dtype, quantization,
                   parallel_sizes):
    if (not persistent or not refill or not eager or async_scheduling
            or prefix_caching or lora or model != 'Qwen3ForCausalLM'
            or dtype != torch.bfloat16 or quantization is not None
            or len(parallel_sizes) != 4
            or any(type(n) is not int or n != 1 for n in parallel_sizes)):
        raise ValueError('Target full Graph requires synchronous eager Qwen3 '
                         'BF16 persistent refill, TP/PP/DP/DCP1, no prefix/LoRA/quantization')


def validate_sequence_floor(floor, max_model_len):
    if (type(floor) is not int or floor < 0 or floor > max_model_len
            or (floor and floor & (floor - 1))):
        raise ValueError('Invalid Target Graph sequence floor')


def graph_key(meta, starts, *, n, capacity, max_query, max_model_len,
              dynamic=False, seq_floor=0):
    """Use the already prepared CPU shadow, never a hot-path D2H reduction."""
    validate_sequence_floor(seq_floor, max_model_len)
    nr = meta.seq_lens.numel()
    if (not 1 <= nr <= capacity or n != meta.num_actual_tokens
            or len(starts) != nr + 1 or starts[0] != 0 or starts[-1] != n
            or meta.block_table.ndim != 2 or meta.block_table.shape[0] != nr
            or meta.block_table.shape[1] < 1
            or meta.query_start_loc.shape != (nr + 1,)
            or meta.seq_lens.ndim != 1 or meta.slot_mapping.shape != (n,)
            or meta.query_start_loc.dtype != torch.int32
            or meta.seq_lens.dtype != torch.int32 or meta.block_table.dtype != torch.int32
            or meta.slot_mapping.dtype != torch.int64):
        return None
    qlens = [b - a for a, b in zip(starts, starts[1:])]
    if (min(qlens) < 1 or max(qlens) != meta.max_query_len
            or max(qlens) > max_query or n > capacity * max_query
            or not 1 <= meta.max_seq_len <= max_model_len):
        return None
    # This scalar is only a launch upper bound. Per-request GPU seq_lens,
    # positions, query offsets and slots remain their exact live values.
    bucket = max(seq_floor, 1 << (meta.max_seq_len - 1).bit_length())
    if bucket > max_model_len:
        return None
    # Query1 stays separate for FA2 packed-GQA. Mixed bounds are explicit;
    # layouts update GPU offsets, sequence lengths and KV on every replay.
    query_max = max_query if dynamic and meta.max_query_len > 1 else meta.max_query_len
    return (n, nr, query_max, bucket, meta.block_table.shape[1],
            meta.max_num_splits, 'bounded' if dynamic else 'exact')


def target_graph_timing_config(env=None):
    """Parse bounded diagnostic timing settings without touching CUDA."""
    env = os.environ if env is None else env
    enabled = env.get('CDB_AV_TARGET_GRAPH_TIMING', '0') == '1'
    if not enabled:
        return False, 5
    validate = env.get('CDB_AV_TARGET_GRAPH_VALIDATE', '0') == '1'
    if not validate:
        raise ValueError('Target Graph timing requires CDB_AV_TARGET_GRAPH_VALIDATE=1')
    try:
        max_samples = int(env.get('CDB_AV_TARGET_GRAPH_TIMING_MAX_SAMPLES', '5'))
    except (TypeError, ValueError) as exc:
        raise ValueError('Invalid Target Graph timing sample limit') from exc
    if not 1 <= max_samples <= 32:
        raise ValueError('Target Graph timing sample limit must be in [1, 32]')
    return True, max_samples


def metadata_has_no_planner(meta):
    # FA2 builder uses scalar0 for the inactive DCP maximum, not None.
    return (not meta.use_cascade and not meta.common_prefix_len and meta.causal
            and meta.max_dcp_context_kv_len in (None, 0)
            and all(getattr(meta, name) is None for name in
                    ('scheduler_metadata', 'prefix_scheduler_metadata',
                     'cu_prefix_query_lens', 'prefix_kv_lens', 'suffix_kv_lens',
                     'dcp_context_kv_lens')))


def clone_metadata(meta, seq_bound, query_bound=None):
    return replace(meta, max_seq_len=seq_bound,
                   max_query_len=meta.max_query_len if query_bound is None else query_bound,
                   query_start_loc=meta.query_start_loc.clone(),
                   seq_lens=meta.seq_lens.clone(),
                   block_table=meta.block_table.clone(),
                   slot_mapping=meta.slot_mapping.clone())


def query_domain_matches(captured, actual, mixed_bound=None):
    if mixed_bound is None:
        return captured == actual
    return ((captured == actual == 1)
            or (captured == mixed_bound and 2 <= actual <= mixed_bound))


def update_metadata(static, live, mixed_bound=None):
    if (static.num_actual_tokens != live.num_actual_tokens
            or not query_domain_matches(static.max_query_len, live.max_query_len, mixed_bound)
            or static.max_num_splits != live.max_num_splits
            or not 1 <= live.max_seq_len <= static.max_seq_len):
        raise RuntimeError('Target Graph metadata scalar domain changed')
    for field in ('query_start_loc', 'seq_lens', 'block_table', 'slot_mapping'):
        dst, src = getattr(static, field), getattr(live, field)
        if dst.shape != src.shape or dst.dtype != src.dtype or dst.device != src.device:
            raise RuntimeError(f'Target Graph metadata storage mismatch: {field}')
        dst.copy_(src)


class TargetFullGraph:
    def __init__(self, runner):
        self.r = runner
        self.model_ref = weakref.ref(runner.model)
        cfg = runner.vllm_config
        pc = cfg.parallel_config
        validate_scope(persistent=runner.cdb_persistent, refill=runner.cdb_refill,
            eager=cfg.model_config.enforce_eager,
            async_scheduling=cfg.scheduler_config.async_scheduling,
            prefix_caching=cfg.cache_config.enable_prefix_caching,
            lora=cfg.lora_config is not None, model=type(runner.model).__name__,
            dtype=cfg.model_config.dtype, quantization=cfg.model_config.quantization,
            parallel_sizes=(pc.tensor_parallel_size, pc.pipeline_parallel_size,
                pc.data_parallel_size, pc.decode_context_parallel_size))
        model_modules = {id(m) for m in runner.model.modules()}
        self.layers = {name: layer for name, layer in
            cfg.compilation_config.static_forward_context.items()
            if id(layer) in model_modules and hasattr(layer, 'kv_cache')}
        if len(self.layers) != cfg.model_config.hf_config.num_hidden_layers:
            raise ValueError('Cannot identify all Target attention layers')
        if any(getattr(layer.impl, 'vllm_flash_attn_version', None) != 2
               for layer in self.layers.values()):
            raise ValueError('Target full Graph is restricted to FlashAttention2')
        self.capacity = cfg.scheduler_config.max_num_seqs
        self.max_query = cfg.speculative_config.num_speculative_tokens + 1
        self.max_model_len = cfg.model_config.max_model_len
        if (self.capacity != 4 or not 1 <= self.max_query <= 9
                or self.max_model_len > 1024 or runner.model.training
                or getattr(cfg.model_config, 'enable_sleep_mode', False)):
            raise ValueError('Unsupported Target full Graph size/training domain')
        self.dynamic = os.getenv('CDB_AV_TARGET_GRAPH_DYNAMIC', '0') == '1'
        self.seq_floor = int(os.getenv('CDB_AV_TARGET_GRAPH_SEQ_FLOOR', '0'))
        validate_sequence_floor(self.seq_floor, self.max_model_len)
        self.limit = int(os.getenv('CDB_AV_TARGET_GRAPH_MAX_KEYS', '16'))
        self.min_hits = int(os.getenv('CDB_AV_TARGET_GRAPH_MIN_HITS', '8'))
        if not 1 <= self.limit <= 32 or self.min_hits < 1:
            raise ValueError('Invalid Target Graph cache budget/frequency')
        self.cache, self.hits, self.counts = {}, Counter(), Counter()
        self.storage = None
        self.timing_enabled, self.timing_max_samples = target_graph_timing_config()
        self.timing_samples = 0
        self.timing_limit_logged = False

    def _storage(self):
        context = self.r.vllm_config.compilation_config.static_forward_context
        if (self.model_ref() is not self.r.model
                or any(context.get(name) is not layer for name,layer in self.layers.items())):
            raise RuntimeError('Target Graph model/attention identity changed before replay')
        caches = {name: layer.kv_cache[0] for name, layer in self.layers.items()}
        if any(kv.ndim != 5 or kv.shape[0] != 2 or kv.dtype != torch.bfloat16
               or not kv.is_cuda for kv in caches.values()):
            raise ValueError('Target full Graph requires BF16 FA2 KV layout')
        storage = (id(self.r.model), *(item for name, kv in caches.items()
                   for item in (name, kv.data_ptr(), tuple(kv.shape), kv.stride(), str(kv.device))))
        if self.storage != storage:
            # Execution is serial: the previous sampler completed before another
            # Target stage. A replacement KV allocation must never reuse a graph.
            self.cache.clear()
            self.hits.clear()
            self.storage = storage
        return caches

    def _record(self, action, key=None, **data):
        self.counts[action] += 1
        self.counts['calls'] += 1
        if os.getenv('CDB_AV_TRACE_TARGET_GRAPH', '0') == '1':
            logger.info('[CDB_TARGET_FULL_GRAPH] %s', json.dumps(dict(
                action=action, key=key, **data)))
        elif self.counts['calls'] % 128 == 0:
            logger.info('[CDB_TARGET_FULL_GRAPH_AGG] %s', json.dumps(dict(self.counts)))

    def forward(self, input_ids, positions, intermediate_tensors, inputs_embeds, kwargs):
        r = self.r
        ctx = get_forward_context()
        if (input_ids is None or positions is None or intermediate_tensors is not None
                or inputs_embeds is not None or kwargs or input_ids.ndim != 1
                or positions.shape != input_ids.shape or not input_ids.is_cuda
                or positions.device != input_ids.device
                or input_ids.dtype not in (torch.int32, torch.int64)
                or positions.dtype != torch.int64
                or ctx.cudagraph_runtime_mode != CUDAGraphMode.NONE
                or ctx.ubatch_slices is not None
                or not isinstance(ctx.attn_metadata, dict)
                or not isinstance(ctx.slot_mapping, dict)):
            self._record('fallback_scope', kwargs=sorted(kwargs),
                mode=str(ctx.cudagraph_runtime_mode),ubatch=str(ctx.ubatch_slices),
                embeddings=inputs_embeds is not None)
            return None
        caches = self._storage()
        try:
            metas = {name: ctx.attn_metadata[name] for name in self.layers}
            slots = {name: ctx.slot_mapping[name] for name in self.layers}
        except KeyError:
            self._record('fallback_metadata')
            return None
        live = next(iter(metas.values()))
        # One attention group only. Optional planner/cascade/DCP tensors would
        # require their own replay update protocol and are explicitly excluded.
        if (type(live).__name__ != 'FlashAttentionMetadata'
                or type(live).__module__ != 'vllm.v1.attention.backends.flash_attn'
                or any(m is not live for m in metas.values())
                or any(getattr(live, field).device != input_ids.device for field in
                       ('query_start_loc','seq_lens','block_table','slot_mapping'))
                or not metadata_has_no_planner(live)
                or any(s.data_ptr() != live.slot_mapping.data_ptr()
                       or s.shape != live.slot_mapping.shape for s in slots.values())):
            self._record('fallback_metadata')
            return None
        n = input_ids.numel()
        nr = live.seq_lens.numel()
        starts = r.query_start_loc.cpu[:nr + 1].tolist()
        key = graph_key(live, starts, n=n, capacity=self.capacity,
                        max_query=self.max_query, max_model_len=self.max_model_len,
                        dynamic=self.dynamic, seq_floor=self.seq_floor)
        if key is None:
            self._record('fallback_layout')
            return None
        entry = self.cache.get(key)
        if entry is None:
            self.hits[key] += 1
            if len(self.cache) >= self.limit or self.hits[key] < self.min_hits:
                self._record('fallback_budget' if len(self.cache) >= self.limit
                             else 'fallback_warmup', key)
                return None
            if torch.cuda.mem_get_info(input_ids.device)[0] < 512 * 1024**2:
                self._record('fallback_memory', key)
                return None
            static = clone_metadata(live, key[3], query_bound=key[2])
            entry = dict(meta=static, ids=input_ids.clone(), positions=positions.clone())
        else:
            static = entry['meta']
            update_metadata(static, live, mixed_bound=self.max_query if self.dynamic else None)
            entry['ids'].copy_(input_ids)
            entry['positions'].copy_(positions)
        static_metas = {name: static for name in self.layers}
        static_slots = {name: static.slot_mapping for name in self.layers}

        def model(meta, slotmap, ids, pos):
            with set_forward_context(meta, r.vllm_config, num_tokens=n,
                    cudagraph_runtime_mode=CUDAGraphMode.NONE, slot_mapping=slotmap):
                return r.model(input_ids=ids, positions=pos,
                    intermediate_tensors=None, inputs_embeds=None)

        if key not in self.cache:
            torch.cuda.synchronize(input_ids.device)
            graph = torch.cuda.CUDAGraph()
            # Private pools avoid cross-key temporary/output lifetime assumptions.
            with torch.cuda.graph(graph):
                hidden = model(static_metas, static_slots, entry['ids'], entry['positions'])
            if not isinstance(hidden, torch.Tensor) or hidden.shape[0] != n:
                raise RuntimeError('Unsupported Target Graph model output')
            entry.update(graph=graph, hidden=hidden)
            self.cache[key] = entry
            action = 'capture'
        else:
            action = 'replay'
        entry['graph'].replay()
        req_ids = list(r.input_batch.req_ids[:nr])
        self._record(action, key, req_ids=req_ids, token_rows=n, request_rows=nr,
                     query_lengths=[b-a for a,b in zip(starts, starts[1:])], padding=0)
        result = entry['hidden']
        if os.getenv('CDB_AV_TARGET_GRAPH_VALIDATE', '0') == '1':
            if self.timing_enabled:
                # In timing mode, sample only warmed Graph replays. Capture calls
                # are not representative of steady-state replay cost.
                should_validate = (action == 'replay'
                                   and self.timing_samples < self.timing_max_samples)
                if not should_validate and self.timing_samples >= self.timing_max_samples:
                    if not self.timing_limit_logged:
                        logger.info('[CDB_TARGET_GRAPH_TIMING_LIMIT] %s', json.dumps(dict(
                            samples=self.timing_samples,
                            sample_limit=self.timing_max_samples)))
                        self.timing_limit_logged = True
            else:
                should_validate = True
            if should_validate:
                result = entry['hidden'].clone()
                self._validate(entry, live, caches, model, metas, slots,
                               input_ids, positions, starts, req_ids, key, action)
                # The normal route follows native graph output lifetime; diagnostic
                # eager calls cannot alter the saved result returned to sampling.
        return result

    def _validate(self, entry, live, caches, model, metas, slots,
                  ids, pos, starts, req_ids, key, action):
        nr, n = key[1], key[0]
        seqs = live.seq_lens.cpu().tolist()
        positions = pos.cpu().tolist()
        qlens = [b-a for a,b in zip(starts, starts[1:])]
        expected = [p for length, q in zip(seqs, qlens) for p in range(length-q, length)]
        if (positions != expected or live.query_start_loc.cpu().tolist() != starts
                or seqs != self.r.seq_lens.cpu[:nr].tolist()):
            raise RuntimeError('Target Graph CPU/GPU query/position/sequence mismatch')
        block_sizes = {kv.shape[2] for kv in caches.values()}
        if len(block_sizes) != 1:
            raise RuntimeError('Target Graph mixed KV block sizes')
        bs = next(iter(block_sizes))
        table = live.block_table.cpu().tolist()
        expected_slots = [table[i][p//bs]*bs+p%bs for i,(length,q) in
            enumerate(zip(seqs,qlens)) for p in range(length-q,length)]
        actual_slots = live.slot_mapping.cpu().tolist()
        if expected_slots != actual_slots or min(actual_slots) < 0 or len(set(actual_slots)) != n:
            raise RuntimeError('Target Graph request/position/KV slot mismatch')
        block = live.slot_mapping.div(bs, rounding_mode='floor')
        offset = live.slot_mapping.remainder(bs)
        saved = {name: kv[:, block, offset].contiguous().clone() for name,kv in caches.items()}
        timing = None
        sample_index = None
        if self.timing_enabled and action == 'replay':
            self.timing_samples += 1
            sample_index = self.timing_samples
            # Diagnostic only: event spans include host submission gaps and are
            # not a CUPTI sum of active Kernel times.
            timing = {}

            def measured(fn):
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                begin.record()
                cpu_begin = time.perf_counter()
                output = fn()
                cpu_ms = (time.perf_counter() - cpu_begin) * 1000
                end.record()
                end.synchronize()
                return output, dict(cpu_enqueue_ms=cpu_ms,
                                    stream_span_ms=begin.elapsed_time(end))

            cpu_begin = time.perf_counter()
            self._storage()
            timing['storage_audit_cpu_ms'] = (time.perf_counter() - cpu_begin) * 1000

            def refresh_inputs():
                update_metadata(entry['meta'], live,
                    mixed_bound=self.max_query if self.dynamic else None)
                entry['ids'].copy_(ids)
                entry['positions'].copy_(pos)

            _, timing['six_buffer_copies'] = measured(refresh_inputs)
            before_hidden = entry['hidden'].clone()
            _, timing['graph_replay'] = measured(entry['graph'].replay)
            if (not torch.equal(before_hidden.view(torch.uint8), entry['hidden'].view(torch.uint8))
                    or not all(torch.equal(saved[name].view(torch.uint8),
                        kv[:, block, offset].contiguous().view(torch.uint8))
                        for name,kv in caches.items())):
                raise RuntimeError('Timed replay changed Target hidden/KV bytes')
            live_tensors = dict(input_ids=ids, positions=pos,
                query_start_loc=live.query_start_loc, seq_lens=live.seq_lens,
                block_table=live.block_table, slot_mapping=live.slot_mapping)
            timing['live_storage'] = {name: dict(pointer=tensor.data_ptr(),
                shape=list(tensor.shape), stride=list(tensor.stride()),
                dtype=str(tensor.dtype), device=str(tensor.device))
                for name,tensor in live_tensors.items()}

        if timing is None:
            graph_logits = self.r.model.compute_logits(entry['hidden']).clone()
        else:
            graph_logits, timing['graph_logits'] = measured(
                lambda: self.r.model.compute_logits(entry['hidden']).clone())
        checks = []
        try:
            for kind, meta, slotmap, x, p in (
                    ('static', {name:entry['meta'] for name in caches},
                     {name:entry['meta'].slot_mapping for name in caches}, entry['ids'],entry['positions']),
                    ('live',metas,slots,ids,pos)):
                if timing is None:
                    eager = model(meta,slotmap,x,p)
                    logits = self.r.model.compute_logits(eager)
                else:
                    eager, timing[kind + '_eager'] = measured(
                        lambda: model(meta,slotmap,x,p))
                    logits, timing[kind + '_logits'] = measured(
                        lambda: self.r.model.compute_logits(eager))
                kv_exact = all(torch.equal(saved[name].view(torch.uint8),
                    kv[:,block,offset].contiguous().view(torch.uint8)) for name,kv in caches.items())
                token_exact = torch.equal(graph_logits.argmax(-1),logits.argmax(-1))
                close = torch.allclose(graph_logits,logits,rtol=1e-3,atol=1e-3)
                checks.append(dict(kind=kind,kv_byte_exact=kv_exact,token_exact=token_exact,
                    logits_close=close,max_abs=float((graph_logits-logits).abs().max())))
        finally:
            for name,kv in caches.items():
                kv[:,block,offset] = saved[name]
        logger.info('[CDB_TARGET_FULL_GRAPH_VALIDATE] %s',json.dumps(dict(
            action=action,key=key,req_ids=req_ids,qlens=qlens,positions=positions,
            slot_mapping=actual_slots,layers=list(caches),checks=checks,extra_eager_forwards=2)))
        if not all(c['kv_byte_exact'] and c['token_exact'] and c['logits_close'] for c in checks):
            raise RuntimeError('Target full Graph/static/live eager mismatch')
        if timing is not None:
            logger.info('[CDB_TARGET_GRAPH_TIMING] %s', json.dumps(dict(
                action=action,key=key,req_ids=req_ids,qlens=qlens,
                actual_max_sequence=max(positions)+1,
                sample_index=sample_index,
                sample_limit=self.timing_max_samples,
                measurements=timing,
                extra_graph_replays=1,extra_eager_forwards=2,
                model_scope='Target backbone and LM Head separately; excludes sampling/commit',
                kv_and_hidden_exact=True)))
