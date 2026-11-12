# SPDX-License-Identifier: Apache-2.0
"""Synchronous, bounded GPU draft windows with dynamically admitted members.

Target input batches remain untouched. Waiting work uses allocator-owned leases;
its prompt is prefetched by the independent draft model, never by Target here.
"""
import json
import os
import time
from collections import Counter
import torch
from vllm.config import CUDAGraphMode
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.spec_decode.cdb_av_draft_stage_profile import get_draft_stage_profiler
from vllm.v1.spec_decode.cdb_av_fingerprint import token_fingerprint
from vllm.v1.spec_decode.cdb_av_acceptance import should_stop
from vllm.v1.spec_decode.cdb_av_config import (
    confidence_stop_flags,
    get_ragged_graph_query_bound,
    validate_ragged_graph_backend,
    get_k_stop_reason,
)

logger = init_logger(__name__)


_REFILL_SUMMARY_EVENTS = {
    'k_policy_config',
    'window_begin',
    'window_end',
    'ready',
    'p2_runtime_config',
    'p2_target_begin',
    'p2_ar_target_bypass',
    'target_ready_select',
    'p2_defer_prefill_budget',
    'p2_window_begin',
    'p2_target_commit',
    'p2_verified',
    'k_replay_decision',
    'k_request_average_decision',
    'p2_handoff',
    'p2_boundary',
    'p2_reuse_plan',
    'reuse_fallback',
    'reuse_prepared',
    'reuse_pending_prefill',
    'reuse_cache_consumed',
    'reuse_lease_release',
}


_LOW_LOG_COUNTS = Counter()
_LOW_LOG_K = Counter()
_LOW_LOG_REASONS = Counter()
_LOW_LOG_REUSE = Counter()
_LOW_LOG_REPLAY = Counter()


def emit(event, **data):
    # The opt-in performance mode retains cumulative Worker evidence while
    # avoiding one JSON log record per Target/Draft step. Detailed mode is
    # unchanged and remains the default for correctness audits.
    if (os.getenv('CDB_AV_LOW_LOG', '0') == '1'
            and os.getenv('CDB_AV_REFILL_TRACE', '0') != '1'):
        if event in _REFILL_SUMMARY_EVENTS:
            _LOW_LOG_COUNTS[event] += 1
            _LOW_LOG_COUNTS['__total__'] += 1
            if event == 'ready':
                _LOW_LOG_K[data['k_actual']] += 1
                _LOW_LOG_REASONS[data['reason']] += 1
            elif event == 'p2_reuse_plan':
                _LOW_LOG_REUSE['reused_tokens'] += data['reused_tokens']
                _LOW_LOG_REUSE['prefill_tokens'] += data['prefill_tokens']
            elif event == 'p2_verified':
                _LOW_LOG_REUSE['accepted_tokens'] += data['accepted']
                _LOW_LOG_REUSE['drafted_tokens'] += len(data['drafts'])
            elif event in ('k_replay_decision', 'k_request_average_decision'):
                _LOW_LOG_REPLAY[data['source']] += 1
            if _LOW_LOG_COUNTS['__total__'] % 1024 == 0:
                logger.info('[CDB_AGG] %s', json.dumps(dict(
                    events=dict(_LOW_LOG_COUNTS),
                    k_actual=dict(_LOW_LOG_K),
                    stop_reasons=dict(_LOW_LOG_REASONS),
                    token_counts=dict(_LOW_LOG_REUSE),
                    replay=dict(_LOW_LOG_REPLAY),
                )))
        return
    # Keep request/verification/reuse summaries available by default, while
    # making high-frequency per-forward and per-step diagnostics opt-in.
    # This preserves the counters needed for performance analysis without
    # serializing every intermediate GPU step into the server log.
    if (
        os.getenv('CDB_AV_REFILL_TRACE', '0') == '1'
        or event in _REFILL_SUMMARY_EVENTS
    ):
        logger.info('[CDB_REFILL] %s', json.dumps(dict(event=event, **data)))


def _log_draft_graph_runtime_once(
    proposer, action, *, phase, request_rows, token_rows,
    graph_key=None, reason=None,
):
    """Log each Draft Graph action/shape once, capped at 32 records per process."""
    seen = getattr(proposer, '_cdb_draft_graph_runtime_seen', None)
    if seen is None:
        seen = set()
        proposer._cdb_draft_graph_runtime_seen = seen
    # Once the diagnostic quota is full, avoid serializing a key on every
    # later Forward. This does not change graph dispatch or model execution.
    if len(seen) >= 32:
        return
    # Keep enough detail to distinguish graph keys while bounding log volume.
    # Include the phase and live shape because the same key/action can be used
    # by different forward phases or request layouts.
    graph_key_id = (json.dumps(graph_key, sort_keys=True)
                    if graph_key is not None else None)
    log_key = (action, reason, phase, request_rows, token_rows, graph_key_id)
    if log_key in seen:
        return
    seen.add(log_key)

    env_defaults = (
        ('CDB_AV_DRAFT_DECODE_GRAPH', '0'),
        ('CDB_AV_DRAFT_RAGGED_GRAPH', '0'),
        ('CDB_AV_DRAFT_RAGGED_DYNAMIC', '0'),
        ('CDB_AV_DRAFT_RAGGED_MAX_QUERY', '2'),
        ('CDB_AV_RAGGED_GRAPH_MAX_KEYS', '4'),
        ('CDB_AV_RAGGED_GRAPH_MIN_HITS', '8'),
    )
    config = {name: os.getenv(name, default)
              for name, default in env_defaults}
    logger.info('[CDB_DRAFT_GRAPH_RUNTIME] %s', json.dumps(dict(
        pid=os.getpid(),
        action=action,
        phase=phase,
        request_rows=request_rows,
        token_rows=token_rows,
        graph_key=graph_key,
        reason=reason,
        config=config,
    )))


class RefillSession:
    def __init__(self, proposer, kwargs):
        self.p = proposer
        self.kwargs = kwargs
        self.members = {}
        self.active = []
        # Pending Prompt rows are consumed by the next Draft forward.
        self.pending_prefill = []
        self.steps_completed = 0
        self.done = False
        self.window = None
        self.initialized = False
        self.forward_count = 0
        self.row_sum = 0
        self.ready_records = []
        self._draft_stage_profiler = get_draft_stage_profiler(proposer)
        self._draft_stage_pending = None
        self._draft_stage_metadata_ms = 0.0
        self.graph_query_bound = get_ragged_graph_query_bound()
        self.builder = proposer.attn_metadata_builder or proposer._get_attention_metadata_builder()
        validate_ragged_graph_backend(self.graph_query_bound, self.builder)
        self.block_size = self.builder.kv_cache_spec.block_size
        self.policy = os.getenv('CDB_AV_K_POLICY', 'manual').strip().lower()
        self.threshold = float(os.getenv('CDB_AV_HF_THRESHOLD', '0.4'))
        if self.policy not in ('manual', 'hf_confidence', 'replay_acceptance',
                               'request_avg_acceptance') or not 0 <= self.threshold <= 1:
            raise ValueError('Invalid refill stopping policy')
        emit(
            'k_policy_config',
            policy=self.policy,
            threshold=self.threshold,
            server_max_k=getattr(proposer, 'num_speculative_tokens', None),
        )

    def configure(self, window, horizon, contexts, capacity):
        if self.window is not None or (
                horizon < max(self.kwargs['request_draft_lengths'])
                and os.getenv('CDB_P1_VIRTUAL_H', '0') != '1'):
            raise ValueError('Refill needs a new window and H >= initial K cap')
        if capacity != self.p.vllm_config.scheduler_config.max_num_seqs:
            raise ValueError('Refill capacity must equal max_num_seqs')
        self.window, self.horizon, self.capacity = window, horizon, capacity
        self.contexts = contexts
        emit('window_begin', window=window, H=horizon, contexts=contexts)

    def _metadata(self, blocks, seqs, qlens, slots):
        profiler = getattr(self, '_draft_stage_profiler', None)
        tick = (time.perf_counter() if profiler is not None
                and profiler.has_capacity else None)
        dev = self.p.device
        if os.getenv('CDB_AV_PACKED_METADATA', '0') == '1':
            from vllm.v1.spec_decode.cdb_av_metadata import packed_metadata_tensors
            fields = packed_metadata_tensors(blocks, seqs, qlens, slots, dev)
            if not getattr(self.p, '_cdb_packed_metadata_recorded', False):
                logger.info('[CDB_METADATA_RUNTIME] %s', json.dumps(dict(
                    pid=os.getpid(), caller_source=__file__,
                    factory_source=packed_metadata_tensors.__code__.co_filename,
                    device=str(dev), single_transfer=True)))
                self.p._cdb_packed_metadata_recorded = True
            result = CommonAttentionMetadata(
                **fields, num_reqs=len(blocks), num_actual_tokens=sum(qlens),
                max_query_len=max(qlens), max_seq_len=max(seqs), causal=True,
                _seq_lens_cpu=None, _num_computed_tokens_cpu=None)
        else:
            width = max(len(b) for b in blocks)
            bt = torch.zeros((len(blocks), width), dtype=torch.int32, device=dev)
            for i, row in enumerate(blocks):
                bt[i, :len(row)] = torch.tensor(row, dtype=torch.int32, device=dev)
            qcpu = torch.tensor([0] + list(torch.tensor(qlens).cumsum(0).tolist()), dtype=torch.int32)
            result = CommonAttentionMetadata(query_start_loc=qcpu.to(dev), query_start_loc_cpu=qcpu,
                seq_lens=torch.tensor(seqs, dtype=torch.int32, device=dev),
                num_reqs=len(blocks), num_actual_tokens=sum(qlens), max_query_len=max(qlens),
                max_seq_len=max(seqs), block_table_tensor=bt,
                slot_mapping=torch.tensor(slots, dtype=torch.int64, device=dev), causal=True,
                _seq_lens_cpu=None, _num_computed_tokens_cpu=None)
        if tick is not None:
            self._draft_stage_metadata_ms = (time.perf_counter() - tick) * 1000
        return result

    def _decode_graph_key(self, cad, n):
        """Only one-token-per-request rows are eligible for the Draft graph."""
        if (n != cad.num_reqs or cad.num_actual_tokens != n
                or cad.max_query_len != 1 or n > self.capacity):
            return None
        required = max(cad.max_seq_len,
                       cad.block_table_tensor.shape[1] * self.block_size)
        bucket = max(self.block_size, 1 << (required - 1).bit_length())
        if bucket > self.p.max_model_len:
            return None
        return n, bucket

    def _mixed_graph_key(self, cad, n, dynamic=False, max_query=2):
        if max_query not in (2, 4, 8):
            raise ValueError('Draft Graph query bound must be 2, 4 or 8')
        # The existing exact-layout path retains its original query1/2 scope.
        limit = max_query if dynamic else 2
        if (not 1 <= cad.num_reqs <= self.capacity
                or n != cad.num_actual_tokens
                or not 2 <= cad.max_query_len <= limit
                or n > limit * self.capacity):
            return None
        starts = cad.query_start_loc_cpu.tolist()
        if len(starts) != cad.num_reqs + 1 or starts[0] != 0 or starts[-1] != n:
            return None
        qlens = tuple(end - start for start, end in zip(starts, starts[1:]))
        if not all(1 <= qlen <= limit for qlen in qlens):
            return None
        if max(qlens) != cad.max_query_len:
            return None
        required = max(cad.max_seq_len,
                       cad.block_table_tensor.shape[1] * self.block_size)
        bucket = max(self.block_size, 1 << (required - 1).bit_length())
        if bucket > self.p.max_model_len:
            return None
        if dynamic:
            return n, bucket, cad.num_reqs, f'dynamic{limit}'
        return n, bucket, qlens

    @staticmethod
    def _graph_static_query_bound(key, qlens):
        if not qlens or min(qlens) < 1 or sum(qlens) != key[0]:
            raise ValueError('Invalid Graph query layout')
        if len(key) == 4:
            if key[3] not in ('dynamic2', 'dynamic4', 'dynamic8') or len(qlens) != key[2]:
                raise ValueError('Invalid dynamic Graph query key')
            bound = {'dynamic2': 2, 'dynamic4': 4, 'dynamic8': 8}[key[3]]
        else:
            bound = max(qlens)
        if max(qlens) > bound:
            raise ValueError('Graph query exceeds captured upper bound')
        return bound

    def _decode_graph_storage(self):
        """Identify every model/cache buffer whose address a capture retains."""
        p = self.p
        layers = p.vllm_config.compilation_config.static_forward_context
        return (p.input_ids.data_ptr(), p._get_positions(1).data_ptr(),
                p._slot_mapping_buffer.data_ptr(),
                *(layers[name].kv_cache[0].data_ptr()
                  for name in p.attn_layer_names))

    @staticmethod
    def _graph_phase_eligible(phase, last_indices):
        # Mixed Prefill joins the same capture only when every request has one
        # query token. _decode_graph_key enforces this shape separately.
        return (last_indices is not None and phase in
                ('decode', 'mixed_prefill_decode', 'mixed_prefill'))

    def _decode_graph_forward(self, cad, n, key, phase='decode',
                              last_indices=None, profile=None):
        """Replay a Draft-only graph with persistent, request-independent buffers.

        The graph stores pointers to these buffers, never request-specific KV
        pages. Before each replay, copy the current row mapping and slot IDs.
        First capture uses the same live input and owned KV slots as its replay.
        """
        p = self.p
        cache = getattr(p, '_cdb_decode_graphs', None)
        storage = self._decode_graph_storage()
        if (cache is None
                or getattr(p, '_cdb_decode_graph_storage', None) != storage):
            # A Worker wake/reset may replace the model buffers or Draft KV.
            # Discard captures before they can reference the old allocation.
            cache = p._cdb_decode_graphs = {}
            p._cdb_decode_graph_storage = storage
        entry = cache.get(key)
        if entry is None:
            bucket = key[1]
            dynamic_layout = len(key) == 4
            starts = cad.query_start_loc_cpu.tolist() if dynamic_layout else None
            qlens = (tuple(b - a for a, b in zip(starts, starts[1:]))
                     if dynamic_layout else
                     key[2] if len(key) == 3 else (1,) * n)
            num_reqs = len(qlens)
            width = (bucket + self.block_size - 1) // self.block_size
            query_cpu = torch.tensor(
                [0] + list(torch.tensor(qlens).cumsum(0).tolist()),
                dtype=torch.int32)
            static_last_indices = torch.tensor(
                [end - 1 for end in query_cpu[1:].tolist()],
                dtype=torch.long, device=p.device)
            static_cad = CommonAttentionMetadata(
                query_start_loc=query_cpu.to(p.device),
                query_start_loc_cpu=query_cpu,
                seq_lens=torch.empty(num_reqs, dtype=torch.int32, device=p.device),
                num_reqs=num_reqs, num_actual_tokens=n,
                max_query_len=self._graph_static_query_bound(key, qlens),
                max_seq_len=bucket,
                block_table_tensor=torch.zeros(
                    (num_reqs, width), dtype=torch.int32, device=p.device),
                slot_mapping=torch.empty(n, dtype=torch.int64, device=p.device),
                causal=True, _seq_lens_cpu=None,
                _num_computed_tokens_cpu=None,
            )
            metadata = self.builder.build_for_drafting(
                static_cad, draft_index=0)
            per_layer = {name: metadata for name in p.attn_layer_names}
            entry = dict(cad=static_cad, per_layer=per_layer,
                         capture_qlens=qlens,
                         last_indices=(None if dynamic_layout
                                       else static_last_indices))
        else:
            static_cad = entry['cad']

        if profile is not None:
            profile.start('metadata')
        dynamic_layout = len(key) == 4
        if dynamic_layout:
            # FlashAttention reads GPU query offsets at replay time. The key
            # fixes request/token counts and the query upper bound; layouts move.
            static_cad.query_start_loc.copy_(cad.query_start_loc)
            static_cad.query_start_loc_cpu.copy_(cad.query_start_loc_cpu)

        width = cad.block_table_tensor.shape[1]
        static_cad.block_table_tensor.zero_()
        static_cad.block_table_tensor[:, :width].copy_(cad.block_table_tensor)
        static_cad.seq_lens.copy_(cad.seq_lens)
        static_cad.slot_mapping.copy_(cad.slot_mapping)
        if profile is not None:
            profile.end('metadata')

        if key not in cache:
            # Capture only after all live metadata and input buffers are valid.
            # The next replay performs the current forward with those values.
            torch.cuda.synchronize(p.device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                with set_forward_context(
                    entry['per_layer'], p.vllm_config, num_tokens=n,
                    cudagraph_runtime_mode=CUDAGraphMode.NONE,
                    slot_mapping=p._get_slot_mapping(
                        n, static_cad.slot_mapping),
                ):
                    ret = p.model(input_ids=p.input_ids[:n],
                                  positions=p._get_positions(n),
                                  inputs_embeds=None)
            entry['graph'] = graph
            entry['hidden'] = ret[0] if p.model_returns_tuple() else ret
            cache[key] = entry
            action = 'capture'
        else:
            action = 'replay'
        _log_draft_graph_runtime_once(
            p, action, phase=phase, request_rows=cad.num_reqs,
            token_rows=n, graph_key=key)
        if profile is not None:
            profile.start('backbone')
        entry['graph'].replay()
        if profile is not None:
            profile.end('backbone')
        if os.getenv('CDB_AV_TRACE_GRAPH', '0') == '1':
            logger.info('[CDB_DRAFT_GRAPH] %s', json.dumps(dict(
                action=action, phase=phase, rows=n, seq_bucket=key[1],
                query_lengths=(tuple(b - a for a, b in zip(
                    cad.query_start_loc_cpu.tolist(),
                    cad.query_start_loc_cpu.tolist()[1:]))
                    if len(key) >= 3 else None),
                capture_query_lengths=(entry['capture_qlens']
                                       if dynamic_layout else None),
                static_max_query_len=static_cad.max_query_len,
                dynamic_layout=dynamic_layout)))
        hidden = entry['hidden']
        if profile is not None:
            profile.start('logits')
        if len(key) >= 3:
            hidden = hidden[last_indices if dynamic_layout
                            else entry['last_indices']]
        logits = p.model.compute_logits(hidden)
        if profile is not None:
            profile.end('logits')
        if os.getenv('CDB_AV_DRAFT_GRAPH_VALIDATE', '0') == '1':
            # Diagnostic only. The eager replay writes the same owned slots;
            # compare them before allowing a later Draft step to observe them.
            slot = static_cad.slot_mapping
            block = torch.div(slot, self.block_size, rounding_mode='floor')
            offset = slot.remainder(self.block_size)
            layers = p.vllm_config.compilation_config.static_forward_context
            graph_kv = [layers[name].kv_cache[0][:, block, offset].clone()
                        for name in p.attn_layer_names]
            with set_forward_context(
                entry['per_layer'], p.vllm_config, num_tokens=n,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                slot_mapping=p._get_slot_mapping(n, static_cad.slot_mapping),
            ):
                eager_ret = p.model(input_ids=p.input_ids[:n],
                                    positions=p._get_positions(n),
                                    inputs_embeds=None)
            eager_hidden = (eager_ret[0] if p.model_returns_tuple()
                            else eager_ret)
            if len(key) >= 3:
                eager_hidden = eager_hidden[last_indices if dynamic_layout
                                            else entry['last_indices']]
            eager_logits = p.model.compute_logits(eager_hidden)
            kv_exact = all(torch.equal(before,
                layers[name].kv_cache[0][:, block, offset])
                for name, before in zip(p.attn_layer_names, graph_kv))
            token_exact = torch.equal(logits.argmax(-1),
                                      eager_logits.argmax(-1))
            logits_close = torch.allclose(logits, eager_logits,
                                          rtol=1e-3, atol=1e-3)
            max_abs = (logits - eager_logits).abs().max().item()
            # Also compare against the original eager path, whose metadata
            # uses the live (unbucketed) sequence length and block width.
            dynamic_metadata = self.builder.build_for_drafting(
                cad, draft_index=0)
            with set_forward_context(
                {name: dynamic_metadata for name in p.attn_layer_names},
                p.vllm_config, num_tokens=n,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                slot_mapping=p._get_slot_mapping(n, cad.slot_mapping),
            ):
                dynamic_ret = p.model(input_ids=p.input_ids[:n],
                                      positions=p._get_positions(n),
                                      inputs_embeds=None)
            dynamic_hidden = (dynamic_ret[0] if p.model_returns_tuple()
                              else dynamic_ret)
            if len(key) >= 3:
                dynamic_hidden = dynamic_hidden[last_indices if dynamic_layout
                                                else entry['last_indices']]
            dynamic_logits = p.model.compute_logits(dynamic_hidden)
            dynamic_token_exact = torch.equal(
                logits.argmax(-1), dynamic_logits.argmax(-1))
            dynamic_kv_exact = all(torch.equal(before,
                layers[name].kv_cache[0][:, block, offset])
                for name, before in zip(p.attn_layer_names, graph_kv))
            dynamic_max_abs = (logits - dynamic_logits).abs().max().item()
            # The diagnostic forwards wrote these slots; return them to the
            # graph result so subsequent Draft steps observe the tested path.
            for name, before in zip(p.attn_layer_names, graph_kv):
                layers[name].kv_cache[0][:, block, offset] = before
            logger.info('[CDB_DRAFT_GRAPH_VALIDATE] %s', json.dumps(dict(
                action=action, phase=phase, rows=n, seq_bucket=key[1],
                query_lengths=(tuple(b - a for a, b in zip(
                    cad.query_start_loc_cpu.tolist(),
                    cad.query_start_loc_cpu.tolist()[1:]))
                    if len(key) >= 3 else None),
                capture_query_lengths=(entry['capture_qlens']
                                       if dynamic_layout else None),
                static_max_query_len=static_cad.max_query_len,
                dynamic_layout=dynamic_layout,
                kv_exact=kv_exact, token_exact=token_exact,
                logits_close=logits_close, max_abs=max_abs,
                dynamic_token_exact=dynamic_token_exact,
                dynamic_kv_exact=dynamic_kv_exact,
                dynamic_max_abs=dynamic_max_abs)))
            if not (kv_exact and token_exact and logits_close):
                raise RuntimeError('Draft Decode Graph/eager validation mismatch')
        return logits

    def _forward(self, ids, positions, cad, last_indices, req_ids, phase):
        p = self.p
        n = ids.numel()
        if n <= 0 or len(req_ids) > self.capacity:
            raise RuntimeError('Invalid refill forward dimensions')
        input_capacity = p.input_ids.shape[0]
        if n > input_capacity:
            raise RuntimeError(
                f'Refill packed input has {n} tokens, exceeding Draft '
                f'input buffer capacity {input_capacity}'
            )
        mode, desc = p.cudagraph_dispatcher.dispatch(n)
        if mode != CUDAGraphMode.NONE or desc.num_tokens != n:
            raise RuntimeError('Refill requires eager, no graph padding')
        p.input_ids[:n] = ids
        p._set_positions(n, positions)
        # Decode and single-query mixed Prefill share the exact-row graph.
        # Small ragged Prefill uses gated static buffers; unsupported rows stay eager.
        graph_key = None
        graph_candidate_key = None
        graph_reason = None
        graph_enabled = os.getenv('CDB_AV_DRAFT_DECODE_GRAPH', '0') == '1'
        if not graph_enabled:
            graph_reason = 'draft_graph_disabled'
        elif not self._graph_phase_eligible(phase, last_indices):
            graph_reason = 'phase_ineligible'
        else:
            graph_key = self._decode_graph_key(cad, n)
            graph_candidate_key = graph_key
            if graph_key is None:
                ragged_enabled = (
                    os.getenv('CDB_AV_DRAFT_RAGGED_GRAPH', '0') == '1')
                if not ragged_enabled:
                    graph_reason = 'ragged_graph_disabled'
                else:
                    graph_key = self._mixed_graph_key(
                        cad, n,
                        dynamic=os.getenv(
                            'CDB_AV_DRAFT_RAGGED_DYNAMIC', '0') == '1',
                        max_query=self.graph_query_bound)
                    graph_candidate_key = graph_key
                    if graph_key is None:
                        graph_reason = 'ragged_shape_ineligible'
                    elif graph_key not in getattr(
                            p, '_cdb_decode_graphs', {}):
                        cache = getattr(p, '_cdb_decode_graphs', {})
                        hits = getattr(p, '_cdb_ragged_graph_hits', None)
                        if hits is None:
                            hits = p._cdb_ragged_graph_hits = Counter()
                        hits[graph_key] += 1
                        limit = int(os.getenv(
                            'CDB_AV_RAGGED_GRAPH_MAX_KEYS', '4'))
                        min_hits = int(os.getenv(
                            'CDB_AV_RAGGED_GRAPH_MIN_HITS', '8'))
                        captured = sum(len(key) >= 3 for key in cache)
                        if captured >= limit:
                            graph_key = None
                            graph_reason = 'ragged_key_limit'
                        elif hits[graph_key] < min_hits:
                            graph_key = None
                            graph_reason = 'ragged_warmup'
        if graph_key is None:
            _log_draft_graph_runtime_once(
                p, 'eager', phase=phase, request_rows=len(req_ids),
                token_rows=n, graph_key=graph_candidate_key,
                reason=graph_reason)
        sample = None
        profiler = self._draft_stage_profiler
        if profiler is not None and profiler.has_capacity:
            if graph_key is None:
                graph_action = 'eager'
            else:
                cache = getattr(p, '_cdb_decode_graphs', {})
                if (graph_key in cache and
                        getattr(p, '_cdb_decode_graph_storage', None)
                        == self._decode_graph_storage()):
                    graph_action = 'replay'
                else:
                    graph_action = 'capture'
            # Capture includes synchronization and is not a steady-state sample.
            if graph_action != 'capture':
                sample = profiler.begin(
                    window=self.window, step=self.steps_completed + 1,
                    phase=phase, request_rows=len(req_ids), token_rows=n,
                    graph_action=graph_action, graph_key=graph_key,
                    metadata_pack_cpu_ms=self._draft_stage_metadata_ms)
        self._draft_stage_metadata_ms = 0.0
        if graph_key is not None:
            logits = self._decode_graph_forward(
                cad, n, graph_key, phase=phase,
                last_indices=last_indices, profile=sample)
        else:
            if sample is not None:
                sample.start('metadata')
            metadata = self.builder.build_for_drafting(cad, draft_index=0)
            per_layer = {name: metadata for name in p.attn_layer_names}
            if sample is not None:
                sample.end('metadata')
                sample.start('backbone')
            with set_forward_context(per_layer, p.vllm_config, num_tokens=n,
                    cudagraph_runtime_mode=CUDAGraphMode.NONE,
                    slot_mapping=p._get_slot_mapping(n, cad.slot_mapping)):
                ret = p.model(input_ids=p.input_ids[:n], positions=p._get_positions(n), inputs_embeds=None)
            if sample is not None:
                sample.end('backbone')
            hidden = ret[0] if p.model_returns_tuple() else ret
            if sample is not None:
                sample.start('logits')
            logits = None if last_indices is None else p.model.compute_logits(hidden[last_indices])
            if sample is not None:
                sample.end('logits')
        if sample is not None:
            if logits is None or phase == 'audit_replay':
                sample.finish()
            else:
                if self._draft_stage_pending is not None:
                    raise RuntimeError('Draft stage profile pending sample overwritten')
                self._draft_stage_pending = sample
        self.forward_count += 1
        self.row_sum += len(req_ids)
        if os.getenv('CDB_AV_VALIDATE_METADATA') == '1':
            if not torch.equal(cad.query_start_loc.cpu(), cad.query_start_loc_cpu):
                raise RuntimeError('Refill CPU/GPU query offsets mismatch')
            if logits is not None and not bool(torch.isfinite(logits).all()):
                raise RuntimeError('Nonfinite refill logits')
        emit('forward', window=self.window, step=self.steps_completed+1,
             phase=phase, req_ids=req_ids, forward_rows=n, logical_active=len(req_ids),
             graph_padding=0)
        return logits

    def _initialize(self):
        p, kw = self.p, self.kwargs
        n, last, base = p.set_inputs_first_pass(kw['target_token_ids'], kw['next_token_ids'],
            kw['target_positions'], kw['last_token_indices'], kw['common_attn_metadata'],
            kw.get('num_rejected_tokens_gpu'))
        # Capture merged inputs before any preparation forward reuses buffers.
        input_ids = p.input_ids[:n].clone()
        positions = p._get_positions(n).clone()
        request_ids = kw['request_ids']
        q = base.query_start_loc_cpu.tolist()
        rows, chunks, sample_indices, offset = [], [], [], 0
        block_rows, seqs, qlens, slots = [], [], [], []
        last_cpu = last.cpu().tolist()
        for i, rid in enumerate(request_ids):
            ctx = self.contexts.get(rid)
            if ctx is None:  # finished by the preceding Target commit
                continue
            start, end = q[i:i+2]
            pos = int(positions[last_cpu[i]])
            if pos != ctx['context_tokens']-1:
                raise RuntimeError(f'Refill committed position mismatch: {rid}')
            cap = kw['request_draft_lengths'][i]
            block_count = (ctx['context_tokens']+cap+self.block_size-1)//self.block_size
            block_ids = base.block_table_tensor[i, :block_count].cpu().tolist()
            self.members[rid] = dict(**ctx, req_id=rid, source='running',
                cap=kw['request_draft_lengths'][i], progress=0, tokens=[],
                blocks=block_ids, next_position=pos, ready=False, ready_at=None)
            rows.append(rid); chunks.extend(range(start, end))
            block_rows.append(block_ids); seqs.append(int(base.seq_lens[i]))
            qlens.append(end-start); slots.extend(base.slot_mapping[start:end].cpu().tolist())
            sample_indices.append(offset+last_cpu[i]-start); offset += end-start
        self.active = rows
        self.initialized = True
        if not rows:
            return None, []
        ix = torch.tensor(chunks, dtype=torch.long, device=p.device)
        cad = self._metadata(block_rows, seqs, qlens, slots)
        logits = self._forward(input_ids[ix], positions[ix], cad,
            torch.tensor(sample_indices, device=p.device), rows, 'initial')
        return logits, rows

    def admit(self, plans):
        if not self.initialized or self.done or self.steps_completed >= self.horizon:
            raise RuntimeError('Refill admission outside an open GPU window')
        if len(self.active)+len(plans) > self.capacity:
            raise RuntimeError('Refill exceeds active capacity')
        for plan in plans:
            rid = plan['req_id']
            prompt = plan['prompt']
            if rid in self.members or not prompt or token_fingerprint(prompt) != plan['fingerprint']:
                raise RuntimeError('Invalid or duplicate refill context')
            if plan['window'] != self.window or plan['cap'] > self.horizon-self.steps_completed:
                raise RuntimeError('Stale window or insufficient H for refill K')
            if len(prompt)+plan['cap'] > self.p.max_model_len:
                raise RuntimeError('Refill exceeds context capacity')
            blocks = plan['blocks']
            if len(set(blocks)) != len(blocks) or any(b<=0 for b in blocks):
                raise RuntimeError('Invalid leased KV blocks')
            used = {b for m in self.members.values() for b in m['blocks'] if b > 0}
            if used.intersection(blocks):
                raise RuntimeError('Cross-request KV lease collision')
            m = dict(req_id=rid, source='waiting', cap=plan['cap'], tokens=[], progress=0,
                blocks=blocks, next_position=len(prompt)-1, next_token=prompt[-1], ready=False,
                context_tokens=len(prompt), fingerprint=plan['fingerprint'], ready_at=None,
                prompt=list(prompt), audit_blocks=plan.get('audit_blocks'))
            # Prefill all except the final prompt token. The final token participates
            # in the NEXT mixed decode forward alongside surviving original rows.
            count = len(prompt)-1
            if count:
                positions = list(range(count))
                slots = [blocks[x//self.block_size]*self.block_size+x%self.block_size for x in positions]
                cad = self._metadata([blocks], [count], [count], slots)
                self._forward(torch.tensor(prompt[:-1], dtype=torch.int32, device=self.p.device),
                    torch.tensor(positions, dtype=torch.int64, device=self.p.device), cad, None, [rid], 'prepare')
            self.members[rid] = m
            self.active.append(rid)
            emit('prepared', window=self.window, after_step=self.steps_completed, req_id=rid,
                context_tokens=len(prompt), fingerprint=m['fingerprint'], blocks=blocks,
                kv_valid_upto=count, next_position=m['next_position'], cap=m['cap'])

    def step(self):
        if self.window is None or self.done:
            raise RuntimeError('Refill step outside configured window')

        if not self.initialized:
            if getattr(self, 'pending_prefill', None):
                raise RuntimeError('Cannot mix pending Prefill before initial Target pass')
            logits, worked = self._initialize()
        else:
            decode_ids = list(self.active)
            pending = list(getattr(self, 'pending_prefill', ()))
            pending_ids = [entry[4] for entry in pending]
            if len(set(decode_ids + pending_ids)) != len(decode_ids) + len(pending_ids):
                raise RuntimeError('Duplicate request in mixed Draft batch')
            worked = decode_ids + pending_ids
            if not worked:
                raise RuntimeError('Refill cannot execute an empty forward')

            ids_parts, pos_parts = [], []
            block_rows, seqs, qlens, slots = [], [], [], []
            last_indices = []
            offset = 0
            slot_owners = {}
            decode_assoc = []

            # Existing requests contribute exactly one Decode input token.
            for rid in decode_ids:
                m = self.members[rid]
                if m['ready']:
                    raise RuntimeError(f'Ready request re-entered Decode: {rid}')
                pos = m['next_position']
                if pos < 0 or pos // self.block_size >= len(m['blocks']):
                    raise RuntimeError(f'Invalid Decode KV position for {rid}: {pos}')
                block = m['blocks'][pos // self.block_size]
                slot = block * self.block_size + pos % self.block_size
                if slot in slot_owners:
                    raise RuntimeError(f'Duplicate mixed-batch KV slot {slot}')
                slot_owners[slot] = rid
                ids_parts.append([m['next_token']])
                pos_parts.append([pos])
                block_rows.append(m['blocks'])
                seqs.append(pos + 1)
                qlens.append(1)
                slots.append(slot)
                last_indices.append(offset)
                offset += 1
                decode_assoc.append(dict(req_id=rid, position=pos,
                    input_token=m['next_token'], slot=slot,
                    progress_before=m['progress']))

            # New requests contribute the uncached Prompt suffix, including its
            # final token so the corresponding hidden state yields first Draft logits.
            for prefix, start, end, blocks, rid, phase in pending:
                if rid not in self.members or self.members[rid]['ready']:
                    raise RuntimeError(f'Invalid pending Prefill member: {rid}')
                if not (0 <= start < end <= len(prefix)):
                    raise RuntimeError(
                        f'Invalid pending Prefill range for {rid}: {start}:{end}')
                if len(blocks) * self.block_size < end:
                    raise RuntimeError(f'Insufficient KV lease for pending Prefill: {rid}')
                row_positions = list(range(start, end))
                row_slots = [
                    blocks[pos // self.block_size] * self.block_size + pos % self.block_size
                    for pos in row_positions
                ]
                for slot in row_slots:
                    if slot in slot_owners:
                        raise RuntimeError(
                            f'Duplicate mixed-batch KV slot {slot}: '
                            f'{slot_owners[slot]} and {rid}')
                    slot_owners[slot] = rid
                ids_parts.append(list(prefix[start:end]))
                pos_parts.append(row_positions)
                block_rows.append(blocks)
                seqs.append(end)
                qlen = end - start
                qlens.append(qlen)
                slots.extend(row_slots)
                last_indices.append(offset + qlen - 1)
                offset += qlen

            ids = torch.tensor(
                [token for part in ids_parts for token in part],
                dtype=torch.int32, device=self.p.device)
            positions = torch.tensor(
                [pos for part in pos_parts for pos in part],
                dtype=torch.int64, device=self.p.device)
            cad = self._metadata(block_rows, seqs, qlens, slots)
            phase = ('mixed_prefill_decode' if pending and decode_ids else
                     'mixed_prefill' if pending else 'decode')
            if len(last_indices) != len(worked):
                raise RuntimeError('Mixed Batch Logits/request alignment mismatch')

            logits = self._forward(
                ids, positions, cad,
                torch.tensor(last_indices, dtype=torch.long, device=self.p.device),
                worked, phase)
            if decode_assoc:
                emit('association', window=self.window,
                     step=self.steps_completed + 1, rows=decode_assoc)
            if pending:
                emit('mixed_batch', window=self.window,
                     step=self.steps_completed + 1,
                     prefill_req_ids=pending_ids, decode_req_ids=decode_ids,
                     query_lengths=qlens, total_tokens=len(ids),
                     last_indices=last_indices)

        self.steps_completed += 1
        completed = []
        sample = self._draft_stage_pending
        update_cpu_start = None
        if logits is not None:
            if logits.ndim != 2 or logits.shape[0] != len(worked):
                raise RuntimeError('Draft logits/request alignment mismatch')

            confidence_values = None
            need_confidence = (
                self.policy == 'hf_confidence'
                or (self.policy in ('replay_acceptance',
                                    'request_avg_acceptance') and any(
                    self.members[rid].get('k_source') not in
                    ('replay', 'request_average')
                    for rid in worked
                ))
            )
            if need_confidence:
                if sample is not None:
                    sample.start('confidence_gpu')
                # The confidence of the greedy token is exp(max(logits) -
                # logsumexp(logits)). Avoid materializing a full log_softmax
                # tensor for every vocabulary row.
                logits_fp32 = logits.float()
                max_logits, tokens = logits_fp32.max(dim=-1)
                log_normalizer = torch.logsumexp(logits_fp32, dim=-1)
                confidence = (max_logits - log_normalizer).exp()
                # Pack the small per-request result before D2H. A single
                # synchronization is cheaper than separately materializing
                # confidence.cpu().tolist() and tokens.tolist().
                packed_gpu = torch.stack(
                    (tokens.to(torch.float64), confidence.to(torch.float64)),
                    dim=1,
                )
                if sample is not None:
                    sample.end('confidence_gpu')
                    d2h_total_start = time.perf_counter()
                    # Split the blocking host wait from the tiny device-to-host
                    # materialization. packed_gpu was produced on this stream.
                    gpu_wait_start = time.perf_counter()
                    torch.cuda.current_stream(
                        device=packed_gpu.device).synchronize()
                    sample.cpu_span(
                        'pre_d2h_gpu_wait_cpu_ms', gpu_wait_start)
                    d2h_copy_start = time.perf_counter()
                packed = packed_gpu.cpu().tolist()
                if sample is not None:
                    sample.cpu_span('d2h_materialize_cpu_ms', d2h_copy_start)
                    sample.cpu_span('d2h_wait_cpu_ms', d2h_total_start)
                token_values = [int(row[0]) for row in packed]
                confidence_values = [float(row[1]) for row in packed]
                stops = confidence_stop_flags(confidence_values, self.threshold)
            else:
                if sample is not None:
                    sample.start('confidence_gpu')
                tokens = logits.argmax(-1)
                if sample is not None:
                    sample.end('confidence_gpu')
                    d2h_total_start = time.perf_counter()
                    gpu_wait_start = time.perf_counter()
                    torch.cuda.current_stream(
                        device=tokens.device).synchronize()
                    sample.cpu_span(
                        'pre_d2h_gpu_wait_cpu_ms', gpu_wait_start)
                    d2h_copy_start = time.perf_counter()
                token_values = tokens.tolist()
                if sample is not None:
                    sample.cpu_span('d2h_materialize_cpu_ms', d2h_copy_start)
                    sample.cpu_span('d2h_wait_cpu_ms', d2h_total_start)
                stops = [False] * len(worked)

            if sample is not None:
                update_cpu_start = time.perf_counter()
            k_caps = {}
            k_actual = {}
            stop_reasons = {}
            for index, (rid, tok, stop) in enumerate(
                zip(worked, token_values, stops)
            ):
                m = self.members[rid]
                if self.policy in ('replay_acceptance', 'request_avg_acceptance'):
                    stop = should_stop(self.policy, m.get('k_source', 'hf_fallback'), stop)
                if m['ready']:
                    raise RuntimeError('Completed request reactivated')
                # Retain the sampled candidate before evaluating the stop rule.
                m['tokens'].append(tok)
                m['progress'] += 1
                m['next_token'] = tok
                m['next_position'] += 1
                k_caps[rid] = m['cap']
                k_actual[rid] = m['progress']

                reason = get_k_stop_reason(m['progress'], m['cap'], stop)
                if reason is not None:
                    m['ready'] = True
                    m['ready_at'] = time.monotonic()
                    completed.append(rid)
                    stop_reasons[rid] = reason
                    emit(
                        'ready',
                        window=self.window,
                        step=self.steps_completed,
                        req_id=rid,
                        ready_at=m['ready_at'],
                        tokens=m['tokens'],
                        fingerprint=m['fingerprint'],
                        source=m['source'],
                        reason=reason,
                        k_source=m.get('k_source', 'configured'),
                        k_cap=m['cap'],
                        k_actual=m['progress'],
                        confidence=(
                            None if confidence_values is None
                            else confidence_values[index]
                        ),
                        threshold=(self.threshold if
                                   (self.policy == 'hf_confidence' or
                                    (self.policy in ('replay_acceptance',
                                                     'request_avg_acceptance') and
                                     m.get('k_source') not in
                                     ('replay', 'request_average')))
                                   else None),
                    )

            if (
                self.policy in ('hf_confidence', 'replay_acceptance',
                                'request_avg_acceptance')
                and os.getenv('CDB_AV_TRACE_STEPS', '0') == '1'
            ):
                logger.info(
                    '[CDB_HF_CONF] %s',
                    json.dumps(
                        dict(
                            window=self.window,
                            step=self.steps_completed,
                            threshold=self.threshold,
                            confidence=dict(zip(worked, confidence_values)),
                            K_cap=k_caps,
                            K_actual=k_actual,
                            stop_reasons=stop_reasons,
                        )
                    ),
                )

        # These rows have now been materialized by the successful mixed forward.
        if hasattr(self, 'pending_prefill'):
            self.pending_prefill = []
        self.active = [r for r in worked if not self.members[r]['ready']]
        if (self.steps_completed == self.horizon and self.active
                and not getattr(self, 'supports_carry', False)
                and os.getenv('CDB_P1_VIRTUAL_H', '0') != '1'):
            raise RuntimeError('H exhausted with unfinished admitted work')
        result = dict(
            pending=True, done=False, window=self.window,
            steps_completed=self.steps_completed,
            active_req_ids=list(self.active),
            newly_completed_req_ids=completed,
            ready_req_ids=[r for r, m in self.members.items() if m['ready']],
            progress={r: m['progress'] for r, m in self.members.items()},
            slots=self.capacity - len(self.active),
            remaining_steps=self.horizon - self.steps_completed)
        if sample is not None:
            if update_cpu_start is not None:
                sample.cpu_span('batch_update_cpu_ms', update_cpu_start)
            sample.finish()
            self._draft_stage_pending = None
        return result

    def resume_virtual_h(self):
        """Continue Draft in the same session; NO Target execution in between.

        The member dictionaries and Draft KV physical pages stay in place.
        This is a GPU continuity test, not cross-Target Carry.
        """
        if os.getenv('CDB_P1_VIRTUAL_H', '0') != '1':
            raise RuntimeError('Virtual H continuation is disabled')
        if not self.initialized or self.done or not self.active:
            raise RuntimeError('No active Draft to carry')
        if self.steps_completed != self.horizon:
            raise RuntimeError('Virtual H boundary not reached')
        snapshot = {rid: (self.members[rid]['progress'],
                          tuple(self.members[rid]['tokens']),
                          self.members[rid]['next_position'],
                          tuple(self.members[rid]['blocks']),
                          self.members[rid]['fingerprint'])
                    for rid in self.active}
        self.steps_completed = 0
        self.virtual_epochs = getattr(self, 'virtual_epochs', 0) + 1
        for rid, original in snapshot.items():
            m = self.members[rid]
            actual = (m['progress'], tuple(m['tokens']), m['next_position'],
                      tuple(m['blocks']), m['fingerprint'])
            if actual != original:
                raise RuntimeError('Carry snapshot changed unexpectedly')
        emit('p1_virtual_resume', window=self.window,
             epoch=self.virtual_epochs, active=list(self.active),
             progress={rid: self.members[rid]['progress'] for rid in self.active},
             context_fingerprints={rid: self.members[rid]['fingerprint'] for rid in self.active})
        return {'resumed': True, 'active_req_ids': list(self.active)}

    def finalize(self):
        if self.active:
            raise RuntimeError('Cannot discard unfinished refill work')
        # Independent teacher-forced full-prefix replay, using separately leased
        # pages, validates tokens/positions without relying on the incremental KV.
        for rid, m in self.members.items():
            if m.get('audit_blocks'):
                prefix = m['prompt'] + m['tokens'][:-1]
                blocks = m['audit_blocks']
                pos = list(range(len(prefix)))
                slots = [blocks[x//self.block_size]*self.block_size+x%self.block_size for x in pos]
                cad = self._metadata([blocks], [len(prefix)], [len(prefix)], slots)
                indices = torch.arange(len(m['prompt'])-1, len(prefix), device=self.p.device)
                logits = self._forward(torch.tensor(prefix,dtype=torch.int32,device=self.p.device),
                    torch.tensor(pos,dtype=torch.int64,device=self.p.device),cad,indices,[rid],'audit_replay')
                replay = logits.argmax(-1).tolist()
                emit('independent_replay',window=self.window,req_id=rid,
                     incremental_tokens=m['tokens'],replay_tokens=replay,exact=replay==m['tokens'],
                     replay_blocks=blocks,incremental_blocks=m['blocks'],
                     fingerprint=token_fingerprint(m['prompt']),positions=pos,
                     teacher_forced_prefix=prefix)
        self.done = True
        now = time.monotonic()
        records = [dict(req_id=r,tokens=m['tokens'],context_tokens=m['context_tokens'],
            fingerprint=m['fingerprint'],window=self.window,source=m['source'],
            ready_at=m['ready_at'],ready_wait=now-m['ready_at']) for r,m in self.members.items()]
        emit('window_end',window=self.window,steps=self.steps_completed,H=self.horizon,
            forwards=self.forward_count,effective_rows=self.row_sum,records=records)
        return records
