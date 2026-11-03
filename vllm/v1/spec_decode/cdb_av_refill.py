# SPDX-License-Identifier: Apache-2.0
"""Synchronous, bounded GPU draft windows with dynamically admitted members.

Target input batches remain untouched. Waiting work uses allocator-owned leases;
its prompt is prefetched by the independent draft model, never by Target here.
"""
import json
import os
import time
import torch
from vllm.config import CUDAGraphMode
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.spec_decode.cdb_av_fingerprint import token_fingerprint
from vllm.v1.spec_decode.cdb_av_config import (
    confidence_stop_flags,
    get_k_stop_reason,
)

logger = init_logger(__name__)


def emit(event, **data):
    if os.getenv('CDB_AV_REFILL_TRACE', '1') == '1':
        logger.info('[CDB_REFILL] %s', json.dumps(dict(event=event, **data)))


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
        self.builder = proposer.attn_metadata_builder or proposer._get_attention_metadata_builder()
        self.block_size = self.builder.kv_cache_spec.block_size
        self.policy = os.getenv('CDB_AV_K_POLICY', 'manual').strip().lower()
        self.threshold = float(os.getenv('CDB_AV_HF_THRESHOLD', '0.4'))
        if self.policy not in ('manual', 'hf_confidence') or not 0 <= self.threshold <= 1:
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
        dev = self.p.device
        width = max(len(b) for b in blocks)
        bt = torch.zeros((len(blocks), width), dtype=torch.int32, device=dev)
        for i, row in enumerate(blocks):
            bt[i, :len(row)] = torch.tensor(row, dtype=torch.int32, device=dev)
        qcpu = torch.tensor([0] + list(torch.tensor(qlens).cumsum(0).tolist()), dtype=torch.int32)
        return CommonAttentionMetadata(query_start_loc=qcpu.to(dev), query_start_loc_cpu=qcpu,
            seq_lens=torch.tensor(seqs, dtype=torch.int32, device=dev),
            num_reqs=len(blocks), num_actual_tokens=sum(qlens), max_query_len=max(qlens),
            max_seq_len=max(seqs), block_table_tensor=bt,
            slot_mapping=torch.tensor(slots, dtype=torch.int64, device=dev), causal=True,
            _seq_lens_cpu=None, _num_computed_tokens_cpu=None)

    def _forward(self, ids, positions, cad, last_indices, req_ids, phase):
        p = self.p
        n = ids.numel()
        if n <= 0 or len(req_ids) > self.capacity:
            raise RuntimeError('Invalid refill forward dimensions')
        mode, desc = p.cudagraph_dispatcher.dispatch(n)
        if mode != CUDAGraphMode.NONE or desc.num_tokens != n:
            raise RuntimeError('Refill requires eager, no graph padding')
        p.input_ids[:n] = ids
        p._set_positions(n, positions)
        metadata = self.builder.build_for_drafting(cad, draft_index=0)
        per_layer = {name: metadata for name in p.attn_layer_names}
        with set_forward_context(per_layer, p.vllm_config, num_tokens=n,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                slot_mapping=p._get_slot_mapping(n, cad.slot_mapping)):
            ret = p.model(input_ids=p.input_ids[:n], positions=p._get_positions(n), inputs_embeds=None)
        hidden = ret[0] if p.model_returns_tuple() else ret
        logits = None if last_indices is None else p.model.compute_logits(hidden[last_indices])
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
        if logits is not None:
            if logits.ndim != 2 or logits.shape[0] != len(worked):
                raise RuntimeError('Draft logits/request alignment mismatch')

            tokens = logits.argmax(-1)
            token_values = tokens.tolist()
            confidence_values = None
            if self.policy == 'hf_confidence':
                confidence = logits.float().log_softmax(-1).gather(
                    1, tokens[:, None]
                ).exp().squeeze(1)
                # One batched D2H transfer for all confidences; avoid per-request .item() sync.
                confidence_values = confidence.detach().cpu().tolist()
                stops = confidence_stop_flags(confidence_values, self.threshold)
            else:
                stops = [False] * len(worked)

            k_caps = {}
            k_actual = {}
            stop_reasons = {}
            for index, (rid, tok, stop) in enumerate(
                zip(worked, token_values, stops)
            ):
                m = self.members[rid]
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
                        k_cap=m['cap'],
                        k_actual=m['progress'],
                        confidence=(
                            None if confidence_values is None
                            else confidence_values[index]
                        ),
                        threshold=(
                            self.threshold if self.policy == 'hf_confidence'
                            else None
                        ),
                    )

            if (
                self.policy == 'hf_confidence'
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
        return dict(
            pending=True, done=False, window=self.window,
            steps_completed=self.steps_completed,
            active_req_ids=list(self.active),
            newly_completed_req_ids=completed,
            ready_req_ids=[r for r, m in self.members.items() if m['ready']],
            progress={r: m['progress'] for r, m in self.members.items()},
            slots=self.capacity - len(self.active),
            remaining_steps=self.horizon - self.steps_completed)

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
