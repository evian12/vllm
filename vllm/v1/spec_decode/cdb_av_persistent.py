# SPDX-License-Identifier: Apache-2.0
"""P2: exclusive Draft pages, persistent members and real Target boundaries.

The scheduler owns KVCacheBlock objects. The worker owns only Draft contents;
release is legal only after the worker acknowledges dropping a member. Target
never sees these pages or unverified tokens as part of the official prefix.
"""
import os
import time
import torch

from vllm.v1.spec_decode.cdb_av_refill import RefillSession, emit
from vllm.v1.spec_decode.cdb_av_fingerprint import token_fingerprint
from vllm.v1.spec_decode.cdb_av_refill_queue import RefillQueue
from vllm.v1.outputs import DraftTokenIds
from vllm.v1.request import RequestStatus


def snapshot(m):
    return {k: m[k] for k in ('req_id', 'context_tokens', 'fingerprint', 'cap',
        'progress', 'next_token', 'next_position', 'blocks', 'tokens', 'origin_window')}


class PersistentDraftSession(RefillSession):
    supports_carry = True

    def __init__(self, proposer):
        super().__init__(proposer, {})
        self.initialized = True
        self.capacity = proposer.vllm_config.scheduler_config.max_num_seqs
        self.suspended = {}
        self.target_epoch = 0
        self.kv_snapshots = {}

    def open_window(self, payload, requests):
        if self.window is not None and payload['window'] <= self.window:
            raise RuntimeError('P2 window must advance')
        self.window, self.horizon = payload['window'], payload['H']
        if not 1 <= self.horizon <= 64:
            raise ValueError('P2 H must be in [1,64]')
        self.steps_completed = 0
        for rid, old in self.suspended.items():
            m = self.members[rid]
            state = requests[rid]
            prefix = list(state.prompt_token_ids) + list(state.output_token_ids)
            if snapshot(m) != old or token_fingerprint(prefix) != m['fingerprint']:
                raise RuntimeError('P2 Carry state/context changed across Target')
            emit('carry_resume', window=self.window, target_epoch=payload['target_epoch'],
                 previous_target_epoch=self.target_epoch, **snapshot(m))
        if self.kv_snapshots:
            for rid, saved in self.kv_snapshots.items():
                current = self.kv_bytes(self.members[rid])
                if len(saved) != len(current):
                    raise RuntimeError('P2 Carry Draft KV layer count changed across Target')
                if any(not torch.equal(a, b) for a,b in zip(saved,current)):
                    raise RuntimeError('P2 Carry Draft KV bytes changed across Target')
                emit('carry_kv_unchanged', window=self.window, req_id=rid,
                     target_epoch=payload['target_epoch'], previous_target_epoch=self.target_epoch,
                     checked_bytes=sum(t.numel() for t in saved), layers=len(saved))
        self.kv_snapshots = {}
        self.suspended = {}
        self.target_epoch = payload['target_epoch']
        emit('p2_window_begin', window=self.window, H=self.horizon,
             active=list(self.active), target_epoch=self.target_epoch)
        return {'opened': True}

    def prepare(self, plans, requests):
        if len(self.active) + len(plans) > self.capacity:
            raise RuntimeError('P2 active Draft capacity exceeded')
        for plan in plans:
            rid, prefix = plan['req_id'], plan['prefix']
            state = requests[rid]
            actual = list(state.prompt_token_ids) + list(state.output_token_ids)
            if (rid in self.members or not state.output_token_ids or actual != prefix
                    or token_fingerprint(prefix) != plan['fingerprint']):
                raise RuntimeError('P2 requires a matching Target-confirmed context')
            blocks = plan['blocks']
            used = {b for m in self.members.values() for b in m['blocks']}
            if (used.intersection(blocks) or len(set(blocks)) != len(blocks)
                    or any(b <= 0 for b in blocks)):
                raise RuntimeError('Cross-request KV lease collision')
            if len(blocks)*self.block_size < len(prefix)+plan['cap']:
                raise RuntimeError('P2 insufficient leased KV capacity')
            # A once-per-context prefill into exclusively owned Draft pages.
            # It is never repeated for a Carry resume, even after Target ran.
            count = len(prefix)-1
            if count:
                positions = list(range(count))
                slots = [blocks[x//self.block_size]*self.block_size+x%self.block_size for x in positions]
                cad = self._metadata([blocks], [count], [count], slots)
                self._forward(torch.tensor(prefix[:-1], dtype=torch.int32, device=self.p.device),
                    torch.tensor(positions, dtype=torch.int64, device=self.p.device),
                    cad, None, [rid], 'p2_prepare')
            self.members[rid] = dict(req_id=rid, source='target_first',
                context_tokens=len(prefix), fingerprint=plan['fingerprint'],
                cap=plan['cap'], progress=0, tokens=[], blocks=blocks,
                next_position=count, next_token=prefix[-1], ready=False, ready_at=None,
                origin_window=self.window)
            self.active.append(rid)
            emit('p2_prepared', window=self.window, req_id=rid,
                 context_tokens=len(prefix), fingerprint=plan['fingerprint'],
                 blocks=blocks, kv_valid_upto=count, cap=plan['cap'])
        return {'prepared': [p['req_id'] for p in plans]}

    def kv_bytes(self, m):
        # Validation only: compare exact bytes, including NaN payload bits in
        # unused slots. Production Carry correctness relies on exclusive leases.
        layers = self.p.vllm_config.compilation_config.static_forward_context
        ids = torch.tensor(m['blocks'], device=self.p.device, dtype=torch.long)
        result = []
        for name in self.p.attn_layer_names:
            kv = layers[name].kv_cache[0]
            if kv.ndim != 5 or kv.shape[0] != 2:
                raise ValueError('Carry byte audit currently requires FLASH_ATTN KV layout')
            result.append(kv.index_select(1, ids).contiguous().view(torch.uint8).clone())
        return result

    def boundary(self):
        # Synchronize before mutating live Member ownership.
        torch.cuda.synchronize(self.p.device)

        now = time.monotonic()
        records = []
        ready_ids = set()

        for rid, m in self.members.items():
            if m['ready']:
                ready_ids.add(rid)
                records.append(dict(
                    req_id=rid,
                    tokens=list(m['tokens']),
                    context_tokens=m['context_tokens'],
                    fingerprint=m['fingerprint'],
                    window=self.window,
                    origin_window=m['origin_window'],
                    ready_at=m['ready_at'],
                    ready_wait=now-m['ready_at'],
                ))

        # Stage snapshots before changing the live dictionaries.
        next_members = {
            rid: m for rid, m in self.members.items()
            if rid not in ready_ids
        }

        next_suspended = {
            rid: snapshot(m)
            for rid, m in next_members.items()
        }

        next_kv_snapshots = self.kv_snapshots
        if os.getenv('CDB_AV_CARRY_VALIDATE', '0') == '1':
            next_kv_snapshots = {
                rid: self.kv_bytes(m)
                for rid, m in next_members.items()
            }

        # Commit only after synchronization and snapshot creation succeed.
        self.members = next_members
        self.suspended = next_suspended
        self.kv_snapshots = next_kv_snapshots

        for m in self.suspended.values():
            emit(
                'carry_suspend',
                window=self.window,
                target_epoch=self.target_epoch,
                **m,
            )

        emit(
            'draft_partial',
            window=self.window,
            steps=self.steps_completed,
            ready=[r['req_id'] for r in records],
            carry=list(self.active),
        )

        return {'records': records, 'carry': self.suspended}

    def drop(self, ids):
        # Do not discard ownership records before synchronization succeeds.
        torch.cuda.synchronize(self.p.device)

        for rid in ids:
            self.members.pop(rid, None)
            self.suspended.pop(rid, None)
            self.kv_snapshots.pop(rid, None)

        self.active = [r for r in self.active if r not in ids]
        return {'dropped': ids}

def worker_rpc(runner, operation, payload):
    session = getattr(runner, '_cdb_persistent_session', None)
    if session is None:
        cls = PersistentDraftSession
        if os.getenv('CDB_AV_KV_REUSE', '0') == '1':
            from vllm.v1.spec_decode.cdb_av_kv_reuse import ReuseDraftSession
            cls = ReuseDraftSession
        session = runner._cdb_persistent_session = cls(runner.drafter)
    if operation == 'p2_open':
        return session.open_window(payload, runner.requests)
    if operation == 'p2_prepare':
        return session.prepare(payload['plans'], runner.requests)
    if operation == 'p2_step':
        return session.step()
    if operation == 'p2_boundary':
        return session.boundary()
    if operation == 'p2_replay':
        return session.replay(payload['mapping'])
    if operation == 'p2_drop':
        return session.drop(payload['ids'])
    raise ValueError(operation)


class PersistentDraftQueue:
    """CPU owner of exclusive pages and official-context version snapshots."""
    def __init__(self, engine):
        self.engine, self.s = engine, engine.scheduler
        if os.getenv('CDB_AV_INTRA_WINDOW', '0') == '1':
            if os.getenv('CDB_AV_KV_REUSE', '0') != '1':
                raise RuntimeError(
                    'CDB_AV_INTRA_WINDOW requires CDB_AV_KV_REUSE=1 so new '
                    'Draft Prefill rows join the next mixed Decode forward'
                )
            if os.getenv('CDB_AV_REUSE_VALIDATE', '0') == '1':
                raise RuntimeError(
                    'CDB_AV_INTRA_WINDOW is incompatible with '
                    'CDB_AV_REUSE_VALIDATE=1 because the audit path uses '
                    'a separate Prefill forward'
                )
        # Reuse the existing allocator compatibility checks; not its fresh queue.
        checked = RefillQueue(self.s)
        self.block_size = checked.block_size
        self.leases, self.contexts, self.carry = {}, {}, {}
        self.ready, self.inflight = {}, {}
        self.window = self.target_epoch = 0
        self.horizon = int(os.getenv('CDB_AV_WINDOW_H', '2'))
        if not 1 <= self.horizon <= 64:
            raise ValueError('P2 H must be in [1,64]')
        self._cdb_admit_limit = None
        self.s._cdb_p2 = self

    @property
    def frozen(self):
        return set(self.leases)

    def rpc(self, operation, payload=None):
        return self.engine.model_executor.collective_rpc('cdb_refill_rpc',
            args=('p2_'+operation, payload or {}), single_value=True)

    def release(self, rid, reason):
        # Validate and free before removing the ownership record. If validation
        # or free_blocks raises, retain the record so cleanup/reconciliation can
        # still identify the lease instead of silently losing its block IDs.
        blocks = self.leases[rid]
        if any(b.ref_cnt != 1 or b.is_null for b in blocks):
            raise RuntimeError('P2 lease ownership lost')
        self.s.kv_cache_manager.block_pool.free_blocks(blocks)
        self.leases.pop(rid)
        self.contexts.pop(rid, None)
        self.carry.pop(rid, None)
        emit('p2_lease_release', window=self.window, req_id=rid, reason=reason,
             live_leases=len(self.leases), blocks=[b.block_id for b in blocks])

    def reconcile(self):
        invalid = []
        for rid, ctx in self.contexts.items():
            req = self.s.requests.get(rid)
            if req is None or req.is_finished() or req.status != RequestStatus.RUNNING:
                invalid.append(rid)
            elif RefillQueue.context(req) != ctx['context']:
                # A frozen RUNNING context must never advance silently.
                raise RuntimeError('P2 Target advanced a frozen Carry context')
            elif req.num_computed_tokens != ctx['computed']:
                raise RuntimeError('P2 Target changed Carry computed length')
        if invalid:
            ack = self.rpc('drop', {'ids': invalid})
            if ack['dropped'] != invalid:
                raise RuntimeError('P2 invalidation acknowledgement mismatch')
            for rid in invalid:
                self.release(rid, 'invalid')
                emit('p2_invalidated', window=self.window, req_id=rid)
        for rid in list(self.inflight):
            req = self.s.requests.get(rid)
            if req is None or req.is_finished():
                self.inflight.pop(rid)
                emit('p2_inflight_cancelled', window=self.window, req_id=rid)
        for rid in list(self.ready):
            req = self.s.requests.get(rid)
            if req is None or req.is_finished() or req.status != RequestStatus.RUNNING:
                self.ready.pop(rid)
                emit('p2_ready_invalidated', window=self.window, req_id=rid)

    def before_target(self, output):
        self.reconcile()
        owned = [b.block_id for bs in self.leases.values() for b in bs]
        formal = {b for req in self.s.running
                  for group in self.s.kv_cache_manager.get_block_ids(req.request_id) for b in group}
        if len(set(owned)) != len(owned) or formal.intersection(owned):
            raise RuntimeError('P2 Draft lease collides with formal Target ownership')
        if any(b.ref_cnt != 1 for bs in self.leases.values() for b in bs):
            raise RuntimeError('P2 exclusive lease reference count changed')
        if self.frozen.intersection(output.num_scheduled_tokens):
            raise RuntimeError('P2 scheduled frozen Carry in Target')
        if not output.total_num_scheduled_tokens:
            return
        self.target_epoch += 1
        for rid, drafts in output.scheduled_spec_decode_tokens.items():
            record = self.ready.pop(rid, None)
            if record is None or record['tokens'] != drafts:
                raise RuntimeError('P2 missing/duplicate Ready handoff')
            if RefillQueue.context(self.s.requests[rid]) != {
                    'context_tokens': record['context_tokens'], 'fingerprint': record['fingerprint']}:
                raise RuntimeError('P2 stale Ready context')
            self.inflight[rid] = record
            emit('target_verify', target_epoch=self.target_epoch, window=self.window,
                 req_id=rid, tokens=drafts, fingerprint=record['fingerprint'],
                 ready_wait=time.monotonic()-record['ready_at'])
        if set(self.ready).intersection(output.num_scheduled_tokens):
            raise RuntimeError('P2 scheduled Ready without its complete candidates')
        emit('p2_target_begin', target_epoch=self.target_epoch,
             scheduled=dict(output.num_scheduled_tokens), frozen=sorted(self.frozen),
             verify=list(output.scheduled_spec_decode_tokens))

    def committed(self, req, drafts, generated):
        record = self.inflight.pop(req.request_id, None)
        if record is None:
            if drafts:
                raise RuntimeError('P2 unexpected verification result')
            return
        accepted = len(generated)-1
        if drafts != record['tokens'] or not 0 <= accepted <= len(drafts) or generated[:accepted] != drafts[:accepted]:
            raise RuntimeError('P2 verification accepted-prefix mismatch')
        emit('p2_verified', target_epoch=self.target_epoch, req_id=req.request_id,
             window=record['window'], drafts=drafts, generated=generated,
             accepted=accepted, fingerprint=record['fingerprint'])

    def plans(self):
        plans = []
        pool = self.s.kv_cache_manager.block_pool
        for req in self.s.running:
            if self._cdb_admit_limit is not None and len(plans) >= self._cdb_admit_limit:
                break
            rid = req.request_id
            if (rid in self.leases or rid in self.ready or rid in self.inflight
                    or req.status != RequestStatus.RUNNING or req.is_finished()
                    or not req.num_output_tokens or req.spec_token_ids
                    or req.num_computed_tokens != req.num_tokens-1):
                continue
            params = req.sampling_params
            if (params is None or params.temperature != 0 or req.has_encoder_inputs
                    or req.lora_request is not None or req.structured_output_request
                    or params.prompt_logprobs is not None):
                raise ValueError('P2 supports text-only greedy requests without grammar/LoRA')
            cap = (params.extra_args or {}).get('cdb_draft_k', self.s.num_spec_tokens)
            if type(cap) is not int or not 1 <= cap <= self.s.num_spec_tokens:
                raise ValueError('Invalid P2 request K')
            if cap > self.horizon - self._p3_steps:
                continue  # Do not admit work that cannot finish before this boundary.
            # Near length limits, safely use native Target-only generation.
            if req.num_tokens+cap >= self.s.max_model_len:
                continue
            nblocks = (req.num_tokens+cap+self.block_size-1)//self.block_size
            # Preserve room for Target progress. Allocation failure falls back to
            # native Target-only generation, never a fabricated Draft context.
            if pool.get_num_free_blocks() < nblocks + self.s.max_num_running_reqs*2:
                continue
            blocks = pool.get_new_blocks(nblocks)
            if any(b.ref_cnt != 1 or b.is_null for b in blocks):
                raise RuntimeError('P2 new lease not exclusive')
            self.leases[rid] = blocks
            ctx = RefillQueue.context(req)
            self.contexts[rid] = {'context': ctx, 'computed': req.num_computed_tokens}
            plans.append(dict(req_id=rid, prefix=list(req.prompt_token_ids)+list(req.output_token_ids),
                cap=cap, blocks=[b.block_id for b in blocks], **ctx))
            emit('p2_selected', window=self.window, req_id=rid, cap=cap,
                 blocks=[b.block_id for b in blocks], context=ctx,
                 official_output_tokens=req.num_output_tokens)
        return plans

    def begin_window(self, model_executed):
        """Open a P2 window; no Draft Forward is performed here."""
        if getattr(self, '_p3_window_open', False):
            raise RuntimeError('P3 window already open')
        self.reconcile()
        if self.inflight:
            raise RuntimeError('P2 Target did not return every Ready result')
        if not model_executed and not self.leases:
            return False
        if model_executed:
            emit('p2_target_commit', target_epoch=self.target_epoch,
                 frozen=sorted(self.frozen))
        self.window += 1
        self._p3_steps = 0
        self.rpc('open', dict(window=self.window, H=self.horizon, target_epoch=self.target_epoch))
        intra = os.getenv('CDB_AV_INTRA_WINDOW', '0') == '1'
        self._cdb_intra_window = intra
        capacity = self.s.max_num_running_reqs
        first_limit = int(os.getenv('CDB_AV_INITIAL_ADMIT', '3'))
        if intra and not 1 <= first_limit <= capacity:
            raise ValueError('CDB_AV_INITIAL_ADMIT must be in [1, max_num_running_reqs]')
        # Do not overwrite the caller's setting after a window; plans() reads it.
        self._cdb_admit_limit = (max(0, min(first_limit - len(self.leases),
                                               capacity - len(self.leases)))
                                 if intra else None)
        try:
            plans = self.plans()
        finally:
            self._cdb_admit_limit = None
        if plans:
            ack = self.rpc('prepare', {'plans': plans})
            if ack['prepared'] != [p['req_id'] for p in plans]:
                raise RuntimeError('P2 preparation acknowledgement mismatch')
            if intra:
                emit('p2_intra_admit', window=self.window, after_step=0,
                     req_ids=[p['req_id'] for p in plans])
        self._p3_window_open = True
        self._p3_steps = 0
        self._p3_done = not bool(self.leases)
        return True

    def advance_one_step(self):
        """Run at most one Draft Forward, preserving existing Carry checks."""
        if not getattr(self, '_p3_window_open', False):
            raise RuntimeError('P3 step outside open window')
        if self._p3_done or self._p3_steps >= self.horizon:
            return False
        if not self.leases:
            self._p3_done = True
            return False
        status = self.rpc('step')
        self._p3_steps += 1
        for rid, progress in status['progress'].items():
            previous = self.carry.get(rid, {}).get('progress', 0)
            if progress < previous:
                raise RuntimeError('P2 progress regressed')
        self._p3_done = not bool(status['active_req_ids'])

        # Ready rows remain materialized until the Target boundary, but they no
        # longer consume an Active Draft slot. Refill those slots with requests
        # whose official Target context is already complete. In ReuseDraftSession,
        # prepare() stages their Prefill rows for the NEXT mixed Draft forward.
        if self._cdb_intra_window and self._p3_steps < self.horizon:
            free_slots = max(
                0, self.s.max_num_running_reqs - len(status['active_req_ids'])
            )
            if free_slots:
                self._cdb_admit_limit = free_slots
                try:
                    plans = self.plans()
                finally:
                    self._cdb_admit_limit = None
                if plans:
                    ack = self.rpc('prepare', {'plans': plans})
                    expected = [p['req_id'] for p in plans]
                    if ack['prepared'] != expected:
                        raise RuntimeError('P2 intra-window admission acknowledgement mismatch')
                    self._p3_done = False
                    emit(
                        'p2_intra_admit',
                        window=self.window,
                        after_step=self._p3_steps,
                        req_ids=expected,
                        active_before=len(status['active_req_ids']),
                        refill_slots=free_slots,
                    )
        return True

    def finish_window(self):
        """Commit the original P2 boundary and ready handoffs."""
        if not getattr(self, '_p3_window_open', False):
            raise RuntimeError('P3 finish outside open window')
        if not self._p3_done and self._p3_steps < self.horizon:
            raise RuntimeError('P3 boundary before draft window is finished')
        result = self.rpc('boundary')
        self.carry = result['carry']
        for record in result['records']:
            rid = record['req_id']
            if rid in self.ready or rid not in self.leases:
                raise RuntimeError('P2 duplicate/unowned Ready')
            if self.contexts[rid]['context'] != {
                    'context_tokens':record['context_tokens'], 'fingerprint':record['fingerprint']}:
                raise RuntimeError('P2 handoff provenance mismatch')
            self.ready[rid] = record
            self.release(rid, 'ready_acknowledged')
            self.s.update_draft_token_ids(DraftTokenIds([rid], [record['tokens']],
                source_context_tokens=[record['context_tokens']],
                context_fingerprints=[record['fingerprint']]))
            emit('p2_handoff', **record)
        if set(self.carry) != self.frozen:
            raise RuntimeError('P2 Carry/lease membership mismatch')
        emit('p2_boundary', window=self.window, ready=list(self.ready),
             carry=list(self.carry), leases=len(self.leases),
             leased_pages=sum(len(b) for b in self.leases.values()))
        self._p3_window_open = False

    def advance(self, model_executed):
        """Compatibility wrapper: still synchronous until EngineCore is changed."""
        if not self.begin_window(model_executed):
            return
        while self.advance_one_step():
            pass
        self.finish_window()
