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
        occupied = len(self.active) + len(
            getattr(self, 'pending_prefill', ())
        )
        if occupied + len(plans) > self.capacity:
            raise RuntimeError('P2 active capacity exceeded')
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

    def step(self):
        """Advance exactly one Draft-window step, including Ready-only idle ticks."""
        if self.window is None or self.done:
            raise RuntimeError("P2 Draft step outside a configured window")
        if self.steps_completed >= self.horizon:
            raise RuntimeError(
                f"P2 Draft window already reached H={self.horizon}"
            )

        pending_prefill = getattr(self, "pending_prefill", ())
        if not self.active and not pending_prefill:
            unfinished = [
                rid for rid, member in self.members.items()
                if not member["ready"]
            ]
            if unfinished:
                raise RuntimeError(
                    "P2 unfinished members are missing from the Active batch: "
                    f"{unfinished}"
                )

            self.steps_completed += 1
            ready_ids = [
                rid for rid, member in self.members.items()
                if member["ready"]
            ]
            emit(
                "p2_idle_step",
                window=self.window,
                step=self.steps_completed,
                ready=ready_ids,
                reason="ready_barrier",
            )
            return {
                "pending": True,
                "done": False,
                "window": self.window,
                "steps_completed": self.steps_completed,
                "active_req_ids": [],
                "newly_completed_req_ids": [],
                "ready_req_ids": ready_ids,
                "progress": {
                    rid: member["progress"]
                    for rid, member in self.members.items()
                },
                "slots": self.capacity,
                "remaining_steps": self.horizon - self.steps_completed,
            }

        status = super().step()
        if status["steps_completed"] != self.steps_completed:
            raise RuntimeError("P2 Worker step counter diverged")
        return status

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

        ids_set = set(ids)
        for rid in ids_set:
            self.members.pop(rid, None)
            self.suspended.pop(rid, None)
            self.kv_snapshots.pop(rid, None)
            if hasattr(self, "cached"):
                self.cached.pop(rid, None)

        if hasattr(self, "pending_prefill"):
            self.pending_prefill = [
                entry for entry in self.pending_prefill if entry[4] not in ids_set
            ]
        if hasattr(self, "ready_records"):
            self.ready_records = [
                record for record in self.ready_records
                if record.get("req_id") not in ids_set
            ]
        self.active = [rid for rid in self.active if rid not in ids_set]
        return {"dropped": ids}

    def reset(self):
        """Synchronously discard all persistent Draft state."""
        ids = set(self.members)
        ids.update(self.suspended)
        ids.update(self.kv_snapshots)
        ids.update(self.active)
        ids.update(getattr(self, 'cached', {}).keys())
        ids = sorted(ids)

        # Synchronize CUDA before discarding Worker ownership.
        ack = self.drop(ids)
        if ack['dropped'] != ids:
            raise RuntimeError('P2 Worker reset acknowledgement mismatch')

        self.pending_prefill.clear()
        self.ready_records.clear()
        if hasattr(self, 'history'):
            self.history.clear()

        self.steps_completed = 0
        self.done = False

        emit('p2_worker_reset', dropped=ids, window=self.window,
             target_epoch=self.target_epoch)

        return {
            'reset': True,
            'dropped': ids,
            'window': self.window,
            'target_epoch': self.target_epoch,
        }


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
    if operation == 'p2_reset':
        return session.reset()
    raise ValueError(operation)


class PersistentDraftQueue:
    """CPU owner of exclusive pages and official-context version snapshots."""
    def __init__(self, engine):
        self.engine, self.s = engine, engine.scheduler
        # Reuse the existing allocator compatibility checks; not its fresh queue.
        checked = RefillQueue(self.s)
        self.block_size = checked.block_size
        self.leases, self.contexts, self.carry = {}, {}, {}
        self.ready, self.inflight = {}, {}
        self.window = self.target_epoch = 0
        from vllm.v1.spec_decode.cdb_av_config import get_window_h
        self.horizon = get_window_h()
        self.max_k = self.s.num_spec_tokens
        if type(self.max_k) is not int or self.max_k < 1:
            raise ValueError(
                "CDB-AV persistent mode requires num_speculative_tokens >= 1"
            )
        emit(
            "p2_runtime_config",
            H=self.horizon,
            max_k=self.max_k,
            k_policy=os.getenv("CDB_AV_K_POLICY", "manual").strip().lower(),
            hf_threshold=os.getenv("CDB_AV_HF_THRESHOLD", "0.4"),
            kv_reuse=os.getenv("CDB_AV_KV_REUSE", "0"),
            intra_window=os.getenv("CDB_AV_INTRA_WINDOW", "1"),
        )
        self._cdb_admit_limit = None
        self._cdb_admit_token_limit = None
        self._uses_mixed_prefill = (
            os.getenv("CDB_AV_KV_REUSE", "0") == "1"
        )
        self._target_selected_ready = set()
        self._p3_window_open = False
        self._p3_steps = 0
        self._p3_done = True
        self._p3_ready_seen = False
        self.s._cdb_p2 = self

    @property
    def frozen(self):
        """Current-window Draft members and unselected published Ready work.

        self.ready contains only candidates published by a completed H
        boundary. They remain eligible for Target scheduling even while a
        later Draft window is open. Members of the current window are still in
        self.leases and cannot enter self.ready until that window hits H.
        """
        from vllm.v1.spec_decode.cdb_av_config import get_frozen_request_ids

        return get_frozen_request_ids(
            self.leases,
            self.ready,
            getattr(self, "_target_selected_ready", set()),
        )

    @property
    def preempt_protected(self):
        """All P2-owned/in-flight requests whose state must survive preemption."""
        return set(self.leases) | set(self.ready) | set(self.inflight)

    def begin_target_schedule(self):
        """Select published Ready work, including backlog from earlier windows.

        Current-window members are still protected by self.leases and are not
        published to self.ready before H. An open Draft window must therefore
        not block verification of Ready work from older windows.
        """
        if self.inflight:
            raise RuntimeError("P2 Target scheduling began with in-flight verification")

        from vllm.v1.spec_decode.cdb_av_config import (
            select_ready_request_ids,
        )

        running_ids = [req.request_id for req in self.s.running]
        ready_ids = set(self.ready)
        occupied = set(self.leases) | set(self.inflight)
        non_ready_running = set(running_ids) - ready_ids - occupied
        selected = select_ready_request_ids(
            running_ids,
            ready_ids,
            occupied,
            self.s.max_num_running_reqs,
        )
        eligible_ready_ids = {
            req.request_id
            for req in self.s.running
            if req.status == RequestStatus.RUNNING and not req.is_finished()
        }
        selected = [rid for rid in selected if rid in eligible_ready_ids]

        self._target_selected_ready = set(selected)
        emit(
            "target_ready_select",
            window=self.window,
            capacity=self.s.max_num_running_reqs,
            non_ready_running=len(non_ready_running),
            ready_queue=len(ready_ids),
            selected=selected,
            deferred=sorted(ready_ids - set(selected)),
        )

    def can_schedule_verify(self, request, num_new_tokens):
        """Require the complete candidate sequence in one Target forward."""
        record = self.ready.get(request.request_id)
        if record is None:
            return True

        expected = list(record["tokens"])
        if list(request.spec_token_ids) != expected:
            raise RuntimeError(
                "P2 Ready candidate list changed before Target verification"
            )

        scheduled = (
            num_new_tokens
            + request.num_computed_tokens
            - request.num_tokens
            - request.num_output_placeholders
        )
        required = len(expected)
        if scheduled < required:
            emit(
                "target_verify_deferred",
                req_id=request.request_id,
                required_spec_tokens=required,
                schedulable_spec_tokens=max(0, scheduled),
                ready_window=record["window"],
            )
            return False
        return True

    def update_target_selection(self, scheduled_ready_ids):
        """Freeze Ready candidates that the running-request pass did not schedule."""
        scheduled = set(scheduled_ready_ids)
        selected = getattr(self, "_target_selected_ready", set())
        if scheduled - selected:
            raise RuntimeError(
                "P2 Target scheduled Ready requests outside its selected subset"
            )
        self._target_selected_ready = scheduled

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

    @staticmethod
    def unsupported_reason(req):
        """Return why a request must stay on native Target-only generation."""
        params = req.sampling_params
        if params is None:
            return "missing_sampling_params"
        if params.temperature != 0:
            return "non_greedy_sampling"
        if req.has_encoder_inputs:
            return "encoder_inputs"
        if req.lora_request is not None:
            return "lora"
        if req.structured_output_request:
            return "structured_output"
        if params.prompt_logprobs is not None:
            return "prompt_logprobs"
        return None

    def rollback_plans(self, req_ids, reason):
        """Drop Worker state before releasing leases staged for a failed window."""
        ids = [rid for rid in req_ids if rid in self.leases]
        if not ids:
            return
        ack = self.rpc("drop", {"ids": ids})
        if ack.get("dropped") != ids:
            raise RuntimeError("P2 plan rollback acknowledgement mismatch")

        pool = self.s.kv_cache_manager.block_pool
        for rid in ids:
            self.release(rid, reason)
            commits = getattr(self, "commits", None)
            if commits is not None:
                commits.pop(rid, None)
            audit_leases = getattr(self, "audit_leases", None)
            if audit_leases is not None and rid in audit_leases:
                blocks = audit_leases[rid]
                if any(block.ref_cnt != 1 or block.is_null for block in blocks):
                    raise RuntimeError("P2 audit lease ownership lost during rollback")
                pool.free_blocks(blocks)
                audit_leases.pop(rid)

    def reset(self):
        """Reset persistent Draft state with ownership-safe cleanup."""

        pool = self.s.kv_cache_manager.block_pool

        # Phase 1: Preflight every CPU-owned allocation.
        # No Worker or CPU ownership state is modified in this phase.
        owned = []
        seen = {}

        mappings = [
            ("leases", self.leases),
            ("cached_leases", getattr(self, "cached_leases", {})),
            ("audit_leases", getattr(self, "audit_leases", {})),
        ]

        groups = []
        for name, mapping in mappings:
            for rid, blocks in mapping.items():
                groups.append((name, rid, blocks))

        replay_blocks = getattr(self, "replay_blocks", [])
        if replay_blocks:
            groups.append(("replay_blocks", "temporary", replay_blocks))

        for index, blocks in enumerate(
            getattr(self, "pending_allocations", [])
        ):
            groups.append(("pending_allocations", index, blocks))

        for category, owner, blocks in groups:
            for block in blocks:
                bid = block.block_id

                if bid in seen:
                    raise RuntimeError(
                        f"P2 duplicate KV ownership: block={bid}, "
                        f"first={seen[bid]}, second={(category, owner)}"
                    )

                if block.is_null or block.ref_cnt != 1:
                    raise RuntimeError(
                        f"P2 invalid KV ownership before reset: "
                        f"category={category}, owner={owner}, "
                        f"block={bid}, ref_cnt={block.ref_cnt}, "
                        f"is_null={block.is_null}"
                    )

                if (
                    bid < 0
                    or bid >= len(pool.blocks)
                    or pool.blocks[bid] is not block
                ):
                    raise RuntimeError(
                        f"P2 stale KV block reference: "
                        f"category={category}, owner={owner}, block={bid}"
                    )

                seen[bid] = (category, owner)
                owned.append(block)

        # Phase 2: Synchronize and reset Worker state.
        # CPU ownership remains intact until Worker acknowledges.
        ack = self.rpc("reset", {})

        if not ack.get("reset"):
            raise RuntimeError("P2 Worker reset failed")

        # Phase 3: Release all validated blocks exactly once.
        if owned:
            pool.free_blocks(owned)

        # Phase 4: Clear CPU-side ownership records.
        for _, mapping in mappings:
            mapping.clear()

        self.contexts.clear()
        self.carry.clear()
        self.ready.clear()
        self.inflight.clear()

        if hasattr(self, "replay_blocks"):
            self.replay_blocks.clear()

        if hasattr(self, "pending_allocations"):
            self.pending_allocations.clear()

        if hasattr(self, "commits"):
            self.commits.clear()

        if hasattr(self, "history_pages"):
            self.history_pages.clear()

        if hasattr(self, "replayed"):
            self.replayed.clear()

        self._cdb_admit_limit = None
        self._cdb_admit_token_limit = None
        self._p3_window_open = False
        self._p3_steps = 0
        self._p3_done = True
        self._p3_ready_seen = False
        self._target_selected_ready.clear()

        emit(
            "p2_reset",
            window=self.window,
            target_epoch=self.target_epoch,
            released_blocks=len(owned),
            live_leases=len(self.leases),
            ready=len(self.ready),
            inflight=len(self.inflight),
        )

        return {
            "reset": True,
            "released_blocks": len(owned),
        }

    def close(self):
        """Release persistent Worker state and CPU-owned leases at shutdown."""
        if (
            self.leases
            or self.contexts
            or self.carry
            or self.ready
            or self.inflight
            or self._p3_window_open
        ):
            self.reset()

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

        scheduled_ready = set(output.scheduled_spec_decode_tokens)
        # Only candidates published by finish_window() are in self.ready.
        # Current-window members remain in self.leases until H is reached.
        if scheduled_ready - getattr(self, "_target_selected_ready", set()):
            raise RuntimeError("P2 scheduler admitted an unselected Ready request")

        owned = [b.block_id for bs in self.leases.values() for b in bs]
        formal = {
            b
            for req in self.s.running
            for group in self.s.kv_cache_manager.get_block_ids(req.request_id)
            for b in group
        }
        if len(set(owned)) != len(owned) or formal.intersection(owned):
            raise RuntimeError(
                "P2 Draft lease collides with formal Target ownership"
            )
        if any(
            b.ref_cnt != 1
            for bs in self.leases.values()
            for b in bs
        ):
            raise RuntimeError("P2 exclusive lease reference count changed")
        if self.frozen.intersection(output.num_scheduled_tokens):
            raise RuntimeError("P2 scheduled a frozen Draft/Ready request")

        if not output.total_num_scheduled_tokens:
            self._target_selected_ready = set()
            return

        self.target_epoch += 1
        for rid, drafts in output.scheduled_spec_decode_tokens.items():
            record = self.ready.pop(rid, None)
            if record is None or record["tokens"] != drafts:
                raise RuntimeError("P2 missing/duplicate Ready handoff")
            if RefillQueue.context(self.s.requests[rid]) != {
                "context_tokens": record["context_tokens"],
                "fingerprint": record["fingerprint"],
            }:
                raise RuntimeError("P2 stale Ready context")
            self.inflight[rid] = record
            emit(
                "target_verify",
                target_epoch=self.target_epoch,
                window=record["window"],
                req_id=rid,
                tokens=drafts,
                fingerprint=record["fingerprint"],
                ready_wait=time.monotonic() - record["ready_at"],
            )

        if set(self.ready).intersection(output.num_scheduled_tokens):
            raise RuntimeError("P2 scheduled Ready without its complete candidates")

        self._target_selected_ready = scheduled_ready
        emit(
            "p2_target_begin",
            target_epoch=self.target_epoch,
            scheduled=dict(output.num_scheduled_tokens),
            frozen=sorted(self.frozen),
            verify=list(output.scheduled_spec_decode_tokens),
        )

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
        planned_prefill_tokens = 0
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
            reason = self.unsupported_reason(req)
            if reason is not None:
                emit(
                    "p2_skip_unsupported",
                    window=self.window,
                    req_id=rid,
                    reason=reason,
                )
                continue
            params = req.sampling_params
            from vllm.v1.spec_decode.cdb_av_config import get_request_k_cap
            cap = get_request_k_cap(params.extra_args, self.max_k)
            # Near length limits, safely use native Target-only generation.
            if req.num_tokens+cap >= self.s.max_model_len:
                continue
            nblocks = (req.num_tokens+cap+self.block_size-1)//self.block_size
            # Preserve room for Target progress. Allocation failure falls back to
            # native Target-only generation, never a fabricated Draft context.
            if pool.get_num_free_blocks() < nblocks + self.s.max_num_running_reqs*2:
                continue
            ctx = RefillQueue.context(req)
            prefix = list(req.prompt_token_ids) + list(req.output_token_ids)
            prefill_cost = (
                ctx["context_tokens"]
                if self._uses_mixed_prefill
                else max(0, ctx["context_tokens"] - 1)
            )
            from vllm.v1.spec_decode.cdb_av_config import (
                can_admit_prefill_tokens,
            )
            token_limit = self._cdb_admit_token_limit
            if not can_admit_prefill_tokens(
                planned_prefill_tokens,
                prefill_cost,
                token_limit,
                self.s.max_num_scheduled_tokens,
            ):
                emit(
                    "p2_defer_prefill_budget",
                    window=self.window,
                    req_id=rid,
                    prefill_tokens=prefill_cost,
                    planned_tokens=planned_prefill_tokens,
                    token_limit=token_limit,
                    target_token_budget=self.s.max_num_scheduled_tokens,
                )
                continue
            blocks = pool.get_new_blocks(nblocks)
            if len(blocks) != nblocks:
                if blocks and all(
                    block.ref_cnt == 1 and not block.is_null for block in blocks
                ):
                    pool.free_blocks(blocks)
                raise RuntimeError("P2 allocator returned an incomplete lease")
            if any(block.ref_cnt != 1 or block.is_null for block in blocks):
                raise RuntimeError("P2 new lease not exclusive")

            self.leases[rid] = blocks
            self.contexts[rid] = {
                "context": ctx,
                "computed": req.num_computed_tokens,
            }
            plans.append(
                dict(
                    req_id=rid,
                    prefix=prefix,
                    cap=cap,
                    blocks=[block.block_id for block in blocks],
                    **ctx,
                )
            )
            planned_prefill_tokens += prefill_cost
            emit('p2_selected', window=self.window, req_id=rid, cap=cap,
                 blocks=[b.block_id for b in blocks], context=ctx,
                 official_output_tokens=req.num_output_tokens)
        return plans

    def begin_window(self, model_executed):
        """Open a fixed-H Draft window after a Target scheduling opportunity."""
        if getattr(self, "_p3_window_open", False):
            raise RuntimeError("P3 window already open")
        self.reconcile()
        if self.inflight:
            raise RuntimeError("P2 Target did not return every Ready result")
        if not model_executed and not self.leases:
            return False

        if model_executed:
            emit(
                "p2_target_commit",
                target_epoch=self.target_epoch,
                frozen=sorted(self.frozen),
            )

        self.window += 1
        from vllm.v1.spec_decode.cdb_av_config import (
            get_initial_admit_limit,
            is_enabled,
        )
        intra = is_enabled("CDB_AV_INTRA_WINDOW", default=True)
        capacity = self.s.max_num_running_reqs
        first_limit = get_initial_admit_limit(capacity) if intra else capacity

        # Plan only Target-confirmed requests. A new request becomes eligible
        # after Target has prefetched it and produced its first output token.
        self._cdb_admit_limit = max(
            0,
            min(first_limit - len(self.leases), capacity - len(self.leases)),
        )
        self._cdb_admit_token_limit = (
            max(0, self.s.max_num_scheduled_tokens - len(self.carry))
            if self._uses_mixed_prefill
            else None
        )
        previous_leases = set(self.leases)
        try:
            plans = self.plans()
        except BaseException:
            staged_ids = sorted(set(self.leases) - previous_leases)
            try:
                self.rollback_plans(staged_ids, "plan_failed")
            except BaseException as cleanup_error:
                raise RuntimeError(
                    "P2 planning failed and staged lease rollback also failed"
                ) from cleanup_error
            self.window -= 1
            raise
        finally:
            self._cdb_admit_limit = None
            self._cdb_admit_token_limit = None

        # Do not create an empty Worker window. A Ready-only queue can continue
        # to be serviced by Target when there is no Draft Carry or new work.
        if not self.leases and not plans:
            self.window -= 1
            self._target_selected_ready = set()
            return False

        staged_ids = [plan["req_id"] for plan in plans]
        try:
            self.rpc(
                "open",
                dict(window=self.window, H=self.horizon, target_epoch=self.target_epoch),
            )
            if plans:
                ack = self.rpc("prepare", {"plans": plans})
                expected = [plan["req_id"] for plan in plans]
                if ack["prepared"] != expected:
                    raise RuntimeError("P2 preparation acknowledgement mismatch")
                if intra:
                    emit(
                        "p2_intra_admit",
                        window=self.window,
                        after_step=0,
                        req_ids=expected,
                    )
        except BaseException:
            try:
                self.rollback_plans(staged_ids, "window_prepare_failed")
            except BaseException as cleanup_error:
                raise RuntimeError(
                    "P2 Worker preparation failed and staged lease rollback also failed"
                ) from cleanup_error
            raise

        self._p3_window_open = True
        self._p3_steps = 0
        self._p3_done = False
        self._p3_ready_seen = False
        self._target_selected_ready = set()
        return True

    def advance_one_step(self):
        """Execute one Draft step and refill every newly available Active slot."""
        if not getattr(self, "_p3_window_open", False):
            raise RuntimeError("P3 step outside open window")
        if self._p3_steps >= self.horizon:
            return False

        # The Worker returns an idle barrier tick when every current member is
        # Ready. This keeps Ready queued until H without issuing an empty GPU
        # forward, and gives Target a chance to prefill new requests meanwhile.
        status = self.rpc("step")
        self._p3_steps += 1
        if status.get("steps_completed") != self._p3_steps:
            raise RuntimeError("P2 Worker/Scheduler Draft-step counter mismatch")
        self._p3_ready_seen = bool(status.get("ready_req_ids"))

        for rid, progress in status["progress"].items():
            previous = self.carry.get(rid, {}).get("progress", 0)
            if progress < previous:
                raise RuntimeError("P2 progress regressed")

        admitted = []
        from vllm.v1.spec_decode.cdb_av_config import is_enabled
        intra = is_enabled("CDB_AV_INTRA_WINDOW", default=True)

        # Ready members have left Active, but they stay in the Worker/CPU lease
        # set until H. Their freed Active slots may be filled by other requests
        # whose Target first token was produced in an intervening Target step.
        if intra and self._p3_steps < self.horizon:
            slots = max(0, int(status.get("slots", 0)))
            if slots:
                previous_leases = set(self.leases)
                self._cdb_admit_limit = slots
                self._cdb_admit_token_limit = (
                    max(
                        0,
                        self.s.max_num_scheduled_tokens
                        - len(status["active_req_ids"]),
                    )
                    if self._uses_mixed_prefill
                    else None
                )
                try:
                    plans = self.plans()
                    if plans:
                        ack = self.rpc("prepare", {"plans": plans})
                        expected = [p["req_id"] for p in plans]
                        if ack["prepared"] != expected:
                            raise RuntimeError(
                                "P2 intra-window preparation acknowledgement mismatch"
                            )
                        admitted = expected
                        emit(
                            "p2_intra_admit",
                            window=self.window,
                            after_step=self._p3_steps,
                            slots=slots,
                            req_ids=admitted,
                        )
                except BaseException:
                    staged_ids = sorted(set(self.leases) - previous_leases)
                    try:
                        self.rollback_plans(
                            staged_ids, "intra_window_prepare_failed"
                        )
                    except BaseException as cleanup_error:
                        raise RuntimeError(
                            "P2 intra-window preparation failed and staged "
                            "lease rollback also failed"
                        ) from cleanup_error
                    raise
                finally:
                    self._cdb_admit_limit = None
                    self._cdb_admit_token_limit = None

        self._p3_done = False
        emit(
            "p2_draft_step",
            window=self.window,
            step=self._p3_steps,
            active=status["active_req_ids"],
            ready=status["ready_req_ids"],
            admitted=admitted,
            slots=status["slots"],
            remaining_steps=status["remaining_steps"],
        )
        return True

    def finish_window(self):
        """Publish Ready candidates only after exactly H Draft steps."""
        if not getattr(self, "_p3_window_open", False):
            raise RuntimeError("P3 finish outside open window")
        if self._p3_steps != self.horizon:
            raise RuntimeError(
                f"P2 window must complete exactly H={self.horizon} Draft steps "
                f"(completed {self._p3_steps})"
            )

        result = self.rpc("boundary")
        records = result["records"]
        record_ids = [record["req_id"] for record in records]
        ready_ids = set(record_ids)
        carry = result["carry"]

        # Validate the complete Worker boundary before publishing any request
        # or transferring/freeing any CPU-owned KV lease.
        if len(ready_ids) != len(record_ids):
            raise RuntimeError("P2 Worker boundary contains duplicate Ready IDs")
        if ready_ids.intersection(self.ready) or ready_ids - set(self.leases):
            raise RuntimeError("P2 duplicate/unowned Ready")
        if set(carry) != set(self.leases) - ready_ids:
            raise RuntimeError("P2 Worker Carry/Ready partition mismatch")

        for record in records:
            rid = record["req_id"]
            context = self.contexts.get(rid)
            if context is None or context["context"] != {
                "context_tokens": record["context_tokens"],
                "fingerprint": record["fingerprint"],
            }:
                raise RuntimeError("P2 handoff provenance mismatch")
            if record["window"] != self.window:
                raise RuntimeError("P2 Ready record belongs to a stale window")
            tokens = record["tokens"]
            if not 1 <= len(tokens) <= self.max_k or record["ready_at"] is None:
                raise RuntimeError("P2 Worker returned an invalid Ready record")
            if any(
                block.ref_cnt != 1 or block.is_null
                for block in self.leases[rid]
            ):
                raise RuntimeError("P2 Ready lease ownership lost at boundary")
            if rid in getattr(self, "cached_leases", {}):
                raise RuntimeError("P2 Ready request already has a cached lease")

        # Publish the entire Ready batch atomically to the Scheduler. This is
        # one CPU-side handoff; Target still verifies each request in a single
        # later forward, subject to its running/token capacity.
        if records:
            self.s.update_draft_token_ids(
                DraftTokenIds(
                    record_ids,
                    [list(record["tokens"]) for record in records],
                    source_context_tokens=[
                        record["context_tokens"] for record in records
                    ],
                    context_fingerprints=[
                        record["fingerprint"] for record in records
                    ],
                )
            )

        self.carry = carry
        for record in records:
            self.ready[record["req_id"]] = record
        for record in records:
            self.release(record["req_id"], "ready_acknowledged")
            emit("p2_handoff", **record)

        if set(self.carry) != set(self.leases):
            raise RuntimeError("P2 Carry/lease membership mismatch")
        emit(
            "p2_boundary",
            window=self.window,
            ready=list(self.ready),
            carry=list(self.carry),
            leases=len(self.leases),
            leased_pages=sum(len(blocks) for blocks in self.leases.values()),
        )
        self._p3_window_open = False
        self._p3_done = True
        self._p3_ready_seen = False
        self._target_selected_ready = set()

    def advance(self, model_executed):
        """Advance one Draft step per EngineCore iteration.

        Target and Draft never execute GPU forwards concurrently. EngineCore
        first completes its Target forward (if any), then this hook executes
        one Draft step. The next EngineCore iteration may process AR/Prefill
        before the next Draft step. Ready verification is released only at H.
        """
        if not getattr(self, "_p3_window_open", False):
            if not self.begin_window(model_executed):
                return

        if self._p3_steps >= self.horizon:
            self.finish_window()
            return

        advanced = self.advance_one_step()
        if advanced and self._p3_steps == self.horizon:
            self.finish_window()
