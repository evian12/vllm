# SPDX-License-Identifier: Apache-2.0
"""Opt-in accepted-prefix reuse on exclusive P2 Draft leases.

A cache token describes an INPUT actually forwarded, not a sampled candidate.
Ready caches remain owned but are not scheduler-frozen. A commit certificate is
required before they can become active again. Rejection invalidates actual slots.
"""
import os
from collections import Counter
import torch
from vllm.v1.request import RequestStatus
from vllm.v1.spec_decode.cdb_av_persistent import PersistentDraftQueue, PersistentDraftSession
from vllm.v1.spec_decode.cdb_av_refill_queue import RefillQueue
from vllm.v1.spec_decode.cdb_av_refill import emit
from vllm.v1.spec_decode.cdb_av_fingerprint import token_fingerprint
from vllm.v1.spec_decode.cdb_av_kv_metrics import kv_segment_errors


def reuse_span(cached_inputs, prefix, old_prefix, drafts, certificate):
    """Return safe input prefix length; invalid provenance means full fallback."""
    if (certificate is None or certificate['source_fp'] != token_fingerprint(old_prefix)
            or certificate['drafts'] != drafts):
        return 0, 'provenance'
    generated = certificate['generated']
    accepted = len(generated)-1
    if (not 0 <= accepted <= len(drafts) or generated[:accepted] != drafts[:accepted]
            or prefix != old_prefix+generated):
        return 0, 'commit_prefix'
    # The final predicted candidate has no KV unless separately forwarded.
    if cached_inputs != old_prefix+drafts[:-1]:
        return 0, 'cached_input_lineage'
    common = 0
    for a,b in zip(cached_inputs,prefix[:-1]):
        if a != b:break
        common += 1
    bound = len(old_prefix)+accepted
    return min(common,bound), None


class ReuseDraftSession(PersistentDraftSession):
    def __init__(self, proposer):
        super().__init__(proposer)
        self.cached = {}
        self.counters = Counter()
        self.history = []

    def _forward(self, ids, positions, cad, last_indices, req_ids, phase):
        result = super()._forward(ids, positions, cad, last_indices, req_ids, phase)
        if os.getenv('CDB_AV_REUSE_REPLAY', '0') == '1' and phase != 'reuse_reference_full':
            q = cad.query_start_loc_cpu.tolist()
            self.history.append(dict(ids=ids.cpu().tolist(), positions=positions.cpu().tolist(),
                blocks=cad.block_table_tensor.cpu().tolist(), seqs=cad.seq_lens.cpu().tolist(),
                qlens=[b-a for a,b in zip(q,q[1:])], slots=cad.slot_mapping.cpu().tolist(),
                req_ids=list(req_ids), phase=phase))
        return result

    def replay(self, mapping):
        mapping = {int(k):v for k,v in mapping.items()}
        for op in self.history:
            if 'invalidate' in op:
                blocks, start, end = op['invalidate']
                self._clear_slots([mapping[b] for b in blocks], start, end)
                continue
            blocks = [[mapping[b] if b else 0 for b in bs] for bs in op['blocks']]
            slots = [mapping[x//self.block_size]*self.block_size+x%self.block_size for x in op['slots']]
            cad = self._metadata(blocks,op['seqs'],op['qlens'],slots)
            # Teacher-forced recomputation from empty, separate pages. No KV copy.
            super()._forward(torch.tensor(op['ids'],dtype=torch.int32,device=self.p.device),
                torch.tensor(op['positions'],dtype=torch.int64,device=self.p.device),cad,
                None,op['req_ids'],'reuse_shape_replay')
        checks=[]
        for rid,m in self.members.items():
            count=len(m['kv_inputs'])
            actual=self.values(m['blocks'],count)
            reference=self.values([mapping[b] for b in m['blocks']],count)
            layers=[dict(layer=name,exact=bool(torch.equal(a.view(torch.uint8),b.view(torch.uint8))),
                         different_bytes=int((a.view(torch.uint8)!=b.view(torch.uint8)).sum()),bytes=a.numel()*a.element_size())
                    for (name,_),a,b in zip(self.layers(),actual,reference)]
            checks.append(dict(req_id=rid,fingerprint=m['fingerprint'],valid_tokens=count,layers=layers))
        emit('reuse_shape_replay',window=self.window,operations=len(self.history),checks=checks,
             exact=all(x['exact'] for c in checks for x in c['layers']))
        if not all(x['exact'] for c in checks for x in c['layers']):
            raise RuntimeError('Independent same-shape full-history KV replay mismatch')
        return {'exact':True,'checks':checks}

    def layers(self):
        context = self.p.vllm_config.compilation_config.static_forward_context
        result = []
        for name in self.p.attn_layer_names:
            kv = context[name].kv_cache[0]
            if kv.ndim != 5 or kv.shape[0] != 2:
                raise ValueError('KV reuse currently requires FLASH_ATTN (2,blocks,slots,heads,dim)')
            result.append((name,kv))
        return result

    def invalidate(self, blocks, start, end):
        if start != end and os.getenv('CDB_AV_REUSE_REPLAY','0') == '1':
            self.history.append({'invalidate':(list(blocks),start,end)})
        self._clear_slots(blocks,start,end)
        self.counters['invalidated_suffix_tokens'] += end-start

    def _clear_slots(self, blocks, start, end):
        if start == end:return
        pos = list(range(start,end))
        b = torch.tensor([blocks[x//self.block_size] for x in pos],device=self.p.device)
        off = torch.tensor([x%self.block_size for x in pos],device=self.p.device)
        for _,kv in self.layers():
            kv[:,b,off] = 0  # assignment into the real tensor, not a sliced copy

    def values(self, blocks, count):
        b = torch.tensor([blocks[x//self.block_size] for x in range(count)],device=self.p.device)
        off = torch.tensor([x%self.block_size for x in range(count)],device=self.p.device)
        return [kv[:,b,off].contiguous() for _,kv in self.layers()]

    def fill(self, prefix, start, end, blocks, rid, phase):
        if start == end:return
        pos = list(range(start,end))
        slots = [blocks[x//self.block_size]*self.block_size+x%self.block_size for x in pos]
        cad = self._metadata([blocks],[end],[end-start],slots)
        self._forward(torch.tensor(prefix[start:end],dtype=torch.int32,device=self.p.device),
            torch.tensor(pos,dtype=torch.int64,device=self.p.device),cad,None,[rid],phase)

    def fill_batch(self, entries):
        """Prefill multiple independent requests in one Draft forward.

        Each entry contains:
            prefix, start, end, blocks, req_id, phase

        Positions and KV slots remain request-local, while token inputs
        are concatenated into one packed forward.
        """
        if not entries:
            return

        # Validate the whole packed batch before touching GPU state.
        seen_req_ids = set()
        block_owners = {}

        for prefix, start, end, blocks, rid, phase in entries:
            if rid in seen_req_ids:
                raise RuntimeError(f"Duplicate request ID: {rid}")
            seen_req_ids.add(rid)

            if len(set(blocks)) != len(blocks) or any(b <= 0 for b in blocks):
                raise RuntimeError(f"Invalid KV block lease for {rid}")

            for block in blocks:
                owner = block_owners.get(block)
                if owner is not None and owner != rid:
                    raise RuntimeError(
                        f"Block collision: block={block}, "
                        f"requests={owner},{rid}"
                    )
                block_owners[block] = rid

        ids_parts = []
        pos_parts = []
        slots = []
        block_rows = []
        seqs = []
        qlens = []
        req_ids = []

        for prefix, start, end, blocks, rid, phase in entries:
            if not 0 <= start < end <= len(prefix):
                raise ValueError(
                    f"Invalid Prefill range for {rid}: "
                    f"start={start}, end={end}, prefix={len(prefix)}"
                )

            if len(blocks) * self.block_size < end:
                raise RuntimeError(f"Insufficient KV blocks for {rid}")

            positions = list(range(start, end))
            ids_parts.extend(prefix[start:end])
            pos_parts.extend(positions)
            slots.extend(
                blocks[pos // self.block_size] * self.block_size
                + pos % self.block_size
                for pos in positions
            )

            block_rows.append(blocks)
            seqs.append(end)
            qlens.append(end - start)
            req_ids.append(rid)

        ids = torch.tensor(
            ids_parts, dtype=torch.int32, device=self.p.device
        )
        positions = torch.tensor(
            pos_parts, dtype=torch.int64, device=self.p.device
        )

        cad = self._metadata(block_rows, seqs, qlens, slots)

        self._forward(
            ids, positions, cad, None, req_ids, "reuse_batch_prefill"
        )

        emit(
            "reuse_batch_prefill",
            window=self.window,
            req_ids=req_ids,
            forward_rows=len(ids_parts),
            query_lengths=qlens,
            seq_lens=seqs,
            query_start_loc=cad.query_start_loc_cpu.tolist(),
        )

    def prepare(self, plans, requests):
        """Prepare all members with one packed Prefill forward."""
        if len(self.members) + len(plans) > self.capacity:
            raise RuntimeError("P2 member capacity exceeded")

        staged = []
        entries = []

        # Phase 1: validate leases, determine reusable prefixes, and
        # invalidate stale suffixes. Do not register new members yet.
        for plan in plans:
            rid = plan["req_id"]
            prefix = plan["prefix"]
            blocks = plan["blocks"]
            state = requests[rid]

            actual = (
                list(state.prompt_token_ids)
                + list(state.output_token_ids)
            )
            if (
                rid in self.members
                or not state.output_token_ids
                or actual != prefix
                or token_fingerprint(prefix) != plan["fingerprint"]
            ):
                raise RuntimeError(
                    "P2 requires a matching Target-confirmed context"
                )

            old = self.cached.pop(rid, None)
            used = {
                b
                for m in [*self.members.values(), *self.cached.values()]
                for b in m["blocks"]
            }
            if (
                used.intersection(blocks)
                or len(set(blocks)) != len(blocks)
                or any(b <= 0 for b in blocks)
            ):
                raise RuntimeError("Cross-request KV lease collision")

            if len(blocks) * self.block_size < len(prefix) + plan["cap"]:
                raise RuntimeError("P2 insufficient leased KV capacity")

            start = 0
            reason = None
            old_valid = 0

            if old is not None:
                if blocks[:len(old["blocks"])] != old["blocks"]:
                    raise RuntimeError(
                        "KV lease changed before reuse acknowledgement"
                    )
                old_valid = len(old["kv_inputs"])
                start, reason = reuse_span(
                    old["kv_inputs"],
                    prefix,
                    old["prefix"],
                    old["tokens"],
                    plan.get("commit"),
                )
            elif plan.get("reuse"):
                reason = "missing_worker_cache"

            end = len(prefix) - 1
            preserved = (
                self.values(blocks, start)
                if plan.get("audit_blocks") and start
                else []
            )

            # Only invalidate the suffix that is no longer trustworthy.
            self.invalidate(blocks, start, old_valid)

            phase = "reuse_incremental" if start else "reuse_full"
            if start < end:
                entries.append(
                    (prefix, start, end, blocks, rid, phase)
                )

            staged.append(
                dict(
                    plan=plan,
                    rid=rid,
                    prefix=prefix,
                    blocks=blocks,
                    start=start,
                    end=end,
                    old_valid=old_valid,
                    reason=reason,
                    preserved=preserved,
                    phase=phase,
                )
            )

        # Phase 2: all requests share a single Draft forward.
        # If it raises, do not register any new members.
        for entry in entries:
            self.fill(*entry)

        # Phase 3: publish member state only after successful Prefill.
        for item in staged:
            plan = item["plan"]
            rid = item["rid"]
            prefix = item["prefix"]
            blocks = item["blocks"]
            start = item["start"]
            end = item["end"]
            old_valid = item["old_valid"]
            reason = item["reason"]
            preserved = item["preserved"]

            if reason:
                self.counters["fallback_count"] += 1
                emit(
                    "reuse_fallback",
                    window=self.window,
                    req_id=rid,
                    reason=reason,
                )

            self.counters["reused_kv_tokens"] += start
            self.counters[
                "incremental_prefill_tokens" if start
                else "full_prefill_tokens"
            ] += end - start
            self.counters[
                "reuse_count" if start else "full_prefill_count"
            ] += 1

            m = dict(
                req_id=rid,
                source="target_first",
                context_tokens=len(prefix),
                fingerprint=plan["fingerprint"],
                cap=plan["cap"],
                progress=0,
                tokens=[],
                blocks=blocks,
                next_position=end,
                next_token=prefix[-1],
                ready=False,
                ready_at=None,
                origin_window=self.window,
                prefix=list(prefix),
                kv_inputs=list(prefix[:-1]),
            )
            self.members[rid] = m
            self.active.append(rid)

            branch = (plan.get("commit") or {}).get("branch", "initial")
            emit(
                "reuse_prepared",
                window=self.window,
                req_id=rid,
                fingerprint=m["fingerprint"],
                context_tokens=len(prefix),
                reused=start,
                incremental=end - start if start else 0,
                full=end if not start else 0,
                invalidated=old_valid - start,
                branch=branch,
                blocks=blocks,
                next_position=end,
                kv_valid_upto=end,
            )

            # Keep the independent full-prefix audit outside the normal
            # packed Prefill forward so its cost remains separately visible.
            if plan.get("audit_blocks"):
                reference = plan["audit_blocks"]
                self.fill(
                    prefix, 0, end, reference, rid,
                    "reuse_reference_full",
                )

                actual_kv = self.values(blocks, end)
                ref = self.values(reference, end)
                layers = []

                layer_tensors = self.layers()
                if not len(layer_tensors) == len(actual_kv) == len(ref):
                    raise ValueError("KV audit layer count mismatch")
                if start and len(preserved) != len(layer_tensors):
                    raise ValueError("KV audit preserved layer count mismatch")
                for (name, _), a, b in zip(
                    layer_tensors, actual_kv, ref
                ):
                    # Validate gathered layout before any legacy byte comparison.
                    segments = kv_segment_errors(a, b, start, end)
                    # KV layout: (K/V, tokens, heads, head_dim).
                    for kv_index, kv_name in enumerate(("K", "V")):
                        actual = a[kv_index]
                        reference_kv = b[kv_index]

                        # Compare the original cached prefix captured
                        # BEFORE incremental preparation.
                        if start:
                            old_before = preserved[
                                len(layers) // 2
                            ][kv_index]
                            old_reference = reference_kv[:start]
                            old_prefix_exact = bool(torch.equal(
                                old_before.contiguous().view(torch.uint8),
                                old_reference.contiguous().view(torch.uint8),
                            ))
                        else:
                            old_prefix_exact = None

                        # Compare only tokens computed in this preparation.
                        new_actual = actual[start:end]
                        new_reference = reference_kv[start:end]
                        new_suffix_exact = bool(torch.equal(
                            new_actual.contiguous().view(torch.uint8),
                            new_reference.contiguous().view(torch.uint8),
                        ))

                        aa = actual.float()
                        bb = reference_kv.float()
                        delta = aa - bb
                        max_abs = delta.abs().max()
                        rms = delta.square().mean().sqrt()
                        ref_norm = bb.norm()
                        rel_l2 = delta.norm() / ref_norm.clamp_min(1e-12)

                        av = actual.contiguous().view(torch.uint8)
                        bv = reference_kv.contiguous().view(torch.uint8)

                        layers.append(
                            dict(
                                layer=name,
                                kv=kv_name,
                                exact=bool(torch.equal(av, bv)),
                                old_prefix_exact=old_prefix_exact,
                                new_suffix_exact=new_suffix_exact,
                                different_bytes=int((av != bv).sum()),
                                bytes=av.numel(),
                                max_abs=float(max_abs),
                                rms=float(rms),
                                rel_l2=float(rel_l2),
                                **segments[kv_index],
                            )
                        )

                if preserved:
                    now = self.values(blocks, start)
                    if any(
                        not torch.equal(
                            a.view(torch.uint8), b.view(torch.uint8)
                        )
                        for a, b in zip(preserved, now)
                    ):
                        raise RuntimeError(
                            "Accepted-prefix bytes were modified "
                            "during incremental preparation"
                        )

                emit(
                    "reuse_kv_audit",
                    window=self.window,
                    req_id=rid,
                    branch=branch,
                    reused=start,
                    fingerprint=m["fingerprint"],
                    prefix=prefix[:-1],
                    positions=list(range(end)),
                    preserved_exact=True,
                    layers=layers,
                    independent_exact=all(
                        x["exact"] for x in layers
                    ),
                )
                torch.cuda.synchronize(self.p.device)

        return {"prepared": [p["req_id"] for p in plans]}

    def step(self):
        inputs={r:(self.members[r]['next_position'],self.members[r]['next_token']) for r in self.active}
        for rid,(pos,_) in inputs.items():
            if len(self.members[rid]['kv_inputs'])!=pos:
                raise RuntimeError('KV valid input length/next position mismatch')
        status=super().step()
        for rid,(_,tok) in inputs.items():self.members[rid]['kv_inputs'].append(tok)
        return status

    def boundary(self):
        done={r:m for r,m in self.members.items() if m['ready']}
        result=super().boundary()
        for rid,m in done.items():
            if rid in self.cached or m['kv_inputs']!=m['prefix']+m['tokens'][:-1]:
                raise RuntimeError('Duplicate Ready cache or wrong KV input lineage')
            self.cached[rid]=m
        return result

    def drop(self,ids):
        for rid in ids:self.cached.pop(rid,None)
        return super().drop(ids)


class ReuseDraftQueue(PersistentDraftQueue):
    def __init__(self,engine):
        super().__init__(engine)
        self.cached_leases={};self.commits={};self.audit_leases={}
        self.history_pages=set();self.replayed=set();self.replay_blocks=[]
        # Allocations whose rollback failed remain explicitly tracked for close().
        self.pending_allocations=[]

    def release(self,rid,reason):
        if reason!='ready_acknowledged':return super().release(rid,reason)
        if rid in self.cached_leases:raise RuntimeError('Duplicate cached lease')
        # Check ownership before transferring the record between maps. The old
        # order popped the only active record before validating the blocks.
        blocks=self.leases[rid]
        if any(b.ref_cnt!=1 or b.is_null for b in blocks):
            raise RuntimeError('P2 lease ownership lost')
        self.cached_leases[rid]=blocks
        self.leases.pop(rid)
        self.contexts.pop(rid,None);self.carry.pop(rid,None)
        emit('reuse_lease_park',window=self.window,req_id=rid,blocks=[b.block_id for b in blocks])

    def discard_cached(self,ids,reason):
        if not ids:return
        ack=self.rpc('drop',{'ids':ids})
        if ack['dropped']!=ids:raise RuntimeError('Cache drop acknowledgement mismatch')
        for rid in ids:
            # Keep the cache ownership record until validation and allocator
            # release both succeed. This makes failed cleanup retryable/auditable.
            bs=self.cached_leases[rid]
            if any(b.ref_cnt!=1 or b.is_null for b in bs):
                raise RuntimeError('Cached lease ownership lost')
            self.s.kv_cache_manager.block_pool.free_blocks(bs)
            self.cached_leases.pop(rid)
            self.commits.pop(rid,None)
            emit('reuse_lease_release',req_id=rid,reason=reason,blocks=[b.block_id for b in bs])

    def reconcile(self):
        super().reconcile()
        invalid=[rid for rid in self.cached_leases if rid not in self.s.requests
            or self.s.requests[rid].is_finished() or self.s.requests[rid].status!=RequestStatus.RUNNING]
        self.discard_cached(invalid,'ended_or_preempted')

    def before_target(self,output):
        super().before_target(output)
        owned=[b for rows in [*self.leases.values(),*self.cached_leases.values()] for b in rows]
        ids=[b.block_id for b in owned]
        formal={b for req in self.s.running for group in self.s.kv_cache_manager.get_block_ids(req.request_id) for b in group}
        if len(set(ids))!=len(ids) or formal.intersection(ids) or any(b.ref_cnt!=1 for b in owned):
            raise RuntimeError('Reuse cached/active KV ownership collision')

    def committed(self,req,drafts,generated):
        record=self.inflight.get(req.request_id)
        super().committed(req,drafts,generated)
        if record is None:return
        if req.request_id not in self.cached_leases:raise RuntimeError('Verified request lost its cache lease')
        a=len(generated)-1
        self.commits[req.request_id]=dict(source_fp=record['fingerprint'],drafts=list(drafts),generated=list(generated),
            branch='all' if a==len(drafts) else ('zero' if a==0 else 'partial'),accepted=a,k=len(drafts))

    def _rollback_allocations(self, allocations):
        """Free temporary allocations, retaining ownership records on failure."""
        pool=self.s.kv_cache_manager.block_pool
        for blocks in allocations:
            if not blocks:
                continue
            try:
                if any(b.is_null or b.ref_cnt != 1 for b in blocks):
                    raise RuntimeError('P2 temporary allocation ownership lost during rollback')
                pool.free_blocks(blocks)
            except BaseException:
                # Do not silently lose block IDs if validation/free fails.
                self.pending_allocations.append(list(blocks))

    def plans(self):
        plans=[];pool=self.s.kv_cache_manager.block_pool
        for req in self.s.running:
            rid=req.request_id
            if (rid in self.leases or rid in self.ready or rid in self.inflight or req.status!=RequestStatus.RUNNING
                    or req.is_finished() or not req.num_output_tokens or req.spec_token_ids
                    or req.num_computed_tokens!=req.num_tokens-1):continue
            params=req.sampling_params
            if (params is None or params.temperature!=0 or req.has_encoder_inputs or req.lora_request is not None
                    or req.structured_output_request or params.prompt_logprobs is not None):
                raise ValueError('P2 supports text-only greedy requests without grammar/LoRA')
            cap=(params.extra_args or {}).get('cdb_draft_k',self.s.num_spec_tokens)
            if type(cap) is not int or not 1<=cap<=self.s.num_spec_tokens:raise ValueError('Invalid P2 request K')
            if req.num_tokens+cap>=self.s.max_model_len:
                self.discard_cached([rid] if rid in self.cached_leases else [],'length_limit');continue

            n=(req.num_tokens+cap+self.block_size-1)//self.block_size
            cached=self.cached_leases.get(rid,[])
            extra=max(0,n-len(cached))
            audit=os.getenv('CDB_AV_REUSE_VALIDATE','0')=='1'
            if pool.get_num_free_blocks()<extra+(n if audit else 0)+self.s.max_num_running_reqs*2:
                self.discard_cached([rid] if cached else [],'capacity_fallback');continue

            # Stage all fallible work before transferring cached lease ownership,
            # publishing an active lease, or consuming the commit certificate.
            new_blocks=[]
            audit_blocks=[]
            try:
                if extra:
                    new_blocks=pool.get_new_blocks(extra)
                    if len(new_blocks) != extra:
                        raise RuntimeError('P2 allocator returned an incomplete new lease')
                blocks=list(cached)+list(new_blocks)
                if len(blocks) < n:
                    raise RuntimeError('P2 staged lease has insufficient block capacity')
                if any(b.ref_cnt!=1 or b.is_null for b in blocks):
                    raise RuntimeError('P2 new lease not exclusive')
                ctx=RefillQueue.context(req)
                prefix=list(req.prompt_token_ids)+list(req.output_token_ids)
                if audit:
                    audit_blocks=pool.get_new_blocks(n)
                    if len(audit_blocks) != n:
                        raise RuntimeError('P2 allocator returned an incomplete audit lease')
                    if any(b.ref_cnt!=1 or b.is_null for b in audit_blocks):
                        raise RuntimeError('P2 audit lease not exclusive')
                commit=self.commits.get(rid)
                plan=dict(req_id=rid,prefix=prefix,cap=cap,
                    blocks=[b.block_id for b in blocks],
                    audit_blocks=[b.block_id for b in audit_blocks],
                    reuse=bool(cached),commit=commit,**ctx)
            except BaseException:
                self._rollback_allocations([audit_blocks,new_blocks])
                raise

            # Commit the staged ownership transition only after every allocation,
            # ownership check, context construction, and plan construction succeeds.
            if cached:
                if self.cached_leases.get(rid) is not cached:
                    self._rollback_allocations([audit_blocks,new_blocks])
                    raise RuntimeError('P2 cached lease changed during plan staging')
                self.cached_leases.pop(rid)
            self.leases[rid]=blocks
            self.history_pages.update(b.block_id for b in blocks)
            self.contexts[rid]={'context':ctx,'computed':req.num_computed_tokens}
            if audit_blocks:
                self.audit_leases[rid]=audit_blocks
            if commit is not None:
                self.commits.pop(rid,None)
            plans.append(plan)
        return plans

    def rpc(self,operation,payload=None):
        result=super().rpc(operation,payload)
        if operation=='prepare' and os.getenv('CDB_AV_REUSE_REPLAY','0')=='1':
            keys={(p['commit']['k'],p['commit']['accepted']) for p in payload['plans'] if p.get('commit')}
            if keys-self.replayed:
                self.replay_blocks=self.s.kv_cache_manager.block_pool.get_new_blocks(len(self.history_pages))
                mapping={k:b.block_id for k,b in zip(sorted(self.history_pages),self.replay_blocks)}
                super().rpc('replay',{'mapping':mapping})
                self.s.kv_cache_manager.block_pool.free_blocks(self.replay_blocks)
                self.replay_blocks=[];self.replayed.update(keys)
        if operation=='prepare':
            # Validation RPC copied comparisons to CPU and synchronized before ACK.
            for bs in self.audit_leases.values():self.s.kv_cache_manager.block_pool.free_blocks(bs)
            self.audit_leases.clear()
        return result

    def close(self):
        ids=list(set(self.leases)|set(self.cached_leases)|set(self.audit_leases))
        if ids:
            ack=self.rpc('drop',{'ids':ids})
            if ack['dropped']!=ids:raise RuntimeError('Reuse cleanup acknowledgement mismatch')
        for rid in list(self.leases):super().release(rid,'close')
        for mapping in (self.cached_leases,self.audit_leases):
            for bs in mapping.values():
                if any(b.ref_cnt!=1 for b in bs):raise RuntimeError('Cleanup lost KV ownership')
                self.s.kv_cache_manager.block_pool.free_blocks(bs)
            mapping.clear()
        if self.replay_blocks:
            self.s.kv_cache_manager.block_pool.free_blocks(self.replay_blocks)
            self.replay_blocks=[]
        pool=self.s.kv_cache_manager.block_pool
        while self.pending_allocations:
            bs=self.pending_allocations[0]
            if any(b.is_null or b.ref_cnt!=1 for b in bs):
                raise RuntimeError('P2 pending allocation ownership lost during close')
            pool.free_blocks(bs)
            self.pending_allocations.pop(0)
        self.commits.clear();self.ready.clear();self.inflight.clear()

    def advance(self,model_executed):
        try:super().advance(model_executed)
        except BaseException:
            self.close()
            raise
