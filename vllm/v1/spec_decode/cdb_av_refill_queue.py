# SPDX-License-Identifier: Apache-2.0
"""Persistent scheduler-side draft work, provenance and exclusive KV leases."""

import os
import time

from vllm.v1.outputs import DraftTokenIds
from vllm.v1.request import RequestStatus
from vllm.v1.spec_decode.cdb_av_fingerprint import token_fingerprint
from vllm.v1.spec_decode.cdb_av_refill import emit
from vllm.v1.spec_decode.cdb_av_work_queue import CDBDraftWorkQueue, DraftWorkStatus


class RefillQueue:
    def __init__(self, scheduler):
        self.s = scheduler
        self.queue = CDBDraftWorkQueue()
        self.leases = {}
        self.ready = {}
        self.inflight = {}
        self.window = 0
        if (
            scheduler.cache_config.enable_prefix_caching
            or scheduler.connector is not None
        ):
            raise ValueError(
                "Refill v1 requires prefix caching and KV connector disabled"
            )
        groups = scheduler.kv_cache_manager.kv_cache_config.kv_cache_groups
        if len(groups) != 1:
            raise ValueError("Refill v1 requires a single full-attention KV group")
        self.block_size = groups[0].kv_cache_spec.block_size

    @staticmethod
    def context(req):
        tokens = list(req.prompt_token_ids) + list(req.output_token_ids)
        return dict(context_tokens=len(tokens), fingerprint=token_fingerprint(tokens))

    def begin(self, window):
        if window <= self.window or self.leases:
            raise RuntimeError("Invalid refill window lifecycle or outstanding leases")
        self.window = window
        # Pending and unverified Ready survive windows. Retire only truly ended work.
        for rid in list(self.queue.items):
            if rid not in self.s.requests or (
                self.s.requests[rid].status != RequestStatus.WAITING
                and rid not in self.ready
                and rid not in self.inflight
            ):
                self.queue.retire(rid)
                self.ready.pop(rid, None)
        return {
            rid: self.context(req)
            for rid, req in self.s.requests.items()
            if req.status == RequestStatus.RUNNING and not req.is_finished()
        }

    def select(self, status):
        if os.getenv("CDB_AV_ALLOW_FRESH_WAITING", "1") == "0":
            return []
        if os.getenv("CDB_AV_ALLOW_FRESH_WAITING", "1") == "0":
            return []
        if status["window"] != self.window:
            raise RuntimeError("Refill selection against stale snapshot")
        plans = []
        pool = self.s.kv_cache_manager.block_pool
        slots = status["slots"]
        for req in self.s.waiting:
            rid = req.request_id
            params = req.sampling_params
            if (
                req.status != RequestStatus.WAITING
                or req.num_computed_tokens != 0
                or req.num_output_tokens != 0
                or req.has_encoder_inputs
                or req.lora_request is not None
                or params is None
                or params.temperature != 0
                or req.structured_output_request is not None
                or req.prompt_token_ids is None
                or params.prompt_logprobs is not None
            ):
                continue
            cap = (params.extra_args or {}).get("cdb_draft_k", self.s.num_spec_tokens)
            if type(cap) is not int or not 1 <= cap <= self.s.num_spec_tokens:
                raise ValueError(f"Invalid waiting draft K: {rid}")

            fresh_k = int(os.getenv("CDB_AV_FRESH_K", "0"))
            if fresh_k < 0 or fresh_k > self.s.num_spec_tokens:
                raise ValueError("Invalid CDB_AV_FRESH_K")
            if fresh_k:
                cap = min(cap, fresh_k)
            ctx = self.context(req)
            item = self.queue.items.get(rid)
            if item is None:
                item = self.queue.register(
                    rid, ctx["context_tokens"], ctx["context_tokens"], cap, "waiting"
                )
                item.fingerprint = ctx["fingerprint"]
            if item.fingerprint != ctx["fingerprint"]:
                raise RuntimeError("Pending waiting context changed")
            if item.status != DraftWorkStatus.PREPARE_REQUIRED:
                continue
            if (
                slots <= 0
                or cap > status["remaining_steps"]
                or ctx["context_tokens"] + cap >= self.s.max_model_len
                or ctx["context_tokens"] + cap > self.s.max_num_scheduled_tokens
            ):
                continue
            nblocks = (
                ctx["context_tokens"] + cap + self.block_size - 1
            ) // self.block_size
            audit = os.getenv("CDB_AV_REFILL_VALIDATE", "0") == "1"
            if pool.get_num_free_blocks() < nblocks * (2 if audit else 1):
                continue
            blocks = pool.get_new_blocks(nblocks)
            if any(b.ref_cnt != 1 or b.is_null for b in blocks):
                raise RuntimeError("Refill KV lease is not exclusive")
            audit_blocks = pool.get_new_blocks(nblocks) if audit else []
            self.leases[rid] = blocks + audit_blocks
            item.status = DraftWorkStatus.ELIGIBLE
            plans.append(
                dict(
                    req_id=rid,
                    prompt=list(req.prompt_token_ids),
                    cap=cap,
                    blocks=[b.block_id for b in blocks],
                    audit_blocks=[b.block_id for b in audit_blocks],
                    fingerprint=ctx["fingerprint"],
                    window=self.window,
                )
            )
            slots -= 1
            emit(
                "selected",
                window=self.window,
                after_step=status["steps_completed"],
                req_id=rid,
                slots_before=slots + 1,
                blocks=[b.block_id for b in blocks],
                target_status=req.status.name,
                context=ctx,
                cap=cap,
            )
        return plans

    def prepared(self, plans):
        for p in plans:
            item = self.queue.items[p["req_id"]]
            if item.status != DraftWorkStatus.ELIGIBLE:
                raise RuntimeError("Invalid preparation acknowledgement")
            item.status = DraftWorkStatus.ACTIVE
            item.kv_validated = True

    def step(self, status):
        for rid, current in status["progress"].items():
            item = self.queue.items.get(rid)
            if item is None or rid not in self.leases:
                continue  # normal running request managed by Target scheduler
            if current != item.progress and (
                current < item.progress or item.status == DraftWorkStatus.READY
            ):
                raise RuntimeError("Invalid refill progress or Ready reactivation")
            item.progress = current
            if rid in status["ready_req_ids"]:
                item.status = DraftWorkStatus.READY
        emit(
            "slots",
            window=self.window,
            step=status["steps_completed"],
            active=status["active_req_ids"],
            ready=status["ready_req_ids"],
            slots=status["slots"],
            waiting=len(self.s.waiting),
        )
        # Read-only probe: eligible means a Target-confirmed prefix exists,
        # NOT that a reusable Draft KV state is available for GPU admission.
        if os.getenv("CDB_AV_ELIGIBILITY_TRACE", "0") == "1":
            occupied = (
                set(status["active_req_ids"])
                | set(status["ready_req_ids"])
                | set(self.ready)
                | set(self.inflight)
                | set(self.leases)
            )
            eligible = []
            rejected: dict[str, int] = {}
            for req in self.s.running:
                rid = req.request_id
                if req.is_finished() or req.status != RequestStatus.RUNNING:
                    why = "not_running"
                elif rid in occupied:
                    why = "already_drafting_or_queued"
                elif req.num_output_tokens < 1:
                    why = "no_target_output"
                elif req.spec_token_ids:
                    why = "existing_spec_tokens"
                elif req.has_encoder_inputs or req.lora_request is not None:
                    why = "unsupported_request_type"
                else:
                    why = None
                if why is None:
                    eligible.append(rid)
                else:
                    rejected[why] = rejected.get(why, 0) + 1
            slots = int(status["slots"])
            emit(
                "eligibility",
                window=self.window,
                step=status["steps_completed"],
                target_running=len(self.s.running),
                target_waiting=len(self.s.waiting),
                draft_active=len(status["active_req_ids"]),
                draft_ready=len(status["ready_req_ids"]),
                draft_slots=slots,
                target_first_eligible=len(eligible),
                potential_refill=min(slots, len(eligible)),
                eligible_req_ids=eligible,
                rejection_counts=rejected,
            )

    def accept(self, records):
        ids, tokens, lengths, fingerprints = [], [], [], []
        for record in records:
            rid = record["req_id"]
            req = self.s.requests.get(rid)
            if req is None or req.is_finished():
                continue
            if record["window"] != self.window or self.context(req) != {
                "context_tokens": record["context_tokens"],
                "fingerprint": record["fingerprint"],
            }:
                raise RuntimeError("Refill handoff stale context/window")
            if record["source"] == "waiting":
                if rid in self.ready or rid not in self.leases:
                    raise RuntimeError("Duplicate or unowned refill result")
                self.ready[rid] = record
            ids.append(rid)
            tokens.append(record["tokens"])
            lengths.append(record["context_tokens"])
            fingerprints.append(record["fingerprint"])
            emit("handoff", **record)
        self.s.update_draft_token_ids(
            DraftTokenIds(
                ids,
                tokens,
                source_context_tokens=lengths,
                context_fingerprints=fingerprints,
            )
        )
        # Worker converted candidates to CPU before this acknowledgement. No GPU
        # operation can still reference these leases; Target never uses their KV.
        for rid, blocks in list(self.leases.items()):
            self.s.kv_cache_manager.block_pool.free_blocks(blocks)
            del self.leases[rid]

    def scheduled(self, req, candidates):
        record = self.ready.pop(req.request_id)
        if (
            self.context(req)["fingerprint"] != record["fingerprint"]
            or candidates != record["tokens"]
        ):
            raise RuntimeError("Refill candidates changed before Target prefill/verify")
        self.inflight[req.request_id] = record
        emit(
            "verify_scheduled",
            req_id=req.request_id,
            window=record["window"],
            tokens=candidates,
            fingerprint=record["fingerprint"],
            ready_wait=time.monotonic() - record["ready_at"],
            target_prefill_tokens=req.num_tokens,
        )

    def committed(self, req, drafts, generated):
        record = self.inflight.pop(req.request_id, None)
        if record is None:
            return
        if drafts != record["tokens"] or not generated:
            raise RuntimeError("Refill verification lost candidate batch")
        accepted = len(generated) - 1
        if generated[:accepted] != drafts[:accepted]:
            raise RuntimeError("Refill accepted token prefix mismatch")
        emit(
            "verified",
            req_id=req.request_id,
            window=record["window"],
            drafts=drafts,
            accepted=accepted,
            rejected=len(drafts) - accepted,
            generated=generated,
            fingerprint=record["fingerprint"],
        )
        self.queue.retire(req.request_id)
