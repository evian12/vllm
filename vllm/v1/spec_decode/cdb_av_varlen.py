# SPDX-License-Identifier: Apache-2.0
"""Shrinking, synchronous draft_model sessions. No refill or cross-forward carry."""

import json
import os
import time
from dataclasses import dataclass

import torch

from vllm.config import CUDAGraphMode
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.v1.attention.backend import CommonAttentionMetadata

logger = init_logger(__name__)


def validate_lengths(request_ids, lengths, maximum):
    if len(request_ids) != len(lengths) or len(set(request_ids)) != len(request_ids):
        raise ValueError("CDB request IDs and lengths must be unique and aligned")
    if not request_ids or any(
        type(k) is not int or not 1 <= k <= maximum for k in lengths
    ):
        raise ValueError(f"CDB per-request K must be integers in [1, {maximum}]")
    return tuple(lengths)


@dataclass(frozen=True)
class CDBDraftResult:
    # Padding is internal only. GPUModelRunner trims each row BEFORE DraftTokenIds.
    token_ids: torch.Tensor
    valid_lengths: tuple[int, ...]
    request_ids: tuple[str, ...]

    def __post_init__(self):
        if self.token_ids.ndim != 2 or self.token_ids.shape[0] != len(self.request_ids):
            raise ValueError("Invalid CDB draft matrix")
        validate_lengths(self.request_ids, self.valid_lengths, self.token_ids.shape[1])


def ragged_rows(cpu_tokens, lengths):
    if len(cpu_tokens) != len(lengths):
        raise ValueError("CDB row/length mismatch")
    if any(
        type(k) is not int or not 0 <= k <= len(row)
        for row, k in zip(cpu_tokens, lengths)
    ):
        raise ValueError("Invalid CDB valid length")
    return [row[:k] for row, k in zip(cpu_tokens, lengths)]


def compact_decode_metadata(base, original_indices, positions, seq_lens, block_size):
    """Fresh metadata; all row tensors use ORIGINAL request indices, never local IDs.

    seq_lens is already rejection-adjusted and incremented. CPU shadows and derived
    caches must not be inherited from a differently shaped / older batch.
    """
    n = original_indices.numel()
    blocks = base.block_table_tensor.index_select(0, original_indices).contiguous()
    block_ids = blocks.gather(1, (positions // block_size).view(-1, 1)).view(-1)
    slots = block_ids * block_size + positions % block_size
    q_cpu = torch.arange(n + 1, dtype=torch.int32)
    return CommonAttentionMetadata(
        query_start_loc=q_cpu.to(positions.device),
        query_start_loc_cpu=q_cpu,
        seq_lens=seq_lens,
        num_reqs=n,
        num_actual_tokens=n,
        max_query_len=1,
        # Upper bound avoids a D2H reduction on every step.
        max_seq_len=base.max_seq_len,
        block_table_tensor=blocks,
        slot_mapping=slots,
        causal=True,
        _seq_lens_cpu=None,
        _num_computed_tokens_cpu=None,
    )


def propose_varlen_steps(
    self,
    target_token_ids,
    target_positions,
    target_hidden_states,
    next_token_ids,
    last_token_indices,
    common_attn_metadata,
    sampling_metadata,
    mm_embed_inputs=None,
    num_rejected_tokens_gpu=None,
    slot_mappings=None,
    num_speculative_tokens=None,
    request_ids=None,
    request_draft_lengths=None,
):
    maximum = (
        self.num_speculative_tokens
        if num_speculative_tokens is None
        else num_speculative_tokens
    )
    ids = tuple(request_ids or ())
    lengths = validate_lengths(ids, request_draft_lengths or (), maximum)
    b = common_attn_metadata.num_reqs
    if len(ids) != b:
        raise ValueError("CDB request-ID mapping does not match attention batch")
    if self.pass_hidden_states_to_model or self.uses_mrope or self.supports_mm_inputs:
        raise NotImplementedError(
            "CDB varlen supports independent text draft_model only"
        )
    if self.draft_indexer_metadata_builder or self.indexer_layer_names:
        raise NotImplementedError("CDB varlen does not support indexer attention")
    if any(
        getattr(common_attn_metadata, f) is not None
        for f in ("encoder_seq_lens", "dcp_local_seq_lens", "logits_indices_padded")
    ):
        raise NotImplementedError(
            "CDB varlen does not support cross/DCP/indexed metadata"
        )
    trace = os.getenv("CDB_AV_TRACE_STEPS", "0") == "1"
    check = os.getenv("CDB_AV_VALIDATE_METADATA", "0") == "1"
    k_policy = os.getenv("CDB_AV_K_POLICY", "manual").strip().lower()
    if k_policy not in ("manual", "hf_confidence"):
        raise ValueError(f"Unsupported CDB_AV_K_POLICY={k_policy!r}")
    hf_dynamic = k_policy == "hf_confidence"
    hf_threshold = float(os.getenv("CDB_AV_HF_THRESHOLD", "0.4"))
    if hf_dynamic and not 0.0 <= hf_threshold <= 1.0:
        raise ValueError("CDB_AV_HF_THRESHOLD must be in [0, 1]")
    self._cdb_round = getattr(self, "_cdb_round", 0) + 1
    round_id = self._cdb_round
    if os.getenv("CDB_AV_MIXED_TRACE", "0") == "1":
        query_lens = common_attn_metadata.query_start_loc_cpu.diff().tolist()
        logger.info(
            "[CDB_MIXED] round=%d req_ids=%s query_lens=%s total=%d",
            round_id,
            list(ids),
            query_lens,
            sum(query_lens),
        )

    builder = self.attn_metadata_builder or self._get_attention_metadata_builder()
    block_size = builder.kv_cache_spec.block_size
    # Use the existing rejection-aware merge kernel for the first (multi-token)
    # draft forward. All requests still need step 1.
    num_tokens, last_indices, base = self.set_inputs_first_pass(
        target_token_ids,
        next_token_ids,
        target_positions,
        last_token_indices,
        common_attn_metadata,
        num_rejected_tokens_gpu,
    )
    positions = self.positions[last_indices].clone()
    seq_lens = base.seq_lens.clone()
    if num_rejected_tokens_gpu is not None:
        seq_lens = seq_lens - num_rejected_tokens_gpu
    # No mutation of the target's metadata, cached shadows or persistent InputBatch.
    base = base.replace(
        _seq_lens_cpu=None,
        _num_computed_tokens_cpu=None,
        _num_computed_tokens_cache=None,
    )
    result = torch.zeros(
        (b, maximum), dtype=torch.int64, device=target_token_ids.device
    )
    active = list(range(b))
    progress = [0] * b
    actual_lengths = [0] * b
    ready = []
    previous_ids: torch.Tensor | None = None
    cad = base
    reference_logits: list[tuple[tuple[int, ...], torch.Tensor]] | None = (
        [] if getattr(self, "_cdb_capture_logits", False) else None
    )
    if reference_logits is not None:
        self._cdb_varlen_logits = reference_logits
    for step in range(1, max(lengths) + 1):
        original = torch.tensor(active, dtype=torch.long, device=positions.device)
        if step > 1:
            # Gather ALL request-indexed state by original indices. Completed
            # requests are absent from both tensors and actual model forward.
            current_positions = positions.index_select(0, original) + step - 1
            current_lens = seq_lens.index_select(0, original) + step - 1
            cad = compact_decode_metadata(
                base.replace(
                    max_seq_len=min(base.max_seq_len + step - 1, self.max_model_len)
                ),
                original,
                current_positions,
                current_lens,
                block_size,
            )
            num_tokens = len(active)
            self.input_ids[:num_tokens] = previous_ids.index_select(0, original).int()
            self._set_positions(num_tokens, current_positions)
            last_indices = torch.arange(num_tokens, device=positions.device)
        if check:
            assert int(cad.query_start_loc[-1]) == num_tokens
            assert cad.num_reqs == len(active)
            assert cad.block_table_tensor.shape[0] == len(active)
            assert torch.equal(cad.query_start_loc.cpu(), cad.query_start_loc_cpu)
            if step > 1:
                assert torch.equal(cad.seq_lens, current_positions + 1)
                assert bool((current_positions < self.max_model_len).all())
                assert bool((cad.slot_mapping >= 0).all())
                assert torch.equal(
                    cad.block_table_tensor, base.block_table_tensor[original]
                )
                assert torch.equal(cad.num_computed_tokens_cpu, cad.seq_lens.cpu() - 1)
        mode, desc = self.cudagraph_dispatcher.dispatch(num_tokens)
        if mode != CUDAGraphMode.NONE or desc.num_tokens != num_tokens:
            raise RuntimeError(
                "CDB varlen requires eager forwards without graph padding"
            )
        metadata = builder.build_for_drafting(cad, draft_index=step - 1)
        per_layer = {name: metadata for name in self.attn_layer_names}
        inputs = self.input_ids[:num_tokens]
        with set_forward_context(
            per_layer,
            self.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            slot_mapping=self._get_slot_mapping(num_tokens, cad.slot_mapping),
        ):
            ret = self.model(
                input_ids=inputs,
                positions=self._get_positions(num_tokens),
                inputs_embeds=None,
            )
        hidden = ret[0] if self.model_returns_tuple() else ret
        logits = self.model.compute_logits(hidden[last_indices])
        if logits is None:
            raise RuntimeError("CDB Draft model returned no logits")
        if check and not bool(torch.isfinite(logits).all()):
            raise RuntimeError("Non-finite CDB draft logits")
        if reference_logits is not None:
            reference_logits.append((tuple(active), logits.detach().clone()))
        tokens = logits.argmax(dim=-1)

        # Compute the probability of the selected (greedy) draft token.
        # FP32 softmax improves numerical stability for BF16 logits.
        # Kept off in manual mode so the validated baseline is unchanged.
        confidence = None
        stop_flags = None
        if hf_dynamic:
            confidence = (
                torch.log_softmax(logits.float(), dim=-1)
                .gather(dim=-1, index=tokens.unsqueeze(-1))
                .exp()
                .squeeze(-1)
            )
            # Single batched D2H transfer per step, not one .item() sync per row.
            # Diagnostic only: preserve the GPU-to-CPU synchronization.
            stop_flags = (confidence < hf_threshold).detach().cpu().tolist()

        result[original, step - 1] = tokens
        # A full original-row token vector makes the next gather unambiguous.
        previous_ids = result[:, step - 1]
        completed = []
        stop_reasons = {}
        for local_idx, original_idx in enumerate(active):
            progress[original_idx] += 1
            actual_lengths[original_idx] = progress[original_idx]
            cap_reached = progress[original_idx] >= lengths[original_idx]
            low_confidence = hf_dynamic and bool(stop_flags[local_idx])
            if cap_reached or low_confidence:
                completed.append(original_idx)
                ready.append(original_idx)
                stop_reasons[ids[original_idx]] = (
                    "max_k" if cap_reached else "low_confidence"
                )
        if trace and completed:
            logger.info(
                "[CDB_READY_TIMING] %s",
                json.dumps(
                    dict(ready_at=time.monotonic(), req_ids=[ids[i] for i in completed])
                ),
            )
        if trace and hf_dynamic:
            logger.info(
                "[CDB_HF_CONF] %s",
                json.dumps(
                    dict(
                        round=round_id,
                        step=step,
                        threshold=hf_threshold,
                        confidence=dict(
                            zip(
                                (ids[i] for i in active),
                                confidence.detach().cpu().tolist(),
                            )
                        ),
                        stop_reasons=stop_reasons,
                    )
                ),
            )
        report = dict(
            step=step,
            logical_batch=len(active),
            input_rows=int(inputs.shape[0]),
            draft_token_ids=tokens,
            request_ids=ids,
            active_indices=tuple(active),
            completed_indices=tuple(completed),
            ready_indices=tuple(ready),
            draft_progress=tuple(progress),
            request_draft_lengths=lengths,
            actual_draft_lengths=tuple(actual_lengths),
            k_policy=k_policy,
            round=round_id,
        )
        if trace:
            logger.info(
                "[CDB_VARLEN] %s",
                json.dumps(
                    dict(
                        diagnostic=getattr(self, "_cdb_diagnostic", None),
                        round=round_id,
                        step=step,
                        active=[ids[i] for i in active],
                        original_indices=active,
                        completed=[ids[i] for i in completed],
                        logical_active=len(active),
                        forward_rows=int(inputs.shape[0]),
                        graph_padding=0,
                        progress=dict(zip(ids, progress)),
                        K_cap=dict(zip(ids, lengths)),
                        K_actual=dict(zip(ids, actual_lengths)),
                        k_policy=k_policy,
                        stop_reasons=stop_reasons,
                        ready_to_verify=[ids[i] for i in ready],
                    )
                ),
            )
        yield report

        completed_set = set(completed)
        active = [i for i in active if i not in completed_set]

        # All requests have completed drafting.
        # Never issue a zero-sized Draft Forward.
        if not active:
            break
    if active or any(not 1 <= k <= cap for k, cap in zip(actual_lengths, lengths)):
        raise RuntimeError("Incomplete or invalid CDB draft session")
    logger.info(
        "[CDB_DEBUG] round=%d generator_return lengths=%s",
        round_id,
        actual_lengths,
    )

    return CDBDraftResult(result, tuple(actual_lengths), ids)


def validate_fixed_prefix(self, args, kwargs):
    """Opt-in GPU oracle, NEVER a performance mode. Replays original fixed-K
    proposer at the same target boundary, then the shrinking path. Future KV
    writes from the oracle are causally masked and rewritten before use.
    Any prefix mismatch is reported as unresolved, not waved away as numerics.
    """
    if os.getenv("CDB_AV_K_POLICY", "manual").strip().lower() == "hf_confidence":
        raise RuntimeError(
            "Fixed-K prefix oracle is not compatible with dynamic stopping; "
            "disable CDB_AV_VALIDATE_PREFIX for HF mode"
        )
    from dataclasses import fields

    from vllm.v1.spec_decode.eagle import SpecDecodeBaseProposer

    if args:
        raise ValueError("CDB prefix oracle expects named proposer arguments")
    ref = dict(kwargs)
    ref.pop("request_ids", None)
    ref.pop("request_draft_lengths", None)
    cad = ref["common_attn_metadata"]
    if cad is None:
        raise ValueError("CDB prefix oracle requires attention metadata")
    ref["common_attn_metadata"] = cad.replace(
        **{
            f.name: getattr(cad, f.name).clone()
            for f in fields(cad)
            if isinstance(getattr(cad, f.name), torch.Tensor)
        }
    )
    reference_logits = []
    compute_logits = self.model.compute_logits

    def capture(hidden, *a, **kw):
        logits = compute_logits(hidden, *a, **kw)
        reference_logits.append(logits.detach().clone())
        return logits

    self.model.compute_logits = capture
    try:
        reference_result = SpecDecodeBaseProposer.propose(self, **ref)
        if reference_result is None:
            raise RuntimeError("CDB fixed-prefix reference returned no tokens")
        reference = reference_result.clone()
    finally:
        self.model.compute_logits = compute_logits
    self._cdb_capture_logits = True
    try:
        session = self.begin_draft(**kwargs)
        result = self.finish_draft(session)
        diffs = []
        max_logit_delta = 0.0
        for step, (indices, logits) in enumerate(self._cdb_varlen_logits):
            ref_logits = reference_logits[step][list(indices)]
            delta = float((ref_logits.float() - logits.float()).abs().max())
            max_logit_delta = max(max_logit_delta, delta)
        for i, k in enumerate(result.valid_lengths):
            left, right = reference[i, :k].tolist(), result.token_ids[i, :k].tolist()
            if left != right:
                first = next(j for j in range(k) if left[j] != right[j])
                indices, actual = self._cdb_varlen_logits[first]
                ref_top = reference_logits[first][i].float().topk(2)
                got_top = actual[indices.index(i)].float().topk(2)
                diffs.append(
                    dict(
                        req_id=result.request_ids[i],
                        first_diff=first,
                        fixed=left,
                        varlen=right,
                        fixed_top2=ref_top.indices.tolist(),
                        fixed_margin=float(ref_top.values[0] - ref_top.values[1]),
                        varlen_top2=got_top.indices.tolist(),
                        varlen_margin=float(got_top.values[0] - got_top.values[1]),
                    )
                )
        controls = None
        checked_round = self._cdb_round
        if diffs:
            # Hold the target boundary fixed; isolate the compacted batch shape.
            # Original fixed-K replay -> new generator with NO contraction ->
            # compacted replay. Restore compacted KV writes before returning.
            self._cdb_diagnostic = "prefix_replay_control"
            repeated_logits = []

            def repeat_capture(hidden, *a, **kw):
                logits = compute_logits(hidden, *a, **kw)
                repeated_logits.append(logits.detach().clone())
                return logits

            self.model.compute_logits = repeat_capture
            try:
                repeated = SpecDecodeBaseProposer.propose(self, **ref).clone()
            finally:
                self.model.compute_logits = compute_logits
            full_kwargs = dict(kwargs)
            full_kwargs["request_draft_lengths"] = [reference.shape[1]] * len(
                result.request_ids
            )
            full = self.finish_draft(self.begin_draft(**full_kwargs))
            full_logits = list(self._cdb_varlen_logits)
            repeated_fixed_exact = torch.equal(reference, repeated) and all(
                torch.equal(a, b) for a, b in zip(reference_logits, repeated_logits)
            )
            uncontracted_exact = torch.equal(reference, full.token_ids) and all(
                torch.equal(a, b[1]) for a, b in zip(reference_logits, full_logits)
            )
            restored = self.finish_draft(self.begin_draft(**kwargs))
            compact_repeat_exact = torch.equal(result.token_ids, restored.token_ids)
            controls = dict(
                original_fixed_replay_exact=repeated_fixed_exact,
                new_uncontracted_matches_original_logits_exact=uncontracted_exact,
                compacted_replay_tokens_exact=compact_repeat_exact,
                all_metadata_invariants_checked=os.getenv("CDB_AV_VALIDATE_METADATA")
                == "1",
            )
            self._cdb_diagnostic = None
            result = restored
        classification = "match"
        if diffs:
            classification = (
                "batch_shape_sensitive"
                if controls and all(controls.values())
                else "mismatch_unresolved"
            )
        logger.info(
            "[CDB_PREFIX_CHECK] %s",
            json.dumps(
                dict(
                    round=checked_round,
                    req_ids=result.request_ids,
                    lengths=result.valid_lengths,
                    status=classification,
                    controls=controls,
                    max_abs_logit_delta=max_logit_delta,
                    differences=diffs,
                )
            ),
        )
        return result
    finally:
        self._cdb_diagnostic = None
        self._cdb_capture_logits = False
        self._cdb_varlen_logits = []
