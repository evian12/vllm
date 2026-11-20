# SPDX-License-Identifier: Apache-2.0
"""Helpers for a single Draft backbone/head/HF CUDA Graph.

No scheduler, request or KV ownership state. Outputs are borrowed only until
next replay; the original blocking D2H consumes packets before that replay.
"""
import torch


def native_hf_packet(logits):
    fp32 = logits.float()
    maximum, tokens = fp32.max(dim=-1)
    confidence = (maximum - torch.logsumexp(fp32, dim=-1)).exp()
    return torch.stack((tokens.to(torch.float64), confidence.to(torch.float64)), dim=1)


def chain_storage(model):
    head = model.lm_head
    weight = head.weight
    processor = model.logits_processor
    return (id(model), id(head), id(head.quant_method), weight.data_ptr(),
            tuple(weight.shape), tuple(weight.stride()), weight.dtype, weight.device,
            id(processor), processor.scale, processor.soft_cap, processor.org_vocab_size)


def validate_chain_scope(proposer, policy):
    from vllm.v1.spec_decode.cdb_av_inputs import validate_staging_scope
    validate_staging_scope(proposer.vllm_config)
    model = proposer.model
    if (policy != 'hf_confidence' or type(model).__name__ != 'LlamaForCausalLM'
            or type(model.lm_head.quant_method).__name__ != 'UnquantizedEmbeddingMethod'
            or tuple(model.lm_head.weight.shape) != (128256, 2048)
            or model.lm_head.weight.dtype != torch.bfloat16
            or model.training):
        raise ValueError('Draft chain Graph requires greedy HF Llama1B BF16 static scope')


def consume_packet(pending, logits):
    if pending is None:
        return None
    source, packet = pending
    if source is not logits or packet.shape != (logits.shape[0], 2):
        raise RuntimeError('Draft chain packet/logits identity or row mismatch')
    return packet
