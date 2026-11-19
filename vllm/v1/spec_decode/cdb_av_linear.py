# SPDX-License-Identifier: Apache-2.0
"""Opt-in Llama Draft down-projection GEMM experiment, not batch padding.

Only internal projection rows are padded. All model/attention/KV/token rows
retain their exact sizes. Original weights and unsupported GEMMs are untouched.
"""
import functools
import json
import os
import threading

import torch
import torch.nn.functional as F

from vllm.logger import init_logger

logger = init_logger(__name__)
PADDED_ROWS = frozenset((2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16))


def enabled(env=None):
    value = (os.environ if env is None else env).get('CDB_AV_DRAFT_DOWN_PAD32', '0')
    if value not in ('0', '1'):
        raise ValueError('CDB_AV_DRAFT_DOWN_PAD32 must be 0 or 1')
    return value == '1'


def validate_scope(cfg, model_name):
    c, p = cfg.model_config, cfg.parallel_config
    hf = c.hf_text_config
    if (model_name != 'LlamaForCausalLM' or c.dtype != torch.bfloat16
            or c.quantization is not None or not c.enforce_eager
            or cfg.scheduler_config.async_scheduling or cfg.lora_config is not None
            or getattr(c, 'enable_sleep_mode', False)
            or not 1 <= c.max_model_len <= 1024
            or (hf.hidden_size, hf.intermediate_size, hf.num_hidden_layers,
                hf.vocab_size) != (2048, 8192, 16, 128256)
            or any(type(n) is not int or n != 1 for n in
                   (p.tensor_parallel_size, p.pipeline_parallel_size,
                    p.data_parallel_size, p.decode_context_parallel_size))):
        raise ValueError('Draft down GEMM requires eager BF16 Llama1B/TP1, no async/LoRA/quantization/sleep')


def padded_call(original, layer, x, bias):
    """Capture/launch full FP32 accumulation; restore the process flag on error.

    The serialized single-thread engine is a required scope. Graph replay uses
    the kernels selected during capture and does not execute these CPU setters.
    No device synchronization, KV operation or precision relaxation is added.
    """
    previous = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    try:
        if previous:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        padded = F.pad(x, (0, 0, 0, 32 - x.shape[0]))
        return original(layer, padded, bias)[:x.shape[0]]
    finally:
        if previous:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = previous


def wrap_projection(original, owner, audit=None):
    @functools.wraps(original)
    def apply(layer, x, bias=None):
        if threading.get_ident() != owner:
            raise RuntimeError('Draft linear experiment used from a different thread')
        if (x.ndim != 2 or not x.is_cuda or x.dtype != torch.bfloat16
                or x.shape[1] != 8192 or x.shape[0] not in PADDED_ROWS
                or not x.is_contiguous()):
            return original(layer, x, bias)
        output = padded_call(original, layer, x, bias)
        if audit is not None and audit[0] < 4 and not torch.cuda.is_current_stream_capturing():
            audit[0] += 1
            native = original(layer, x, bias)
            if output.shape != native.shape or not bool(torch.isfinite(output).all()):
                raise RuntimeError('Draft projection shape/nonfinite mismatch')
            difference = output.float() - native.float()
            reference_norm = native.float().norm()
            norm = float(reference_norm)
            logger.info('[CDB_DRAFT_LINEAR_AUDIT] %s', json.dumps(dict(
                sample=audit[0], projection=layer.prefix, live_M=x.shape[0],
                gemm_M=32, output_shape=list(output.shape), weight_shape=list(layer.weight.shape),
                accumulation='fp32', comparison='unchanged native projection, same actual input/weight',
                byte_exact=torch.equal(output.contiguous().view(torch.uint8),native.contiguous().view(torch.uint8)),
                max_abs=float(difference.abs().max()),rms=float(difference.square().mean().sqrt()),
                rel_l2=float(difference.norm())/norm if norm else None,
                projected_argmax_equal_rows=int((output.argmax(-1)==native.argmax(-1)).sum()),
                note='Projection argmax is not a generated token; diagnostic extra GEMM excluded from performance.')))
        return output
    return apply


def configure(model, cfg):
    if not enabled():
        return 0
    validate_scope(cfg, type(model).__name__)
    if model.training:
        raise ValueError('Draft linear experiment requires inference mode')
    layers = [(name, layer) for name, layer in model.named_modules()
              if name.endswith('.mlp.down_proj')]
    if len(layers) != 16:
        raise ValueError('Expected exactly16 Draft down projections')
    methods = []
    for name, layer in layers:
        weight = layer.weight
        method = layer.quant_method
        if (weight.shape != (2048, 8192) or not weight.is_cuda
                or weight.dtype != torch.bfloat16 or not weight.is_contiguous()
                or type(method).__name__ != 'UnquantizedLinearMethod'
                or getattr(layer, '_cdb_down_pad32_installed', False)):
            raise ValueError(f'Unsupported Draft projection storage/method: {name}')
        methods.append(method)
    if len({id(m) for m in methods}) != 16:
        raise ValueError('Draft projection methods unexpectedly shared')
    # Validate the entire model before changing any module. Installation adds
    # no weight allocation or mutation, and happens before all Graph captures.
    owner = threading.get_ident()
    audit = [0] if os.getenv('CDB_AV_DRAFT_LINEAR_AUDIT', '0') == '1' else None
    for (_, layer), method in zip(layers, methods):
        method.apply = wrap_projection(method.apply, owner, audit)
        layer._cdb_down_pad32_installed = True
    logger.info('[CDB_DRAFT_DOWN_PAD32] %s', json.dumps(dict(
        layers=16, weight_shape=[2048, 8192], live_M=sorted(PADDED_ROWS), gemm_M=32,
        accumulation='fp32', weights_changed=False, model_rows_changed=False)))
    return 16
