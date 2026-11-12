# SPDX-License-Identifier: Apache-2.0
"""Opt-in experimental exact-BF16 QK norm/RoPE; static Qwen3 CDB only."""
import functools
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import torch
from vllm.logger import init_logger

logger = init_logger(__name__)
ROOT = Path(__file__).resolve().parents[3]
NAME = 'cdb_exact_qknorm_20261113'
_EXECUTED_ROLES = set()

def validate_scope(*, dtype, eager, asynchronous, prefix_cache, lora, quantized,
                   parallel_sizes, capacity, batch_invariant=False):
    if (dtype != torch.bfloat16 or not eager or asynchronous or prefix_cache
            or lora or quantized or batch_invariant or capacity != 4 or len(parallel_sizes) != 4
            or any(type(x) is not int or x != 1 for x in parallel_sizes)):
        raise ValueError('Exact QK requires static eager BF16 CDB,capacity4,TP/PP/DP/DCP1,no async/prefix/LoRA/quantization')

def validate_receipt(path):
    path = Path(path).resolve()
    receipt = json.loads(path.with_suffix('.json').read_text())
    if receipt.get('format') != 1 or receipt.get('module') != NAME:
        raise ValueError('Invalid exact QK binary receipt')
    if receipt.get('torch') != torch.__version__:
        raise ValueError('Exact QK Torch version changed')
    if hashlib.sha256(path.read_bytes()).hexdigest() != receipt.get('binary_sha256'):
        raise ValueError('Exact QK binary hash mismatch')
    required = ('experiments/cdb_exact_qknorm/binding.cpp',
                'experiments/cdb_exact_qknorm/kernel.cu',
                'csrc/cub_helpers.h', 'csrc/quantization/vectorization_utils.cuh',
                'csrc/quantization/vectorization.cuh')
    sources = receipt.get('sources', {})
    if set(sources) != set(required):
        raise ValueError('Incomplete exact QK source receipt')
    for name in required:
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != sources[name]:
            raise ValueError('Exact QK source hash mismatch: ' + name)
    return path, receipt

@functools.lru_cache(maxsize=1)
def _load(path):
    path, receipt = validate_receipt(path)
    if torch.cuda.get_device_capability() != (8, 9):
        raise ValueError('Exact QK experimental binary requires SM89')
    spec = importlib.util.spec_from_file_location(NAME, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    logger.info('[CDB_EXACT_QK] %s', json.dumps(dict(
        event='runtime', pid=os.getpid(), source=__file__, binary=str(path),
        binary_sha256=receipt['binary_sha256'])))
    return module

def native_qk(attention, positions, qkv):
    q, k, _ = qkv.split([attention.q_size,attention.kv_size,attention.kv_size], dim=-1)
    qshape, kshape = q.shape, k.shape
    q = attention.q_norm(q.view(*q.shape[:-1], attention.num_heads,128)).view(qshape)
    k = attention.k_norm(k.view(*k.shape[:-1], attention.num_kv_heads,128)).view(kshape)
    return attention.rotary_emb(positions,q,k)

class ExactQK:
    def __init__(self, kernel, role, prefix, validate=False, limit=8):
        self.kernel=kernel;self.role=role;self.prefix=prefix
        self.validate=validate;self.limit=limit;self.samples=0
        self.fallback_seen=set()

    def __call__(self, att, positions, qkv):
        if att.training:
            raise RuntimeError('Exact QK is inference only')
        if qkv.ndim != 2 or qkv.shape[0] > 1024 or not qkv.is_contiguous():
            reason='row_or_layout_domain'
            if reason not in self.fallback_seen:
                self.fallback_seen.add(reason)
                logger.info('[CDB_EXACT_QK] %s',json.dumps(dict(
                    event='fallback',role=self.role,prefix=self.prefix,
                    reason=reason,shape=list(qkv.shape))))
            return native_qk(att,positions,qkv)
        att.rotary_emb._match_cos_sin_cache_dtype(qkv)
        q=qkv.new_empty((qkv.shape[0],att.q_size))
        k=qkv.new_empty((qkv.shape[0],att.kv_size))
        self.kernel.forward(qkv,att.q_norm.weight,att.k_norm.weight,
            att.rotary_emb.cos_sin_cache,positions,q,k,
            att.q_norm.variance_epsilon,att.num_heads,att.num_kv_heads)
        if self.role not in _EXECUTED_ROLES:
            _EXECUTED_ROLES.add(self.role)
            logger.info('[CDB_EXACT_QK] %s',json.dumps(dict(
                event='forward_enqueued',role=self.role,prefix=self.prefix,
                rows=qkv.shape[0],device=str(qkv.device))))
        if self.validate and self.samples < self.limit and not torch.cuda.is_current_stream_capturing():
            rq,rk=native_qk(att,positions,qkv)
            exact=torch.equal(q.view(torch.uint8),rq.view(torch.uint8)) and torch.equal(k.view(torch.uint8),rk.view(torch.uint8))
            self.samples+=1
            logger.info('[CDB_EXACT_QK] %s',json.dumps(dict(
                event='validate',role=self.role,prefix=self.prefix,
                sample=self.samples,rows=qkv.shape[0],Q_shape=list(q.shape),
                K_shape=list(k.shape),positions=positions.cpu().tolist(),
                byte_exact=exact)))
            if not exact:
                raise RuntimeError('Exact QK model/native byte mismatch')
        return q,k

def make_exact_qknorm(attention, prefix, quant_config):
    flag=os.getenv('CDB_AV_EXACT_QKNORM','0')
    if flag not in ('0','1'):
        raise ValueError('Invalid exact QK switch')
    if flag=='0':
        return None
    if any(os.getenv(k,'0')!='1' for k in ('CDB_AV_REFILL','CDB_AV_PERSISTENT')):
        raise ValueError('Exact QK requires CDB persistent refill')
    from vllm.config import get_current_vllm_config
    cfg=get_current_vllm_config();pc=cfg.parallel_config
    from vllm.model_executor.layers.batch_invariant import vllm_is_batch_invariant
    validate_scope(dtype=cfg.model_config.dtype,eager=cfg.model_config.enforce_eager,
        asynchronous=cfg.scheduler_config.async_scheduling,
        prefix_cache=cfg.cache_config.enable_prefix_caching,
        lora=cfg.lora_config is not None,
        quantized=quant_config is not None or cfg.model_config.quantization is not None,
        parallel_sizes=(pc.tensor_parallel_size,pc.pipeline_parallel_size,
            pc.data_parallel_size,pc.decode_context_parallel_size),
        capacity=cfg.scheduler_config.max_num_seqs,batch_invariant=vllm_is_batch_invariant())
    if (attention.head_dim != 128 or attention.num_kv_heads != 8
            or (attention.hidden_size,attention.num_heads) not in ((2048,16),(4096,32))
            or type(attention.rotary_emb).__name__!='RotaryEmbedding'
            or attention.rotary_emb.rotary_dim!=128
            or not attention.rotary_emb.is_neox_style
            or attention.rotary_emb.use_flashinfer
            or attention.rotary_emb._forward_method.__name__!='forward_cuda'
            or any(not norm.has_weight or norm.variance_size_override is not None
                   or norm.weight.dtype!=torch.bfloat16
                   or norm._forward_method.__name__!='forward_cuda'
                   for norm in (attention.q_norm,attention.k_norm))
            or attention.q_norm.variance_epsilon!=attention.k_norm.variance_epsilon):
        raise ValueError('Unsupported exact QK head/norm/RoPE domain')
    role='draft' if attention.hidden_size==2048 else 'target'
    mode=os.getenv('CDB_AV_EXACT_QKNORM_MODE','both')
    if mode not in ('both','draft','target'):
        raise ValueError('Invalid exact QK model selection')
    if mode not in ('both',role):
        return None
    validate=os.getenv('CDB_AV_EXACT_QK_VALIDATE','0')=='1'
    limit=int(os.getenv('CDB_AV_EXACT_QK_VALIDATE_SAMPLES','8'))
    if not 1<=limit<=32:
        raise ValueError('Invalid exact QK audit limit')
    path=os.environ.get('CDB_AV_EXACT_QK_LIB')
    if not path:
        raise ValueError('CDB_AV_EXACT_QK_LIB is required')
    return ExactQK(_load(path),role,prefix,validate,limit)
