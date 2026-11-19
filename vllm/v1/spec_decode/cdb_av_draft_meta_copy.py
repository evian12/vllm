# SPDX-License-Identifier: Apache-2.0
"""Opt-in integer-only copy into existing Draft Graph metadata buffers.

No request, context, KV lease, model input or capture is owned here. All writes
are to the same four destinations as the original zero_/copy_ sequence.
"""
import json
import os

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)
FIELDS = ('query_start_loc', 'seq_lens', 'block_table_tensor', 'slot_mapping')


def flag(name, env=None):
    value = (os.environ if env is None else env).get(name, '0')
    if value not in ('0', '1'):
        raise ValueError(f'{name} must be 0 or 1')
    return value == '1'


def layout(source, destination):
    """Check all copy extents before launching any GPU write."""
    nr, n = source.num_reqs, source.num_actual_tokens
    if not (type(nr) is int and type(n) is int
            and 1 <= nr <= 4 and nr <= n <= 32
            and destination.num_reqs == nr
            and destination.num_actual_tokens == n):
        raise ValueError('Unsupported Draft metadata row counts')
    if (len(source.block_table_tensor.shape) != 2
            or len(destination.block_table_tensor.shape) != 2):
        raise ValueError('Draft metadata block tables must be matrices')
    sw = source.block_table_tensor.shape[1]
    dw = destination.block_table_tensor.shape[1]
    if not 1 <= sw <= dw <= 1024:
        raise ValueError('Invalid Draft metadata block-table widths')
    shapes = [(nr+1,), (nr,), (nr, sw), (n,)]
    target_shapes = [(nr+1,), (nr,), (nr, dw), (n,)]
    dtypes = [torch.int32, torch.int32, torch.int32, torch.int64]
    device = source.slot_mapping.device
    for name, shape, target_shape, dtype in zip(FIELDS, shapes, target_shapes, dtypes):
        a, b = getattr(source, name), getattr(destination, name)
        if (tuple(a.shape) != shape or tuple(b.shape) != target_shape
                or a.dtype != dtype or b.dtype != dtype
                or not a.is_cuda or not b.is_cuda
                or a.device != device or b.device != device
                or not a.is_contiguous() or not b.is_contiguous()):
            raise ValueError(f'Unsupported Draft metadata tensor: {name}')
    return nr, n, sw, dw


@triton.jit(do_not_specialize=['nr', 'n', 'sw', 'dw'])
def _copy_metadata_kernel(query, seq, table, slots, out_query, out_seq,
                          out_table, out_slots, nr, n, sw, dw,
                          BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    q = tl.load(query+i, i < nr+1, other=0)
    tl.store(out_query+i, q, i < nr+1)
    s = tl.load(seq+i, i < nr, other=0)
    tl.store(out_seq+i, s, i < nr)
    row, col = i // dw, i % dw
    value = tl.load(table + row*sw + col,
                    (i < nr*dw) & (col < sw), other=0)
    tl.store(out_table+i, value, i < nr*dw)
    slot = tl.load(slots+i, i < n, other=0)
    tl.store(out_slots+i, slot, i < n)


class DraftMetadataCopier:
    def __init__(self):
        self.audit = flag('CDB_AV_DRAFT_META_COPY_VALIDATE')
        self.calls = 0

    def __call__(self, source, destination):
        nr, n, sw, dw = layout(source, destination)
        refs = None
        if self.audit and self.calls < 32:
            # Independent original operations, extra work only in diagnostics.
            refs = {name: torch.empty_like(getattr(destination, name)) for name in FIELDS}
            refs['query_start_loc'].copy_(source.query_start_loc)
            refs['seq_lens'].copy_(source.seq_lens)
            refs['block_table_tensor'].zero_()
            refs['block_table_tensor'][:, :sw].copy_(source.block_table_tensor)
            refs['slot_mapping'].copy_(source.slot_mapping)
        _copy_metadata_kernel[(1,)](
            *(getattr(source, name) for name in FIELDS),
            *(getattr(destination, name) for name in FIELDS),
            nr, n, sw, dw, BLOCK=triton.next_power_of_2(max(nr*dw, n, nr+1)),
            num_warps=4,
        )
        self.calls += 1
        if refs is not None:
            checks = {name: torch.equal(getattr(destination, name), refs[name])
                      for name in FIELDS}
            logger.info('[CDB_DRAFT_META_COPY_VALIDATE] %s', json.dumps(dict(
                call=self.calls, requests=nr, tokens=n, source_width=sw,
                destination_width=dw, checks=checks)))
            if not all(checks.values()):
                raise RuntimeError('Fused/native Draft Graph metadata mismatch')
        if self.calls == 1 or self.calls % 1024 == 0:
            logger.info('[CDB_DRAFT_META_COPY] %s', json.dumps(dict(
                source=__file__, calls=self.calls, metadata_kernel_calls=self.calls,
                integer_copy_only=True, extra_model_rows=0)))


def get_metadata_copier(proposer, builder):
    if not flag('CDB_AV_DRAFT_META_COPY'):
        return None
    from vllm.v1.spec_decode.cdb_av_inputs import validate_staging_scope
    from vllm.v1.spec_decode.cdb_av_config import validate_ragged_graph_backend
    validate_staging_scope(proposer.vllm_config)
    validate_ragged_graph_backend(8, builder)
    if (type(proposer.model).__name__ != 'LlamaForCausalLM'
            or proposer.vllm_config.model_config.quantization is not None
            or not flag('CDB_AV_PINNED_INPUTS')
            or not flag('CDB_AV_DRAFT_DECODE_GRAPH')):
        raise ValueError('Fused Draft metadata requires staged unquantized Llama Graphs')
    return DraftMetadataCopier()
