# SPDX-License-Identifier: Apache-2.0
"""Audit-only numerical metrics; imports torch, never the serving/model stack.

Byte-exact checks remain in the caller and use the original storage dtype.
Only numerical reductions use float64 copies; KV inputs are never mutated.
"""
import math

import torch


def _validate_pair(actual, reference, ndim):
    if not isinstance(actual, torch.Tensor) or not isinstance(reference, torch.Tensor):
        raise TypeError('KV audit requires tensors')
    if actual.ndim != ndim or reference.ndim != ndim:
        raise ValueError(f'KV audit requires rank {ndim}, got {actual.ndim}/{reference.ndim}')
    if actual.shape != reference.shape:
        raise ValueError(f'KV audit shape mismatch: {actual.shape} vs {reference.shape}')
    if actual.dtype != reference.dtype or actual.device != reference.device:
        raise ValueError('KV audit dtype/device mismatch; no implicit conversion or broadcasting')
    if not actual.is_floating_point() or not reference.is_floating_point():
        raise ValueError('KV numerical metrics require real floating-point tensors')


def tensor_error_metrics(actual, reference):
    """Compare a single K or V interval [tokens, heads, head_dim].

    None means undefined/unavailable, never zero. Zero reference norm leaves
    max_abs and rms valid, but rel_l2 undefined (including identical zero KV).
    Nonfinite input or reductions make all three metrics unavailable.
    Shape/layout errors raise rather than silently broadcasting/truncating.
    """
    _validate_pair(actual, reference, 3)
    if actual.shape[1] == 0 or actual.shape[2] == 0:
        raise ValueError('KV head dimensions must be nonempty')
    unavailable = dict(max_abs=None, rms=None, rel_l2=None)
    if actual.shape[0] == 0:
        return dict(**unavailable, error_status='empty')
    if not bool(torch.isfinite(actual).all()) or not bool(torch.isfinite(reference).all()):
        return dict(**unavailable, error_status='nonfinite_input')
    # detach/to return reduction inputs, not a cast of the underlying KV cache.
    aa = actual.detach().to(dtype=torch.float64)
    bb = reference.detach().to(dtype=torch.float64)
    delta = aa - bb
    maximum = float(delta.abs().max())
    rms = float(delta.square().mean().sqrt())
    difference_norm = float(torch.linalg.vector_norm(delta))
    reference_norm = float(torch.linalg.vector_norm(bb))
    if not all(math.isfinite(v) for v in (maximum, rms, difference_norm, reference_norm)):
        return dict(**unavailable, error_status='nonfinite_statistics')
    if reference_norm == 0.0:
        return dict(max_abs=maximum, rms=rms, rel_l2=None,
                    error_status='zero_reference_norm')
    relative = difference_norm / reference_norm
    if not math.isfinite(relative):
        return dict(max_abs=maximum, rms=rms, rel_l2=None,
                    error_status='nonfinite_relative_error')
    return dict(max_abs=maximum, rms=rms, rel_l2=relative, error_status='ok')


def kv_segment_errors(actual, reference, start, end):
    """Return two dicts (K, V) for gathered [2, end, heads, head_dim] KV.

    old_prefix is actual[:, 0:start] against reference[:, 0:start].
    new_suffix is actual[:, start:end] against reference[:, start:end].
    Reject a physical block-cache tensor: caller must gather token positions
    first. No implicit clamp, broadcasting, or flattening across K/V.
    """
    _validate_pair(actual, reference, 4)
    if actual.shape[0] != 2:
        raise ValueError('Expected leading K/V axis of size 2')
    if type(start) is not int or type(end) is not int or not 0 <= start <= end:
        raise ValueError('Invalid KV audit interval')
    if actual.shape[1] != end:
        raise ValueError('Gathered token extent must equal end')
    result = []
    for kv_index in range(2):
        item = {}
        for name, lo, hi in (('old_prefix', 0, start), ('new_suffix', start, end)):
            metrics = tensor_error_metrics(actual[kv_index, lo:hi], reference[kv_index, lo:hi])
            item.update({f'{name}_{key}': value for key, value in metrics.items()})
        result.append(item)
    return result
