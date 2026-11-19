# SPDX-License-Identifier: Apache-2.0
"""Scope tests only; independent Llama GPU verification remains mandatory."""
import pytest
import torch

from vllm.v1.spec_decode.cdb_av_target_graph import llama_graph_opt_in, validate_scope


def args():
    return dict(persistent=True, refill=True, eager=True, async_scheduling=False,
                prefix_caching=False, lora=False, model='LlamaForCausalLM',
                dtype=torch.bfloat16, quantization=None, parallel_sizes=(1, 1, 1, 1))


def test_llama_stays_disabled_without_explicit_opt_in():
    assert not llama_graph_opt_in({})
    with pytest.raises(ValueError, match='Target full Graph'):
        validate_scope(**args())
    validate_scope(**args(), allow_llama=True)


@pytest.mark.parametrize('value,expected', [('0', False), ('1', True)])
def test_flag_is_explicit(value, expected):
    assert llama_graph_opt_in({'CDB_AV_TARGET_GRAPH_LLAMA': value}) is expected


@pytest.mark.parametrize('value', ['', 'true', '2', 'false'])
def test_invalid_flag_is_rejected(value):
    with pytest.raises(ValueError, match='CDB_AV_TARGET_GRAPH_LLAMA'):
        llama_graph_opt_in({'CDB_AV_TARGET_GRAPH_LLAMA': value})


@pytest.mark.parametrize('change', [
    dict(async_scheduling=True), dict(prefix_caching=True), dict(lora=True),
    dict(eager=False), dict(persistent=False), dict(refill=False),
    dict(dtype=torch.float16), dict(quantization='fp8'),
    dict(parallel_sizes=(2, 1, 1, 1)), dict(model='MistralForCausalLM'),
    dict(allow_llama='1'),
])
def test_llama_does_not_bypass_other_safety_restrictions(change):
    fields = dict(args(), allow_llama=True)
    fields.update(change)
    with pytest.raises(ValueError, match='Target full Graph'):
        validate_scope(**fields)


def test_qwen_scope_is_unchanged():
    fields = args()
    fields['model'] = 'Qwen3ForCausalLM'
    validate_scope(**fields)
    validate_scope(**fields, allow_llama=True)
