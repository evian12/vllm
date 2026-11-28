# SPDX-License-Identifier: Apache-2.0
"""Reject unsupported strict-CDB requests before they can poison EngineCore."""
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from vllm.sampling_params import SamplingParams
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.spec_decode.cdb_av_config import (
    get_persistent_request_unsupported_reason,
)
from vllm.v1.spec_decode.cdb_av_persistent import PersistentDraftQueue


@pytest.mark.parametrize("changes,reason", [
    ({}, None),
    ({"sampling_params": None}, "missing_sampling_params"),
    ({"sampling_params": NS(temperature=.7)}, "non_greedy_sampling"),
    ({"prompt_token_ids": None}, "missing_prompt_token_ids"),
    ({"prompt_embeds": object()}, "prompt_embeds"),
    ({"mm_features": [object()]}, "multimodal_inputs"),
    ({"has_encoder_inputs": True}, "encoder_inputs"),
    ({"lora_request": object()}, "lora"),
    ({"structured_output_request": object()}, "structured_output"),
    ({"sampling_params": NS(temperature=0, prompt_logprobs=0)}, "prompt_logprobs"),
    ({"sampling_params": NS(temperature=0, prompt_logprobs=2)}, "prompt_logprobs"),
])
def test_ingress_and_queue_share_the_existing_request_domain(changes, reason):
    fields=dict(sampling_params=NS(temperature=0, prompt_logprobs=None),
                prompt_token_ids=[1, 2], prompt_embeds=None, mm_features=None,
                has_encoder_inputs=False, lora_request=None,
                structured_output_request=None)
    fields.update(changes)
    assert get_persistent_request_unsupported_reason(**fields) == reason
    assert PersistentDraftQueue.unsupported_reason(NS(**fields)) == reason


@pytest.fixture
def processor(monkeypatch):
    # Exercise the real process_inputs boundary without loading a tokenizer/model.
    p=InputProcessor.__new__(InputProcessor)
    p.vllm_config=NS(parallel_config=NS(data_parallel_size=1,
        data_parallel_size_local=1,local_engines_only=False))
    p.model_config=NS(multimodal_config=None,max_model_len=1024)
    p.cache_config=NS(enable_prefix_caching=False)
    p.generation_config_fields={}
    p.input_preprocessor=NS(tokenizer=None,
        preprocess=Mock(return_value={"type":"token","prompt_token_ids":[1,2,3]}),
        get_eos_token_id=lambda:0)
    p._validate_lora=Mock()
    p._validate_params=Mock()
    p._validate_mm_uuids=Mock()
    p._validate_model_inputs=Mock()
    from vllm.platforms import current_platform
    monkeypatch.setattr(current_platform,"validate_request",Mock())
    monkeypatch.setenv("CDB_AV_PERSISTENT","1")
    monkeypatch.setenv("CDB_AV_AR_OUTPUT_AUDIT","strict")
    return p


def test_real_ingress_rejects_non_greedy_before_core_request(processor, monkeypatch):
    constructor=Mock()
    monkeypatch.setattr("vllm.v1.engine.input_processor.EngineCoreRequest",constructor)
    with pytest.raises(ValueError,match="non_greedy_sampling"):
        processor.process_inputs("bad",[1,2,3],SamplingParams(temperature=.7,max_tokens=4))
    constructor.assert_not_called()


def test_rejected_request_does_not_poison_next_supported_request(processor):
    with pytest.raises(ValueError,match="non_greedy_sampling"):
        processor.process_inputs("bad",[1,2,3],SamplingParams(temperature=.7,max_tokens=4))
    request=processor.process_inputs("good",[1,2,3],SamplingParams(temperature=0,max_tokens=16))
    assert request.request_id == "good"
    assert request.sampling_params.temperature == 0
    assert request.prompt_token_ids == [1,2,3]


@pytest.mark.parametrize("persistent,audit",[("0","strict"),("1","off"),("0","off")])
def test_native_and_non_strict_modes_keep_their_existing_behavior(processor,monkeypatch,persistent,audit):
    monkeypatch.setenv("CDB_AV_PERSISTENT",persistent)
    monkeypatch.setenv("CDB_AV_AR_OUTPUT_AUDIT",audit)
    request=processor.process_inputs("native",[1,2,3],SamplingParams(temperature=.7,max_tokens=4))
    assert request.sampling_params.temperature == .7


def test_zero_prompt_logprobs_is_rejected_explicitly(processor):
    with pytest.raises(ValueError,match="prompt_logprobs"):
        processor.process_inputs("prompt-logprobs",[1,2,3],SamplingParams(temperature=0,max_tokens=4,prompt_logprobs=0))
