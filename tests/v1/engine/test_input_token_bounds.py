# SPDX-License-Identifier: Apache-2.0
"""Reject negative prompt IDs before building an EngineCore request."""
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.engine.input_processor import InputProcessor


@pytest.fixture
def processor(monkeypatch):
    p = InputProcessor.__new__(InputProcessor)
    p.vllm_config = NS(
        parallel_config=NS(
            data_parallel_size=1, data_parallel_size_local=1, local_engines_only=False
        )
    )
    p.model_config = NS(
        multimodal_config=None,
        max_model_len=1024,
        is_multimodal_model=False,
        runner_type="generate",
        get_vocab_size=lambda: 128,
    )
    p.cache_config = NS(enable_prefix_caching=False)
    p.generation_config_fields = {}
    p.input_preprocessor = NS(
        tokenizer=None,
        preprocess=lambda prompt, **kwargs: {
            "type": "token",
            "prompt_token_ids": prompt,
        },
        get_eos_token_id=lambda: 0,
    )
    p._validate_lora = Mock()
    p._validate_params = Mock()
    p._validate_mm_uuids = Mock()
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "validate_request", Mock())
    return p


@pytest.mark.parametrize("ids", [[-1], [1, -1, 2], [1, -1000000]])
@pytest.mark.parametrize("has_tokenizer", [False, True])
@pytest.mark.parametrize("prompt_type", ["encoder", "decoder"])
def test_negative_ids_fail_model_input_validation(
    processor, ids, has_tokenizer, prompt_type
):
    if has_tokenizer:
        processor.input_preprocessor.tokenizer = NS(max_token_id=100)
    with pytest.raises(ValueError, match="negative|non-negative"):
        processor._validate_model_input(
            {"type": "token", "prompt_token_ids": ids}, prompt_type=prompt_type
        )


@pytest.mark.parametrize(
    "persistent,audit", [("1", "strict"), ("1", "off"), ("0", "off")]
)
def test_invalid_request_never_enters_core_and_next_valid_input_works(
    processor, monkeypatch, persistent, audit
):
    monkeypatch.setenv("CDB_AV_PERSISTENT", persistent)
    monkeypatch.setenv("CDB_AV_AR_OUTPUT_AUDIT", audit)
    constructor = Mock()
    with monkeypatch.context() as patch:
        patch.setattr("vllm.v1.engine.input_processor.EngineCoreRequest", constructor)
        with pytest.raises(ValueError, match="negative|non-negative"):
            processor.process_inputs(
                "invalid", [1, -1], SamplingParams(temperature=0, max_tokens=4)
            )
        constructor.assert_not_called()
    request = processor.process_inputs(
        "valid", [0, 1, 2], SamplingParams(temperature=0, max_tokens=4)
    )
    assert request.request_id == "valid"
    assert request.prompt_token_ids == [0, 1, 2]


def test_existing_positive_vocab_union_is_preserved(processor):
    processor.input_preprocessor.tokenizer = NS(max_token_id=140)
    processor._validate_model_input(
        {"type": "token", "prompt_token_ids": [0, 127, 140]}, prompt_type="decoder"
    )
    with pytest.raises(ValueError, match="out of vocabulary"):
        processor._validate_model_input(
            {"type": "token", "prompt_token_ids": [141]}, prompt_type="decoder"
        )


def test_prompt_embeddings_have_no_token_ids(processor):
    processor._validate_model_input(
        {"type": "embeds", "prompt_embeds": torch.zeros(2, 4)}, prompt_type="decoder"
    )


@pytest.mark.parametrize("prompt_type", ["encoder", "decoder"])
@pytest.mark.parametrize("token_id", [128, 2147483647])
def test_upper_bound_without_tokenizer(processor, prompt_type, token_id):
    with pytest.raises(ValueError, match="out of vocabulary"):
        processor._validate_model_input(
            {"type": "token", "prompt_token_ids": [1, token_id]},
            prompt_type=prompt_type,
        )


@pytest.mark.parametrize("prompt_type", ["encoder", "decoder"])
def test_model_vocab_boundary_without_tokenizer(processor, prompt_type):
    processor._validate_model_input(
        {"type": "token", "prompt_token_ids": [0, 127]},
        prompt_type=prompt_type,
    )


@pytest.mark.parametrize(
    "persistent,audit", [("1", "strict"), ("1", "off"), ("0", "off")]
)
def test_oversized_request_never_enters_core_without_tokenizer(
    processor, monkeypatch, persistent, audit
):
    monkeypatch.setenv("CDB_AV_PERSISTENT", persistent)
    monkeypatch.setenv("CDB_AV_AR_OUTPUT_AUDIT", audit)
    constructor = Mock()
    with monkeypatch.context() as patch:
        patch.setattr("vllm.v1.engine.input_processor.EngineCoreRequest", constructor)
        with pytest.raises(ValueError, match="out of vocabulary"):
            processor.process_inputs(
                "invalid", [1, 128], SamplingParams(temperature=0, max_tokens=4)
            )
        constructor.assert_not_called()
    request = processor.process_inputs(
        "valid", [0, 1, 2], SamplingParams(temperature=0, max_tokens=4)
    )
    assert request.prompt_token_ids == [0, 1, 2]
