# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.spec_decode.cdb_av_chain_graph import (
    chain_storage, consume_packet, native_hf_packet,
)


@pytest.mark.parametrize('rows', [1, 2, 3, 4])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
def test_packet_preserves_native_math_and_token_precision(rows, dtype):
    logits = torch.full((rows, 128256), -10., dtype=dtype)
    logits[:, 128255] = 4.
    fp32 = logits.float()
    maximum, tokens = fp32.max(dim=-1)
    confidence = (maximum - torch.logsumexp(fp32, dim=-1)).exp()
    expected = torch.stack((tokens.to(torch.float64), confidence.to(torch.float64)), dim=1)
    packet = native_hf_packet(logits)
    assert torch.equal(packet.view(torch.uint8), expected.view(torch.uint8))
    assert packet[:, 0].tolist() == [128255.] * rows
    assert consume_packet((logits, packet), logits) is packet
    assert consume_packet(None, logits) is None


def test_reject_stale_or_wrong_row_packet():
    logits = torch.zeros(3, 8)
    with pytest.raises(RuntimeError, match='identity or row'):
        consume_packet((logits.clone(), torch.zeros(3, 2)), logits)
    with pytest.raises(RuntimeError, match='identity or row'):
        consume_packet((logits, torch.zeros(2, 2)), logits)


def test_storage_invalidates_replaced_head_and_processor():
    model = SimpleNamespace(lm_head=SimpleNamespace(weight=torch.zeros(8, 4), quant_method=object()),
        logits_processor=SimpleNamespace(scale=1., soft_cap=None, org_vocab_size=8))
    initial = chain_storage(model)
    assert initial == chain_storage(model)
    model.lm_head.weight = torch.ones(8, 4)
    assert initial != chain_storage(model)
    initial = chain_storage(model)
    model.logits_processor.scale = 2.
    assert initial != chain_storage(model)
