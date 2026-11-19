import pytest
from plan import cases


CFG = dict(model_type='llama', hidden_size=4096, intermediate_size=14336,
           num_attention_heads=32, num_key_value_heads=8,
           num_hidden_layers=32, vocab_size=128256)


def test_actual_target_shapes_and_streaming_bank():
    rows = cases(CFG)
    assert len(rows) == 20
    assert {(x['projection'], x['n'], x['k']) for x in rows} == {
        ('qkv',6144,4096), ('o',4096,4096), ('gate_up',28672,4096),
        ('down',4096,14336), ('lm_head',128256,4096)}
    assert all(x['streamed_bytes'] >= 256*1024**2 for x in rows)
    assert all(1 <= x['bank'] <= 32 for x in rows)


def test_explicit_head_dimension_uses_actual_kv_heads():
    config = dict(CFG, head_dim=64, hidden_size=2048,
                  intermediate_size=8192, num_hidden_layers=32)
    assert next(x for x in cases(config) if x['projection']=='qkv')['n']==3072


def test_rejects_uninspected_family_or_insufficient_bank():
    with pytest.raises(ValueError): cases(dict(CFG, model_type='qwen3'))
    with pytest.raises(ValueError): cases(dict(CFG, num_hidden_layers=1))
