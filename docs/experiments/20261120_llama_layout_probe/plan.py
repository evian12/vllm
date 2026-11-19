"""Bounded physical-layout probe; no model/runtime mutation or TPS inference."""
import math


def cases(config):
    h, i = config['hidden_size'], config['intermediate_size']
    d = config.get('head_dim', h // config['num_attention_heads'])
    kv = config['num_key_value_heads'] * d
    if config['model_type'] != 'llama' or h % d:
        raise ValueError('Expected inspected Llama dimensions')
    shapes = [('qkv', h + 2 * kv, h), ('o', h, h),
              ('gate_up', 2 * i, h), ('down', h, i),
              ('lm_head', config['vocab_size'], h)]
    result = []
    for projection, n, k in shapes:
        bank = max(1, math.ceil(256 * 1024**2 / (n * k * 2)))
        if bank > config['num_hidden_layers']:
            raise ValueError('Insufficient independent layer bank')
        for m in [1, 4, 8, 16]:
            result.append(dict(projection=projection, m=m, n=n, k=k,
                               bank=bank, streamed_bytes=bank*n*k*2))
    return result
