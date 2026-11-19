"""Read actual configs and safetensor headers only; never load/modify weights."""
from pathlib import Path
import hashlib
import json
import struct

MODELS={
 'target':Path('/disk14T/yypan/models/Llama-3.1-8B-Instruct'),
 'draft':Path('/disk14T/yypan/CDB-AV-runtime/baseline-comparison-20261113/models/Llama-3.2-1B-Instruct')}
R=Path(__file__).resolve().parent
facts={}
for kind,folder in MODELS.items():
    raw_config=(folder/'config.json').read_bytes();cfg=json.loads(raw_config)
    index=folder/'model.safetensors.index.json'
    mapping=json.loads(index.read_text())['weight_map'] if index.exists() else None
    singles=list(folder.glob('*.safetensors'));headers={}
    def header(path):
        if path not in headers:
            with path.open('rb') as f:
                size=struct.unpack('<Q',f.read(8))[0];assert 1<size<16*1024*1024
                data=f.read(size);assert len(data)==size
            headers[path]=(json.loads(data),hashlib.sha256(data).hexdigest())
        return headers[path][0]
    names=['model.layers.0.self_attn.'+n+'.weight' for n in ['q_proj','k_proj','v_proj','o_proj']]
    names+=['model.layers.0.mlp.'+n+'.weight' for n in ['gate_proj','up_proj','down_proj']]
    names+=['lm_head.weight','model.embed_tokens.weight']
    tensors={}
    for requested in names:
        name=requested
        available=mapping if mapping is not None else header(singles[0])
        if name not in available:
            # llama.py config branch explicitly ties this weight and skips
            # lm_head.* during loading. Do not invent a separate raw tensor.
            assert name=='lm_head.weight' and cfg.get('tie_word_embeddings') is True
            name='model.embed_tokens.weight'
        path=folder/mapping[name] if mapping else singles[0]
        item=header(path)[name]
        tensors[requested]=dict(shape=item['shape'],dtype=item['dtype'],raw_key=name,
                                logical_tied_alias=requested!=name)
    H=cfg['hidden_size'];I=cfg['intermediate_size'];L=cfg['num_hidden_layers']
    heads=cfg['num_attention_heads'];kv=cfg['num_key_value_heads'];D=cfg.get('head_dim',H//heads)
    assert (H,I,L,heads,kv,D)==((4096,14336,32,32,8,128) if kind=='target' else (2048,8192,16,32,8,64))
    for n,shape in {'q_proj':[H,H],'k_proj':[kv*D,H],'v_proj':[kv*D,H],'o_proj':[H,H]}.items():
        assert tensors['model.layers.0.self_attn.'+n+'.weight']['shape']==shape
    for n,shape in {'gate_proj':[I,H],'up_proj':[I,H],'down_proj':[H,I]}.items():
        assert tensors['model.layers.0.mlp.'+n+'.weight']['shape']==shape
    assert tensors['lm_head.weight']['shape']==[cfg['vocab_size'],H]
    assert all(t['dtype']=='BF16' for t in tensors.values())
    assert raw_config==(folder/'config.json').read_bytes()
    facts[kind]=dict(path=str(folder),config_sha256=hashlib.sha256(raw_config).hexdigest(),
                     dimensions=dict(H=H,I=I,layers=L,heads=heads,kv_heads=kv,head_dim=D,vocab=cfg['vocab_size']),
                     tie_word_embeddings=cfg.get('tie_word_embeddings'),tensors=tensors,
                     safetensor_header_sha256={str(p):h[1] for p,h in headers.items()},
                     weights_data_checksummed=False,runtime_alias_pointer_probed=False)
(R/'ACTUAL_MODEL_HEADERS.json').write_text(json.dumps(facts,indent=2)+'\n')
print(json.dumps({k:dict(dimensions=v['dimensions'],tie_word_embeddings=v['tie_word_embeddings']) for k,v in facts.items()},indent=2))
