from pathlib import Path
import json,struct,math
out={}
for label,base in [('target',Path('/disk14T/yypan/models/Llama-3.1-8B-Instruct')),('draft',Path('/disk14T/yypan/CDB-AV-runtime/baseline-comparison-20261113/models/Llama-3.2-1B-Instruct'))]:
 weights={}
 for p in base.glob('*.safetensors'):
  with p.open('rb') as f:
   n=struct.unpack('<Q',f.read(8))[0];head=json.loads(f.read(n))
  for k,v in head.items():
   if k!='__metadata__':weights[k]=v
 cfg=json.loads((base/'config.json').read_text());layer={k:v for k,v in weights.items() if k.startswith('model.layers.')}
 projections={k:v for k,v in layer.items() if any(s in k for s in ['q_proj.weight','k_proj.weight','v_proj.weight','o_proj.weight','gate_proj.weight','up_proj.weight','down_proj.weight'])}
 def size(rows):return sum(v['data_offsets'][1]-v['data_offsets'][0] for v in rows.values())
 out[label]=dict(path=str(base),config={k:cfg.get(k) for k in ['hidden_size','intermediate_size','num_hidden_layers','num_attention_heads','num_key_value_heads','tie_word_embeddings','vocab_size']},projection_count=len(projections),projection_bytes=size(projections),layer_bytes=size(layer),all_checkpoint_bytes=size(weights),layer0={k:dict(shape=v['shape'],dtype=v['dtype'],bytes=v['data_offsets'][1]-v['data_offsets'][0]) for k,v in projections.items() if k.startswith('model.layers.0.')})
print(json.dumps(out,indent=2))
