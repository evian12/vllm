from pathlib import Path
import json
r=Path('outputs/kernel_audit');a=json.loads((r/'STAGE_SHAPE_KERNELS.json').read_text());w=json.loads((r/'MODEL_WEIGHT_HEADERS.json').read_text())
result={}
for stage,model,calls in [('target_forward','target',64),('draft_forward','draft',151)]:
 g=a['stage_kernel_totals'][stage];b=w[model]['projection_bytes'];time=g['gemm_us']/calls
 result[stage]=dict(calls=calls,projection_bytes=b,mean_backbone_ms=g['total_us']/calls/1000,mean_gemm_ms=time/1000,projection_byte_equivalent_GBs=b/(time*1e-6)/1e9,gemm_fraction=g['gemm_us']/g['total_us'])
for stage,model,calls,h in [('target_logits','target',64,4096),('draft_logits','draft',151,2048)]:
 g=a['stage_kernel_totals'][stage];b=128256*h*2;time=g['gemm_us']/calls
 result[stage]=dict(head_bytes=b,mean_gemm_ms=time/1000,weight_byte_equivalent_GBs=b/(time*1e-6)/1e9)
result['limits']='Weight byte equivalents assume one logical weight pass per invocation; they are NOT measured DRAM traffic, bandwidth saturation proof, or standalone throughput. Shapes vary and profiler perturbs execution.'
(r/'WEIGHT_TIME_EQUIVALENTS.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
