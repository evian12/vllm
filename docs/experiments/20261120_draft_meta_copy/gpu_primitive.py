"""Actual CUDA integer-copy check; NOT a model or performance acceptance test."""
from pathlib import Path
from types import SimpleNamespace as NS
import hashlib
import json
import os
import subprocess

import torch
from vllm.v1.spec_decode.cdb_av_draft_meta_copy import DraftMetadataCopier, FIELDS
import vllm.v1.spec_decode.cdb_av_draft_meta_copy as module

R=Path(__file__).resolve().parent
assert os.environ['CUDA_VISIBLE_DEVICES']=='3'
assert os.environ.get('CDB_AV_DRAFT_META_COPY_VALIDATE','0')=='0'
torch.manual_seed(42)
copy=DraftMetadataCopier()
retained=[]
records=[]
for nr in range(1,5):
    for q in [1,2,4,8]:
        for width,dw in [(1,1),(1,2),(2,2),(1,17),(17,17),
                         (1,64),(17,64),(64,64),(17,1024),(1024,1024)]:
            qlens=[max(1,q-i%2) if q in (2,4) else q for i in range(nr)]
            starts=[0]
            for size in qlens:starts.append(starts[-1]+size)
            n=starts[-1]
            src=NS(num_reqs=nr,num_actual_tokens=n,
                   query_start_loc=torch.tensor(starts,dtype=torch.int32,device='cuda'),
                   seq_lens=torch.arange(100,100+nr,dtype=torch.int32,device='cuda'),
                   block_table_tensor=torch.arange(nr*width,dtype=torch.int32,device='cuda').reshape(nr,width),
                   slot_mapping=torch.arange(n,dtype=torch.int64,device='cuda')*2**40-1)
            dst=NS(num_reqs=nr,num_actual_tokens=n,
                   query_start_loc=torch.full((nr+1,),-999,dtype=torch.int32,device='cuda'),
                   seq_lens=torch.full((nr,),-999,dtype=torch.int32,device='cuda'),
                   block_table_tensor=torch.full((nr,dw),-999,dtype=torch.int32,device='cuda'),
                   slot_mapping=torch.full((n,),-999,dtype=torch.int64,device='cuda'))
            # This is the independent original zero_/copy_ chain.
            ref={name:torch.empty_like(getattr(dst,name)) for name in FIELDS}
            ref['query_start_loc'].copy_(src.query_start_loc)
            ref['seq_lens'].copy_(src.seq_lens)
            ref['block_table_tensor'].zero_()
            ref['block_table_tensor'][:,:width].copy_(src.block_table_tensor)
            ref['slot_mapping'].copy_(src.slot_mapping)
            source_before={name:getattr(src,name).clone() for name in FIELDS}
            copy(src,dst)
            retained.append((src,dst,ref,source_before))
            records.append(dict(requests=nr,tokens=n,source_width=width,destination_width=dw))
torch.cuda.synchronize()
for src,dst,ref,before in retained:
    assert all(torch.equal(getattr(dst,k),ref[k]) and getattr(dst,k).dtype==ref[k].dtype for k in FIELDS)
    assert all(torch.equal(getattr(src,k),before[k]) for k in FIELDS)
record=dict(cases=len(records),all_four_fields_byte_exact=True,source_unchanged=True,
            retained_destinations=len(retained),records=records,torch=torch.__version__,
            cuda=torch.version.cuda,source=module.__file__,
            source_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
            commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            scope='Integer metadata primitive only. No model/KV/throughput acceptance claim.')
(R/'GPU_PRIMITIVE.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:v for k,v in record.items() if k!='records'}),flush=True)
