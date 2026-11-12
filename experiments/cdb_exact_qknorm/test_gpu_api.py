# SPDX-License-Identifier: Apache-2.0
"""Explicit isolated-GPU negative API tests; no model/HTTP load."""
import importlib.util
import os
import pytest
import torch

@pytest.fixture
def arguments():
    if os.getenv('CDB_EXACT_NEGATIVE_GPU') != '1':
        pytest.skip('explicit isolated GPU reservation required')
    assert os.environ['CUDA_VISIBLE_DEVICES']=='3'
    device='cuda'
    qkv=torch.randn(2,4096,device=device,dtype=torch.bfloat16)
    qw=torch.ones(128,device=device,dtype=torch.bfloat16)
    kw=qw.clone()
    cache=torch.zeros(1024,128,device=device,dtype=torch.bfloat16)
    positions=torch.tensor([0,1023],device=device,dtype=torch.int64)
    qo=torch.empty(2,2048,device=device,dtype=torch.bfloat16)
    ko=torch.empty(2,1024,device=device,dtype=torch.bfloat16)
    return [qkv,qw,kw,cache,positions,qo,ko,1e-6,16,8]

@pytest.fixture
def extension():
    name='cdb_exact_qknorm_20261113'
    path=os.path.join(os.environ['CDB_EXACT_BUILD_DIR'],name+'.so')
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

@pytest.mark.parametrize('index',[0,1,2])
def test_vector_inputs_require_16_byte_alignment(arguments,extension,index):
    old=arguments[index]
    shifted=torch.empty(old.numel()+1,device=old.device,dtype=old.dtype)[1:].view(old.shape)
    shifted.copy_(old)
    assert shifted.is_contiguous() and shifted.data_ptr()%16 != 0
    arguments[index]=shifted
    with pytest.raises(RuntimeError,match='16-byte aligned'):
        extension.forward(*arguments)
    torch.cuda.synchronize()

@pytest.mark.parametrize('case,match',[
 ('dtype','contiguous BF16'),('layout','contiguous BF16'),
 ('positions_dtype','Invalid positions'),('positions_rows','Invalid QKV shape'),
 ('weights_shape','Invalid weight/token shape'),('cache_shape','NeoX cache'),
 ('output_shape','Invalid output shape'),('alias','Output aliases input'),
 ('overlap','Outputs overlap'),('epsilon','Invalid epsilon'),('heads','Unsupported head counts'),
])
def test_invalid_arguments_fail_before_launch(arguments,extension,case,match):
    if case=='dtype':arguments[0]=arguments[0].float()
    elif case=='layout':arguments[0]=torch.empty(4096,2,device='cuda',dtype=torch.bfloat16).t()
    elif case=='positions_dtype':arguments[4]=arguments[4].int()
    elif case=='positions_rows':arguments[4]=arguments[4][:1]
    elif case=='weights_shape':arguments[1]=arguments[1][:127]
    elif case=='cache_shape':arguments[3]=arguments[3][:,:64].contiguous()
    elif case=='output_shape':arguments[5]=arguments[5][:1]
    elif case=='alias':arguments[5]=arguments[0].view(-1)[:4096].view(2,2048)
    elif case=='overlap':arguments[6]=arguments[5].view(-1)[:2048].view(2,1024)
    elif case=='epsilon':arguments[7]=float('nan')
    elif case=='heads':arguments[8]=33
    with pytest.raises(RuntimeError,match=match):extension.forward(*arguments)
    torch.cuda.synchronize()
