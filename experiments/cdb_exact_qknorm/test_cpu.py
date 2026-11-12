# SPDX-License-Identifier: Apache-2.0
import importlib.util
import os
from pathlib import Path
import pytest
import torch

def _module():
    path = Path(os.environ['CDB_EXACT_BUILD_DIR']) / 'cdb_exact_qknorm_20261113.so'
    spec = importlib.util.spec_from_file_location('cdb_exact_qknorm_20261113',path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

@pytest.mark.skipif(not os.environ.get('CDB_EXACT_BUILD_DIR'), reason='explicit experimental binary required')
@pytest.mark.parametrize('rows',[0,1])
def test_cpu_inputs_rejected_before_cuda_launch(rows):
    module = _module()
    qkv=torch.empty((rows,4096),dtype=torch.bfloat16)
    weight=torch.ones(128,dtype=torch.bfloat16)
    cache=torch.zeros((1024,128),dtype=torch.bfloat16)
    positions=torch.zeros(rows,dtype=torch.int64)
    with pytest.raises(RuntimeError,match='QKV must be CUDA'):
        module.forward(qkv,weight,weight,cache,positions,
            torch.empty((rows,2048),dtype=torch.bfloat16),
            torch.empty((rows,1024),dtype=torch.bfloat16),1e-6,16,8)
