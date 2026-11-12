# SPDX-License-Identifier: Apache-2.0
# Build only the isolated experimental operator; no installed package changes.
import importlib.util
import json
import hashlib
import os
import subprocess
import sysconfig
from pathlib import Path
import torch
from torch.utils.cpp_extension import include_paths, library_paths, _get_pybind11_abi_build_flags

SOURCE = Path(__file__).resolve().parent
NAME = 'cdb_exact_qknorm_20261113'

def module_path():
    return Path(os.environ['CDB_EXACT_BUILD_DIR']) / (NAME + '.so')

def load_built():
    path = module_path()
    spec = importlib.util.spec_from_file_location(NAME, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def build():
    directory = module_path().parent
    directory.mkdir(parents=True, exist_ok=True)
    cuda = Path(os.environ['CUDA_HOME'])
    includes = [*include_paths(device_type='cuda'),
                sysconfig.get_path('include'), str(SOURCE.parents[1]/'csrc')]
    inc = [arg for path in includes for arg in ('-I', path)]
    flags = ['-O3','-std=c++17','-fPIC', f'-DTORCH_EXTENSION_NAME={NAME}',
             '-DTORCH_API_INCLUDE_EXTENSION_H',
             f'-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}',
             *_get_pybind11_abi_build_flags()]
    commands = [
      ['g++', *flags, *inc, '-c',str(SOURCE/'binding.cpp'),'-o',str(directory/'binding.o')],
      [str(cuda/'bin/nvcc'),'-O3','-std=c++17','-lineinfo','--expt-relaxed-constexpr',
       '-Xcompiler','-fPIC','-gencode','arch=compute_89,code=sm_89',
       f'-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}',
       *inc,'-c',str(SOURCE/'kernel.cu'),'-o',str(directory/'kernel.o')],
      ['g++','-shared',str(directory/'binding.o'),str(directory/'kernel.o'),
       *[arg for path in library_paths(device_type='cuda') for arg in ('-L',path)],
       '-lc10','-ltorch','-ltorch_cpu','-ltorch_python','-lc10_cuda','-ltorch_cuda',
       '-lcudart', '-o',str(module_path())]
    ]
    (directory/'commands.json').write_text(json.dumps(dict(
        commands=commands,torch=torch.__version__,torch_cuda=torch.version.cuda,
        cuda_home=str(cuda),arch='8.9',note='Manual existing-toolchain build; Ninja missing. CUDA12.4/Torch12.9 minor version difference.'),indent=2))
    for command in commands:
        print(command, flush=True)
        subprocess.run(command,check=True)
    required = ['experiments/cdb_exact_qknorm/binding.cpp',
                'experiments/cdb_exact_qknorm/kernel.cu',
                'csrc/cub_helpers.h','csrc/quantization/vectorization_utils.cuh',
                'csrc/quantization/vectorization.cuh']
    root = SOURCE.parents[1]
    receipt = dict(format=1,module=NAME,torch=torch.__version__,
        binary_sha256=hashlib.sha256(module_path().read_bytes()).hexdigest(),
        sources={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in required})
    module_path().with_suffix('.json').write_text(json.dumps(receipt,indent=2))
    return load_built()

if __name__ == '__main__':
    print(build().__file__)
