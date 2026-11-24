#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build one optional CPU prefix extension without installing packages or CUDA."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import sysconfig


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--module', choices=('types', 'pack'), default='types')
    args=parser.parse_args()
    if platform.python_implementation()!='CPython' or sys.version_info[:2]!=(3,11):
        raise RuntimeError('Native prefix extensions require CPython 3.11')
    root=Path(__file__).resolve().parents[1]
    module=f'cdb_av_prefix_{args.module}'
    source=root/'csrc'/f'{module}.c'
    header=Path(sysconfig.get_path('include'))/'Python.h'
    if not header.is_file():raise RuntimeError('Existing Python.h required; no dependency installation')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    output=args.output_dir/(f'_{module}'+sysconfig.get_config_var('EXT_SUFFIX'))
    if output.exists():raise FileExistsError(output)
    command=shlex.split(sysconfig.get_config_var('CC'))+[
        '-shared','-fPIC','-O3','-std=c11','-Wall','-Wextra','-Werror',
        '-I'+str(header.parent),str(source),'-o',str(output)]
    subprocess.run(command,check=True)
    facts=dict(command=command,python=sys.version,implementation=platform.python_implementation(),
        soabi=sysconfig.get_config_var('SOABI'),source=str(source),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        python_h_sha256=hashlib.sha256(header.read_bytes()).hexdigest(),
        output=str(output.resolve()),output_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        scope=f'CPU-only CPython prefix {args.module} extension')
    (args.output_dir/'BUILD.json').write_text(json.dumps(facts,indent=2)+'\n')
    print(json.dumps(facts,indent=2))


if __name__=='__main__':main()
