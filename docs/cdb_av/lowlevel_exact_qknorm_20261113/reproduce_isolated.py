#!/usr/bin/env python3
"""Replay one isolated run into a new directory; never overwrite existing results."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
ARCHIVE=Path('/home/yypan/CDB-AV/results/exact_qknorm_20261113')
W=Path('/home/yypan/CDB-AV/worktrees/cdb-exact-qknorm-rope-20261113')
PY=Path('/home/yypan/CDB-AV/.venv/bin/python')
def main():
 p=argparse.ArgumentParser()
 p.add_argument('--out',type=Path,required=True)
 p.add_argument('--mode',choices=('validate','perf'),default='validate')
 p.add_argument('--dry-run',action='store_true')
 args=p.parse_args()
 if not args.out.is_absolute():p.error('--out must be absolute')
 if args.out.exists():p.error('refusing to overwrite an existing directory')
 files=('harness_v2.py','run_model_v2.py','run_pair_v2.py','audit.py','analyze_model_v2.py')
 copied={}
 for name in files:
  text=(ARCHIVE/name).read_text()
  if name in ('run_model_v2.py','run_pair_v2.py'):
   old="D=ROOT/'results/exact_qknorm_20261113'"
   assert text.count(old)==1
   text=text.replace(old,"D=Path(__file__).resolve().parent")
  compile(text,name,'exec');copied[name]=text
 runner='run_model_v2.py' if args.mode=='validate' else 'run_pair_v2.py'
 command=[str(PY),'-u',str(args.out/runner),args.mode,'8','0','24','1']
 if args.mode=='perf':command.append('0')
 receipt=dict(command=command,source=str(W),mode=args.mode,
  requests=32 if args.mode=='validate' else 512,
  warmup_requests=0 if args.mode=='validate' else 128,
  binary=str(ARCHIVE/'build_v2/cdb_exact_qknorm_20261113.so'),
  generated_script_sha256={k:hashlib.sha256(v.encode()).hexdigest() for k,v in copied.items()})
 if args.dry_run:
  print(json.dumps(receipt,indent=2));return
 args.out.mkdir(parents=True,exist_ok=False)
 for name,text in copied.items():(args.out/name).write_text(text)
 (args.out/'REPRODUCE.json').write_text(json.dumps(receipt,indent=2))
 with (args.out/'controller.log').open('w') as log:
  result=subprocess.run(command,cwd=W,stdout=log,stderr=subprocess.STDOUT)
 if result.returncode:raise SystemExit(result.returncode)
 if args.mode=='validate':
  target=args.out/'guard_validate8_meta0_target24_qk1'
  with (args.out/'graph_audit.log').open('w') as log:
   subprocess.run([str(PY),str(args.out/'audit.py'),str(target)],check=True,stdout=log,stderr=subprocess.STDOUT)
  with (args.out/'model_audit.log').open('w') as log:
   subprocess.run([str(PY),str(args.out/'analyze_model_v2.py'),target.name],check=True,stdout=log,stderr=subprocess.STDOUT)
 print('Finished isolated run;GPU release evidence is in its CLEANUP.json:',args.out)
if __name__=='__main__':main()
