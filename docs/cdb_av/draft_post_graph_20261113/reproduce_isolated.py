#!/usr/bin/env python3
"""Copy validated drivers to a NEW result root;never overwrite experiments."""
import argparse,hashlib,json,subprocess
from pathlib import Path
ARCHIVE=Path('/home/yypan/CDB-AV/results/draft_post_graph_20261113')
W=Path('/home/yypan/CDB-AV/worktrees/cdb-draft-post-graph-20261113')
PY=Path('/home/yypan/CDB-AV/.venv/bin/python')
def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--out',type=Path,required=True)
 ap.add_argument('--mode',choices=('validate','perf'),default='validate')
 ap.add_argument('--on',choices=(0,1),type=int,default=1)
 ap.add_argument('--dry-run',action='store_true')
 args=ap.parse_args()
 if not args.out.is_absolute():ap.error('--out must be absolute')
 if args.out.exists():ap.error('refusing to overwrite existing output')
 copied={}
 for name in ('harness.py','run_model.py','run_pair.py','audit.py','analyze_model.py'):
  text=(ARCHIVE/name).read_text()
  if name in ('run_model.py','run_pair.py'):
   old="D=ROOT/'results/draft_post_graph_20261113'"
   assert text.count(old)==1
   text=text.replace(old,"D=Path(__file__).resolve().parent")
  compile(text,name,'exec');copied[name]=text
 if args.mode=='validate':
  command=[str(PY),'-u',str(args.out/'run_model.py'),'validate','8','0','24','1',str(args.on)]
 else:
  command=[str(PY),'-u',str(args.out/'run_pair.py'),'perf','8','0','24','1','0',str(args.on)]
 receipt=dict(command=command,requests=32 if args.mode=='validate' else 512,
  warmup=0 if args.mode=='validate' else 128,source=str(W),async_scheduling=False,
  generated_script_sha256={k:hashlib.sha256(v.encode()).hexdigest() for k,v in copied.items()})
 if args.dry_run:
  print(json.dumps(receipt,indent=2));return
 args.out.mkdir(parents=True,exist_ok=False)
 for name,text in copied.items():(args.out/name).write_text(text)
 (args.out/'REPRODUCE.json').write_text(json.dumps(receipt,indent=2))
 with (args.out/'controller.log').open('w') as log:
  subprocess.run(command,cwd=W,check=True,stdout=log,stderr=subprocess.STDOUT)
 if args.mode=='validate':
  target=args.out/f'model_validate_post{args.on}'
  with (args.out/'audit.log').open('w') as log:
   subprocess.run([str(PY),str(args.out/'audit.py'),str(target)],check=True,stdout=log,stderr=subprocess.STDOUT)
  if args.on:
   with (args.out/'input_audit.log').open('w') as log:
    subprocess.run([str(PY),str(args.out/'analyze_model.py')],check=True,stdout=log,stderr=subprocess.STDOUT)
 print('Run complete;own-service release evidence is in CLEANUP.json:',args.out)
if __name__=='__main__':main()
