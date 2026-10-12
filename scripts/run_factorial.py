#!/usr/bin/env python3
"""Run fixed/adaptive × flat/dictionary/shared allocation controls."""
from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path
from spectree.experiments.ablation import experiment_grid


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graph',type=Path,required=True);p.add_argument('--questions',type=Path,required=True)
    p.add_argument('--outdir',type=Path,required=True);p.add_argument('--config',default='configs/default.json')
    p.add_argument('--backend',choices=['demo','hf'],default='demo');p.add_argument('--model',default='Qwen/Qwen3.5-9B')
    p.add_argument('--drafter-checkpoint',type=Path);p.add_argument('--cost-profile',type=Path)
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    args.outdir.mkdir(parents=True,exist_ok=True)
    commands=[]
    script=Path(__file__).resolve().parent/'run_kgqa.py'
    for e in experiment_grid():
        cmd=[sys.executable,str(script),'--graph',str(args.graph),'--questions',str(args.questions),
             '--output',str(args.outdir/(e['name']+'.jsonl')),'--config',str(args.config),
             '--encoding',e['encoding'],'--allocation',e['allocation'],
             '--backend',args.backend,'--model',args.model]
        if args.drafter_checkpoint:cmd+=['--drafter-checkpoint',str(args.drafter_checkpoint)]
        if args.cost_profile:cmd+=['--cost-profile',str(args.cost_profile)]
        commands.append(cmd)
        if not args.dry_run:subprocess.run(cmd,check=True)
    (args.outdir/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
    print(json.dumps({'configuration_count':len(commands),'directory':str(args.outdir),
                      'dry_run':args.dry_run},indent=2))

if __name__=='__main__':main()
