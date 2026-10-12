#!/usr/bin/env python3
"""Compute bootstrap prefix overlap and encoding–allocation interactions by stratum."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from spectree.core.codec import parse_bank,encode
from spectree.experiments.io import read_jsonl,write_jsonl
from spectree.experiments.ablation import interaction_f1


def overlap(bank):
    if not bank:return 0.0
    flat=sum(len(c.path) for c in bank)
    packet,_=encode(bank,'shared')
    return 1-len(packet['records'])/flat if flat else 0.0


def strata(value):
    return 'low' if value<.25 else ('medium' if value<.5 else 'high')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True,
                   help='JSONL: question_id, bootstrap_bank; optionally scores map')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    rows=read_jsonl(args.input)
    out=[];groups={k:[] for k in ('low','medium','high')}
    for r in rows:
        rho=overlap(parse_bank(r['bootstrap_bank']))
        scores=r.get('scores',{})
        contribution=interaction_f1(scores) if scores else None
        item={'question_id':r['question_id'],'overlap':rho,
              'group':strata(rho),'interaction_f1':contribution}
        out.append(item);groups[item['group']].append(item)
    write_jsonl(args.output,out)
    summary={g:{'n':len(v), 'mean_interaction_f1':
                (sum(x['interaction_f1'] for x in v)/len(v)
                 if v and all(x['interaction_f1'] is not None for x in v) else None)}
             for g,v in groups.items()}
    path=args.output.with_suffix('.summary.json');path.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
