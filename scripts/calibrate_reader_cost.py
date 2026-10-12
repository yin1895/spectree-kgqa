#!/usr/bin/env python3
"""Profile complete frozen-reader calls and build monotone input-length bins."""
from __future__ import annotations
import argparse,json,math,statistics
from pathlib import Path
from time import perf_counter
from spectree.selector.hf import FrozenHFSelector
from spectree.experiments.io import read_jsonl


def aggregate(rows,bin_size=256):
    bins={}
    for tokens,ms in rows:
        bound=int(math.ceil(tokens/bin_size)*bin_size)
        bins.setdefault(bound,[]).append(ms)
    limits=sorted(bins)
    maximums=[];running=0.0
    for k in limits:
        running=max(running,max(bins[k]));maximums.append(round(running,3))
    return {'upper_token_bounds':limits,'conservative_ms':maximums,
            'n_profile_calls':len(rows),'bin_size':bin_size}


def synchronize(torch):
    if torch.cuda.is_available(): torch.cuda.synchronize()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True);p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--repeats',type=int,default=3);p.add_argument('--warmup',type=int,default=2)
    p.add_argument('--bin-size',type=int,default=256)
    a=p.parse_args()
    if min(a.repeats,a.bin_size)<1 or a.warmup<0:raise ValueError('Invalid sampling options')
    reader=FrozenHFSelector(a.model)
    rows=read_jsonl(a.inputs)
    if not rows: raise ValueError('No tuning packets')
    samples=[]
    for row in rows:
        packet=row.get('packet_text',row.get('text'))
        if not isinstance(packet,str) or not packet:raise ValueError('Missing packet_text/text')
        for _ in range(a.warmup): reader.select(packet)
        for _ in range(a.repeats):
            synchronize(reader.torch)
            t=perf_counter();reader.select(packet);synchronize(reader.torch)
            ms=(perf_counter()-t)*1000
            samples.append((reader.count_complete_chat(packet),ms))
    output=aggregate(samples,a.bin_size)
    output['model']=a.model;output['input_file']=a.inputs.name
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))

if __name__=='__main__':main()
