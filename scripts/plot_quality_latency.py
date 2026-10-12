#!/usr/bin/env python3
"""Plot measured F1 versus observed median latency from run summaries."""
from __future__ import annotations
import argparse,json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True,
                   help='JSON list with name, f1, and median_ms')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    records=json.loads(args.input.read_text())
    fig,ax=plt.subplots(figsize=(6.1,3.4))
    for r in records:
        x=float(r['median_ms']);y=float(r['f1'])
        ax.scatter([x],[y],s=35)
        ax.annotate(str(r['name']),(x,y),xytext=(4,5),textcoords='offset points',fontsize=8)
    ax.set(xlabel='Median end-to-end latency (ms)',ylabel='Macro set-F1 (%)')
    ax.grid(alpha=.2)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output)
    print(args.output)

if __name__=='__main__':main()
