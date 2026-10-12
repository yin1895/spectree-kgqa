#!/usr/bin/env python3
"""Record SHA-256 hashes of files used by a run."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('files',type=Path,nargs='+');p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    result={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in args.files}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
