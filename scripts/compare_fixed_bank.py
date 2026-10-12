#!/usr/bin/env python3
"""Evaluate three encodings on an identical supplied bank and fact inventory."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from spectree.core.codec import parse_bank
from spectree.experiments.fixed_bank import compare_encodings
from spectree.experiments.io import read_jsonl,write_jsonl
from spectree.selector.demo import toy_complete_chat_counter


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True,
                   help='JSONL rows: question_id, question, candidates, optional_facts')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--backend',choices=['demo','hf'],default='demo')
    p.add_argument('--model',default='Qwen/Qwen3.5-9B')
    p.add_argument('--input-cap',type=int,default=2048)
    args=p.parse_args()
    if args.backend=='hf':
        from spectree.selector.hf import FrozenHFSelector
        reader=FrozenHFSelector(args.model)
        count,select=reader.count_complete_chat,reader.select
    else:
        from spectree.demo import demo_selector
        count,select=toy_complete_chat_counter,demo_selector
    output=[]
    for row in read_jsonl(args.input):
        bank=sorted(parse_bank(row['candidates']),key=lambda c:-c.score)
        result=compare_encodings(row['question'],bank,row.get('optional_facts',[]),
                                 input_cap=args.input_cap,count_complete_chat=count,selector=select)
        output.append({'question_id':row['question_id'],**result})
    write_jsonl(args.output,output)
    print(json.dumps({'questions':len(output),'file':str(args.output)},indent=2))

if __name__=='__main__':main()
