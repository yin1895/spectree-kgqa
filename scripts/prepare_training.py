#!/usr/bin/env python3
"""Create multi-positive graph action supervision from training split questions."""
from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
from collections import defaultdict
from spectree.experiments.io import read_jsonl,load_graph,write_jsonl
from spectree.core.graph_protocol import sample_training_paths,label_paths
from spectree.core.codec import canonical
from spectree.drafter.search import FrontierState,STOP
from spectree.drafter.adapter import state_to_text,action_to_text


def build(graph, questions,seed=42,max_depth=4,frontier_cap=256,complete_cap=256,
          max_actions=32):
    training=[]
    for row in questions:
        if row.get('split') in ('test','evaluation'):
            raise ValueError('Training supervision must use training split questions')
        qid,question,roots,answers=(row['question_id'],row['question'],row['topic_entities'],row['answers'])
        for root in roots:
            paths=sample_training_paths(graph,root,qid,seed,max_depth,frontier_cap,complete_cap)
            relevant=label_paths(graph,root,paths,answers)['supported']
            by_prefix=defaultdict(set)
            saved={}
            for path in relevant:
                for i in range(len(path)+1):
                    prefix=tuple(path[:i]); key=canonical(prefix)
                    saved[key]=prefix
                    if i==len(path):by_prefix[key].add('STOP')
                    else:by_prefix[key].add(canonical(path[i]))
            for key in sorted(by_prefix):
                prefix=saved[key]
                current=prefix[-1]['tail'] if prefix else root
                visited={root}|{a['tail'] for a in prefix}
                choices=graph.actions(current,visited)
                if prefix:choices.append(STOP)
                positives=by_prefix[key]
                required=[x for x in choices if (canonical(x) if x!=STOP else 'STOP') in positives]
                others=[x for x in choices if x not in required]
                others=sorted(others,key=lambda x: hashlib.sha256(
                     f'{seed}:{qid}:{key}:{canonical(x)}'.encode()).hexdigest())
                selected=required+others[:max(0,max_actions-len(required))]
                state=FrontierState(root,current,prefix,0.0,
                                     {'graph':graph.sha256,'root':root,'executor':'relation-program-v1'})
                training.append({'question_id':qid,'state_text':state_to_text(question,state),
                                 'actions':[action_to_text(a) for a in selected],
                                 'positives':[(canonical(a) if a!=STOP else 'STOP') in positives for a in selected]})
    return training


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graph',type=Path,required=True);p.add_argument('--questions',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--max-depth',type=int,default=4)
    p.add_argument('--frontier-cap',type=int,default=256);p.add_argument('--complete-cap',type=int,default=256)
    p.add_argument('--max-actions',type=int,default=32)
    a=p.parse_args()
    rows=build(load_graph(a.graph),read_jsonl(a.questions),a.seed,a.max_depth,
               a.frontier_cap,a.complete_cap,a.max_actions)
    write_jsonl(a.output,rows)
    print(json.dumps({'training_states':len(rows),'output':str(a.output)},indent=2))

if __name__=='__main__':main()
