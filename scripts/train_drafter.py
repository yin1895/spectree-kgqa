#!/usr/bin/env python3
"""LoRA dual-projection drafter training with multi-positive action loss."""
from __future__ import annotations
import argparse,json,random
from pathlib import Path
from spectree.experiments.io import read_jsonl


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train',type=Path,required=True)
    p.add_argument('--checkpoint',default='Qwen/Qwen3-Embedding-0.6B')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--lr',type=float,default=2e-4)
    p.add_argument('--lora-rank',type=int,default=8)
    p.add_argument('--temperature',type=float,default=.07)
    a=p.parse_args()
    import torch
    from spectree.drafter.neural import DualScoringDrafter
    torch.manual_seed(a.seed);random.seed(a.seed)
    rows=read_jsonl(a.train)
    rows=[r for r in rows if any(r['positives'])]
    if not rows:raise ValueError('No positive training states')
    model=DualScoringDrafter(a.checkpoint,lora_rank=a.lora_rank,temperature=a.temperature)
    model.train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr)
    history=[]
    for epoch in range(a.epochs):
        indices=list(range(len(rows)));random.Random(a.seed+epoch).shuffle(indices)
        epoch_losses=[]
        for i in range(0,len(indices),a.batch_size):
            batch=[rows[j] for j in indices[i:i+a.batch_size]]
            loss=model.train_step([r['state_text'] for r in batch],
                                  [r['actions'] for r in batch],
                                  [r['positives'] for r in batch],optimizer)
            epoch_losses.append(loss)
        history.append({'epoch':epoch+1,'loss':sum(epoch_losses)/len(epoch_losses)})
        print(f"epoch {epoch+1}/{a.epochs}: loss={history[-1]['loss']:.5f}")
    model.save(a.out)
    (a.out/'history.json').write_text(json.dumps({'seed':a.seed,'training_states':len(rows),
                                                  'loss':history},indent=2)+'\n')

if __name__=='__main__':main()
