#!/usr/bin/env python3
"""Pair official per-question F1 records and compute cluster-bootstrap intervals.

Input JSONL requires question_id, cluster_id, and official_f1 (0--100).
Supply scores emitted by the benchmark's own evaluator; this program does not
silently replace entity/alias matching with a different answer scorer.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from typing import Any
import numpy as np

def load(path: Path, score_field: str = "official_f1") -> dict[str, dict[str, Any]]:
    rows = {}
    for line_number, line in enumerate(path.read_text(encoding="utf8").splitlines(), 1):
        if not line.strip(): continue
        r = json.loads(line); q = r["question_id"]
        if not isinstance(q, str) or not q: raise ValueError(f"Invalid question ID at {line_number}")
        if q in rows: raise ValueError(f"Duplicate question ID {q} in {path}")
        score = float(r[score_field])
        if not math.isfinite(score) or not 0 <= score <= 100: raise ValueError(f"Invalid F1 for {q}")
        if not isinstance(r.get("cluster_id"), str) or not r["cluster_id"]:
            raise ValueError(f"Missing cluster_id for {q}; do not invent source-family grouping")
        r = dict(r); r["_score"] = score; rows[q] = r
    if not rows: raise ValueError("Empty record file")
    return rows

def cluster_interval(left: np.ndarray, right: np.ndarray, clusters: np.ndarray,
                     draws: int = 20000, seed: int = 20261008) -> dict[str, Any]:
    if left.ndim != 1 or left.shape != right.shape or len(clusters) != len(left):
        raise ValueError("Unpaired vector shapes")
    if len(left) == 0 or draws < 100: raise ValueError("Empty vectors or too few draws")
    _, inverse = np.unique(clusters, return_inverse=True)
    n_groups = inverse.max()+1
    if n_groups < 2: raise ValueError("At least two independent resampling clusters are required")
    sums = np.bincount(inverse, weights=left-right, minlength=n_groups)
    sizes = np.bincount(inverse, minlength=n_groups)
    rng = np.random.default_rng(seed); boot = np.empty(draws)
    for a in range(0, draws, 128):
        k = min(128, draws-a)
        ix = rng.integers(0, n_groups, size=(k, n_groups))
        boot[a:a+k] = sums[ix].sum(axis=1) / sizes[ix].sum(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {"n_questions": len(left), "n_clusters": int(n_groups), "left_mean": float(left.mean()),
            "right_mean": float(right.mean()), "delta_f1": float((left-right).mean()),
            "ci95": [float(lo), float(hi)], "bootstrap_draws": draws, "bootstrap_seed": seed,
            "estimand": "question-weighted macro-F1 difference conditional on the supplied checkpoints"}

def compare(left: dict, right: dict, draws: int, seed: int, same_bank: bool = False) -> dict:
    if left.keys() != right.keys():
        raise ValueError("Question sets differ; failures and missing outputs must remain explicitly scored")
    qids = sorted(left)
    for q in qids:
        a,b = left[q],right[q]
        if a['cluster_id'] != b['cluster_id']: raise ValueError(f"Cluster mismatch: {q}")
        if a.get('reference_hash') != b.get('reference_hash'): raise ValueError(f"Reference mismatch: {q}")
        if same_bank:
            for field in ('bank_hash','inventory_hash'):
                if not a.get(field) or a[field] != b.get(field): raise ValueError(f"{field} mismatch: {q}")
    return cluster_interval(np.array([left[q]['_score'] for q in qids]),
                            np.array([right[q]['_score'] for q in qids]),
                            np.array([left[q]['cluster_id'] for q in qids]),draws,seed)

def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--left',required=True,type=Path);p.add_argument('--right',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--expected-count',required=True,type=int)
    p.add_argument('--draws',default=20000,type=int);p.add_argument('--seed',default=20261008,type=int)
    p.add_argument('--score-field',default='official_f1');p.add_argument('--same-bank',action='store_true')
    a=p.parse_args();l=load(a.left,a.score_field);r=load(a.right,a.score_field)
    if len(l)!=a.expected_count or len(r)!=a.expected_count:
        p.error('Record count differs from the prespecified evaluation set')
    out=compare(l,r,a.draws,a.seed,a.same_bank)
    out.update(left_file=str(a.left),right_file=str(a.right),score_field=a.score_field)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__': main()
