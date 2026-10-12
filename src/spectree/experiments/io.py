"""Strict, dataset-agnostic JSON/JSONL readers for canonical KGQA records."""
from __future__ import annotations
from pathlib import Path
import json
from collections.abc import Iterable
from ..core.graph_protocol import DirectedGraph
from ..core.codec import parse_bank


def read_jsonl(path: str | Path) -> list[dict]:
    rows=[]
    with Path(path).open(encoding='utf-8') as handle:
        for ln,line in enumerate(handle,1):
            if not line.strip(): continue
            row=json.loads(line)
            if not isinstance(row,dict): raise ValueError(f'Line {ln} is not an object')
            rows.append(row)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n')


def load_graph(path: str | Path) -> DirectedGraph:
    data=json.loads(Path(path).read_text(encoding='utf-8'))
    triples=data['triples'] if isinstance(data,dict) else data
    return DirectedGraph(tuple(tuple(t) for t in triples))


def load_candidate_bank(path: str | Path) -> dict[str,list]:
    rows=read_jsonl(path)
    ids=set();result={}
    for row in rows:
        q=row['question_id']
        if q in ids: raise ValueError(f'Duplicate question_id {q}')
        ids.add(q)
        result[q]=parse_bank(row['candidates'])
    return result


def assert_disjoint_splits(manifests: dict[str,list[str]]) -> None:
    seen={}
    for split,ids in manifests.items():
        for q in ids:
            if q in seen: raise ValueError(f'Question {q} in {seen[q]} and {split}')
            seen[q]=split
