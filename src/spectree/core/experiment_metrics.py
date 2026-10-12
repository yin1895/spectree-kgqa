"""Explicit evaluation units for answer sets and auxiliary path diagnostics.

No function invents references, labels, or model scores. Callers supply a frozen
reference inventory; unannotated questions are reported separately.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Iterable, Mapping, Sequence, Hashable
import hashlib, math


def answer_metrics(predictions: Mapping[str, Sequence[str]], references: Mapping[str, Sequence[str]]) -> dict:
    if set(predictions) - set(references):
        raise ValueError('Prediction IDs outside the evaluation population')
    if not references:
        raise ValueError('Empty evaluation population')
    hits, total = 0, 0.0
    for q, gold in references.items():
        gold, pred = set(gold), set(predictions.get(q, ()))
        if not gold:
            raise ValueError('Reference answer sets must be nonempty')
        overlap = len(gold & pred)
        hits += int(overlap > 0)
        total += 2 * overlap / (len(pred) + len(gold))
    n = len(references)
    return {'n_questions': n, 'hit_count': hits, 'f1_sum': total,
            'hit': 100 * hits / n, 'f1': 100 * total / n}


def next_relation_accuracy(states: Iterable[dict], population_ids: Iterable[str]) -> dict:
    population = set(population_ids)
    by_q = defaultdict(list)
    seen = set()
    for row in states:
        q = row['question_id']; state_id = (q, row['state_id'])
        if q not in population or state_id in seen:
            raise ValueError('Unknown question or duplicate diagnostic state')
        seen.add(state_id)
        allowed = {tuple(r) for r in row['outgoing_relations']}
        gold = {tuple(r) for r in row['reference_next_relations']}
        if not allowed or not gold or not gold <= allowed:
            raise ValueError('Every eligible state requires executable reference relations')
        scores = {tuple(x[:2]): float(x[2]) for x in row['scores']}
        if set(scores) != allowed or any(not math.isfinite(x) for x in scores.values()):
            raise ValueError('Scores must cover all outgoing types before shortlisting')
        if any(d not in ('+', '-') for _, d in allowed):
            raise ValueError('Directed relation must explicitly use + or -')
        best = min(allowed, key=lambda x: (-scores[x], x))
        by_q[q].append(float(best in gold))
    n = len(by_q)
    return {'eligible_questions': n, 'eligible_states': len(seen),
            'unannotated_questions': len(population) - n,
            'accuracy': 100 * sum(sum(v) / len(v) for v in by_q.values()) / n if n else None}


def canonical_program(program: dict) -> tuple:
    path = tuple((str(r), str(d)) for r, d in program['relations'])
    if any(d not in ('+', '-') for _, d in path):
        raise ValueError('Unknown relation direction')
    return str(program['root']), path


def plan_recall(predicted: Mapping[str, Sequence[dict]], reference: Mapping[str, Sequence[dict]],
                population_ids: Iterable[str], k: int = 16) -> dict:
    if k < 1:
        raise ValueError('k must be positive')
    population = set(population_ids)
    if set(predicted) - population or set(reference) - population:
        raise ValueError('Inventory outside the declared population')
    eligible = matched = 0
    for q in sorted(population):
        refs = {canonical_program(r) for r in reference.get(q, ()) if r.get('path_only', True)}
        if not refs:
            continue
        eligible += 1
        cand = {canonical_program(r) for r in predicted.get(q, ())[:k]}
        matched += bool(cand & refs)
    return {'population': len(population), 'eligible': eligible, 'unannotated': len(population)-eligible,
            'matched': matched, 'recall': 100*matched/eligible if eligible else None}


def fixed_sample(rows: Sequence[dict], quotas: Mapping[str, int], seed: int = 42) -> list[str]:
    """GrailQA-like stratified sample; only IDs and level labels are inspected."""
    groups = defaultdict(list); seen = set()
    for r in rows:
        q = str(r['question_id']); level = r['level']
        if q in seen:
            raise ValueError('Duplicate question ID')
        seen.add(q); groups[level].append(q)
    out = []
    for label, n in quotas.items():
        if n < 0 or len(groups[label]) < n:
            raise ValueError('Insufficient stratum for requested quota')
        ranked = sorted(groups[label], key=lambda q: (hashlib.sha256(f'{seed}:{q}'.encode()).hexdigest(), q))
        out.extend(ranked[:n])
    return out


def hash_split(ids: Sequence[str], seed: int = 42) -> dict[str, list[str]]:
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate input IDs')
    ids = sorted(ids, key=lambda q: (hashlib.sha256(f'{seed}:{q}'.encode()).hexdigest(), q))
    n = len(ids); a, b = n*8//10, n//10
    return {'train': ids[:a], 'tune': ids[a:a+b], 'evaluation': ids[a+b:]}
