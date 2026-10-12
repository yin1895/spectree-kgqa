"""Graph execution, canonical KG identity and path supervision."""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable
import hashlib
import json

Triple = tuple[str, str, str]

def key_hash(*parts: object) -> str:
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()

@dataclass
class DirectedGraph:
    triples: tuple[Triple, ...]

    def __post_init__(self) -> None:
        if any(len(t) != 3 or not all(isinstance(x, str) and x for x in t) for t in self.triples):
            raise ValueError("Every triple needs three nonempty canonical strings")
        self.triples = tuple(sorted(set(tuple(t) for t in self.triples)))
        self.forward = defaultdict(set)
        self.reverse = defaultdict(set)
        for h, r, t in self.triples:
            self.forward[h, r].add(t)
            self.reverse[t, r].add(h)
        self.sha256 = key_hash(self.triples)

    def neighbors(self, entity: str, relation: str, direction: str) -> tuple[str, ...]:
        if direction not in ("+", "-"):
            raise ValueError("Direction must be + or -")
        index = self.forward if direction == "+" else self.reverse
        return tuple(sorted(index.get((entity, relation), ())))

    def execute(self, root: str, program: Iterable[tuple[str, str]]) -> tuple[str, ...]:
        frontier = {root}
        for relation, direction in program:
            frontier = {v for e in frontier for v in self.neighbors(e, relation, direction)}
        return tuple(sorted(frontier))

    def actions(self, entity: str, visited: set[str]) -> list[dict]:
        out = []
        for h, r, t in self.triples:
            if h == entity and t not in visited:
                out.append(dict(head=h, relation=r, direction="+", tail=t))
            if t == entity and h not in visited:
                out.append(dict(head=t, relation=r, direction="-", tail=h))
        return sorted(out, key=lambda a: (a['relation'], a['direction'], a['tail']))

    def incident_facts(self, entities: Iterable[str]) -> list[dict]:
        es = set(entities)
        return [dict(head=h, relation=r, direction="+", tail=t)
                for h, r, t in self.triples if h in es or t in es]

def sample_training_paths(graph: DirectedGraph, root: str, question_id: str, seed: int,
                          max_depth: int = 4, frontier_cap: int = 256,
                          complete_cap: int = 256) -> list[tuple[dict, ...]]:
    """Outcome-independent sampling; answers enter only the label step below."""
    if min(max_depth, frontier_cap, complete_cap) <= 0:
        raise ValueError("Caps must be positive")
    active = [tuple()]
    completed = []
    for depth in range(1, max_depth + 1):
        successors = []
        for path in active:
            current = path[-1]['tail'] if path else root
            visited = {root} | {a['tail'] for a in path}
            successors.extend(path + (a,) for a in graph.actions(current, visited))
        successors.sort(key=lambda p: key_hash(seed, question_id, depth, p))
        active = successors[:frontier_cap]
        completed.extend(active)
        if not active:
            break
    completed.sort(key=lambda path: key_hash(seed, question_id, 'complete', path))
    return completed[:complete_cap]

def label_paths(graph: DirectedGraph, root: str, paths: list[tuple[dict, ...]],
                training_answers: Iterable[str], max_positive: int = 32) -> dict:
    gold = set(training_answers)
    supported, distractors = [], []
    for path in paths:
        answers = graph.execute(root, [(a['relation'], a['direction']) for a in path])
        (supported if gold.intersection(answers) else distractors).append(path)
    return {'supported': supported[:max_positive], 'distractors': distractors}

def select_optional_facts(graph: DirectedGraph, answer_examples: Iterable[str],
                          relation_scores: dict[tuple[str, str], float], limit: int = 64) -> list[dict]:
    """Scores are question-conditioned; this function never accepts gold answers."""
    if limit < 0:
        raise ValueError("limit must be nonnegative")
    facts = graph.incident_facts(answer_examples)
    facts.sort(key=lambda a: (-relation_scores.get((a['relation'], a['direction']), 0.0),
                             a['head'], a['relation'], a['direction'], a['tail']))
    return facts[:limit]
