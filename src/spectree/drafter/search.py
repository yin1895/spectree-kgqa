"""Resumable grounded KG search with a plug-in question-conditioned action scorer.

The search owns the frontier. The scheduler decides when to resume it. STOP
produces a complete candidate without deleting longer path continuations.
This implementation is independent of the neural scoring backend.
"""
from __future__ import annotations
from dataclasses import dataclass
from heapq import heappush, heappop
from itertools import count
from math import exp, isfinite, log
from typing import Callable, Sequence

from ..core.codec import Candidate, canonical
from ..core.graph_protocol import DirectedGraph

STOP = "STOP"

@dataclass(frozen=True)
class FrontierState:
    root: str
    current: str
    path: tuple[dict, ...]
    log_score: float
    context: dict

    @property
    def depth(self) -> int:
        return len(self.path)


# Return one logit for each executable outgoing action and the optional STOP.
ActionScorer = Callable[[str, FrontierState, Sequence[dict | str]], Sequence[float]]


def _log_softmax(logits: Sequence[float]) -> list[float]:
    if not logits or any(not isfinite(x) for x in logits):
        raise ValueError("An action scorer must return finite logits")
    m = max(logits)
    logz = m + log(sum(exp(x - m) for x in logits))
    return [x - logz for x in logits]


class ResumableGraphSearch:
    """Best-first frontier; at most `beam_width` states survive each depth.

    `max_quanta` counts expanded frontier states, not accepted candidates.
    Candidate IDs are allocated on completion and are never reassigned.
    """
    def __init__(self, question: str, graph: DirectedGraph, root: str,
                 scorer: ActionScorer, *, graph_id: str = "graph",
                 executor_id: str = "relation-program-v1", max_depth: int = 4,
                 beam_width: int = 32, relation_k: int = 8):
        if max_depth < 1 or beam_width < 1 or relation_k < 1:
            raise ValueError("Search limits must be positive")
        self.question, self.graph, self.root, self.scorer = question, graph, root, scorer
        self.max_depth, self.beam_width, self.relation_k = max_depth, beam_width, relation_k
        self.context = dict(graph=graph_id, root=root, executor=executor_id,
                            graph_sha256=graph.sha256)
        self._seq = count()
        self._next_id = count(1)
        self._queue: list[tuple[float, int, FrontierState]] = []
        self.expansion_steps = 0
        self.generated = 0
        self._push(FrontierState(root, root, (), 0., self.context))

    def _push(self, state: FrontierState) -> None:
        heappush(self._queue, (-state.log_score, next(self._seq), state))
        # Cap per depth to maintain a finite graph-side frontier.
        same = [q for q in self._queue if q[2].depth == state.depth]
        if len(same) > self.beam_width:
            worst = max(same, key=lambda x: (x[0], x[1]))
            self._queue.remove(worst)
            from heapq import heapify
            heapify(self._queue)

    @property
    def has_frontier(self) -> bool:
        return bool(self._queue)

    def frontier_summary(self) -> list[dict]:
        return [dict(depth=s.depth, log_score=s.log_score, path=list(s.path),
                     current=s.current) for _, _, s in sorted(self._queue)]

    def next_frontier(self) -> FrontierState | None:
        return self._queue[0][2] if self._queue else None

    def expand(self, quanta: int = 8, completed_cap: int = 2) -> tuple[Candidate, ...]:
        if quanta < 1 or completed_cap < 1:
            raise ValueError("Expansion budget must be positive")
        completed: list[Candidate] = []
        for _ in range(quanta):
            if not self._queue or len(completed) >= completed_cap:
                break
            _, _, state = heappop(self._queue)
            self.expansion_steps += 1
            visited = {state.root} | {a["tail"] for a in state.path}
            actions: list[dict | str] = []
            if state.depth < self.max_depth:
                out = self.graph.actions(state.current, visited)
                # Rank directed relation types with the intent tower, then
                # evaluate grounded successors with the transition tower.
                relation_types = sorted({(a["relation"], a["direction"]) for a in out})
                if len(relation_types) > self.relation_k:
                    rank_relations = getattr(self.scorer, "rank_relations", None)
                    if callable(rank_relations):
                        scores = list(rank_relations(self.question, state, relation_types))
                        if len(scores) != len(relation_types):
                            raise ValueError("Relation scorer returned a different length")
                        ranked = sorted(zip(relation_types, scores), key=lambda x:(-x[1], x[0]))
                        allowed = {r for r, _ in ranked[:self.relation_k]}
                    else:
                        scores = list(self.scorer(self.question, state, out))
                        if len(scores) != len(out):
                            raise ValueError("Scorer returned a different action count")
                        best = {}
                        for a, value in zip(out, scores):
                            k = (a["relation"], a["direction"])
                            best[k] = max(best.get(k, float('-inf')), value)
                        allowed = set(sorted(best, key=lambda k:(-best[k], k))[:self.relation_k])
                    out = [a for a in out if (a["relation"],a["direction"]) in allowed]
                actions.extend(out)
            if state.path:
                actions.append(STOP)
            if not actions:
                continue
            logits = list(self.scorer(self.question, state, actions))
            if len(logits) != len(actions):
                raise ValueError("Scorer returned a different action count")
            logps = _log_softmax(logits)
            ranked = sorted(zip(actions, logps), key=lambda row: (-row[1],
                            canonical(row[0])))
            for a, logp in ranked:
                if a == STOP:
                    program = [(x["relation"], x["direction"]) for x in state.path]
                    answers = self.graph.execute(state.root, program)
                    if answers:
                        c = Candidate(f"c{next(self._next_id)}", state.context,
                                      state.path, state.log_score + logp, answers)
                        c.validate()
                        completed.append(c)
                        self.generated += 1
                else:
                    self._push(FrontierState(state.root, a["tail"],
                               (*state.path, a), state.log_score + logp, state.context))
        return tuple(completed[:completed_cap])


def toy_action_scorer(question: str, state: FrontierState,
                      actions: Sequence[dict | str]) -> list[float]:
    """Hand-written scorer used by the self-contained Mira example."""
    q = question.lower()
    logits = []
    for action in actions:
        if action == STOP:
            logits.append(0.3 if state.depth >= 2 else -1.5)
            continue
        rel = action["relation"].replace("_", " ").lower()
        preferred = ("university" in q and rel == "studied at") or (
            "country" in q and rel in {"in country", "located in"})
        logits.append(1.0 if preferred else -0.2)
    return logits


class MultiRootSearch:
    """Merge rooted search frontiers while keeping globally stable candidate IDs."""
    def __init__(self, searches: Sequence[ResumableGraphSearch]):
        if not searches:
            raise ValueError("At least one linked root is required")
        self.searches = list(searches)
        self._ids = count(1)

    @property
    def has_frontier(self) -> bool:
        return any(s.has_frontier for s in self.searches)

    def next_frontier(self) -> FrontierState | None:
        choices=[s.next_frontier() for s in self.searches]
        choices=[x for x in choices if x is not None]
        return max(choices,key=lambda x:(x.log_score,-x.depth,x.root)) if choices else None

    def expand(self, quanta: int = 8, completed_cap: int = 2) -> tuple[Candidate,...]:
        if quanta < 1 or completed_cap < 1: raise ValueError("Positive limits required")
        out=[]
        for _ in range(quanta):
            choices=[s for s in self.searches if s.has_frontier]
            if not choices or len(out)>=completed_cap: break
            search=max(choices,key=lambda s:(s.next_frontier().log_score,
                                             -s.next_frontier().depth,s.root))
            for c in search.expand(1,completed_cap-len(out)):
                out.append(Candidate(f"c{next(self._ids)}",c.context,c.path,c.score,c.answers))
        return tuple(out)
