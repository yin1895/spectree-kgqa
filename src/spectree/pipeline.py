"""Bind resumable KG search, shared compilation, budget control and selector.

No dataset knowledge or reference answers are accessed by this module.
Deployment supplies scoring, tokenization and cost profiles from its own setup.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import exp
from typing import Callable
from .core.budget_scheduler import (
    Settings, Compiler, State, Fact, Offer, Acquired, Outcome, run,
    expansion_gain_estimate, utility,
)
from .core.codec import canonical
from .core.graph_protocol import DirectedGraph
from .drafter.search import ResumableGraphSearch


@dataclass(frozen=True)
class PipelineConfig:
    bootstrap_k: int = 4
    expand_quanta: int = 8
    expand_batch_cap: int = 2
    fact_pool_cap: int = 64
    estimated_expand_ms: float = 12.0
    estimated_enrich_ms: float = 2.0
    estimated_new_record_tokens: int = 100


class SpecTreePipeline:
    def __init__(self, graph: DirectedGraph, search: ResumableGraphSearch,
                 compiler: Compiler, cfg: Settings, selector: Callable[[str], str],
                 final_cost_ms: Callable[[int], float], *, pcfg: PipelineConfig = PipelineConfig(),
                 fact_relevance: Callable[[dict], float] | None = None):
        self.graph, self.search, self.compiler, self.cfg = graph, search, compiler, cfg
        self.selector, self.final_cost_ms, self.pcfg = selector, final_cost_ms, pcfg
        self.fact_relevance = fact_relevance or (lambda _: 0.0)
        self._fact_pool: dict[str, Fact] = {}

    def bootstrap(self) -> tuple[State, int]:
        bank = []
        attempts = 0
        # Trial calls cost real wall time and are included in run(...started_at).
        while len(bank) < self.pcfg.bootstrap_k and attempts < self.cfg.max_trials:
            if not self.search.has_frontier: break
            attempts += 1
            proposals = self.search.expand(self.pcfg.expand_quanta,
                                           self.pcfg.expand_batch_cap)
            for c in proposals:
                if len(bank) >= min(self.pcfg.bootstrap_k, self.cfg.k_max): break
                state = self.compiler.pack([*bank, c], (), self.cfg)
                if state is not None:
                    bank.append(c)
            if not proposals and not self.search.has_frontier: break
        return State(tuple(bank)), attempts

    def _refresh_pool(self, state: State) -> None:
        examples = {a for c in state.bank for a in c.answers[:self.cfg.examples]}
        mandatory = {canonical(a) for c in state.bank for a in c.path}
        for row in self.graph.incident_facts(examples):
            if canonical(row) in mandatory: continue
            relevance = float(self.fact_relevance(row))
            if not 0 <= relevance <= 1: raise ValueError("Invalid fact relevance")
            fid = "fact-" + canonical(row)
            self._fact_pool[fid] = Fact(fid, state.bank[0].context["graph"],
                                        row["head"],row["relation"],row["tail"],relevance,row["direction"])
        best = sorted(self._fact_pool.values(), key=lambda x: (-x.relevance, x.id))
        self._fact_pool = {f.id:f for f in best[:self.pcfg.fact_pool_cap]}

    def _offers(self, state: State, *, fixed_target: int | None) -> list[Offer]:
        self._refresh_pool(state)
        offers = []
        can_expand = self.search.has_frontier and len(state.bank) < self.cfg.k_max
        if fixed_target is not None and len(state.bank) >= fixed_target:
            can_expand = False
        if can_expand:
            top = self.search.next_frontier()
            assert top is not None
            represented = {tuple(canonical(a) for a in c.path[:top.depth])
                           for c in state.bank if len(c.path) >= top.depth}
            prefix = tuple(canonical(a) for a in top.path)
            novelty = prefix not in represented
            gain = expansion_gain_estimate(top.log_score,top.depth,novelty,
                                           self.pcfg.expand_batch_cap,self.cfg)
            offers.append(Offer("expand:next-frontier", "expand", gain,
                                self.pcfg.estimated_expand_ms,
                                state.tokens+self.pcfg.estimated_new_record_tokens))
        present = {f.id for f in state.evidence}
        for f in self._fact_pool.values():
            if f.id in present: continue
            gain = utility(state.bank,(*state.evidence,f),self.cfg)-utility(
                state.bank,state.evidence,self.cfg)
            if gain > self.cfg.gain_tolerance:
                offers.append(Offer("enrich:"+f.id,"enrich",gain,
                                    self.pcfg.estimated_enrich_ms,
                                    state.tokens+self.compiler.fact_length(f)))
        if fixed_target is not None and can_expand:
            return [a for a in offers if a.kind=='expand']
        return offers

    def _acquire(self, offer: Offer, _: State) -> Acquired:
        if offer.kind == 'expand':
            return Acquired(self.search.expand(self.pcfg.expand_quanta,
                                               self.pcfg.expand_batch_cap))
        fid = offer.key.split(':',1)[1]
        return Acquired(facts=(self._fact_pool[fid],))

    def run(self, *, fixed_target: int | None = None, clock=None) -> Outcome:
        from time import perf_counter
        clock = perf_counter if clock is None else clock
        started = clock()
        initial, n_boot = self.bootstrap()
        if not initial.bank:
            return Outcome(State(), (), n_boot, 0, "empty")
        return run(initial, self.compiler, self.cfg,
                   lambda s:self._offers(s,fixed_target=fixed_target),
                   self._acquire,self.selector,self.final_cost_ms,
                   clock=clock,started_at=started,bootstrap_trials=n_boot)
