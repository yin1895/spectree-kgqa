"""Deterministic scheduling, utility and full-input compilation interfaces."""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from math import exp, isfinite
from time import perf_counter
from typing import Callable, Sequence

from .codec import Candidate, canonical, encode, recover, render_text
from .incremental_codec import SharedPrefixIndex


@dataclass(frozen=True)
class Fact:
    id: str
    graph: str
    head: str
    relation: str
    tail: str
    relevance: float
    direction: str = "+"

    def __post_init__(self) -> None:
        if not all(isinstance(x, str) and x for x in
                   (self.id, self.graph, self.head, self.relation, self.tail)) or self.direction not in ("+", "-"):
            raise ValueError("Invalid fact identity, content or direction")
        if not isfinite(self.relevance) or not 0 <= self.relevance <= 1:
            raise ValueError("Fact relevance must be finite and in [0,1]")

    def record(self) -> dict:
        return {"graph": self.graph, "head": self.head, "relation": self.relation,
                "direction": self.direction, "tail": self.tail}


@dataclass(frozen=True)
class Settings:
    k_max: int = 32
    token_cap: int = 2048
    time_target_ms: float = 550.0
    candidate_weight: float = 0.5
    exploration_floor: float = 0.25
    gain_tolerance: float = 1e-8
    time_floor_ms: float = 1.0
    max_trials: int = 128
    examples: int = 2

    def __post_init__(self) -> None:
        if self.k_max < 2 or self.token_cap < 1 or self.max_trials < 1 or self.examples < 1:
            raise ValueError("Invalid finite resource limit")
        if not 0 < self.candidate_weight < 1 or not 0 < self.exploration_floor <= 1:
            raise ValueError("Invalid utility weight")
        if not all(isfinite(x) and x > 0 for x in
                   (self.time_target_ms, self.time_floor_ms, self.gain_tolerance)):
            raise ValueError("Invalid time/gain setting")


def candidate_relevance(c: Candidate) -> float:
    c.validate()
    if c.score > 0 or not c.answers:
        raise ValueError("A proposal requires a nonpositive log-score and nonempty denotation")
    return exp(c.score / (len(c.path) + 1))


def program_key(c: Candidate) -> str:
    return canonical([c.context, [(a["relation"], a["direction"]) for a in c.path]])


def outcome_key(c: Candidate) -> str:
    return canonical([c.context["graph"], c.context["executor"], sorted(c.answers)])


def _validate_bank(bank: Sequence[Candidate], cfg: Settings) -> None:
    if len(bank) > cfg.k_max or len({c.id for c in bank}) != len(bank):
        raise ValueError("Bank exceeds capacity or repeats an ID")
    for c in bank:
        candidate_relevance(c)


def utility(bank: Sequence[Candidate], evidence: Sequence[Fact], cfg: Settings) -> float:
    """Candidate value, evidence discrimination and combined utility."""
    _validate_bank(bank, cfg)
    programs: dict[str, float] = {}
    outcomes: dict[str, tuple[float, str, tuple[str, ...]]] = {}
    for c in bank:
        r = candidate_relevance(c)
        pk, ak = program_key(c), outcome_key(c)
        programs[pk] = max(programs.get(pk, 0.0), r)
        old = outcomes.get(ak, (0.0, c.context["graph"], ()))
        outcomes[ak] = (max(old[0], r), c.context["graph"], tuple(sorted(c.answers)[:cfg.examples]))
    cv = (sum(programs.values()) + sum(x[0] for x in outcomes.values())) / (2 * cfg.k_max)
    dv = 0.0
    for z, zz in combinations(outcomes.values(), 2):
        best = 0.0
        for f in evidence:
            incident_z = f.graph == z[1] and bool({f.head, f.tail}.intersection(z[2]))
            incident_zz = f.graph == zz[1] and bool({f.head, f.tail}.intersection(zz[2]))
            if incident_z != incident_zz:
                best = max(best, f.relevance)
        dv += z[0] * zz[0] * best
    dv /= cfg.k_max * (cfg.k_max - 1) / 2
    return cfg.candidate_weight * cv + (1 - cfg.candidate_weight) * dv


def expansion_gain_estimate(prefix_score: float, depth: int, novelty: bool,
                            batch_cap: int, cfg: Settings) -> float:
    if not isfinite(prefix_score) or prefix_score > 0 or depth < 0 or batch_cap < 1:
        raise ValueError("Invalid frontier score/depth")
    r = exp(prefix_score / max(1, depth))
    return (cfg.candidate_weight * batch_cap * r / cfg.k_max *
            (cfg.exploration_floor + (1 - cfg.exploration_floor) * float(novelty)))


@dataclass(frozen=True)
class State:
    bank: tuple[Candidate, ...] = ()
    evidence: tuple[Fact, ...] = ()
    text: str = ""
    tokens: int = 0


class Compiler:
    """Full-chat admission oracle plus cached isolated fact lengths.

    format can be flat/dictionary/shared for controlled comparisons. append-only
    bank order makes the shared encoder's first-occurrence node IDs stable.
    The shared format maintains a persistent prefix index. Trial insertions are
    reversible; only accepted banks advance it. Full text rendering and exact
    counting remain part of the measured work.
    """
    def __init__(self, question: str, count_complete_chat: Callable[[str], int],
                 count_record_tokens: Callable[[str], int], fmt: str = "shared"):
        if fmt not in ("flat", "dictionary", "shared"):
            raise ValueError("Unknown encoding")
        self.question, self.count_chat, self.count_record, self.fmt = (
            question, count_complete_chat, count_record_tokens, fmt)
        self.length_cache: dict[str, int] = {}
        self.shared_index: SharedPrefixIndex | None = None

    def reset_index(self) -> None:
        """Start a separate allocation run, preserving only isolated record lengths."""
        self.shared_index = None

    def commit(self, state: State, cfg: Settings) -> None:
        if self.fmt != "shared":
            return
        if self.shared_index is None:
            self.shared_index = SharedPrefixIndex(cfg.examples)
        self.shared_index.commit(state.bank)

    @staticmethod
    def canonical_facts(facts: Sequence[Fact]) -> tuple[Fact, ...]:
        by_id: dict[str, str] = {}
        by_record: dict[str, Fact] = {}
        for f in facts:
            key = canonical(f.record())
            if f.id in by_id and by_id[f.id] != key:
                raise ValueError("A fact ID cannot identify different records")
            by_id[f.id] = key
            if key in by_record:
                prior = by_record[key]
                if prior.relevance != f.relevance:
                    raise ValueError("Duplicate facts require the same relevance")
                if f.id < prior.id:
                    by_record[key] = f
            else:
                by_record[key] = f
        return tuple(by_record[k] for k in sorted(by_record))

    def render(self, bank: Sequence[Candidate], facts: Sequence[Fact], cfg: Settings) -> State:
        _validate_bank(bank, cfg)
        if self.fmt == "shared":
            if self.shared_index is None:
                self.shared_index = SharedPrefixIndex(cfg.examples)
            if self.shared_index.examples != cfg.examples:
                raise ValueError("Answer-summary settings cannot change within a run")
            packet, registry = self.shared_index.trial(bank)
        else:
            packet, registry = encode(list(bank), self.fmt, cfg.examples)
        ordered = self.canonical_facts(facts)
        text = render_text(self.question, packet, [f.record() for f in ordered])
        tokens = self.count_chat(text)
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
            raise ValueError("Complete-chat counter must return a nonnegative integer")
        rebuilt = recover(packet, registry)
        if [c.record() for c in rebuilt] != [c.record() for c in bank]:
            raise AssertionError("Candidate recovery failed")
        return State(tuple(bank), ordered, text, tokens)

    def fact_length(self, f: Fact) -> int:
        key = canonical(f.record())
        if key not in self.length_cache:
            value = self.count_record("Evidence: " + key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("Record counter must return a nonnegative integer")
            self.length_cache[key] = max(1, value)
        return self.length_cache[key]

    def pack(self, bank: Sequence[Candidate], facts: Sequence[Fact], cfg: Settings) -> State | None:
        mandatory = self.render(bank, (), cfg)
        if mandatory.tokens > cfg.token_cap:
            return None
        trial = self.render(bank, facts, cfg)
        while trial.tokens > cfg.token_cap:
            v = utility(trial.bank, trial.evidence, cfg)
            def loss(f: Fact) -> tuple[float, str]:
                remaining = tuple(g for g in trial.evidence if g.id != f.id)
                return ((v - utility(trial.bank, remaining, cfg)) / self.fact_length(f),
                        canonical(f.record()))
            f = min(trial.evidence, key=loss)
            trial = self.render(bank, [g for g in trial.evidence if g.id != f.id], cfg)
        return trial

    def marginal_mandatory(self, bank: Sequence[Candidate], c: Candidate, cfg: Settings) -> int:
        return self.render([*bank, c], (), cfg).tokens - self.render(bank, (), cfg).tokens


@dataclass(frozen=True)
class Offer:
    key: str
    kind: str
    gain_estimate: float
    acquisition_ms: float
    trial_tokens_estimate: int

    def __post_init__(self) -> None:
        if self.kind not in ("expand", "enrich") or not self.key:
            raise ValueError("Invalid offer")
        if (not isfinite(self.gain_estimate) or not isfinite(self.acquisition_ms)
                or self.acquisition_ms < 0 or self.trial_tokens_estimate < 0):
            raise ValueError("Invalid forecast")


@dataclass(frozen=True)
class Acquired:
    additions: tuple[Candidate, ...] = ()
    facts: tuple[Fact, ...] = ()
    allowed_fact_ids: frozenset[str] | None = None


@dataclass
class Outcome:
    state: State
    answers: tuple[str, ...]
    trials: int
    selector_calls: int
    stop_reason: str
    trace: list[dict] = field(default_factory=list)
    selected_id: str | None = None


def run(initial: State, compiler: Compiler, cfg: Settings,
        offers: Callable[[State], Sequence[Offer]],
        acquire: Callable[[Offer, State], Acquired],
        selector: Callable[[str], str], final_cost_ms: Callable[[int], float],
        *, clock: Callable[[], float] = perf_counter,
        started_at: float | None = None, bootstrap_trials: int = 0) -> Outcome:
    """Allocate after bootstrap, using the same start clock and trial counter.

    acquire advances external frontier state even on failed admission. Returned
    hypotheses must be completed/executed; their generation is never assumed
    free. The caller includes bootstrap in started_at and bootstrap_trials.
    """
    start = clock() if started_at is None else started_at
    if bootstrap_trials < 0 or bootstrap_trials > cfg.max_trials:
        raise ValueError("Invalid bootstrap count")
    if not initial.bank:
        return Outcome(State(), (), bootstrap_trials, 0, "empty")
    compiler.reset_index()
    current = compiler.render(initial.bank, initial.evidence, cfg)
    if current.tokens > cfg.token_cap:
        raise ValueError("Bootstrap state violates input cap")
    compiler.commit(current, cfg)
    elapsed = lambda: (clock() - start) * 1000
    if not current.bank:
        return Outcome(current, (), bootstrap_trials, 0, "empty")
    def cost(n: int) -> float:
        value = final_cost_ms(n)
        if not isfinite(value) or value < 0:
            raise ValueError("Invalid final-call cost profile")
        return value
    seen: set[tuple[str, str]] = set()
    traces: list[dict] = []
    reason, trials = "trial_cap", bootstrap_trials
    while trials < cfg.max_trials:
        if elapsed() + cost(current.tokens) >= cfg.time_target_ms:
            reason = "final_reserve"; break
        state_key = canonical([[c.record() for c in current.bank], [f.record() for f in current.evidence]])
        candidates = list(offers(current))  # any offer construction time is now elapsed
        feasible = [a for a in candidates if (state_key, a.key) not in seen
                    and a.gain_estimate > 0
                    and not (a.kind == "expand" and len(current.bank) >= cfg.k_max)
                    and elapsed() + a.acquisition_ms + cost(a.trial_tokens_estimate) <= cfg.time_target_ms]
        if not feasible:
            reason = "no_offer"; break
        def priority(a: Offer) -> tuple[float, str]:
            denominator = (a.acquisition_ms + max(0.0, cost(a.trial_tokens_estimate) - cost(current.tokens))
                           + cfg.time_floor_ms)
            return (-a.gain_estimate / denominator, a.key)
        a = min(feasible, key=priority)
        seen.add((state_key, a.key)); trials += 1
        old_v = utility(current.bank, current.evidence, cfg)
        change = acquire(a, current)
        if a.kind == "enrich" and change.additions:
            raise ValueError("Enrich cannot change candidate membership")
        old = {c.id: c for c in current.bank}
        for c in change.additions:
            if c.id in old:
                raise ValueError("Accepted terminal IDs cannot be redefined or reused")
        bank = (*current.bank, *change.additions)
        keep = [f for f in current.evidence if change.allowed_fact_ids is None or f.id in change.allowed_fact_ids]
        evidence = [*keep, *change.facts]
        trial = None if len(bank) > cfg.k_max else compiler.pack(bank, evidence, cfg)
        gain = 0.0 if trial is None else utility(trial.bank, trial.evidence, cfg) - old_v
        ok = (trial is not None and gain > cfg.gain_tolerance
              and elapsed() + cost(trial.tokens) <= cfg.time_target_ms)
        if ok:
            compiler.commit(trial, cfg)
            current = trial
        traces.append({"kind": a.kind, "key": a.key, "accepted": ok,
                       "gain": gain, "elapsed_ms": elapsed(), "tokens": current.tokens,
                       "accepted_ids": [c.id for c in current.bank]})
    output = selector(current.text)  # latest committed packet, including on a cap exit
    parsed = output.strip() if isinstance(output, str) else ""
    answer_map = {c.id: c.answers for c in current.bank}
    return Outcome(current, answer_map.get(parsed, ()), trials, 1, reason, traces,
                   parsed if parsed in answer_map else None)
