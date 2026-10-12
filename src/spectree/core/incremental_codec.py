"""Append-only shared-prefix index with reversible trial insertions."""
from __future__ import annotations
from copy import deepcopy
from typing import Sequence
from .codec import Candidate, canonical


class SharedPrefixIndex:
    def __init__(self, examples: int = 2):
        if examples < 0:
            raise ValueError('Negative answer-example count')
        self.examples = examples
        self.bank: list[Candidate] = []
        self.namespaces: list[dict] = []
        self.ns_index: dict[str, int] = {}
        self.records: list[dict] = []
        self.prefix_index: dict[str, str] = {}
        self.terminals: list[dict] = []
        self.registry: dict[str, dict] = {}

    def _mark(self) -> tuple[int, int, int]:
        return len(self.bank), len(self.namespaces), len(self.records)

    def _rollback(self, mark: tuple[int, int, int]) -> None:
        n_bank, n_ns, n_records = mark
        for rec in self.records[n_records:]:
            del self.prefix_index[canonical([rec['ns'], rec['parent'], rec['action']])]
        del self.records[n_records:]
        for c in self.bank[n_bank:]:
            del self.registry[c.id]
        del self.bank[n_bank:]
        del self.terminals[n_bank:]
        for context in self.namespaces[n_ns:]:
            del self.ns_index[canonical(context)]
        del self.namespaces[n_ns:]

    def _check_extension(self, bank: Sequence[Candidate]) -> None:
        if len(bank) < len(self.bank):
            raise ValueError('Committed candidates cannot be removed')
        if len({c.id for c in bank}) != len(bank):
            raise ValueError('Duplicate terminal ID')
        for old, new in zip(self.bank, bank):
            if old.record() != new.record():
                raise ValueError('Committed order, IDs and records are immutable')
        for c in bank[len(self.bank):]:
            c.validate()

    def _append(self, candidate: Candidate) -> None:
        c = deepcopy(candidate)
        context_key = canonical(c.context)
        if context_key not in self.ns_index:
            self.ns_index[context_key] = len(self.namespaces)
            self.namespaces.append(c.context)
        ns = self.ns_index[context_key]
        parent = None
        for action in c.path:
            key = canonical([ns, parent, action])
            if key not in self.prefix_index:
                rid = f'n{len(self.records)}'
                self.prefix_index[key] = rid
                self.records.append(dict(id=rid, ns=ns, parent=parent, action=action))
            parent = self.prefix_index[key]
        answer = f'answer:{c.id}'
        self.terminals.append(dict(id=c.id, ns=ns, answer=answer,
                                   count=len(c.answers), examples=sorted(c.answers)[:self.examples], node=parent))
        self.registry[c.id] = dict(context=c.context, score=c.score,
                                   answers=list(c.answers), answer=answer)
        self.bank.append(c)

    def export(self) -> tuple[dict, dict]:
        return deepcopy(dict(format='shared', namespaces=self.namespaces,
                             records=self.records, terminals=self.terminals)), deepcopy(self.registry)

    def trial(self, bank: Sequence[Candidate]) -> tuple[dict, dict]:
        self._check_extension(bank)
        mark = self._mark()
        try:
            for c in bank[mark[0]:]:
                self._append(c)
            return self.export()
        finally:
            self._rollback(mark)

    def commit(self, bank: Sequence[Candidate]) -> None:
        self._check_extension(bank)
        mark = self._mark()
        try:
            for c in bank[mark[0]:]:
                self._append(c)
        except Exception:
            self._rollback(mark)
            raise
