"""Flat, Dictionary and Shared candidate-preserving encoders.

Full answer sets remain in the external registry while the selector receives
ordinary text containing candidate histories and terminal summaries.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Iterable
import hashlib
import json
import math

FORMATS = ("flat", "dictionary", "shared")

def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)

def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()

@dataclass(frozen=True)
class Candidate:
    id: str
    context: dict[str, Any]
    path: tuple[dict[str, Any], ...]
    score: float
    answers: tuple[str, ...]

    def validate(self) -> None:
        if not self.id or not isinstance(self.id, str):
            raise ValueError("Candidate ID must be a nonempty string")
        if not math.isfinite(self.score):
            raise ValueError("Non-finite score")
        for field in ("graph", "root", "executor"):
            if field not in self.context:
                raise ValueError(f"Missing context field: {field}")
        if not self.path:
            raise ValueError("Primary witness candidates require at least one edge")
        node = self.context["root"]
        for action in self.path:
            if not {"head", "relation", "direction", "tail"}.issubset(action):
                raise ValueError("Incomplete grounded action")
            if action["head"] != node or action["direction"] not in ("+", "-"):
                raise ValueError("Discontinuous witness or invalid direction")
            node = action["tail"]
        if len(self.answers) != len(set(self.answers)) or not all(isinstance(a, str) for a in self.answers):
            raise ValueError("Answers must be unique canonical string IDs")
        canonical(self.context); canonical(self.path)

    def record(self) -> dict[str, Any]:
        return {"id": self.id, "context": self.context, "path": list(self.path),
                "score": self.score, "answers": list(self.answers)}

def parse_bank(rows: Iterable[dict[str, Any]]) -> list[Candidate]:
    bank = [Candidate(str(r["id"]), dict(r["context"]), tuple(r["path"]),
                      float(r["score"]), tuple(r["answers"])) for r in rows]
    for c in bank: c.validate()
    if len({c.id for c in bank}) != len(bank):
        raise ValueError("Duplicate candidate IDs; duplicate paths may have DISTINCT IDs")
    return bank

def encode(bank: list[Candidate], fmt: str, examples: int = 2) -> tuple[dict, dict]:
    if fmt not in FORMATS or examples < 0:
        raise ValueError("Unknown format or negative example count")
    for c in bank: c.validate()
    if len({c.id for c in bank}) != len(bank): raise ValueError("Duplicate terminal ID")
    packet: dict[str, Any] = {"format": fmt, "namespaces": [], "records": [], "terminals": []}
    registry: dict[str, Any] = {}
    namespaces: dict[str, int] = {}; facts: dict[str, str] = {}; prefixes: dict[str, str] = {}
    for c in bank:
        context_key = canonical(c.context)
        if context_key not in namespaces:
            namespaces[context_key] = len(namespaces)
            packet["namespaces"].append(c.context)
        ns = namespaces[context_key]
        terminal: dict[str, Any] = {"id": c.id, "ns": ns, "answer": f"answer:{c.id}",
                                    "count": len(c.answers), "examples": sorted(c.answers)[:examples]}
        if fmt == "flat":
            terminal["path"] = list(c.path)
        elif fmt == "dictionary":
            chain = []
            for action in c.path:
                key = canonical([ns, action])
                if key not in facts:
                    rid = f"f{len(facts)}"; facts[key] = rid
                    packet["records"].append({"id": rid, "ns": ns, "action": action})
                chain.append(facts[key])
            terminal["chain"] = chain
        else:
            parent = None
            for action in c.path:
                # The parent ID identifies the COMPLETE prior execution history.
                key = canonical([ns, parent, action])
                if key not in prefixes:
                    rid = f"n{len(prefixes)}"; prefixes[key] = rid
                    packet["records"].append({"id": rid, "ns": ns, "parent": parent, "action": action})
                parent = prefixes[key]
            terminal["node"] = parent
        packet["terminals"].append(terminal)
        registry[c.id] = {"context": c.context, "score": c.score, "answers": list(c.answers),
                          "answer": terminal["answer"]}
    return packet, registry

def recover(packet: dict, registry: dict) -> list[Candidate]:
    fmt = packet["format"]
    if fmt not in FORMATS: raise ValueError("Unknown format")
    records = {r["id"]: r for r in packet["records"]}
    if len(records) != len(packet["records"]): raise ValueError("Duplicate node/record IDs")
    out = []
    for t in packet["terminals"]:
        r = registry[t["id"]]; ns = t["ns"]
        if packet["namespaces"][ns] != r["context"] or t["answer"] != r["answer"]:
            raise ValueError("Namespace or answer-handle mismatch")
        if fmt == "flat": path = t["path"]
        elif fmt == "dictionary":
            if any(records[k]["ns"] != ns for k in t["chain"]): raise ValueError("Cross-namespace reference")
            path = [records[k]["action"] for k in t["chain"]]
        else:
            path = []; node = t["node"]; seen = set()
            while node is not None:
                if node in seen: raise ValueError("Cyclic parent references")
                seen.add(node); rec = records[node]
                if rec["ns"] != ns: raise ValueError("Cross-namespace parent")
                path.append(rec["action"]); node = rec["parent"]
            path.reverse()
        out.append(Candidate(t["id"], r["context"], tuple(path), r["score"], tuple(r["answers"])))
    for c in out: c.validate()
    return out

def render_text(question: str, packet: dict, evidence: Iterable[dict] = ()) -> str:
    """Readable JSON-line records are ordinary text. Answers remain external."""
    lines = ["Select one listed candidate ID that best answers Q. Return only the ID.",
             "Q: " + question, "Namespaces: " + canonical(packet["namespaces"])]
    lines.extend(canonical(r) for r in packet["records"])
    lines.extend(canonical(t) for t in packet["terminals"])
    lines.extend("Evidence: " + canonical(e) for e in evidence)
    return "\n".join(lines)

def pack(question: str, bank: list[Candidate], fmt: str, evidence: list[dict], cap: int,
         count_complete_chat: Callable[[str], int], examples: int = 2) -> dict:
    """Whole-block admission on a given feasible bank.

    count_complete_chat MUST apply the deployed reader tokenizer and full chat
    template. No approximate word/character counter is substituted by this API.
    Evidence must already be deduplicated and gold-independently prioritized.
    """
    if cap < 1: raise ValueError("Input cap must be positive")
    packet, registry = encode(bank, fmt, examples)
    text = render_text(question, packet)
    if count_complete_chat(text) > cap:
        raise ValueError("Mandatory overflow: first select a common feasible bank")
    kept = []
    for block in evidence:
        trial = render_text(question, packet, sorted([*kept, block], key=canonical))
        if count_complete_chat(trial) <= cap:
            kept.append(block); text = trial
    return {"packet": packet, "registry": registry, "text": text, "evidence": sorted(kept, key=canonical),
            "input_tokens": count_complete_chat(text),
            "bank_hash": digest([c.record() for c in bank]),
            "inventory_hash": digest(evidence)}

def common_feasible_bank(question: str, bank: list[Candidate], cap: int,
                         count_complete_chat: Callable[[str], int]) -> list[Candidate]:
    """Remove the same lowest-ranked terminal for ALL encodings; never redraft."""
    chosen = list(bank)
    if any(chosen[i].score < chosen[i+1].score for i in range(len(chosen)-1)):
        raise ValueError("Supply the already ranked bank in descending score order")
    while chosen:
        if all(count_complete_chat(render_text(question, encode(chosen, f)[0])) <= cap for f in FORMATS):
            return chosen
        chosen.pop()
    return []
