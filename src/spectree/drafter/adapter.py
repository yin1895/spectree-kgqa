"""Text serialization and inference adapter for the dual-scoring drafter."""
from __future__ import annotations
from typing import Sequence
from .search import FrontierState, STOP


def state_to_text(question: str, state: FrontierState) -> str:
    history = " ; ".join(f"{a['head']} --{a['relation']}:{a['direction']}--> {a['tail']}"
                         for a in state.path) or "(root)"
    return f"Question: {question}\nRoot: {state.root}\nExecuted path: {history}\nCurrent: {state.current}"


def action_to_text(action: dict | str) -> str:
    if action == STOP: return "STOP"
    return f"{action['head']} --{action['relation']}:{action['direction']}--> {action['tail']}"


class NeuralActionScorer:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def __call__(self, question: str, state: FrontierState,
                 actions: Sequence[dict | str]) -> list[float]:
        from torch import inference_mode
        if not actions: return []
        with inference_mode():
            logits = self.model.action_logits(state_to_text(question,state),
                                               [action_to_text(a) for a in actions])
        return [float(x) for x in logits.cpu().tolist()]

    def rank_relations(self, question: str, state: FrontierState,
                       relations: Sequence[tuple[str,str]]) -> list[float]:
        from torch import inference_mode
        with inference_mode():
            logits=self.model.action_logits(state_to_text(question,state),
                                            [f"relation: {r}, direction: {d}" for r,d in relations])
        return [float(x) for x in logits.cpu().tolist()]
