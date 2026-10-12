"""Deterministic toy selector for testing packet/registry plumbing ONLY."""


def fixed_id_selector(candidate_id: str):
    def select(text: str) -> str:
        return candidate_id
    return select


def toy_complete_chat_counter(text: str) -> int:
    """Lightweight input-length counter for the standalone Mira example."""
    return len(text.encode("utf-8")) + 23
