"""Controlled Flat / Dictionary / Shared comparison on identical candidate banks.

This module evaluates supplied selectors; no purported measured table values are
bundled. A selector must be a real frozen reader for paper comparisons.
"""
from __future__ import annotations
from collections.abc import Callable
from ..core.codec import Candidate, FORMATS, common_feasible_bank, pack


def compare_encodings(question: str, ranked_bank: list[Candidate], facts: list[dict],
                      *, input_cap: int, count_complete_chat: Callable[[str],int],
                      selector: Callable[[str],str], examples: int = 2) -> dict:
    common=common_feasible_bank(question,ranked_bank,input_cap,count_complete_chat)
    if not common:
        return {'status':'no_common_feasible_bank','results':{}}
    out={}
    for fmt in FORMATS:
        packed=pack(question,common,fmt,facts,input_cap,count_complete_chat,examples)
        selected=selector(packed['text']).strip()
        answers=packed['registry'].get(selected,{}).get('answers',[])
        out[fmt]={
            'candidate_ids':[c.id for c in common],
            'selected_id':selected,'answers':answers,
            'input_tokens':packed['input_tokens'],
            'facts_kept':len(packed['evidence']), 'facts_pool':len(facts),
            'bank_hash':packed['bank_hash'], 'inventory_hash':packed['inventory_hash'],
        }
    hashes={(v['bank_hash'],v['inventory_hash']) for v in out.values()}
    if len(hashes)!=1: raise AssertionError('Fixed-bank contract broken')
    return {'status':'ok','results':out}
