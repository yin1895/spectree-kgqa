import json
from pathlib import Path
from spectree.core.budget_scheduler import Compiler, Settings,State,run,Offer,Acquired
from spectree.core.graph_protocol import DirectedGraph
from spectree.drafter.search import ResumableGraphSearch,toy_action_scorer,MultiRootSearch
from spectree.demo import run_demo
from test_encoding import bank


def test_zero_offer_selection():
    cfg=Settings(token_cap=100000,time_target_ms=100000)
    compiler=Compiler('Where?', lambda text:len(text),lambda text:len(text))
    out=run(State(tuple(bank())),compiler,cfg,lambda s:[],lambda a,s:Acquired(),
            lambda text:'c1',lambda n:0)
    assert out.selected_id=='c1'
    assert out.answers==('United Kingdom',)
    assert out.selector_calls==1 and out.stop_reason=='no_offer'


def test_invalid_selector_returns_empty():
    cfg=Settings(token_cap=100000,time_target_ms=100000)
    out=run(State(tuple(bank())),Compiler('Q',len,len),cfg,lambda _:[],
            lambda a,s:Acquired(),lambda _: 'c999',lambda n:0)
    assert out.answers==() and out.selected_id is None


def test_multiroot_unique_ids():
    graph=DirectedGraph(tuple([('a','r','b'),('c','r','d')]))
    searches=[ResumableGraphSearch('q',graph,r,toy_action_scorer,max_depth=2)
              for r in ('a','c')]
    together=MultiRootSearch(searches)
    candidates=[]
    while together.has_frontier:
        candidates.extend(together.expand(3,3))
    assert len({c.id for c in candidates})==len(candidates)
    assert {c.context['root'] for c in candidates}=={'a','c'}


def test_running_demo():
    out=run_demo()
    assert out['selector_calls']==1
    assert out['n_candidates']>=1
    assert out['answer']==['United Kingdom']
