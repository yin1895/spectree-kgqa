from spectree.core.codec import Candidate, encode, recover
from spectree.core.budget_scheduler import Settings, Compiler, State, Offer, Acquired, run
from spectree.core.graph_protocol import DirectedGraph
from spectree.experiments.ablation import experiment_grid
from test_encoding import bank
import pytest


def test_empty_bank_zero_calls():
    calls=[]
    result=run(State(),Compiler('Q',len,len),Settings(),lambda s:[],lambda a,s:Acquired(),
               lambda text:calls.append(text),lambda tokens:0)
    assert result.answers==() and result.selector_calls==0 and calls==[]


def test_accepted_candidate_persists():
    src=bank()
    cfg=Settings(token_cap=100000,time_target_ms=100000,k_max=3,max_trials=3)
    offers_called=[]
    def proposals(s):
        offers_called.append(len(s.bank))
        return [Offer('acquire-c2','expand',0.1,1,len(s.text)+100)] if len(s.bank)==1 else []
    result=run(State((src[0],)),Compiler('Q',len,len),cfg,proposals,
               lambda a,s:Acquired((src[1],)),lambda text:'c2',lambda tokens:0)
    assert result.answers==('Canada',)
    assert result.state.bank[0].record()==src[0].record()
    assert [c.id for c in result.state.bank]==['c1','c2']


def test_selector_contract_strict_id():
    cfg=Settings(token_cap=100000,time_target_ms=100000)
    result=run(State(tuple(bank())),Compiler('Q',len,len),cfg,lambda _:[],
               lambda a,s:Acquired(),lambda _: 'Chosen candidate: c1',lambda n:0)
    assert result.answers==()


def test_shared_prefix_is_grounded():
    src=bank()
    ctx=src[0].context
    alt_path=tuple([{'head':'Mira','relation':'studied_at','direction':'+','tail':'Another university'},
                    {'head':'Another university','relation':'located_in','direction':'+','tail':'Oxford'}])
    alt=Candidate('c3',ctx,alt_path,-.4,('Oxford',))
    packet,reg=encode([src[0],alt],'shared')
    assert len(packet['records'])==len(src[0].path)+len(alt.path)
    assert len(recover(packet,reg))==2


def test_graph_relation_execution_set_valued():
    graph=DirectedGraph(tuple([('Mira','studied_at','U'),('Mira','studied_at','V'),
                               ('U','located_in','A'),('V','located_in','B')]))
    result=graph.execute('Mira',[('studied_at','+'),('located_in','+')])
    assert result==('A','B')


def test_factorial_has_six_unique_cells():
    grid=experiment_grid()
    assert len(grid)==6 and len({r['name'] for r in grid})==6


def test_budget_feasibility_is_exact():
    src=bank(); cfg=Settings(token_cap=10,time_target_ms=100000)
    compiler=Compiler('Q',lambda text:len(text),lambda text:len(text))
    assert compiler.pack(src,(),cfg) is None
