from spectree.core.graph_protocol import DirectedGraph,label_paths
from spectree.experiments.ablation import interaction_f1,experiment_grid
from spectree.experiments.cost import BinnedFinalCost
from spectree.experiments.io import assert_disjoint_splits
from spectree.experiments.fixed_bank import compare_encodings
from test_encoding import bank
import importlib.util
from pathlib import Path


def module(script):
    p=Path(__file__).resolve().parents[1]/'scripts'/script
    s=importlib.util.spec_from_file_location(script.replace('.','_'),p)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    return m


def test_crossed_controls():
    names={x['name'] for x in experiment_grid()}
    assert 'Fixed+Shared' in names and 'Adaptive+Dictionary' in names
    result=interaction_f1({'Fixed+Shared':64.7,'Adaptive+Shared':66.2,
                           'Fixed+Dictionary':63.3,'Adaptive+Dictionary':64.0})
    assert round(result,2)==0.8


def test_generic_set_scorer():
    scorer=module('score_answer_sets.py')
    out=scorer.score_records([{'question_id':'a','prediction':['x'],'reference':['x','y']},
                              {'question_id':'b','prediction':[],'reference':['z']}], {'a','b'})
    assert out['n']==2 and out['set_hit_pct']==50
    assert round(out['macro_f1_pct'],2)==33.33


def test_webqsp_adapter():
    prep=module('prepare_dataset.py')
    src={'Questions':[{'QuestionId':'q1','RawQuestion':'Where is Mira?',
                       'Parses':[{'TopicEntityMid':'mira','Answers':[{'AnswerArgument':'UK'}]}]}]}
    out=prep.convert(src,'webqsp','train')
    assert out[0]['topic_entities']==['mira'] and out[0]['answers']==['UK']


def test_disjoint_splits():
    assert_disjoint_splits({'train':['a'],'test':['b']})


def test_profile_validation():
    from tempfile import TemporaryDirectory
    import json
    with TemporaryDirectory() as d:
        p=Path(d)/'cost.json'
        p.write_text(json.dumps({'upper_token_bounds':[512,2048],'conservative_ms':[100,210]}))
        f=BinnedFinalCost.load(p)
        assert f(400)==100 and f(1700)==210


def test_same_bank_comparison():
    src=bank()
    out=compare_encodings('Which country?',src,[],input_cap=50000,
                          count_complete_chat=lambda text:len(text),selector=lambda text:'c1')
    assert out['status']=='ok' and len(out['results'])==3
    assert all(x['answers']==['United Kingdom'] for x in out['results'].values())
