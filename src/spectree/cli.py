"""Entry point for the public inference runner."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from time import perf_counter
from .core.budget_scheduler import Compiler, Settings
from .drafter.search import ResumableGraphSearch,toy_action_scorer
from .drafter.adapter import NeuralActionScorer
from .pipeline import PipelineConfig,SpecTreePipeline
from .experiments.io import load_graph,read_jsonl,write_jsonl
from .experiments.cost import BinnedFinalCost
from .selector.demo import toy_complete_chat_counter


def parse_args(args=None):
    p=argparse.ArgumentParser(description="SpecTree KGQA inference")
    p.add_argument('--graph',type=Path,required=True)
    p.add_argument('--questions',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--backend',choices=['hf','demo'],default='demo')
    p.add_argument('--model',default='Qwen/Qwen3.5-9B')
    p.add_argument('--drafter-checkpoint',type=Path)
    p.add_argument('--cost-profile',type=Path)
    p.add_argument('--config',type=Path,default=Path('configs/default.json'))
    p.add_argument('--encoding',choices=['flat','dictionary','shared'],default=None)
    p.add_argument('--allocation',choices=['fixed','adaptive'],default='adaptive')
    p.add_argument('--fixed-target',type=int,default=16)
    p.add_argument('--max-questions',type=int)
    p.add_argument('--save-banks',type=Path,
                   help='Export accepted candidate banks for encoding controls')
    p.add_argument('--save-packets',type=Path,
                   help='Export final packets for time-profile calibration')
    return p.parse_args(args)


def run_questions(a):
    config=json.loads(a.config.read_text(encoding='utf8'))
    graph=load_graph(a.graph)
    rows=read_jsonl(a.questions)
    if a.max_questions is not None:
        rows=rows[:a.max_questions]
    if a.backend=='hf':
        from .selector.hf import FrozenHFSelector
        reader=FrozenHFSelector(a.model)
        selector, counter = reader.select, reader.count_complete_chat
        record_counter = lambda text: len(reader.tokenizer(text, add_special_tokens=False)["input_ids"])
        if a.cost_profile is None: raise ValueError('Provide a calibrated --cost-profile')
        final_cost=BinnedFinalCost.load(a.cost_profile)
    else:
        from .demo import demo_selector
        selector=demo_selector
        counter=record_counter=toy_complete_chat_counter
        final_cost=lambda _n: 1.0
    if a.drafter_checkpoint:
        from .drafter.neural import DualScoringDrafter
        model=DualScoringDrafter.load(a.drafter_checkpoint)
        scorer=NeuralActionScorer(model)
    else:
        scorer=toy_action_scorer
    train_cfg=config['drafter'];sched_cfg=config['scheduling']
    outputs=[];bank_rows=[];packet_rows=[];seen=set()
    for row in rows:
        qid=str(row['question_id'])
        if qid in seen: raise ValueError('Duplicate question ID')
        seen.add(qid)
        q=row['question']; roots=row['topic_entities']
        if not roots: raise ValueError(f'{qid}: no linked roots')
        from .drafter.search import MultiRootSearch
        searches=[ResumableGraphSearch(q,graph,r,scorer,graph_id=graph.sha256,
                                      max_depth=train_cfg['max_depth'],
                                      beam_width=train_cfg['beam_width'],
                                      relation_k=train_cfg['relation_k']) for r in roots]
        search=MultiRootSearch(searches)
        cfg=Settings(k_max=sched_cfg['k_max'],token_cap=sched_cfg['token_cap'] if a.backend=='hf' else 30000,
                     time_target_ms=sched_cfg['time_target_ms'] if a.backend=='hf' else 100000,
                     candidate_weight=sched_cfg['candidate_weight'],exploration_floor=sched_cfg['exploration_floor'],
                     max_trials=sched_cfg['max_trials'])
        compiler=Compiler(q,counter,record_counter,a.encoding or sched_cfg['encoding'])
        pcfg=PipelineConfig(bootstrap_k=train_cfg['bootstrap_k'],
                           expand_quanta=train_cfg['expand_quanta'],
                           expand_batch_cap=train_cfg['expand_batch_cap'])
        pipeline=SpecTreePipeline(graph,search,compiler,cfg,selector,final_cost,pcfg=pcfg,
                                  fact_relevance=lambda fact:0.5)
        start=perf_counter()
        result=pipeline.run(fixed_target=a.fixed_target if a.allocation=='fixed' else None)
        event={"question_id":qid,"prediction":list(result.answers),
               "selected_id":result.selected_id,
               "reference":list(row.get('answers',[])),"selector_calls":result.selector_calls,
               "n_candidates":len(result.state.bank),"input_tokens":result.state.tokens,
               "elapsed_ms":(perf_counter()-start)*1000,"trials":result.trials,
               "stop_reason":result.stop_reason,"encoding":compiler.fmt,"allocation":a.allocation,
               "trace":result.trace,"graph_sha256":graph.sha256}
        outputs.append(event)
        bank_rows.append({'question_id':qid, 'question':q,
                          'candidates':[c.record() for c in result.state.bank],
                          'optional_facts':[f.record() for f in result.state.evidence]})
        packet_rows.append({'question_id':qid,'packet_text':result.state.text,
                            'input_tokens':result.state.tokens})
    write_jsonl(a.output,outputs)
    if a.save_banks: write_jsonl(a.save_banks,bank_rows)
    if a.save_packets: write_jsonl(a.save_packets,packet_rows)
    return outputs


def main(argv=None):
    a=parse_args(argv)
    out=run_questions(a)
    print(json.dumps({"questions":len(out),"file":str(a.output),
                      "answers_nonempty":sum(bool(x['prediction']) for x in out)},indent=2))

if __name__=='__main__': main()
