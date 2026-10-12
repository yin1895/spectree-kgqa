"""Self-contained Mira KG walkthrough."""
from __future__ import annotations
import json
from pathlib import Path
from .core.budget_scheduler import Compiler, Settings
from .core.graph_protocol import DirectedGraph
from .drafter.search import ResumableGraphSearch, toy_action_scorer
from .pipeline import PipelineConfig, SpecTreePipeline
from .selector.demo import toy_complete_chat_counter


def demo_selector(packet_text: str) -> str:
    """A fixed-rule reader for the known Mira example packet."""
    terminals = []
    for line in packet_text.splitlines()[3:]:
        try: row=json.loads(line)
        except json.JSONDecodeError: continue
        if isinstance(row, dict) and "answer" in row and "examples" in row:
            terminals.append(row)
    matching=[t for t in terminals if "United Kingdom" in t["examples"]]
    return (matching or terminals or [{}])[0].get("id", "")


def run_demo() -> dict:
    graph_path=Path(__file__).resolve().parents[3]/"examples"/"mira_graph.json"
    if graph_path.exists():
        graph_data=json.loads(graph_path.read_text())
    else:
        graph_data={"triples":[
            ["Mira","studied_at","University of Oxford"],
            ["University of Oxford","located_in","Oxford"],
            ["Oxford","in_country","United Kingdom"],
            ["University of Oxford","academic_partner","University of Toronto"],
            ["University of Toronto","in_country","Canada"],
            ["Mira","born_in","Paris"],["Paris","in_country","France"],
            ["Mira","graduated_from","University of Oxford"]]}
    graph=DirectedGraph(tuple(tuple(t) for t in graph_data["triples"]))
    question="In which country is the university that Mira attended?"
    search=ResumableGraphSearch(question,graph,"Mira",toy_action_scorer)
    cfg=Settings(token_cap=30000,time_target_ms=100000,max_trials=48)
    compiler=Compiler(question,toy_complete_chat_counter,toy_complete_chat_counter,"shared")
    pipeline=SpecTreePipeline(graph,search,compiler,cfg,demo_selector,lambda _n:1.0,
                              pcfg=PipelineConfig(bootstrap_k=3,expand_quanta=6,expand_batch_cap=3),
                              fact_relevance=lambda row: 0.8 if row["relation"]=="graduated_from" else 0.2)
    result=pipeline.run()
    return {"question":question,"answer":list(result.answers),
            "n_candidates":len(result.state.bank),"accepted_ids":[c.id for c in result.state.bank],
            "tokens":result.state.tokens,"trials":result.trials,
            "selector_calls":result.selector_calls,"stop_reason":result.stop_reason,
            "trace":result.trace}


def main():
    print(json.dumps(run_demo(),ensure_ascii=False,indent=2))

if __name__ == "__main__": main()
