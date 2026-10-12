# SpecTree

**Incremental Shared Hypothesis Compilation and Overlap-Aware Budget Reinvestment for Knowledge Graph Question Answering.**

SpecTree uses a small graph-adapted drafter to explore executable KG paths, compiles compatible path prefixes into a candidate-preserving ordinary-text tree, and allocates remaining inference resources between acquiring new candidates and enriching evidence. A frozen language model is called once to select a candidate ID; the external answer registry resolves the associated full answer set.

## Overview

```text
Question + KG
    |                    ┌─────────────────────────────────┐
    v                    |                                 |
Grounded Drafter ---> Incremental Shared Compiler ---> Budget Scheduler
    ^                    |      prefix tree + packet       |      Expand / Enrich / Stop
    |                    |      marginal input cost        |
    └------- Expand -----┴-------- cost feedback ---------┘
                                              |
                                             Stop
                                              v
                                    Frozen LLM (one ID)
                                              |
                                    External answer registry
                                              |
                                         Answer set
```

| Paper | Implementation |
| --- | --- |
| §4.1 Grounded Hypothesis Drafting | `spectree/drafter/` |
| §4.2 Incremental Shared Hypothesis Compilation | `spectree/core/codec.py`, `incremental_codec.py` |
| §4.3 Overlap-Aware Budget Reinvestment | `spectree/core/budget_scheduler.py`, `pipeline.py` |
| §4.4 Global Candidate Selection | `spectree/selector/` |
| §5 Experiments | `spectree/experiments/`, `scripts/` |

## Install

Python 3.10+:

```bash
python -m pip install -e .
python -m pip install -e '.[test,analysis]'
# For training and local frozen LLMs:
python -m pip install -e '.[ml]'
```

## Quick start

The running example uses Mira, the University of Oxford, Oxford, the United Kingdom, the University of Toronto, Canada, and Paris. All relations are supplied by the example KG file.

```bash
spectree-demo
python scripts/run_kgqa.py --graph examples/mira_graph.json \
  --questions examples/mira_questions.jsonl --backend demo \
  --output outputs/mira_predictions.jsonl
```

For a frozen Hugging Face selector, pass `--backend hf` and `--model` (or load a local model path) together with a calibrated time profile (`--cost-profile`). For a trained graph-adapted drafter, pass `--drafter-checkpoint` to the inference runner.

## Dataset preparation

Inputs use canonical JSON/JSONL schemas documented in [docs/data_format.md](docs/data_format.md). `scripts/prepare_dataset.py` handles the common public KGQA source formats and produces consistent question records; KG triples are provided in a separate graph file.

```bash
python scripts/prepare_dataset.py --format webqsp --input path/to/WebQSP.test.json \
  --output data/processed/webqsp_test.jsonl
python scripts/prepare_training.py --graph data/graph.json \
  --questions data/processed/train.jsonl --output data/processed/train_states.jsonl
```

## Training the drafter

The backbone is a shared text embedding encoder with LoRA adaptation and intent/transition projections. Training samples specify executable actions and multi-positive supervision from training answers.

```bash
python scripts/train_drafter.py --train data/processed/train_states.jsonl \
  --checkpoint Qwen/Qwen3-Embedding-0.6B --out checkpoints/drafter --epochs 3
```

## Inference

```bash
python scripts/calibrate_reader_cost.py --model Qwen/Qwen3.5-9B \
  --inputs data/processed/tuning_packets.jsonl --out configs/reader_cost.json
python scripts/run_kgqa.py --graph data/graph.json \
  --questions data/processed/test.jsonl --backend hf \
  --model Qwen/Qwen3.5-9B --drafter-checkpoint checkpoints/drafter \
  --cost-profile configs/reader_cost.json --config configs/default.json \
  --output outputs/predictions.jsonl
python scripts/evaluate_predictions.py outputs/predictions.jsonl \
  --expected-ids data/processed/test_ids.json
```

## Experiments and analysis

| Entry point | Experiment |
| --- | --- |
| `scripts/run_factorial.py` | Flat/Dictionary/Shared × Fixed/Adaptive |
| `scripts/compare_fixed_bank.py` | Same candidate bank, evidence inventory and reader output contract |
| `scripts/overlap_analysis.py` | Initial-overlap strata and paired interaction |
| `scripts/paired_evaluate.py` | Paired bootstrap over supplied per-question scores |
| `scripts/plot_quality_latency.py` | Quality–latency operating points |
| `scripts/evaluate_predictions.py` | Set-Hit, macro-F1 and exact-match accounting |

See [docs/experiments.md](docs/experiments.md) for evaluation contracts and [docs/architecture.md](docs/architecture.md) for data flow and scheduler details.

## Tests

```bash
python -m pytest -q
```

## License

MIT. See [LICENSE](LICENSE).
