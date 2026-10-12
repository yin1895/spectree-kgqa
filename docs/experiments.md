# Experiment protocols

## Main KGQA comparison

Use question-aligned evaluation populations; metrics are any-correct set-Hit and macro set-F1, with all questions (including failed/empty predictions) in the denominator. Model checkpoints, graph snapshots and benchmark-specific normalization are recorded with each run.

## Crossed compilation/scheduling controls

Three encodings (`flat`, `dictionary`, `shared`) crossed with two allocation modes (`fixed`, `adaptive`). Hold the question, graph, reader, drafter seed, controller utility and resource constraints constant within a comparison. Candidate bank membership may differ after acquisition.

## Fixed-bank encoding

Use the same bank, full answer registry, optional-fact inventory and one-ID reader output contract. In full-evidence conditions, all facts remain visible; in the input-capped condition, each format admits evidence under its own representation cost. The `common_feasible_bank` step chooses a candidate subset fitting all compared formats.

## Quality–cost measurement

Time online operations from beginning of graph search to final registry lookup, including rejected search attempts and prompt tokenization. Distinguish mean of disjoint stage timers from median/p90 of whole-query time. Use hardware- and reader-matched final-call cost profiles for adaptive allocation.

## Seeds, intervals, and controls

Specify drafter training seeds (default 17, 42, 73). Use seed-matched banks for paired comparisons and source question clusters for CWQ resampling where labels are available. Never treat multiple timing repeats of one question as independent test questions. `paired_evaluate.py` reads actual per-question scores.

## Split discipline

Store train, tuning and evaluation question manifests. GrailQA public development stratification uses the official IID, compositional and zero-shot split labels; an example quota is 200/250/550. Train/supervision labels must not be computed from evaluation answers.
