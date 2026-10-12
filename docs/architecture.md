# Architecture and method mapping

The pipeline is divided into the four stages of the paper.

## 1. Grounded hypothesis drafting

A state carries graph context, grounded path, current entity, and a cumulative log score. A small encoder shares weights across two projections: intent scores directed relation types, and transition scores executable grounded continuations. A bounded frontier permits the scheduler to resume the graph search. STOP creates a completed candidate while other continuations remain available.

## 2. Incremental shared compilation

Candidates have stable IDs, a grounded witness, an execution context, a score, and the set returned by the directed-relation program. `SharedPrefixIndex` stores a path prefix only when its entire grounded execution history is compatible, and retains independent terminal records. `encode` supports Flat, Dictionary, and Shared controls. `recover` verifies candidate preservation; complete answer sets remain in an external registry. The selector receives ordinary text comprising shared records, terminal summaries, and admitted optional facts.

## 3. Overlap-aware budget reinvestment

A controller maintains the current accepted bank, optional evidence, elapsed time, and search frontier. It ranks Expand/Enrich offers by estimated incremental utility divided by estimated additional online time. A proposed update is committed only after complete-chat input length, net utility, and final call reserve checks. The prompt input cap is exact; online time uses a calibrated cost profile. The accepted tree remains immutable on rejected trial proposals. The controller ends with Stop.

## 4. Global candidate selection

Once the controller stops, one frozen model call emits one candidate terminal ID. Lookup in the external registry returns that candidate's complete set-valued denotation. Unsupported/invalid output yields an empty answer set.

## Scope of metrics

Input token counts include full chat template and instruction text; timed sections can include graph scoring, execution, trial rejection, compilation, prefill, decode, and registry lookup. Training takes place offline. Graph paths and answer sets remain separate objects: the latter may include endpoints beyond the displayed grounded path.
